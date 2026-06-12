# navisight/engine/train_model.py
import os
import random
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from navisight.pipeline.sequence_builder import ContinuityPreservingAISDataset
from navisight.models.transformer_encoder import MaritimeMAE
from navisight.models.loss_masking import NumericallyStableMaskedLoss
from navisight.pipeline.feature_registry import FEATURE_SCHEMA_HASH, generate_production_manifest_hash

logger = logging.getLogger(__name__)

def enforce_strict_system_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _compute_effective_rank(embeddings: torch.Tensor) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    try:
        sv = torch.linalg.svdvals(embeddings)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -(sv_norm * (sv_norm + 1e-10).log()).sum()
        return float(entropy.exp())
    
    except Exception:
        return 0.0

def run_pretraining_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    scaler: torch.amp.GradScaler,
    device: torch.device, 
    checkpoint_tracker: dict, 
    scheduler=None,
    amp_enabled=True,
) -> dict:
    
    model.train()
    total_loss = 0.0
    steps = 0
    cls_emb_buffer = []

    for step, batch in enumerate(train_loader):
        if checkpoint_tracker.get("resume_epoch", -1) == checkpoint_tracker.get("epoch", 0):
            continue

        features = batch["features"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        reliability = batch["reliability"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            # Pure MAE reconstruction track isolates parameters from unaugmented contrastive bugs
            reconstructed, cls_emb, _, random_mask = model(features, attn_mask, mask_ratio=0.40)
            loss = criterion(reconstructed, features, attn_mask, reliability, random_mask)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss intercepted at batch step {step}. Skipping updates.")
            # Scaler updates on skipped iterations keep the internal scale tracking variables consistent
            scaler.update()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step(checkpoint_tracker["global_step"])
        
        total_loss += loss.item()
        
        with torch.no_grad():
            cls_emb_buffer.append(cls_emb.detach().cpu())
            
        steps += 1
        checkpoint_tracker["global_step"] += 1

    if cls_emb_buffer and steps > 0:
        all_cls = torch.cat(cls_emb_buffer, dim=0).float()
        latent_var = all_cls.var(dim=0).mean().item()
        effective_rank = _compute_effective_rank(all_cls)
    else:
        latent_var, effective_rank = 0.0, 0.0

    n = max(steps, 1)
    return {
        "total_loss": total_loss / n,
        "latent_variance": latent_var,
        "effective_rank": effective_rank,
        "steps": steps,
        "physics_mse": total_loss / n
    }

def train_maritime_model(
    processed_dir: str, 
    config_json: str, 
    checkpoint_dir: str, 
    resume_from: str = None,
    file_list: list = None,
):
    if resume_from:
        print("Resuming from : ", resume_from)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    with open(config_json, "r") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    enforce_strict_system_determinism(seed)
    
    model_cfg = cfg.get("model", cfg)
    system_token = generate_production_manifest_hash(cfg["model"], cfg["policies"])

    train_dataset = ContinuityPreservingAISDataset(
        partitioned_root_dir=processed_dir, 
        manifest_json_path=config_json,
        window_size=model_cfg.get("window_size", 60), 
        stride=model_cfg.get("stride", 30),
        file_list=file_list
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["model"]["batch_size"], 
        num_workers=4, 
        pin_memory=True
    )

    model = MaritimeMAE(
        d_model=cfg["model"]["d_model"], 
        n_heads=cfg["model"]["n_heads"], 
        n_layers=cfg["model"]["n_layers"],
    ).to(device)

    criterion = NumericallyStableMaskedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["model"]["lr"], 
        weight_decay=cfg["model"].get("weight_decay", 1e-2)
    )
    
    # Batch-based scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cfg["model"].get("steps_per_cycle", 1000),
        T_mult=1,
        eta_min=cfg["model"]["lr"] * 0.01,
    )
    
    grad_scaler = torch.amp.GradScaler(enabled=amp_enabled)

    tracker = {
        "epoch": 0, 
        "global_step": 0, 
        "resume_epoch": -1, 
        "max_epochs": cfg["model"]["epochs"]
    }
    
    if resume_from and os.path.exists(resume_from):
        
        cp = torch.load(resume_from, map_location=device)
        
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        
        if "grad_scaler_state_dict" in cp:
            grad_scaler.load_state_dict(cp["grad_scaler_state_dict"])
        
        if "scheduler_state_dict" in cp:
            scheduler.load_state_dict(cp["scheduler_state_dict"])
        
        tracker["global_step"] = cp.get("global_step", 0)
        tracker["resume_epoch"] = cp["epoch"]
        
        logger.info(
            f"Resuming from epoch={tracker['resume_epoch']} "
            f"global_step={tracker['global_step']}"
        )
    
    start_epoch = tracker["resume_epoch"] + 1 if tracker["resume_epoch"] >= 0 else 1
    
    # Track tracking performance parameters cleanly across execution frames
    best_loss = float('inf')
    patience = 3  # Triggers early stopping if model stops improving for 3 epochs
    patience_counter = 0
    
    for epoch in range(start_epoch, cfg["model"]["epochs"] + 1):
        tracker["epoch"] = epoch
        metrics = run_pretraining_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            grad_scaler, 
            device, 
            tracker, 
            scheduler=scheduler,
            amp_enabled=amp_enabled,
        )
        
        if epoch == tracker["resume_epoch"]:
            tracker["resume_epoch"] = -1
        
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(
            f"Epoch {epoch:02d}/{cfg['model']['epochs']} | "
            f"LR={current_lr:.2e} | "
            f"Total Loss={metrics['total_loss']:.5f} | "
            f"Eff Rank={metrics['effective_rank']:.1f} | "
            f"Latent Var={metrics['latent_variance']:.4f}"
        )

        # ── EXPLICIT BEST-CHECKPOINT CHECKLINE ──
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            patience_counter = 0  # Reset early stopping counters
            
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            tmp_best_path = best_checkpoint_path + ".tmp"
            
            logging.info(f"🏆 Performance Milestone! Epoch {epoch} achieved lowest loss: {best_loss:.5f}. Saving best state...")
            torch.save({
                "epoch": epoch, 
                "global_step": tracker["global_step"],
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics, 
                "system_token": system_token, 
                "feature_schema_hash": FEATURE_SCHEMA_HASH,
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": grad_scaler.state_dict(),
            }, tmp_best_path)
            os.replace(tmp_best_path, best_checkpoint_path)
        else:
            patience_counter += 1
            logging.info(f"⚠️ Epoch {epoch} did not surpass current best loss baseline ({best_loss:.5f}). Patience: {patience_counter}/{patience}")
            
        # Standard epoch-wise checkpoint backup routine
        backup_filename = f"navisight_mae_checkpoint_epoch_{epoch:02d}.pt"
        backup_path = os.path.join(checkpoint_dir, backup_filename)
        torch.save({"model_state_dict": model.state_dict()}, backup_path)
        
        # Early Stopping Validation Gate
        if patience_counter >= patience:
            logging.warning(f"🛑 Early stopping triggered! Training halted at Epoch {epoch} to prevent representation degradation.")
            break