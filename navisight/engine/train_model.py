# navisight/engine/train_model.py
import os
import random
import logging
import json
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _compute_effective_rank(embeddings: torch.Tensor) -> Tuple[float, float]:
    """
    Computes sub-sampled, centered effective rank and mean pairwise cosine similarity
    safely on the CPU to eliminate multi-gigabyte LAPACK workspace explosions.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return 1.0, 1.0

    # ──► FIX 1: SUB-SAMPLE TO PREVENT 30GB MEMORY ALLOCATION ERROS ◄──
    sample_size = min(n_samples, 2000)
    indices = torch.randperm(n_samples)[:sample_size]
    sampled_embs = embeddings[indices]

    # Calculate exact pairwise cosine similarities across the sample
    # (Vectors are already L2 normalized by the encoder model output layer)
    sim_matrix = torch.matmul(sampled_embs, sampled_embs.T)
    triu_idx = torch.triu_indices(sample_size, sample_size, offset=1)
    mean_cosine = float(sim_matrix[triu_idx[0], triu_idx[1]].mean().item())

    try:
        # ──► FIX 2: CENTER MATRIX BEFORE RUNNING SVD ◄──
        # Removes the global mean shift so SVD measures true coordinate structural variance
        centered_embs = sampled_embs - sampled_embs.mean(dim=0, keepdim=True)
        
        sv = torch.linalg.svdvals(centered_embs)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -(sv_norm * (sv_norm + 1e-10).log()).sum()
        eff_rank = float(entropy.exp())
    except Exception as e:
        logger.warning(f"SVD matrix convergence exception bypassed: {e}")
        eff_rank = 1.0

    return eff_rank, mean_cosine

def compute_ntxent_loss(proj_A: torch.Tensor, proj_B: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Computes standard symmetric NT-Xent contrastive loss over dual masked views.
    Forces calculation into explicit FP32 by safely bypassing active AMP context managers.
    """
    batch_size = proj_A.size(0)
    device = proj_A.device
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ──► DISABLE AUTOCAST EXPLICITLY TO BLOCK AUTOMATIC FP16 DOWNSAMPLING ◄──
    with torch.amp.autocast(device_type=device.type, enabled=False):
        # Enforce high-precision normalization vectors
        proj_A_norm = F.normalize(proj_A.float(), p=2, dim=-1)
        proj_B_norm = F.normalize(proj_B.float(), p=2, dim=-1)

        # Concatenate views for full cross-comparison matrix extraction
        combined_projections = torch.cat([proj_A_norm, proj_B_norm], dim=0) # [2*Batch, 64]
        similarity_matrix = torch.matmul(combined_projections, combined_projections.T) / temperature # [2*Batch, 2*Batch]

        # Generate a diagonal mask to filter out trivial self-similar pairs
        diagonal_identity_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        
        # Enforce float("-inf") masking safely within a stable FP32 calculation matrix
        similarity_matrix = similarity_matrix.masked_fill(diagonal_identity_mask, float("-inf"))

        # Map target indices (sample i in View A aligns with sample i+Batch in View B)
        target_labels = torch.arange(batch_size, device=device)
        target_labels = torch.cat([target_labels + batch_size, target_labels], dim=0) # [2*Batch]

        return F.cross_entropy(similarity_matrix, target_labels)
    
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
        # superclass_id = batch["superclass_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            # ──► COHORT ALIGNMENT FIX 4: SYMMETRIC DUAL MASK PASSES ◄──
            # Pass A: Forward pass evaluating Mask Configuration A
            recon_A, cls_A, proj_A, mask_A = model(features, attn_mask, mask_ratio=0.75)
            # Pass B: Forward pass evaluating independent Mask Configuration B over the same batch
            recon_B, cls_B, proj_B, mask_B = model(features, attn_mask, mask_ratio=0.75)
            
            # Combine reconstruction losses across both views
            recon_loss = (
                criterion(recon_A, features, attn_mask, reliability, mask_A) +
                criterion(recon_B, features, attn_mask, reliability, mask_B)
            ) / 2.0
            
            # Calculate standard cross-view NT-Xent contrastive loss over the projection heads
            contrastive_loss = compute_ntxent_loss(proj_A, proj_B, temperature=0.07)
            
            # Unified Multi-task Objective Loss
            loss = recon_loss + 0.10 * contrastive_loss

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
            cls_emb_buffer.append(cls_A.detach().cpu())
            
        steps += 1
        checkpoint_tracker["global_step"] += 1

    if cls_emb_buffer and steps > 0:
        all_cls = torch.cat(cls_emb_buffer, dim=0).float()
        latent_var = all_cls.var(dim=0).mean().item()
        effective_rank, mean_cosine = _compute_effective_rank(all_cls)
    else:
        latent_var, effective_rank, mean_cosine = 0.0, 1.0, 1.0

    n = max(steps, 1)
    return {
        "total_loss": total_loss / n,
        "latent_variance": latent_var,
        "effective_rank": effective_rank,
        "mean_cosine": mean_cosine,
        "steps": steps,
        "physics_mse": total_loss / n
    }

def train_maritime_model(
    processed_dir: str, 
    config_json: str, 
    checkpoint_dir: str, 
    resume_from: str | None = None,
    file_list: list = [],
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
        T_0=cfg["model"].get("steps_per_cycle", 100000),
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
    patience = cfg['model'].get("patience", 5) # Triggers early stopping if model stops improving for 3 epochs
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
            f"Mean Cos={metrics['mean_cosine']:.4f} | "
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