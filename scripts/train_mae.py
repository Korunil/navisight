# scripts/train_mae.py
import os
import sys
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navisight.pipeline.sequence_builder import ContinuityPreservingAISDataset
from navisight.models.transformer_encoder import MaritimeMAE
from navisight.models.loss_masking import NumericallyStableMaskedLoss
from navisight.engine.train_model import enforce_strict_system_determinism, run_pretraining_epoch

def run_isolated_pretraining_pipeline():
    parser = argparse.ArgumentParser(description="Navisight AI Pretraining Launcher")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--config_json", type=str, default="configs/production_manifest.json")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--epochs", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default=None)
    args = parser.parse_args()

    enforce_strict_system_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_json, 'r') as f:
        cfg = json.load(f)

    epochs_limit = int(args.epochs) if args.epochs else cfg["model"]["epochs"]
    batch_width = int(args.batch_size) if args.batch_size else cfg["model"]["batch_size"]
    system_token = cfg["system_token"]

    dataset = ContinuityPreservingAISDataset(args.processed_dir, args.config_json)
    dataloader = DataLoader(dataset, batch_size=batch_width, num_workers=4, pin_memory=True)

    model = MaritimeMAE(d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"]).to(device)
    criterion = NumericallyStableMaskedLoss(token_floor=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["model"]["lr"], weight_decay=cfg["model"]["weight_decay"])
    gradient_scaler = torch.cuda.amp.GradScaler()

    checkpoint_tracker = {"epoch": 0, "current_step": 0, "resume_epoch": -1, "resume_step": -1}

    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint_payload = torch.load(args.resume_from, map_location=device)
        if checkpoint_payload["system_token"] != system_token:
            raise ValueError("System contract validation token breach. Configuration maps are incompatible.")
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        checkpoint_tracker["resume_epoch"] = checkpoint_payload["epoch"]
        checkpoint_tracker["resume_step"] = checkpoint_payload["step_index"]

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = max(1, checkpoint_tracker["resume_epoch"])

    for epoch in range(start_epoch, epochs_limit + 1):
        checkpoint_tracker["epoch"] = epoch
        mean_epoch_loss = run_pretraining_epoch(model, dataloader, criterion, optimizer, gradient_scaler, device, checkpoint_tracker)
        
        checkpoint_filename = f"navisight_mae_checkpoint_epoch_{epoch:02d}.pt"
        final_checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
        temporary_tmp_path = final_checkpoint_path + ".tmp"

        # FIXED: Resolves the checkpoint payload schema tracking hash configuration across updates
        current_schema_hash = checkpoint_payload.get("feature_schema_hash") if args.resume_from else str(hashlib.md5(system_token.encode()).hexdigest())

        torch.save({
            "epoch": epoch, "step_index": checkpoint_tracker["current_step"],
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "loss_metric_value": mean_epoch_loss, "system_token": system_token,
            "feature_schema_hash": current_schema_hash
        }, temporary_tmp_path)
        os.replace(temporary_tmp_path, final_checkpoint_path)

if __name__ == "__main__":
    run_isolated_pretraining_pipeline()