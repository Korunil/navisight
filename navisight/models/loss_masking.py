# navisight/models/loss_masking.py
import torch
import torch.nn as nn
import json
import os
import logging
from navisight.pipeline.feature_registry import REGISTRY

logger = logging.getLogger(__name__)

class NumericallyStableMaskedLoss(nn.Module):
    """
    Production-grade Masked Loss Criterion.
    Enforces clean tracking filters across unique target feature indices.
    """
    def __init__(self):
        super().__init__()
        stats_path = os.path.normpath("configs/global_stats.json")
        active_indices = []
        
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    global_stats = json.load(f)
            except Exception:
                global_stats = {}
        else:
            global_stats = {}

        cohort_scales = global_stats.get("global_cohort_scales", {})
        base_cohort = cohort_scales.get("0", {})
        weather_scales = global_stats.get("weather_meso_scales", {})

        for feat in REGISTRY.maskable_for_loss:
            if feat in REGISTRY.feature_index:
                feat_idx = REGISTRY.feature_index[feat]
                scale_val = 1.0
                
                if feat in REGISTRY.weather:
                    if feat in weather_scales:
                        scale_val = weather_scales[feat].get("scale", 1.0)
                else:
                    if feat in base_cohort:
                        scale_val = base_cohort[feat].get("scale", 1.0)
                
                if scale_val < 0.01:
                    logger.info(f"Loss Registry: Excluding zero-variance feature from loss mask: {feat}")
                    continue
                    
                active_indices.append(feat_idx)
                
        if not active_indices:
            logger.warning("Loss Registry: No active features found after scaling filters. Falling back to all maskable keys.")
            active_indices = [REGISTRY.feature_index[feat] for feat in REGISTRY.maskable_for_loss]

        self.register_buffer('maskable_indices', torch.tensor(active_indices, dtype=torch.long))
        self.step_counter = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor, 
                reliability: torch.Tensor, random_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted MSE loss exclusively on masked tokens for valid tracking fields.
        Shape Parameters:
            pred:          (B, T, 35) Model reconstructions
            target:        (B, T, 35) Baseline target features
            attention_mask:(B, T)     Valid input context markers
            reliability:   (B, T)     Quality score parameters
            random_mask:   (B, T)     Active MAE mask tracks
        """
        if self.maskable_indices.device != pred.device:
            self.maskable_indices = self.maskable_indices.to(pred.device)

        # Intercept and clean non-finite value anomalies
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=2.0, neginf=-2.0)
            target = torch.nan_to_num(target, nan=0.0, posinf=2.0, neginf=-2.0)        
        
        # Isolate targeted tracking channels
        pred_targets = torch.index_select(pred, dim=-1, index=self.maskable_indices)
        feature_targets = torch.index_select(target, dim=-1, index=self.maskable_indices)

        # Enforce baseline boundary constraints
        pred_clamped = torch.clamp(pred_targets, min=-10.0, max=10.0)
        target_clamped = torch.clamp(feature_targets, min=-10.0, max=10.0)

        # Compute raw element errors
        squared_errors = (pred_clamped - target_clamped) ** 2

        # Expand mask matrices across hidden features size dimensions
        mask_expanded = random_mask.unsqueeze(-1).float()
        
        # Guard reliability parameters against negative value noise flags
        stable_reliability = torch.clamp(reliability, min=1e-5, max=1.0)
        reliability_expanded = stable_reliability.unsqueeze(-1)

        mask_sum = random_mask.sum().item()
        
        if self.step_counter < 5:
            logger.info(f"--- Loss Layer Analytics (Step {self.step_counter}) ---")
            logger.info(f"  -> Active Mask Sum (Elements) : {mask_sum}")
            logger.info(f"  -> Pred Targets Tensor Mean   : {pred_clamped.mean().item():.4f}")
            logger.info(f"  -> Feature Targets Tensor Mean: {target_clamped.mean().item():.4f}")
            self.step_counter += 1

        # ── FIXED: CRITERION RISK HANDLER GATES ───────────────────────────────
        if mask_sum < 1.0:
            # Reconstruct over all visible sequence tracking frames if the active mask is empty
            weighted_errors = squared_errors * reliability_expanded
            loss = weighted_errors.mean()
        else:
            weighted_errors = squared_errors * reliability_expanded * mask_expanded
            # Normalize step divides explicitly by the product element matrix size dimensions
            loss = weighted_errors.sum() / (mask_sum * self.maskable_indices.numel() + 1e-8)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"CRITICAL FAULT: Non-finite loss value derived inside Masked Criteria! "
                f"Raw Errors Tensor Sum Value: {squared_errors.sum().item():.4f} | "
                f"Valid Active Mask Sum Float: {mask_sum:.2f}"
            )

        return loss