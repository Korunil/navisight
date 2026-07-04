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
    Production-grade masked reconstruction loss.

    Design goals:
    - Supports mixed vessel classes inside the same training mini-batch.
    - Ignores deterministic engineered derivatives to prevent algebraic shortcut learning.
    - Drops globally near-constant noise features via cross-cohort max-scale filtering.
    - Implements reliability-aware token weighting based on sensor-fault flags.
    - Custom tailored for MAE-style self-supervised target recovery.
    """
    
    DETERMINISTIC_DERIVATIVE_SET = {
        "acceleration",
        "jerk",
        "turn_rate",
        "turn_rate_speed_ratio",
        "harbor_basin_proximity",
        "distance_to_piraeus",
        "berth_zone_score",
    }

    MIN_SCALE_THRESHOLD = 0.01
    
    def __init__(self):
        super().__init__()
        
        stats_path = os.path.normpath("configs/global_stats.json")
        
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    global_stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading stats file: {e}")
                global_stats = {}
        else:
            global_stats = {}

        self.cohort_scales = global_stats.get("global_cohort_scales", {})
        self.weather_scales = global_stats.get("weather_meso_scales", {})
        
        active_indices = []

        for feat in REGISTRY.maskable_for_loss:
            
            if feat not in REGISTRY.feature_index:
                continue
            
            if feat in self.DETERMINISTIC_DERIVATIVE_SET:
                logger.info(f"Loss Balancing: Excluding engineered feature [{feat}] from reconstruction targets.")
                continue
            
            feat_idx = REGISTRY.feature_index[feat]
            
            # ------------------------------------------------------------------
            # Compute maximum variance scale observed across all cohorts.
            # If every single cohort determines that a feature's variance is near-zero,
            # it is a global constant and can be safely ignored.
            # ------------------------------------------------------------------
            
            max_scale = 0.0
            
            if feat in REGISTRY.weather:
                scale = self.weather_scales[feat].get("scale", 1.0)
                max_scale = scale
            else:
                for _, cohort_data in self.cohort_scales.items():
                    scale = cohort_data.get(feat, {}).get("scale", 0.0)
                    max_scale = max(max_scale, scale)
            
            if max_scale < self.MIN_SCALE_THRESHOLD:
                logger.info(
                    f"Loss Registry: Excluding near-constant feature from loss mask: [{feat}] "
                    f"(max_scale={max_scale:.6f})"
                )
                continue
                    
            active_indices.append(feat_idx)
                
        if not active_indices:
            logger.warning(
                "Loss Registry: No active features found after scaling filters. "
                "Falling back to all maskable keys."
            )
            active_indices = [
                REGISTRY.feature_index[feat] 
                for feat in REGISTRY.maskable_for_loss 
                if f in REGISTRY.feature_index
            ]

        self.register_buffer('maskable_indices', torch.tensor(active_indices, dtype=torch.long))
        self.step_counter = 0
        
        surviving_features = [
            feat
            for feat in REGISTRY.maskable_for_loss
            if REGISTRY.feature_index[feat] in active_indices
        ]

        logger.info(f"Loss targets retained: {len(surviving_features)}")
        logger.info(f"Retained features: {surviving_features}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention_mask: torch.Tensor,
        reliability: torch.Tensor,
        random_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes weighted MSE loss exclusively on masked tokens for primitive tracking fields.
        
        Args:
            pred:           (B, T, feature_dim) Unbounded network reconstructions
            target:         (B, T, feature_dim) Ground-truth normalized incoming primitives
            attention_mask: (B, T) Booleans tagging valid (non-padded) sequence frames
            reliability:    (B, T) Transponder reliability weights from the QA layer
            random_mask:    (B, T) Booleans tagging active MAE corruptions
        """
        if self.maskable_indices.device != pred.device:
            self.maskable_indices = self.maskable_indices.to(pred.device)

        # Protect backpropagation paths from non-finite sensor artifacts
        if not torch.isfinite(pred).all():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

        if not torch.isfinite(target).all():
            target = torch.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-10.0)

        active_idx = self.maskable_indices

        # Isolate target primitive feature coordinates via fast tensor indexing
        pred_targets = torch.index_select(pred, dim=-1, index=active_idx)
        target_targets = torch.index_select(target, dim=-1, index=active_idx)

        # Restrict clamping boundaries to match robust normalized scale bounds
        pred_targets = torch.clamp(pred_targets, min=-10.0, max=10.0)
        target_targets = torch.clamp(target_targets, min=-10.0, max=10.0)

        # Compute pure, unskewed primitive squared error profiles
        squared_errors = (pred_targets - target_targets).pow(2)

        # Guard reliability weights against zero-division risks
        reliability = torch.clamp(reliability, min=1e-5, max=1.0).unsqueeze(-1)
        mask = random_mask.float().unsqueeze(-1)

        # Apply multi-channel masks to isolate losses to valid, corrupted data frames
        valid_mask = attention_mask.float().unsqueeze(-1)
        weighted_errors = squared_errors * reliability * mask * valid_mask
        mask_sum = random_mask.sum()
        
        effective_count = (random_mask.float() * attention_mask.float()).sum()
        if mask_sum < 1:
            logger.info(f"Sum of mask is {float(mask_sum):.2f}, loss is returned with zeros")
            return torch.zeros((), device=pred.device, requires_grad=True)
        else:
            # Normalize strictly across the active elements
            loss = weighted_errors.sum() / (effective_count * active_idx.numel() + 1e-8)

        # Trace and monitor loss layers over early iterations
        if self.step_counter < 5:
            mask_sum_value = mask_sum.item()
            logger.info(
                f"[Loss Debug Pass] step={self.step_counter} "
                f"mask_sum={float(mask_sum_value):.0f} "
                f"active_features_count={active_idx.numel()} "
                f"computed_step_loss={loss.item():.6f}"
            )
            self.step_counter += 1

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"CRITICAL FAULT: Non-finite loss value derived inside Masked Criteria! "
                f"Raw Errors Tensor Sum Value: {squared_errors.sum().item():.4f} | "
                f"Valid Active Mask Sum Float: {mask_sum:.2f}"
            )

        return loss