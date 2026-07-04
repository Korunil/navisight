# navisight/models/feature_projection.py
"""
Multimodal feature projector for Navisight AI.
 
Maps heterogeneous AIS feature groups into a unified d_model=128 embedding space.
Each of the four feature groups (physics, weather, context, quality) has its own
linear projection branch, preserving inductive bias between physically distinct
feature types.
 
Design contract:
    phys_out + weather_out + ctx_out + qual_out == d_model (128)
    Slice boundaries are read from REGISTRY — do not hardcode offsets here.
"""

import torch
import torch.nn as nn
from navisight.pipeline.feature_registry import REGISTRY

class MultimodalProjector(nn.Module):
    """   
    Four-branch linear projector.
 
    Input:  (B, T, 35)  — ALL_FEATURES in registry order
    Output: (B, T, 128) — normalised token embeddings
 
    Branches:
        physics (13d)  → 64d
        weather  (9d)  → 24d
        context  (8d)  → 24d
        quality  (5d)  → 16d
                         ───
                         128d  ==  d_model
    
    """
    def __init__(
        self,
        phys_out:    int = 64,
        weather_out: int = 24,
        ctx_out:     int = 24,
        qual_out:    int = 16,
    ):
        super().__init__()
 
        # Dimensions read from registry — single source of truth
        self.phys_dim    = len(REGISTRY.physics)    # 13
        self.weather_dim = len(REGISTRY.weather)    # 9
        self.ctx_dim     = len(REGISTRY.context)    # 8
        self.qual_dim    = len(REGISTRY.quality)    # 5
 
        self.d_model = phys_out + weather_out + ctx_out + qual_out  # 128
 
        # Validate at construction time — fail loudly if dims drift
        assert self.d_model == 128, (
            f"Projector output d_model={self.d_model} must equal 128. "
            f"Adjust phys_out/weather_out/ctx_out/qual_out so they sum to 128."
        )
        expected_input = self.phys_dim + self.weather_dim + self.ctx_dim + self.qual_dim
        assert expected_input == len(REGISTRY.all_features), (
            f"Sum of group dims ({expected_input}) != len(ALL_FEATURES) "
            f"({len(REGISTRY.all_features)}). Registry and projector are out of sync."
        )
 
        # Four independent projection branches
        self.physics_proj = nn.Linear(self.phys_dim,    phys_out)
        self.weather_proj = nn.Linear(self.weather_dim, weather_out)
        self.context_proj = nn.Linear(self.ctx_dim,     ctx_out)
        self.quality_proj = nn.Linear(self.qual_dim,    qual_out)
 
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(self.d_model)
 
        # Learnable special tokens — prepended / substituted in transformer
        self.cls_token  = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Slice boundaries — read from REGISTRY, not hardcoded ints
    # ──────────────────────────────────────────────────────────────────────────

    def project_sequence_base(self, features: torch.Tensor) -> torch.Tensor:
        
        """
        Project a batch of feature sequences into d_model space.
 
        Args:
            features: (B, T, 35)
 
        Returns:
            projected: (B, T, 128) — GELU-activated, layer-normalised
        """
        p0, p1 = REGISTRY.physics_slice   # (0, 13)
        w0, w1 = REGISTRY.weather_slice   # (13, 22)
        c0, c1 = REGISTRY.context_slice   # (22, 30)
        q0, q1 = REGISTRY.quality_slice   # (30, 35)
 
        phys_x    = features[:, :, p0:p1]  # (B, T, 13)
        weather_x = features[:, :, w0:w1]  # (B, T, 9)
        ctx_x     = features[:, :, c0:c1]  # (B, T, 8)
        qual_x    = features[:, :, q0:q1]  # (B, T, 5)
 
        h = torch.cat([
            self.act(self.physics_proj(phys_x)),    # (B, T, 64)
            self.act(self.weather_proj(weather_x)), # (B, T, 24)
            self.act(self.context_proj(ctx_x)),     # (B, T, 24)
            self.act(self.quality_proj(qual_x)),    # (B, T, 16)
        ], dim=-1)  # (B, T, 128)
 
        return self.norm(h)