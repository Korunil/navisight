# navisight/models/transformer_encoder.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from navisight.models.feature_projection import MultimodalProjector
from navisight.models.temporal_encoding import FoundationSpatiotemporalPositionEncoder
from navisight.pipeline.feature_registry import REGISTRY

class MaritimeMAE(nn.Module):
    """
    Production Transformer architecture locked to d_model=128.
    Features dual-channel output branches supporting joint Reconstruction and Contrastive Alignment.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.projector = MultimodalProjector()
        self.temporal_encoder = FoundationSpatiotemporalPositionEncoder(d_model=self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=self.d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Channel 1 Head: Reconstructs raw physical features
        self.decoder_head = nn.Linear(self.d_model, len(REGISTRY.all_features))
        
        # Channel 2 Head: Non-linear projection branch for Contrastive Alignment
        self.contrastive_projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 64)
        )

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor, mask_ratio: float = 0.15):
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Enforce a strict clipping guard on input features to prevent 
        # extreme outliers from causing FP16 overflows inside self-attention
        features = torch.clamp(features, min=-50.0, max=50.0)
        
        if not torch.isfinite(features).all():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size, seq_len, feature_dim = features.size()
        
        # ── FIXED: FAIL-FAST FEATURE INDEX BOUNDARY CHECK ──
        # Catches tracking or registry index alignment mismatches right at the front gate
        time_idx = REGISTRY.feature_index['log_time_diff']
        
        # Refinement 2: One-time logging flag for inside the forward pass to reveal hidden drift
        if not hasattr(self, "_forward_indices_logged"):
            logging.info(f"🔮 DEBUG | Model Forward Pass: log_time_diff Index = {time_idx} | Input feature_dim = {feature_dim}")
            self._forward_indices_logged = True

        if time_idx >= feature_dim:
            raise RuntimeError(f"log_time_diff index {time_idx} exceeds feature dimension {feature_dim}")
        
        log_time_diffs = features[:, :, time_idx]

        # Verify mask alignment layout configurations explicitly before self-attention execution
        if attention_mask is not None:
            if attention_mask.shape != features.shape[:2]:
                raise RuntimeError(
                    f"🚨 MASK CONFIG MISMATCH: Input mask tracking shape {attention_mask.shape} "
                    f"does not conform to expected data tensor sequence bounds {(batch_size, seq_len)}."
                )
        
        # Project features into hidden embedding spaces
        try:
            x_seq = self.projector.project_sequence_base(features)
        except Exception:
            logging.exception(f"🚨 CRITICAL CRASH inside project_sequence_base! Input Tensor Shape: {tuple(features.shape)}")
            raise

        if self.training and mask_ratio > 0.0:
            rand_tensor = torch.rand(batch_size, seq_len, device=features.device)
            random_mask = (rand_tensor < mask_ratio) * attention_mask.bool()
        else:
            random_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)

        mask_expanded = random_mask.unsqueeze(-1)
        x_seq = torch.where(mask_expanded, self.projector.mask_token, x_seq)
        
        # Applying LayerNorm AFTER mask substitution ensures both masked 
        # and visible tokens share identical variance scales, preventing attention asymmetry!
        x_seq = self.input_norm(x_seq)

        # Prepend the authoritative global representation [CLS] token
        cls_tokens = self.projector.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x_seq], dim=1)

        # Inject spatiotemporal coordinate signals
        x = x + self.temporal_encoder(log_time_diffs)

        # We check self.training and explicitly validate the incoming token layouts 
        # before passing the key padding mask into the CUDA attention kernels.
        if self.training and attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
            full_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)
            
            # Double-check that mask dimensions match the transformer input sequence shape
            if full_padding_mask.size(0) == x.size(0) and full_padding_mask.size(1) == x.size(1):
                x_lat = self.transformer(x, src_key_padding_mask=full_padding_mask)
            else:
                x_lat = self.transformer(x, src_key_padding_mask=None)
        else:
            # Clean, zero-overhead pass-through for out-of-sample real-time tracks evaluation
            x_lat = self.transformer(x, src_key_padding_mask=None)        
        
        # Extract and normalize the global latent vector representation
        raw_cls_embedding = x_lat[:, 0, :]
        cls_embedding = F.normalize(raw_cls_embedding, p=2, dim=-1)

        # Map latent vectors to contrastive space
        contrastive_embeddings = F.normalize(self.contrastive_projection_head(raw_cls_embedding), p=2, dim=-1)

        reconstructed_features = self.decoder_head(x_lat[:, 1:, :])

        return reconstructed_features, cls_embedding, contrastive_embeddings, random_mask