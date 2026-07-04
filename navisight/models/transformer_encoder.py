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
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4, recon_layers=2, dropout: float = 0.1):
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
        
        recon_heads = n_heads // 2
        if recon_heads < 1:
            recon_heads = 1
        assert d_model % recon_heads == 0
        
        reconstruction_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=recon_heads, dim_feedforward=self.d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.reconstruction_refiner = nn.TransformerEncoder(reconstruction_layer, num_layers=recon_layers)
        
        # Channel 1 Head: Reconstructs raw physical features
        self.reconstruction_head = nn.Linear(self.d_model, len(REGISTRY.all_features))
        
        # Channel 2 Head: Non-linear projection branch for Contrastive Alignment
        self.contrastive_projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 64)
        )

    def _generate_mask(self, batch_size, seq_len, attention_mask, mask_ratio, device):
        if mask_ratio <= 0:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # ── 1. explicit budget ──
        target_tokens = max(1, int(seq_len * mask_ratio))
        k_rand  = int(0.30 * target_tokens)
        k_block = int(0.40 * target_tokens)
        k_tail  = int(0.20 * target_tokens)
        k_dual  = target_tokens - (k_rand + k_block + k_tail)

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # ── RANDOM (30%) ──
        rand_scores = torch.rand(batch_size, seq_len, device=device)
        rand_mask = torch.zeros_like(mask)
        if k_rand > 0:
            rand_idx = torch.topk(rand_scores, k=k_rand, dim=1).indices
            rand_mask.scatter_(1, rand_idx, True)
        
        mask |= rand_mask

        # ── BLOCK (40%) ──
        if k_block > 0:
            block_len = torch.full((batch_size,), max(1, k_block), device=device)
            max_start = torch.clamp(seq_len - block_len + 1, min=1)
            start_idx = (torch.rand(batch_size, device=device) * max_start.float()).long()


            block_mask = (
                (positions >= start_idx.unsqueeze(1)) &
                (positions < (start_idx + block_len).unsqueeze(1))
            )
            
            mask |= block_mask

        # ── TAIL (20%) ──
        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1).to(device)
        else:
            valid_lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        
        # ensure at least 1 token remains unmasked
        min_visible = torch.ones_like(valid_lengths)
        max_tail_k = (valid_lengths - min_visible).clamp(min=0)
        
        if k_tail > 0:
            k_tail_vec = torch.full_like(valid_lengths, k_tail)
            safe_tail_k = torch.minimum(k_tail_vec, max_tail_k)
            
            tail_start = torch.clamp(valid_lengths - safe_tail_k, min=0)

            tail_mask = (
                (positions >= tail_start.unsqueeze(1)) &
                (positions < valid_lengths.unsqueeze(1))
            )
            
            mask |= tail_mask
        
        # ── DUAL (10%) ──
        if k_dual > 0:
            block_len2 = torch.full((batch_size,), max(1, k_dual // 2), device=device)
            max_start2 = torch.clamp(seq_len - block_len2 + 1, min=1)
            start_idx2 = (torch.rand(batch_size, device=device) * max_start2.float()).long()
            
            block_mask2 = (
                (positions >= start_idx2.unsqueeze(1)) &
                (positions < (start_idx2 + block_len2).unsqueeze(1))
            )
            
            mask |= block_mask2
        
        # enforce attention mask
        if attention_mask is not None:
            mask &= attention_mask.bool()
        
        # ── MASK LOGGING (added) ──
        with torch.no_grad():
            if not hasattr(self, "_mask_log_step"):
                self._mask_log_step = 0
            
            if not hasattr(self, "_mask_stats"):
                self._mask_stats = {
                    "count": 0,
                    "requested": 0.0,
                    "realized": 0.0,
                    "efficiency": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

            if attention_mask is not None:
                valid = attention_mask.bool()
                per_track_ratio = (
                    mask.float().sum(dim=1)
                    / valid.float().sum(dim=1).clamp(min=1)
                )
            else:
                per_track_ratio = mask.float().mean(dim=1)

            realized_mean = per_track_ratio.mean().item()
            realized_std = per_track_ratio.std().item()
            realized_min = per_track_ratio.min().item()
            realized_max = per_track_ratio.max().item()

            efficiency = (
                realized_mean / mask_ratio
                if mask_ratio > 0
                else 1.0
            )

            self._mask_stats["count"] += 1
            self._mask_stats["requested"] += mask_ratio
            self._mask_stats["realized"] += realized_mean
            self._mask_stats["efficiency"] += efficiency
            self._mask_stats["std"] += realized_std
            self._mask_stats["min"] = min(self._mask_stats["min"], realized_min)
            self._mask_stats["max"] = max(self._mask_stats["max"], realized_max)

            self._mask_log_step += 1
            
            if self._mask_log_step % 20000 == 0:
                n = self._mask_stats["count"]

                if self._mask_stats["min"] < 0.20:
                    logging.warning(
                        f"[MASK] low-mask sample detected "
                        f"(min={self._mask_stats['min']:.3f})"
                    )
                if self._mask_stats["max"] > 0.95:
                    logging.warning(
                        f"[MASK] near-total-mask sample detected "
                        f"(max={self._mask_stats['max']:.3f})"
                    )
                
                logging.info(
                    "[MASK-STATS] "
                    f"steps={self._mask_log_step} "
                    f"requested={self._mask_stats['requested']/n:.3f} "
                    f"realized={self._mask_stats['realized']/n:.3f} "
                    f"efficiency={self._mask_stats['efficiency']/n:.3f} "
                    f"std={self._mask_stats['std']/n:.3f} "
                    f"min={self._mask_stats['min']/n:.3f} "
                    f"max={self._mask_stats['max']/n:.3f}"
                )
                self._mask_stats = {
                    "count": 0,
                    "requested": 0.0,
                    "realized": 0.0,
                    "efficiency": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
        
        return mask

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor, mask_ratio: float = 0.75):
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Safety Guard 1: Sanitize input tensors against NaN and infinite values
        features = torch.nan_to_num(features, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Safety Guard 2: Enforce fixed input scaling boundaries (matching [-50, 50] tracking limits)
        features = torch.clamp(features, min=-50.0, max=50.0)

        batch_size, seq_len, feature_dim = features.size()
        
        # Safety Guard 3: Validate feature schema mapping boundaries
        if feature_dim != len(REGISTRY.all_features):
            raise RuntimeError(
                f"Feature width mismatch: "
                f"{feature_dim} != {len(REGISTRY.all_features)}"
            )
        
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
            mask_matrix = self._generate_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                attention_mask=attention_mask,
                mask_ratio=mask_ratio,
                device=features.device
            )
        else:
            mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)

        mask_expanded = mask_matrix.unsqueeze(-1)
        x_seq = torch.where(mask_expanded, self.projector.mask_token, x_seq)
        
        # Normalize masked and unmasked token embeddings
        # into a comparable feature scale before attention.
        x_seq = self.input_norm(x_seq)

        # Prepend the authoritative global representation [CLS] token
        cls_tokens = self.projector.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x_seq], dim=1)

        # Inject spatiotemporal coordinate signals
        x = x + self.temporal_encoder(log_time_diffs)

        # Build padding mask whenever an attention mask is supplied
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
            full_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)
            
            # Double-check that mask dimensions match the transformer input sequence shape
            if full_padding_mask.shape != x.shape[:2]:
                full_padding_mask=None
        else:
            # Clean, zero-overhead pass-through for out-of-sample real-time tracks evaluation
            full_padding_mask = None            
        
        x_lat = self.transformer(x, src_key_padding_mask=full_padding_mask)
                    
        # Extract and normalize the global latent vector representation
        raw_cls_embedding = x_lat[:, 0, :]
        
        # Future trials for raw_cls_embedding
        # emb_a = x_lat[:,0,:]
        # emb_b = refined_latents[:,0,:]
        cls_embedding = F.normalize(raw_cls_embedding, p=2, dim=-1)

        # Map latent vectors to contrastive space
        contrastive_embeddings = F.normalize(self.contrastive_projection_head(raw_cls_embedding), p=2, dim=-1)

        refined_latents = self.reconstruction_refiner(x_lat, src_key_padding_mask=full_padding_mask)
        reconstructed_features = self.reconstruction_head(refined_latents[:, 1:, :])

        return reconstructed_features, cls_embedding, contrastive_embeddings, mask_matrix