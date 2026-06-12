# navisight/evaluation/inference_engine.py
import torch
import json
import numpy as np
import logging
from navisight.models.transformer_encoder import MaritimeMAE
from navisight.pipeline.feature_registry import REGISTRY, REGISTRY_SCHEMA_HASH

class ContextualDualChannelDetector:
    """Diagnostic Real-Time Anomaly Engine configured for forensic evidence gathering."""
    def __init__(self, checkpoint_path: str, global_stats_json: str, ann_index, profile_engine, device_type: str = "cuda"):
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        self.ann_index = ann_index
        self.profile_engine = profile_engine

        with open(global_stats_json, "r") as f:
            self.stats = json.load(f)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Refinement 5: Print the actual checkpoint keys to verify metadata presence instantly
        logging.info("==========================================================================")
        logging.info(f"📋 DEBUG | Checkpoint Keys Present: {list(checkpoint.keys())}")
        logging.info("==========================================================================")

        self.model = MaritimeMAE(d_model=128, n_heads=8, n_layers=4).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        decoder_out = self.model.decoder_head.out_features
        registry_out = len(REGISTRY.all_features)
        
        self.maskable_indices = torch.tensor(
            [REGISTRY.feature_index[feat] for feat in REGISTRY.maskable_for_loss], dtype=torch.long, device=self.device
        )

        # Refinement 2: Log the exact structural and registry indices at initialization
        logging.info(f"📊 DEBUG | Hardware Nodes Contract: Decoder Outputs = {decoder_out}")
        logging.info(f"📊 DEBUG | Registry Contract      : Total Features  = {registry_out}")
        if self.maskable_indices.numel() > 0:
            logging.info(f"📊 DEBUG | Loss Masking Contract  : Max Maskable Index = {self.maskable_indices.max().item()}")
        logging.info("==========================================================================")

        # Strict Runtime Exception Guards (No asserts)
        if registry_out != decoder_out:
            raise RuntimeError(f"Schema mismatch: registry={registry_out}, model_decoder={decoder_out}")

        if self.maskable_indices.numel() > 0 and self.maskable_indices.max().item() >= registry_out:
            raise RuntimeError(f"Maskable index out of range: max={self.maskable_indices.max().item()} vs registry={registry_out}")

        checkpoint_schema_hash = checkpoint.get("feature_schema_hash", "UNHASHED_LEGACY_WEIGHTS")
        if checkpoint_schema_hash != "UNHASHED_LEGACY_WEIGHTS" and checkpoint_schema_hash != REGISTRY_SCHEMA_HASH:
            raise RuntimeError(f"Schema hash mismatch: checkpoint={checkpoint_schema_hash}, registry={REGISTRY_SCHEMA_HASH}")

        self.mse_elementwise = torch.nn.MSELoss(reduction='none')
        self._shape_logged = False # Refinement 1 Tracker

    def _normalize_tensor_live(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        norm_tensor = raw_tensor.clone()
        weather_block = self.stats.get("weather_meso_scales", {})
        cohort_block = self.stats.get("global_cohort_scales", {}).get("0", {}) # Pull baseline center matrices
        
        for feat_name, idx in REGISTRY.feature_index.items():
            if feat_name in weather_block:
                center = weather_block[feat_name]["center"]
                scale = weather_block[feat_name]["scale"]
                scale_clamped = scale if abs(scale) > 1e-6 else 1.0
                norm_tensor[..., idx] = (norm_tensor[..., idx] - center) / scale_clamped
            elif feat_name in REGISTRY.continuous_scaled and feat_name in cohort_block:
                # Enforces matching training scaling rules over the live streaming tensor blocks
                center = cohort_block[feat_name]["center"]
                scale = cohort_block[feat_name]["scale"]
                scale_clamped = scale if abs(scale) > 1e-6 else 1.0
                norm_tensor[..., idx] = (norm_tensor[..., idx] - center) / scale_clamped
        return norm_tensor

    def evaluate_live_sequence_anomaly_score(self, raw_feature_tensor: torch.Tensor, vessel_id_int: int, context: dict) -> dict:
        if not torch.isfinite(raw_feature_tensor).all():
            raise RuntimeError("🚨 INFERENCE CORRUPTION FAULT: Incoming tracking tensor contains non-finite elements (NaN/Inf).")

        if raw_feature_tensor.dim() == 2:
            raw_feature_tensor = raw_feature_tensor.unsqueeze(0)

        expected_features = len(REGISTRY.all_features)
        if raw_feature_tensor.size(-1) != expected_features:
            raise RuntimeError(f"Input schema width mismatch: got {raw_feature_tensor.size(-1)}, expected {expected_features}")

        # ── FIXED: MAX CONTEXT SEQUENCE LENGTH SLIDING CHUNK WINDOW ──
        # Extract the hard hardware constraints from your model layers (typically 1024)
        # We subtract 1 to leave room for the [CLS] summary token prepended in forward()
        max_model_context_len = 1024 - 1 
        total_sequence_len = raw_feature_tensor.size(1)

        # Normalize features safely over the full tensor matrix
        full_normalized_tensor = self._normalize_tensor_live(raw_feature_tensor).to(self.device).contiguous()

        # Storage buffers for calculating across the windows
        chunk_risk_scores = []
        all_chunk_attributions = []
        last_cls_vector = None

        # Stream individual sub-windows if the input sequence exceeds maximum transformer capacity
        for start_idx in range(0, total_sequence_len, max_model_context_len):
            end_idx = min(start_idx + max_model_context_len, total_sequence_len)
            
            # Slice down your active window sequence block securely
            normalized_tensor = full_normalized_tensor[:, start_idx:end_idx, :]
            attention_mask = None 

            with torch.no_grad():
                reconstructed, cls_embedding, _, _ = self.model(normalized_tensor, attention_mask, mask_ratio=0.0)

                pred_targets = torch.index_select(reconstructed, dim=-1, index=self.maskable_indices)
                actual_fields = torch.index_select(normalized_tensor, dim=-1, index=self.maskable_indices)
                
                # Extract element-wise errors before reducing to preserve feature attributions
                elementwise_error_tensor = self.mse_elementwise(pred_targets, actual_fields)
                reconstruction_error = elementwise_error_tensor.mean().item()

                # Calculate mean error per feature across batch and sequence tokens
                feature_errors_np = elementwise_error_tensor.mean(dim=(0, 1)).cpu().numpy()
                all_chunk_attributions.append(feature_errors_np)

                np_vectors = cls_embedding.cpu().numpy()
                np_vector = np_vectors[0]
                last_cls_vector = np_vector
                
                _ = self.profile_engine.update_profile_and_score_drift(vessel_id_int, np_vector)
                
                # Fetch class identity from the contextual metadata mapping layers
                superclass_id = context.get("superclass_id", 0)
                
                # Pass superclass_id gate to query strictly isolated sub-graphs
                peer_neighborhood = self.ann_index.query_behavior_neighborhood(np_vector, superclass_id=superclass_id, k=3)
                mean_peer_dist = np.mean([p["distance"] for p in peer_neighborhood]) if peer_neighborhood else 1.0

            fused_risk_score = 0.4 * min(reconstruction_error / 2.5, 2.0) + 0.6 * mean_peer_dist
            chunk_risk_scores.append(fused_risk_score)

        # Aggregate window outputs cleanly to compute an accurate overall summary track score
        final_fused_risk_score = float(np.mean(chunk_risk_scores))
        
        # Compile true model-driven feature attributions across chunks
        mean_feature_errors = np.mean(all_chunk_attributions, axis=0).squeeze().tolist()

        # 1. Map to the FULL 41-feature registry first to guarantee index integrity
        full_attribution_map = {
            feat: float(err) for feat, err in zip(REGISTRY.all_features, mean_feature_errors)
        }

        # 2. Filter down strictly to maskable targets
        filtered_attributions = [
            (feat, max(0.0, full_attribution_map[feat])) 
            for feat in REGISTRY.maskable_for_loss if feat in full_attribution_map
        ]

        # 3. NORMALIZATION PASS: Calculate clean relative fractional shares
        total_error_sum = sum(val for _, val in filtered_attributions)
        if total_error_sum > 0:
            normalized_attributions = [
                (feat, float(val / total_error_sum)) 
                for feat, val in filtered_attributions
            ]
        else:
            # Equal distribution fallback if total reconstruction error is perfectly zero
            normalized_attributions = [
                (feat, 1.0 / len(filtered_attributions)) 
                for feat, _ in filtered_attributions
            ]
        
        # 4. Sort by descending relative impact
        normalized_attributions.sort(key=lambda x: x[1], reverse=True)

        return {
            "vessel_id_int": vessel_id_int,
            "timestamp_sec": context.get("timestamp_sec", 0),
            "fused_risk_score": final_fused_risk_score,
            "anomaly_classification": "Anomaly Detected" if final_fused_risk_score > 0.75 else "Normal Operations",
            # Expose true internals to drive dashboard explainability cards
            "cls_embedding": last_cls_vector,
            "top_contributors": normalized_attributions
        }