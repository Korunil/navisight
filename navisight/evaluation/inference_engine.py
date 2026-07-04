# navisight/evaluation/inference_engine.py
import torch
import json
import numpy as np
import logging
import polars as pl
from navisight.models.transformer_encoder import MaritimeMAE
from navisight.pipeline.feature_registry import REGISTRY, REGISTRY_SCHEMA_HASH

class ContextualDualChannelDetector:
    """Production Real-Time Anomaly Engine calibrated against verified ledger scales."""
    def __init__(self, checkpoint_path: str, global_stats_json: str, ann_index, profile_engine, device_type: str = "cuda"):
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        self.ann_index = ann_index
        self.profile_engine = profile_engine

        with open(global_stats_json, "r") as f:
            self.stats_space = json.load(f)

        sb = self.stats_space["spatial_bounds"]
        self.lon_min, self.lon_max = sb["lon_min"], sb["lon_max"]
        self.lat_min, self.lat_max = sb["lat_min"], sb["lat_max"]
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        logging.info("==========================================================================")
        logging.info(f"📋 SYSTEM PLATFORM | Checkpoint Keys Present: {list(checkpoint.keys())}")
        logging.info("==========================================================================")

        self.model = MaritimeMAE(d_model=128, n_heads=8, n_layers=4).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        recon_out = self.model.reconstruction_head.out_features
        registry_out = len(REGISTRY.all_features)
        
        self.maskable_indices = torch.tensor(
            [REGISTRY.feature_index[feat] for feat in REGISTRY.maskable_for_loss], dtype=torch.long, device=self.device
        )

        logging.info("==========================================================================")
        logging.info(f"📊 SYSTEM PLATFORM | Reconstructor Nodes = {recon_out} | Feature Registry Width = {registry_out}")
        logging.info("==========================================================================")

        if registry_out != recon_out:
            raise RuntimeError(f"Schema mismatch: registry={registry_out}, model_recon={recon_out}")

        checkpoint_schema_hash = checkpoint.get("feature_schema_hash", "UNHASHED_LEGACY_WEIGHTS")
        if checkpoint_schema_hash != "UNHASHED_LEGACY_WEIGHTS" and checkpoint_schema_hash != REGISTRY_SCHEMA_HASH:
            raise RuntimeError(f"Schema hash mismatch: checkpoint={checkpoint_schema_hash}, registry={REGISTRY_SCHEMA_HASH}")

        self.mse_elementwise = torch.nn.MSELoss(reduction='none')

    def _equalize_and_normalize_dataframe(self, df_slice: pl.DataFrame, superclass_id: int) -> torch.Tensor:
        """
        Replicates the exact normalization, 30s temporal resampling, and interpolation
        pipeline defined within the training ContinuityPreservingAISDataset layer.
        """
        # Sort and deduplicate timestamps to ensure a clean chronological series sequence
        df_clean = df_slice.sort("timestamp_sec").unique(subset=["timestamp_sec"], keep="first")
        
        timestamps_arr = df_clean["timestamp_sec"].to_numpy().astype(np.int64)
        raw_features = df_clean.select(REGISTRY.all_features).to_numpy().astype(np.float32)
        sub_ts = timestamps_arr.astype(np.float64)

        # ──► STEP 1: REPLICATE COHORT & REGIONAL RANGE NORMALIZATION ◄──
        for idx, feat_name in enumerate(REGISTRY.all_features):
            if feat_name == "lon_raw":
                raw_features[:, idx] = (raw_features[:, idx] - self.lon_min) / (self.lon_max - self.lon_min + 1e-7)
            elif feat_name == "lat_raw":
                raw_features[:, idx] = (raw_features[:, idx] - self.lat_min) / (self.lat_max - self.lat_min + 1e-7)
            elif feat_name in REGISTRY.continuous_scaled:
                if feat_name in REGISTRY.weather:
                    weather_block = self.stats_space.get("weather_meso_scales", {})
                    center = weather_block.get(feat_name, {"center": 0.0})["center"]
                    scale = weather_block.get(feat_name, {"scale": 1.0})["scale"]
                    raw_features[:, idx] = (raw_features[:, idx] - center) / (scale if abs(scale) > 1e-6 else 1.0)
                else:
                    sc_id = str(int(superclass_id))
                    cohorts = self.stats_space.get("global_cohort_scales", {})
                    target_cohort = cohorts.get(sc_id, cohorts.get("0", {}))
                    if feat_name in target_cohort:
                        center = target_cohort[feat_name]["center"]
                        scale = target_cohort[feat_name]["scale"]
                        raw_features[:, idx] = (raw_features[:, idx] - center) / (scale if abs(scale) > 1e-6 else 1.0)

        # ──► STEP 2: RECONSTRUCT THE UNIFORM ISOMETRIC 30-SECOND GRID ◄──
        t_uniform = np.arange(sub_ts[0], sub_ts[-1] + 1.0, 30.0)
        equalized_features = np.zeros((len(t_uniform), len(REGISTRY.all_features)), dtype=np.float32)

        for f_idx, feat_name in enumerate(REGISTRY.all_features):
            feat_slice = raw_features[:, f_idx].astype(np.float64)
            
            if feat_name in REGISTRY.quality or "voyage_phase" in feat_name or "flag" in feat_name:
                idx_nearest = np.clip(np.searchsorted(sub_ts, t_uniform), 0, len(sub_ts) - 1)
                equalized_features[:, f_idx] = feat_slice[idx_nearest].astype(np.float32)
            elif feat_name == "log_time_diff":
                equalized_features[:, f_idx] = float(np.log1p(30.0))
            else:
                equalized_features[:, f_idx] = np.interp(t_uniform, sub_ts, feat_slice).astype(np.float32)

        # ──► STEP 3: CORRECT DIRECTIONAL TRIGONOMETRIC IDENTITIES ◄──
        for s_col, c_col in [("course_sin", "course_cos"), ("hour_sin", "hour_cos"), ("day_sin", "day_cos")]:
            if s_col in REGISTRY.feature_index and c_col in REGISTRY.feature_index:
                s_idx, c_idx = REGISTRY.feature_index[s_col], REGISTRY.feature_index[c_col]
                m_norms = np.sqrt(equalized_features[:, s_idx]**2 + equalized_features[:, c_idx]**2) + 1e-8
                equalized_features[:, s_idx] /= m_norms
                equalized_features[:, c_idx] /= m_norms

        return torch.from_numpy(equalized_features).unsqueeze(0).to(self.device)

    def evaluate_live_sequence_anomaly_score(self, df_slice: pl.DataFrame, vessel_id_int: int, context: dict, mode: str = "production") -> dict:
        """Production scoring pipeline evaluating complete, uncorrupted observation paths."""
        if mode not in ["production", "diagnostic"]:
            raise ValueError(f"Unsupported operational mode policy token specified: {mode}")
            
        superclass_id = context.get("superclass_id", 0)
        
        # Build the exact model input manifold using the unified preprocessing function
        normalized_tensor = self._equalize_and_normalize_dataframe(df_slice, superclass_id)
        
        max_model_context_len = 60
        total_sequence_len = normalized_tensor.size(1)

        chunk_risk_scores = []
        chunk_recons = []
        chunk_drifts = []
        all_chunk_attributions = []

        # Profile Engine Drift
        prototype_drift_values = []
        
        # ──► DASHBOARD FIX: TIMELINE TRACKING NODES ◄──
        risk_timeline = []
        last_cls_vector = None
        
        target_mask_ratio = 0.0 if mode == "production" else 0.75

        for start_idx in range(0, total_sequence_len, max_model_context_len):
            end_idx = min(start_idx + max_model_context_len, total_sequence_len)
            chunk_tensor = normalized_tensor[:, start_idx:end_idx, :]
            attention_mask = None

            with torch.no_grad():
                # Enforce full-sequence reconstruction path at inference time
                reconstructed, cls_embedding, _, random_mask = self.model(chunk_tensor, attention_mask, mask_ratio=target_mask_ratio)

                pred_targets = torch.index_select(reconstructed, dim=-1, index=self.maskable_indices)
                actual_fields = torch.index_select(chunk_tensor, dim=-1, index=self.maskable_indices)
                
                elementwise_errors = self.mse_elementwise(pred_targets, actual_fields)

                # ──► STRATIFIED MANIFOLD SCORING CHANNEL RECONSTRUCTION ◄──
                if mode == "production" or random_mask.sum() == 0:
                    per_feature_mse = elementwise_errors.mean(dim=(0, 1))
                    reconstruction_score = torch.norm(per_feature_mse, p=2).item()
                    # Safe natural log-scale compression for unmasked structures
                    scaled_recon = float(np.clip(np.log1p(reconstruction_score) / 5.0, 0.0, 1.0))
                else:
                    # Diagnostic Mode: Compute error ONLY over the masked token positions
                    mask_expanded = random_mask.unsqueeze(-1)
                    mask_sum = random_mask.sum().item()
                    weighted_errors = elementwise_errors * mask_expanded
                    reconstruction_score = (weighted_errors.sum() / (mask_sum * self.maskable_indices.numel() + 1e-8)).item()
                    # Dynamic log-scale mapping for high-variance masked tokens
                    scaled_recon = float(np.clip(np.log1p(reconstruction_score) / 15.0, 0.0, 1.0))

                feature_errors_np = elementwise_errors.mean(dim=(0, 1)).cpu().numpy()
                all_chunk_attributions.append(feature_errors_np)

                # Vector extraction loop for neighborhood lookup tracking
                np_vectors = cls_embedding.cpu().numpy()[0]
                vec_norm = np.linalg.norm(np_vectors)
                np_vectors = np_vectors / vec_norm if vec_norm > 1e-6 else np_vectors
                last_cls_vector = np_vectors
                
                drift_metrics = self.profile_engine.update_profile_and_score_drift(vessel_id_int, np_vectors)
                prototype_drift = drift_metrics["cosine_drift_score"]

                # Query more raw neighbors (k=100) to allow plenty of room for identity filtering
                raw_neighborhood = self.ann_index.query_behavior_neighborhood(np_vectors, superclass_id=superclass_id, k=100)
                
                distinct_peers = []
                seen_vessel_ids = set([vessel_id_int]) # Automatically filters out self-history matches
                
                for peer in raw_neighborhood:
                    peer_v_id = peer.get("vessel_id_int")
                    if peer_v_id not in seen_vessel_ids:
                        seen_vessel_ids.add(peer_v_id)
                        distinct_peers.append(peer)
                    if len(distinct_peers) >= 3: # Enforce exactly 3 distinct ships
                        break
                
                # Fallback to standard tracking distance if fleet sampling density is low
                mean_peer_dist = np.mean([p["distance"] for p in distinct_peers]) if distinct_peers else 1.0

            scaled_profile_drift = np.clip(prototype_drift / 0.20, 0.0, 1.0)
            prototype_drift_values.append(scaled_profile_drift)

            scaled_drift = min(max(mean_peer_dist, 0.0), 1.0)
            
            behavior_score = 0.5 * scaled_profile_drift + 0.5 * scaled_drift
            
            # THE CORRECTED FUSION CONTRACT
            base_floor = max(scaled_recon, scaled_drift)
            interaction_boost = 0.15 * (scaled_recon * scaled_drift)
            fused_risk_score = float(np.clip(base_floor + interaction_boost, 0.0, 1.0))
            
            chunk_risk_scores.append(fused_risk_score)
            chunk_recons.append(scaled_recon)
            chunk_drifts.append(scaled_drift)
            
            # ──► TARGETED TELEMETRY PROBE ◄──
            # Only prints for the first 2 chunks of the stream to avoid flooding your terminal
            # if len(chunk_risk_scores) < 2:
            #     print(f"\n🕵️ MODE: {mode} | CHUNK INDEX: {len(chunk_risk_scores)}")
            #     print(f"   ├── Raw Recon Score : {reconstruction_score:.6f} -> Scaled Recon (20%): {scaled_recon:.4f}")
            #     print(f"   ├── EMA cosine drift : {prototype_drift:.4f} -> Scaled Profile Drift (80%): {scaled_profile_drift:.4f}")
            #     print(f"   ├── Mean Peer Dist  : {mean_peer_dist:.6f} -> Scaled Drift (80%): {scaled_drift:.4f}")
            #     print(f"   ├── Behavior Score: {behavior_score:.4f}")
            #     print(f"   ├── Fused Risk Score: {fused_risk_score:.4f}")
            #     print(f"   └── HNSW Neighbors Discovered: {len(distinct_peers)} entries")
            #     for i, p in enumerate(distinct_peers):
            #         print(f"       └── Match [{i}]: Vessel ID: {p.get('vessel_id_int')} | Dist: {p['distance']:.6f}")

            # ──► DASHBOARD FIX: APPEND THE TIME-SERIES TIMELINE COMPONENT ◄──
            risk_timeline.append({
                "chunk_start_step": start_idx,
                "chunk_end_step": end_idx,
                "risk_score": float(fused_risk_score),
                "reconstruction_component": float(scaled_recon),
                "drift_component": float(scaled_drift),
                "distinct_peers": distinct_peers,
            })

        # Calculate aggregated trajectory metadata metrics
        final_fused_risk_score = float(np.mean(chunk_risk_scores)) if chunk_risk_scores else 0.0
        mean_recon_score = float(np.mean(chunk_recons)) if chunk_recons else 0.0
        mean_drift_score = float(np.mean(chunk_drifts)) if chunk_drifts else 0.0

        # ──► DASHBOARD FIX: CALIBRATE PROBABILISTIC CONFIDENCE & OOD LOGS ◄──
        # Confidence based on std of chunk risk scores
        risk_std = np.std(chunk_risk_scores)
        confidence = float(np.clip(np.exp(-3.0 * risk_std), 0.0, 1.0))
        # Out-of-Distribution score maps directly to the mean drfit score
        ood_score = float(np.clip(mean_drift_score, 0.0, 1.0))

        mean_feature_errors = np.mean(all_chunk_attributions, axis=0).squeeze().tolist() if all_chunk_attributions else [0.0] * len(REGISTRY.all_features)
        full_attribution_map = {feat: float(err) for feat, err in zip(REGISTRY.all_features, mean_feature_errors)}

        filtered_attributions = [(feat, max(0.0, full_attribution_map[feat])) for feat in REGISTRY.maskable_for_loss if feat in full_attribution_map]
        total_error_sum = sum(val for _, val in filtered_attributions)
        normalized_attributions = [(feat, float(val / total_error_sum)) if total_error_sum > 0 else (feat, 1.0 / len(filtered_attributions)) for feat, val in filtered_attributions]
        normalized_attributions.sort(key=lambda x: x[1], reverse=True)

        return {
            "vessel_id_int": vessel_id_int,
            "timestamp_sec": context.get("timestamp_sec", 0),
            "fused_risk_score": final_fused_risk_score,
            "reconstruction_mse": mean_recon_score,
            
            # ──► NEW DASHBOARD PAYLOAD FIELDS ◄──
            "confidence": confidence,
            "ood_score": ood_score,
            "risk_timeline": risk_timeline,
            
            "anomaly_classification": "Anomaly Detected" if final_fused_risk_score > 0.75 else "Normal Operations",
            "cls_embedding": last_cls_vector,
            "top_contributors": normalized_attributions
        }