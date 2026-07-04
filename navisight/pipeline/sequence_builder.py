# navisight/pipeline/sequence_builder.py
import os
import json
import logging
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from navisight.pipeline.feature_registry import REGISTRY

logger = logging.getLogger(__name__)

class ContinuityPreservingAISDataset(IterableDataset):
    """
    Production out-of-core row group streaming dataset layer.
    Applies class-stratified cohort scale profiles and cubic spline temporal equalization.
    """
    def __init__(self, partitioned_root_dir: str, manifest_json_path: str, window_size: int = 60, stride: int = 30, file_list: list = None):
        self.root_dir = partitioned_root_dir
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = REGISTRY.all_features

        with open(manifest_json_path, "r") as f:
            self.manifest = json.load(f)

        stats_path = self.manifest.get("global_stats_path", "configs/global_stats.json")
        with open(stats_path, "r") as f:
            self.stats_space = json.load(f)

        sb = self.stats_space["spatial_bounds"]
        self.lon_min, self.lon_max = sb["lon_min"], sb["lon_max"]
        self.lat_min, self.lat_max = sb["lat_min"], sb["lat_max"]
        
        if file_list is not None:
            self.all_files = sorted(file_list)
        else:
            self.all_files = sorted([
                os.path.join(root, f) for root, _, files in os.walk(self.root_dir)
                for f in files if f.endswith(".parquet")
            ])
        if not self.all_files:
            raise FileNotFoundError(f"No processed Parquet data shards found under: {self.root_dir}")

    def __len__(self):
        raise TypeError("ContinuityPreservingAISDataset is an IterableDataset. Length is undefined.")
    
    def _stream_vessel_sequences(self, file_list):
        for file_path in file_list:
            if not os.path.exists(file_path):
                continue
            try:
                parquet_file = pq.ParquetFile(file_path)
            except Exception as e:
                logger.warning(f"Failed to read parquet file header: {file_path}. Error: {e}")
                continue

            for rg_idx in range(parquet_file.num_row_groups):
                try:
                    table = parquet_file.read_row_group(rg_idx)
                    df_rg = pl.from_arrow(table)
                except Exception as e:
                    logger.warning(f"Skipping corrupt row group {rg_idx} in {file_path}: {e}")
                    continue

                if df_rg.is_empty():
                    continue

                df_rg = df_rg.sort(["vessel_id_int", "timestamp_sec"]).unique(
                    subset=["vessel_id_int", "timestamp_sec"], keep="first"
                )
                
                vessel_ids = df_rg["vessel_id_int"].to_numpy().astype(np.int64)
                trip_ids = df_rg["trip_id"].to_numpy().astype(object)
                superclasses = df_rg["vessel_superclass_id"].to_numpy().astype(np.float32)
                
                raw_features = df_rg.select(self.feature_cols).to_numpy().astype(np.float32)
                timestamps_arr = df_rg["timestamp_sec"].to_numpy().astype(np.int64)

                diff_vessels = vessel_ids[1:] != vessel_ids[:-1]
                diff_trips = trip_ids[1:] != trip_ids[:-1]
                boundary_marks = np.where(diff_vessels | diff_trips)[0] + 1
                index_splits = np.split(np.arange(len(df_rg)), boundary_marks)

                # ── VECTORIZED COHORT RANGE RANGE NORMALIZATION BASED ON INDEX SPLITS ──
                for idx, feat_name in enumerate(self.feature_cols):
                    if feat_name == "lon_raw":
                        raw_features[:, idx] = (raw_features[:, idx] - self.lon_min) / (self.lon_max - self.lon_min + 1e-7)
                    elif feat_name == "lat_raw":
                        raw_features[:, idx] = (raw_features[:, idx] - self.lat_min) / (self.lat_max - self.lat_min + 1e-7)
                    elif feat_name in REGISTRY.continuous_scaled:
                        if feat_name in REGISTRY.weather:
                            weather_block = self.stats_space.get("weather_meso_scales", {})
                            center = weather_block.get(feat_name, {"center": 0.0})["center"]
                            scale = weather_block.get(feat_name, {"scale": 1.0})["scale"]
                            scale_clamped = scale if abs(scale) > 1e-6 else 1.0
                            raw_features[:, idx] = (raw_features[:, idx] - center) / scale_clamped
                        else:
                            for split_idx in index_splits:
                                if len(split_idx) == 0: continue
                                sc_id = str(int(superclasses[split_idx[0]]))
                                cohorts = self.stats_space.get("global_cohort_scales", {})
                                target_cohort = cohorts.get(sc_id, cohorts.get("0", {}))
                                if feat_name in target_cohort:
                                    center = target_cohort[feat_name]["center"]
                                    scale = target_cohort[feat_name]["scale"]
                                    scale_clamped = scale if abs(scale) > 1e-6 else 1.0
                                    raw_features[split_idx, idx] = (raw_features[split_idx, idx] - center) / scale_clamped

                # ── STEP 2: TEMPORAL WINDOW SLICING & CUBIC-SPLINE EQUALIZATION ──
                for split_idx in index_splits:
                    seq_len = len(split_idx)
                    if seq_len < 5: continue  # Minimum points required for spline stability

                    v_id = int(vessel_ids[split_idx[0]])
                    t_id = str(trip_ids[split_idx[0]])
                    sc_id_int = int(superclasses[split_idx[0]])
                    
                    sub_ts = timestamps_arr[split_idx].astype(np.float64)
                    
                    # Verify total time delta can accommodate window bounds
                    if (sub_ts[-1] - sub_ts[0]) < (self.window_size * 30.0): continue
                    
                    # Synthesize our target uniform timeline grid
                    t_uniform = np.arange(sub_ts[0], sub_ts[-1], 30.0, dtype=np.float64)
                    if len(t_uniform) < self.window_size: continue

                    equalized_features = np.zeros((len(t_uniform), len(self.feature_cols)), dtype=np.float32)
                    sub_feat = raw_features[split_idx]
                    
                    # Compute the causal lookback indices exactly once for all feature passes
                    idx_causal = np.searchsorted(sub_ts, t_uniform, side="right") - 1 
                    idx_causal = np.clip(idx_causal, 0, len(sub_ts) - 1)

                    for f_idx, feat_name in enumerate(self.feature_cols):
                        feat_slice = sub_feat[:, f_idx].astype(np.float64)
                        
                        if feat_name in REGISTRY.quality or "voyage_phase" in feat_name or "flag" in feat_name:
                            # Categorical handling via Last Observation Carried Forward (LOCF)
                            equalized_features[:, f_idx] = feat_slice[idx_causal].astype(np.float32)
                        elif feat_name == "log_time_diff":
                            # Calculate the causal elapsed time since the actual raw AIS transmission occurred
                            # This preserves the missingness signature and irregular sample spacing without look-ahead leakage
                            time_since_last_actual_obs = t_uniform - sub_ts[idx_causal]
                            equalized_features[:, f_idx] = np.log1p(np.clip(time_since_last_actual_obs, 0.0, None)).astype(np.float32)
                        else:
                            # Continuous feature handling via strict Zero-Order Hold (ZOH) / LOCF
                            # Eliminates look-ahead shortcuts by ensuring data at t relies only on observations <= t
                            equalized_features[:, f_idx] = feat_slice[idx_causal].astype(np.float32)

                    # Normalize directional trigonometric identities post-interpolation pass
                    for s_col, c_col in [("course_sin", "course_cos"), ("hour_sin", "hour_cos"), ("day_sin", "day_cos")]:
                        if s_col in REGISTRY.feature_index and c_col in REGISTRY.feature_index:
                            s_idx, c_idx = REGISTRY.feature_index[s_col], REGISTRY.feature_index[c_col]
                            m_norms = np.sqrt(equalized_features[:, s_idx]**2 + equalized_features[:, c_idx]**2) + 1e-8
                            equalized_features[:, s_idx] /= m_norms
                            equalized_features[:, c_idx] /= m_norms

                    rel_equalized = equalized_features[:, REGISTRY.feature_index["reliability_score"]]

                    for start in range(0, len(t_uniform) - self.window_size + 1, self.stride):
                        end = start + self.window_size
                        if rel_equalized[start:end].mean() < 0.40: continue

                        payload_features = equalized_features[start:end].copy()
                        payload_reliability = rel_equalized[start:end].copy()

                        yield {
                            "sequence_id": f"{v_id}_{t_id}_{start}",
                            "vessel_id_int": v_id,
                            "trip_id": t_id,
                            "superclass_id": sc_id_int,
                            "timestamp_sec": int(t_uniform[start]),
                            "features": torch.from_numpy(payload_features),
                            "attention_mask": torch.ones(self.window_size, dtype=torch.bool),
                            "reliability": torch.from_numpy(payload_reliability),
                            "system_token": self.manifest.get("system_token", "")
                        }

    def _get_shard_files_for_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.all_files
        return self.all_files[worker_info.id::worker_info.num_workers]

    def __iter__(self):
        file_list = self._get_shard_files_for_worker()
        return self._stream_vessel_sequences(file_list)