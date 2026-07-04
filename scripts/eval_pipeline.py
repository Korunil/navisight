# scripts/eval_pipeline.py
import os
import sys
import logging
import polars as pl
import torch
import numpy as np
import json
import hashlib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MONTH_MAP = {7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}

# ==========================================================================
# GLOBAL CONFIGURATION CONSTANTS
# ==========================================================================
CALIBRATION_PERCENTILE = 99.5   # Target percentile for anomaly baseline cutoffs
JOINT_WARNING_FACTOR = 0.95     # Sub-threshold multiplier for joint channel contours

CALIBRATION_MONTHS = [7, 8, 9]
BENCHMARK_MONTHS = [10, 11, 12]

MAX_WINDOW_SIZE = 200
STRIDE = 100

ANOMALY_MODES = ["dead_reckoning", "coastal_creep", "loitering"]


def aggregate_score(scores, mean_weight=0.7, peak_percentile=95):
    """
    Applies the canonical scoring rule over an array of window risk profiles.
    Preserved exactly to maintain measuring instrument consistency.
    """
    if not scores:
        return 0.0
    arr = np.array(scores)
    return float(mean_weight * np.mean(arr) + (1 - mean_weight) * np.percentile(arr, peak_percentile))


def evaluate_vessel_windows(df_vessel_chunk, detector, injector, v_id, context_meta, generate_anomalies=True, ann_calibration=None):
    """
    Evaluates a single vessel trajectory dataset using overlapping sliding windows.
    Adheres strictly to the ContextualDualChannelDetector output dictionary signature.
    """
    total_rows = df_vessel_chunk.height
    
    # Storage for separated structural indicators
    window_records_normal = []
    window_records_anomaly = []
    anomaly_type_records = {mode: [] for mode in ANOMALY_MODES}
    
    last_start = None
    base_windows_parsed = 0
    inference_passes_run = 0

    for start_idx in range(0, total_rows - MAX_WINDOW_SIZE + 1, STRIDE):
        last_start = start_idx
        df_slice = df_vessel_chunk.slice(start_idx, MAX_WINDOW_SIZE)

        # 1. Base Evaluation Invariant Mode Pass (Normal)
        res_normal = detector.evaluate_live_sequence_anomaly_score(
            df_slice, v_id, context_meta, mode="production"
        )
        
        recon_val = res_normal["reconstruction_mse"]
        ann_dist  = res_normal["ood_score"]
        fused_val = res_normal["fused_risk_score"]
        
        # REQ 1 & 2: Remove upper clipping limit; keep lower clipping floor at 0
        if ann_calibration is not None:
            ann_p50, ann_p99 = ann_calibration
            ann_score = float(np.maximum(0.0, (ann_dist - ann_p50) / (ann_p99 - ann_p50 + 1e-8)))
        else:
            ann_score = ann_dist

        window_records_normal.append({
            "reconstruction_score": recon_val,
            "ann_score": ann_score,
            "fused_score": fused_val
        })
        base_windows_parsed += 1
        inference_passes_run += 1

        # 2. Continuous Counterfactual Generation and Testing Paths
        if generate_anomalies:
            seed_str = f"{v_id}_{start_idx}"
            seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
            rng = np.random.default_rng(seed_hash & 0xFFFFFFFF)
            
            mode_idx = seed_hash % len(ANOMALY_MODES)
            anomaly_mode = ANOMALY_MODES[mode_idx]
            severity = float(np.clip(rng.normal(0.5, 0.2), 0.1, 0.9))

            df_perturbed = injector.inject_counterfactual_scenario(
                df_vessel_voyage=df_slice, mode=anomaly_mode, severity=severity
            )

            res_anomaly = detector.evaluate_live_sequence_anomaly_score(
                df_perturbed, v_id, context_meta, mode="production"
            )
            
            a_recon_val = res_anomaly["reconstruction_mse"]
            a_ann_dist  = res_anomaly["ood_score"]
            a_fused_val = res_anomaly["fused_risk_score"]
            
            # Apply lower-bounded quantile re-scaling to synthetic anomalies
            if ann_calibration is not None:
                ann_p50, ann_p99 = ann_calibration
                a_ann_score = float(np.maximum(0.0, (a_ann_dist - ann_p50) / (ann_p99 - ann_p50 + 1e-8)))
            else:
                a_ann_score = a_ann_dist

            window_records_anomaly.append({
                "reconstruction_score": a_recon_val,
                "ann_score": a_ann_score,
                "fused_score": a_fused_val
            })
            
            anomaly_type_records[anomaly_mode].append({
                "reconstruction_score": a_recon_val,
                "ann_score": a_ann_score,
                "fused_score": a_fused_val,
                "severity": severity
            })
            inference_passes_run += 1

    # Guard rail coverage tracking constraint - Tail window step
    tail_start = max(0, total_rows - MAX_WINDOW_SIZE)
    if tail_start >= 0 and last_start is not None and tail_start > last_start:
        df_tail = df_vessel_chunk.slice(tail_start, MAX_WINDOW_SIZE)
        
        res_tail = detector.evaluate_live_sequence_anomaly_score(
            df_tail, v_id, context_meta, mode="production"
        )
        t_recon_val = res_tail["reconstruction_mse"]
        t_ann_dist  = res_tail["ood_score"]
        t_fused_val = res_tail["fused_risk_score"]
        
        if ann_calibration is not None:
            ann_p50, ann_p99 = ann_calibration
            t_ann_score = float(np.maximum(0.0, (t_ann_dist - ann_p50) / (ann_p99 - ann_p50 + 1e-8)))
        else:
            t_ann_score = t_ann_dist

        window_records_normal.append({
            "reconstruction_score": t_recon_val,
            "ann_score": t_ann_score,
            "fused_score": t_fused_val
        })
        base_windows_parsed += 1
        inference_passes_run += 1

        if generate_anomalies:
            seed_str_tail = f"{v_id}_{tail_start}"
            seed_hash_tail = int(hashlib.md5(seed_str_tail.encode()).hexdigest(), 16)
            rng_tail = np.random.default_rng(seed_hash_tail & 0xFFFFFFFF)
            
            mode_idx_tail = seed_hash_tail % len(ANOMALY_MODES)
            anomaly_mode_tail = ANOMALY_MODES[mode_idx_tail]
            severity_tail = float(np.clip(rng_tail.normal(0.5, 0.2), 0.1, 0.9))

            df_tail_perturbed = injector.inject_counterfactual_scenario(
                df_vessel_voyage=df_tail, mode=anomaly_mode_tail, severity=severity_tail
            )
            res_tail_anomaly = detector.evaluate_live_sequence_anomaly_score(
                df_tail_perturbed, v_id, context_meta, mode="production"
            )
            
            t_a_recon_val = res_tail_anomaly["reconstruction_mse"]
            t_a_ann_dist  = res_tail_anomaly["ood_score"]
            t_a_fused_val = res_tail_anomaly["fused_risk_score"]
            
            if ann_calibration is not None:
                ann_p50, ann_p99 = ann_calibration
                t_a_ann_score = float(np.maximum(0.0, (t_a_ann_dist - ann_p50) / (ann_p99 - ann_p50 + 1e-8)))
            else:
                t_a_ann_score = t_a_ann_dist

            window_records_anomaly.append({
                "reconstruction_score": t_a_recon_val,
                "ann_score": t_a_ann_score,
                "fused_score": t_a_fused_val
            })
            anomaly_type_records[anomaly_mode_tail].append({
                "reconstruction_score": t_a_recon_val,
                "ann_score": t_a_ann_score,
                "fused_score": t_a_fused_val,
                "severity": severity_tail
            })
            inference_passes_run += 1

    return {
        "window_records_normal": window_records_normal,
        "window_records_anomaly": window_records_anomaly,
        "anomaly_records": anomaly_type_records,
        "base_windows_parsed": base_windows_parsed,
        "inference_passes_run": inference_passes_run
    }


def main():
    logging.info("🚨 INITIATING MULTI-CHANNEL EVALUATION AND COMPREHENSIVE RECOVERY FRAMEWORK")
    
    processed_lakehouse_root    = "data/processed/"
    checkpoint_directory        = "models/checkpoints/"
    global_stats_json_path      = "configs/global_stats.json"
    index_directory             = "models/state/index"
    output_metrics_json         = "configs/evaluation_results.json"
    
    # ==========================================================================
    # DETECTOR INTEGRITY GUARDRAIL
    # ==========================================================================
    db_path = os.path.join(index_directory, "ann_metadata_registry.db")
    if not os.path.exists(index_directory) or not os.listdir(index_directory) or not os.path.exists(db_path):
        logging.error("❌ EVALUATION ABORTED: Missing Populated Background Vector Manifold!")
        print("\n" + "!" * 80)
        print("🚨 CRITICAL PLATFORM FAULT: STRUCTURAL HNSW VECTOR INDEX UNINITIALIZED")
        print("!" * 80)
        print(f"The vector index tracking directory '{index_directory}' is unpopulated or missing core registry assets.")
        print("Running an evaluation loop against an empty index forces a worst-case fallback distance metric,")
        print("causing 100% anomaly detection score saturation across all baseline tracking windows.")
        print("\n👉 DEPLOYMENT REMEDY: You MUST extract baseline latent representations and build the index graph first:")
        print("   python scripts/build_embeddings.py")
        print("!" * 80 + "\n")
        raise RuntimeError(f"Index Void Exception: Missing required populated assets under '{index_directory}'.")

    best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = sorted([
            os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")
        ])[-1]

    logging.info(f"Loading Global Model Network Checkpoint: {best_checkpoint}")
    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    
    initial_shell_profile = RollingBehaviorProfileEngine(alpha=0.1)
    detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, ann_index, initial_shell_profile)
    injector = ManifoldAwareAnomalyPerturbationEngine()

    if hasattr(detector, "model") and detector.model is not None:
        detector.model.eval()
        logging.info("👉 Operational Guardrail Locked: Target neural graph set to model.eval() evaluation state.")

    total_cal_windows_parsed = 0
    total_cal_inference_passes = 0
    total_bench_windows_parsed = 0
    total_bench_inference_passes = 0
    
    # ==========================================
    # PHASE A: MULTI-CHANNEL CALIBRATION (JUL-SEP)
    # ==========================================
    logging.info("📊 PHASE A: Extracting Baseline Background Distributions (Jul-Sep 2019)...")
    
    raw_calib_ann_distances = []
    vessel_window_groups_recon = []
    vessel_window_groups_ann   = []
    vessel_window_groups_fused = []
    
    vessels_processed_cal = 0

    for m_int in CALIBRATION_MONTHS:
        mock_live_path = os.path.join(processed_lakehouse_root, "year=2019", f"month={m_int:02d}")
        if not os.path.exists(mock_live_path): continue

        for root, _, files in os.walk(mock_live_path):
            for file in files:
                if file.endswith(".parquet"):
                    df_vessel_chunk = pl.read_parquet(os.path.join(root, file))
                    if df_vessel_chunk.height < MAX_WINDOW_SIZE: continue
                    
                    v_id = int(df_vessel_chunk["vessel_id_int"][0])
                    context_meta = {
                        "timestamp_sec": int(df_vessel_chunk["timestamp_sec"][0]), 
                        "trip_id": str(df_vessel_chunk["trip_id"][0]),
                        "superclass_id": int(df_vessel_chunk["vessel_superclass_id"][0])
                    }

                    detector.profile_engine = RollingBehaviorProfileEngine(alpha=0.1)

                    res = evaluate_vessel_windows(
                        df_vessel_chunk, detector, injector, v_id, context_meta, generate_anomalies=False, ann_calibration=None
                    )
                    
                    if res["window_records_normal"]:
                        w_recon = [w["reconstruction_score"] for w in res["window_records_normal"]]
                        w_ann   = [w["ann_score"] for w in res["window_records_normal"]]
                        w_fused = [w["fused_score"] for w in res["window_records_normal"]]
                        
                        raw_calib_ann_distances.extend(w_ann)
                        
                        vessel_window_groups_recon.append(w_recon)
                        vessel_window_groups_ann.append(w_ann)
                        vessel_window_groups_fused.append(w_fused)
                        
                        total_cal_windows_parsed += res["base_windows_parsed"]
                        total_cal_inference_passes += res["inference_passes_run"]
                        
                        vessels_processed_cal += 1
                        if vessels_processed_cal % 50 == 0:
                            print(f"   --> [Phase A] Indexed {vessels_processed_cal} vessel profiles into baseline calibration matrices...")

    ann_p50 = float(np.percentile(raw_calib_ann_distances, 50))
    ann_p99 = float(np.percentile(raw_calib_ann_distances, 99))

    ann_spread = ann_p99 - ann_p50
    if ann_spread < 1e-4:
        logging.warning(
            "⚠️ ANN calibration spread is extremely small "
            "(p50=%.6f, p99=%.6f, Δ=%.6e). "
            "Normalized ANN scores may become excessively amplified. "
            "This may indicate an overly concentrated embedding manifold or "
            "insufficient diversity in the calibration dataset.",
            ann_p50,
            ann_p99,
            ann_spread,
        )

    ann_calibration_tuple = (ann_p50, ann_p99)
    
    logging.info(f"🎯 Sequence Reconstruction Manifold Calibration | ANN p50: {ann_p50:.4f} | ANN p99: {ann_p99:.4f}")

    # REQ 3: Apply smooth, noise-resistant mean-only window aggregation for baseline threshold calibration
    calibration_vessel_recon = [aggregate_score(group, mean_weight=1.0) for group in vessel_window_groups_recon]
    calibration_vessel_fused = [aggregate_score(group, mean_weight=1.0) for group in vessel_window_groups_fused]
    
    calibration_vessel_ann = []
    for group in vessel_window_groups_ann:
        scaled_group = [np.maximum(0.0, (d - ann_p50) / (ann_p99 - ann_p50 + 1e-8)) for d in group]
        calibration_vessel_ann.append(aggregate_score(scaled_group, mean_weight=1.0))

    # REQ 8: Dynamic named percentile token configurations
    threshold_recon = float(np.percentile(calibration_vessel_recon, CALIBRATION_PERCENTILE))
    threshold_ann   = float(np.percentile(calibration_vessel_ann, CALIBRATION_PERCENTILE))
    threshold_fused = float(np.percentile(calibration_vessel_fused, CALIBRATION_PERCENTILE))

    # Compute parametric properties for standard reference logging
    mu_recon, sigma_recon = float(np.mean(calibration_vessel_recon)), float(np.std(calibration_vessel_recon))
    mu_ann, sigma_ann     = float(np.mean(calibration_vessel_ann)), float(np.std(calibration_vessel_ann))
    mu_fused, sigma_fused   = float(np.mean(calibration_vessel_fused)), float(np.std(calibration_vessel_fused))

    logging.info(f"🎯 Independent Cutoffs Settled | Recon Threshold: {threshold_recon:.4f} | ANN Threshold: {threshold_ann:.4f} | Fused Threshold: {threshold_fused:.4f}")

    # ==========================================
    # PHASE B: FINAL BENCHMARK TESTING (OCT-DEC)
    # ==========================================
    logging.info("🏁 PHASE B: Ingesting Production Test Splits to Calculate Independent Channel Metrics...")
    
    # Independent statistics metrics tracking arrays
    y_true_recon, y_scores_recon = [], []
    y_true_ann, y_scores_ann     = [], []
    y_true_fused, y_scores_fused = [], []
    
    # Combined strategy score and decision tracking containers
    y_true_or, y_pred_or, y_scores_combined_decision = [], [], []
    y_true_and, y_pred_and = [], []
    y_true_joint, y_pred_joint = [], []
    
    severity_distribution_ledger = {"critical": 0, "reconstruction": 0, "manifold": 0, "normal": 0}
    anomaly_distribution_ledger = {
        mode: {"recon_scores": [], "ann_scores": [], "fused_scores": [], "severities": []} for mode in ANOMALY_MODES
    }

    for m_int in BENCHMARK_MONTHS:
        mock_live_path = os.path.join(processed_lakehouse_root, "year=2019", f"month={m_int:02d}")
        logging.info(f"🚨 [STREAMFEED] Ingesting evaluation frames for timeline partition: month={m_int:02d}")
        if not os.path.exists(mock_live_path): continue    
        
        for root, _, files in os.walk(mock_live_path):
            for file in files:
                if file.endswith(".parquet"):
                    df_vessel_chunk = pl.read_parquet(os.path.join(root, file))
                    if df_vessel_chunk.height < MAX_WINDOW_SIZE: continue
                    
                    v_id = int(df_vessel_chunk["vessel_id_int"][0])
                    context_meta = {
                        "timestamp_sec": int(df_vessel_chunk["timestamp_sec"][0]), 
                        "trip_id": str(df_vessel_chunk["trip_id"][0]),
                        "superclass_id": int(df_vessel_chunk["vessel_superclass_id"][0])
                    }

                    detector.profile_engine = RollingBehaviorProfileEngine(alpha=0.1)

                    res = evaluate_vessel_windows(
                        df_vessel_chunk, detector, injector, v_id, context_meta, generate_anomalies=True, ann_calibration=ann_calibration_tuple
                    )
                    if res["base_windows_parsed"] == 0: continue

                    # 1. Process Normal Baseline Channels (Label = 0) -> REQ 3: Normal maps to pure mean
                    v_recon_normal = aggregate_score([w["reconstruction_score"] for w in res["window_records_normal"]], mean_weight=1.0)
                    v_ann_normal   = aggregate_score([w["ann_score"] for w in res["window_records_normal"]], mean_weight=1.0)
                    v_fused_normal = aggregate_score([w["fused_score"] for w in res["window_records_normal"]], mean_weight=1.0)
                    
                    y_true_recon.append(0); y_scores_recon.append(v_recon_normal)
                    y_true_ann.append(0);   y_scores_ann.append(v_ann_normal)
                    y_true_fused.append(0); y_scores_fused.append(v_fused_normal)
                    
                    alert_recon_n = v_recon_normal >= threshold_recon
                    alert_ann_n   = v_ann_normal >= threshold_ann
                    
                    # REQ 4 & 5: Combined decision scores and joint contour warnings
                    joint_warning_n = (v_recon_normal >= JOINT_WARNING_FACTOR * threshold_recon) and (v_ann_normal >= JOINT_WARNING_FACTOR * threshold_ann)
                    final_joint_alert_n = alert_recon_n or alert_ann_n or joint_warning_n
                    
                    risk_combined_normal = max(v_recon_normal / (threshold_recon + 1e-8), v_ann_normal / (threshold_ann + 1e-8))
                    
                    y_true_or.append(0);  y_pred_or.append(int(alert_recon_n or alert_ann_n)); y_scores_combined_decision.append(risk_combined_normal)
                    y_true_and.append(0); y_pred_and.append(int(alert_recon_n and alert_ann_n))
                    y_true_joint.append(0); y_pred_joint.append(int(final_joint_alert_n))
                    
                    if alert_recon_n and alert_ann_n:   severity_distribution_ledger["critical"] += 1
                    elif alert_recon_n:                 severity_distribution_ledger["reconstruction"] += 1
                    elif alert_ann_n:                   severity_distribution_ledger["manifold"] += 1
                    else:                               severity_distribution_ledger["normal"] += 1

                    # 2. Process Anomalous Channels (Label = 1) -> REQ 3: Anomalies map to peak-sensitive
                    v_recon_anom = aggregate_score([w["reconstruction_score"] for w in res["window_records_anomaly"]], mean_weight=0.4, peak_percentile=95)
                    v_ann_anom   = aggregate_score([w["ann_score"] for w in res["window_records_anomaly"]], mean_weight=0.4, peak_percentile=95)
                    v_fused_anom = aggregate_score([w["fused_score"] for w in res["window_records_anomaly"]], mean_weight=0.4, peak_percentile=95)
                    
                    y_true_recon.append(1); y_scores_recon.append(v_recon_anom)
                    y_true_ann.append(1);   y_scores_ann.append(v_ann_anom)
                    y_true_fused.append(1); y_scores_fused.append(v_fused_anom)
                    
                    alert_recon_a = v_recon_anom >= threshold_recon
                    alert_ann_a   = v_ann_anom >= threshold_ann
                    
                    joint_warning_a = (v_recon_anom >= JOINT_WARNING_FACTOR * threshold_recon) and (v_ann_anom >= JOINT_WARNING_FACTOR * threshold_ann)
                    final_joint_alert_a = alert_recon_a or alert_ann_a or joint_warning_a
                    
                    risk_combined_anom = max(v_recon_anom / (threshold_recon + 1e-8), v_ann_anom / (threshold_ann + 1e-8))
                    
                    y_true_or.append(1);  y_pred_or.append(int(alert_recon_a or alert_ann_a)); y_scores_combined_decision.append(risk_combined_anom)
                    y_true_and.append(1); y_pred_and.append(int(alert_recon_a and alert_ann_a))
                    y_true_joint.append(1); y_pred_joint.append(int(final_joint_alert_a))
                    
                    if alert_recon_a and alert_ann_a:   severity_distribution_ledger["critical"] += 1
                    elif alert_recon_a:                 severity_distribution_ledger["reconstruction"] += 1
                    elif alert_ann_a:                   severity_distribution_ledger["manifold"] += 1
                    else:                               severity_distribution_ledger["normal"] += 1

                    # 3. Update Anomaly Ledger Matrix
                    for mode in ANOMALY_MODES:
                        for record in res["anomaly_records"][mode]:
                            anomaly_distribution_ledger[mode]["recon_scores"].append(record["reconstruction_score"])
                            anomaly_distribution_ledger[mode]["ann_scores"].append(record["ann_score"])
                            anomaly_distribution_ledger[mode]["fused_scores"].append(record["fused_score"])
                            anomaly_distribution_ledger[mode]["severities"].append(record["severity"])

                    total_bench_windows_parsed += res["base_windows_parsed"]
                    total_bench_inference_passes += res["inference_passes_run"]
                    
                    # ROLLING IN-PLACE TELEMETRY INDICATOR
                    # sys.stdout.write(
                    #     f"\r🚀 Evaluating Test Stream: Vessel {v_id:5d} | "
                    #     f"Recon MSE: {v_recon_normal:6.4f} | ANN Risk: {v_ann_normal:4.2f} | "
                    #     f"Decision Pairs: {len(y_true_recon):4d}"
                    # )
                    # sys.stdout.flush()
                    print(
                        f"🚀 Evaluating Test Stream: Vessel {v_id:5d} | "
                        f"Recon MSE: {v_recon_normal:6.4f} | ANN Risk: {v_ann_normal:4.2f} | "
                        f"Decision Pairs: {len(y_true_recon):4d}"
                    )
    print("\n")

    # ==========================================
    # FINAL METRIC CONSOLIDATION & OUTPUT RENDERING
    # ==========================================
    if y_true_recon:
        # REQ 6: Helper function computes metrics alongside structural confusion matrices
        def compute_channel_metrics(y_true, y_scores, cutoff_threshold):
            y_t = np.array(y_true)
            y_s = np.array(y_scores)
            y_p = (y_s >= cutoff_threshold).astype(int)
            return {
                "auroc": float(roc_auc_score(y_t, y_s)) if len(np.unique(y_t)) > 1 else 0.5,
                "f1": float(f1_score(y_t, y_p, zero_division=0)),
                "precision": float(precision_score(y_t, y_p, zero_division=0)),
                "recall": float(recall_score(y_t, y_p, zero_division=0)),
                "confusion_matrix": {
                    "tp": int(np.sum((y_t == 1) & (y_p == 1))),
                    "fp": int(np.sum((y_t == 0) & (y_p == 1))),
                    "tn": int(np.sum((y_t == 0) & (y_p == 0))),
                    "fn": int(np.sum((y_t == 1) & (y_p == 0)))
                }
            }

        def compute_discrete_metrics(y_true, y_pred):
            y_t = np.array(y_true)
            y_p = np.array(y_pred)
            return {
                "f1": float(f1_score(y_t, y_p, zero_division=0)),
                "precision": float(precision_score(y_t, y_p, zero_division=0)),
                "recall": float(recall_score(y_t, y_p, zero_division=0)),
                "confusion_matrix": {
                    "tp": int(np.sum((y_t == 1) & (y_p == 1))),
                    "fp": int(np.sum((y_t == 0) & (y_p == 1))),
                    "tn": int(np.sum((y_t == 0) & (y_p == 0))),
                    "fn": int(np.sum((y_t == 1) & (y_p == 0)))
                }
            }

        def extract_percentiles(scores):
            if not scores: return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
            s = np.array(scores)
            return {
                "p50": round(float(np.percentile(s, 50)), 4),
                "p90": round(float(np.percentile(s, 90)), 4),
                "p95": round(float(np.percentile(s, 95)), 4),
                "p99": round(float(np.percentile(s, 99)), 4)
            }

        # Separate normal vs anomaly slices to pull distribution statistics
        recon_norm_scores = [s for s, label in zip(y_scores_recon, y_true_recon) if label == 0]
        recon_anom_scores = [s for s, label in zip(y_scores_recon, y_true_recon) if label == 1]
        ann_norm_scores   = [s for s, label in zip(y_scores_ann, y_true_ann) if label == 0]
        ann_anom_scores   = [s for s, label in zip(y_scores_ann, y_true_ann) if label == 1]
        fused_norm_scores = [s for s, label in zip(y_scores_fused, y_true_fused) if label == 0]
        fused_anom_scores = [s for s, label in zip(y_scores_fused, y_true_fused) if label == 1]

        metrics_recon = compute_channel_metrics(y_true_recon, y_scores_recon, threshold_recon)
        metrics_ann   = compute_channel_metrics(y_true_ann, y_scores_ann, threshold_ann)
        
        # REQ 5: Shift fusion metrics out of independent detector slots
        metrics_fused_priorities = compute_channel_metrics(y_true_fused, y_scores_fused, threshold_fused)

        y_true_or_arr, y_pred_or_arr = np.array(y_true_or), np.array(y_pred_or)
        y_true_and_arr, y_pred_and_arr = np.array(y_true_and), np.array(y_pred_and)
        y_true_joint_arr, y_pred_joint_arr = np.array(y_true_joint), np.array(y_pred_joint)

        # REQ 3: Renamed metrics tracking indicator
        combined_decision_score_continuous_auroc = float(roc_auc_score(y_true_or_arr, np.array(y_scores_combined_decision)))

        metrics_or_gate = compute_discrete_metrics(y_true_or_arr, y_pred_or_arr)
        metrics_and_gate = compute_discrete_metrics(y_true_and_arr, y_pred_and_arr)
        metrics_joint_contour = compute_discrete_metrics(y_true_joint_arr, y_pred_joint_arr)

        # Format scenario analytics breakdowns safely
        formatted_anomaly_stats = {}
        for mode in ANOMALY_MODES:
            r_sc = anomaly_distribution_ledger[mode]["recon_scores"]
            a_sc = anomaly_distribution_ledger[mode]["ann_scores"]
            f_sc = anomaly_distribution_ledger[mode]["fused_scores"]
            sevs = anomaly_distribution_ledger[mode]["severities"]
            
            def calculate_safely_corr(x, y):
                if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                    return round(float(np.corrcoef(y, x)[0, 1]), 4)
                return 0.0

            formatted_anomaly_stats[mode] = {
                "sample_count": len(r_sc),
                "sequence_reconstruction": {
                    "mean_mse": round(float(np.mean(r_sc)), 4) if r_sc else 0.0,
                    "severity_correlation": calculate_safely_corr(r_sc, sevs)
                },
                "manifold_ann": {
                    "mean_risk": round(float(np.mean(a_sc)), 4) if a_sc else 0.0,
                    "severity_correlation": calculate_safely_corr(a_sc, sevs)
                },
                "prioritization_fusion": {
                    "mean_score": round(float(np.mean(f_sc)), 4) if f_sc else 0.0,
                    "severity_correlation": calculate_safely_corr(f_sc, sevs)
                }
            }

        final_ledger_report = {
            "meta_config_constants": {
                "calibration_percentile_cutoff": CALIBRATION_PERCENTILE,
                "joint_warning_factor_multiplier": JOINT_WARNING_FACTOR,
                # PRESERVE ARCHITECTURE: Save the quantile anchors dynamically to disk
                "ann_p50": round(float(ann_p50), 4),
                "ann_p99": round(float(ann_p99), 4)
            },
            "calibration_baselines": {
                "sequence_reconstruction": {"mu": round(mu_recon, 4), "sigma": round(sigma_recon, 4), "threshold": round(threshold_recon, 4)},
                "manifold_ann": {"mu": round(mu_ann, 4), "sigma": round(sigma_ann, 4), "threshold": round(threshold_ann, 4)},
                "prioritization_fusion": {"mu": round(mu_fused, 4), "sigma": round(sigma_fused, 4), "threshold": round(threshold_fused, 4)}
            },
            "isolated_detector_channels": {
                "sequence_reconstruction": metrics_recon,
                "manifold_ann_distance": metrics_ann
            },
            "combinatorial_decision_analysis": {
                "combined_decision_score_continuous_auroc": round(combined_decision_score_continuous_auroc, 4),
                "or_gate_discrete_behavior": metrics_or_gate,
                "and_gate_discrete_behavior": metrics_and_gate,
                "joint_contour_discrete_behavior": metrics_joint_contour
            },
            "prioritization_ranking_metrics": {
                "fused_prioritization_index": metrics_fused_priorities
            },
            "score_distribution_percentiles": {
                "sequence_reconstruction": {
                    "normal_profile": extract_percentiles(recon_norm_scores),
                    "anomaly_profile": extract_percentiles(recon_anom_scores)
                },
                "manifold_ann_distance": {
                    "normal_profile": extract_percentiles(ann_norm_scores),
                    "anomaly_profile": extract_percentiles(ann_anom_scores)
                },
                "prioritization_fusion": {
                    "normal_profile": extract_percentiles(fused_norm_scores),
                    "anomaly_profile": extract_percentiles(fused_anom_scores)
                }
            },
            "severity_taxonomy_counters": severity_distribution_ledger,
            "per_scenario_manifold_breakdown": formatted_anomaly_stats,
            "telemetry_execution_counters": {
                "calibration_windows": total_cal_windows_parsed,
                "benchmark_windows": total_bench_windows_parsed,
                "total_transformer_forward_passes": total_bench_inference_passes + total_cal_inference_passes
            }
        }

        with open(output_metrics_json, "w") as f:
            json.dump(final_ledger_report, f, indent=4)

        print("\n" + "="*75)
        print("📊 NAVISIGHT PLATFORM INDEPENDENT MULTI-CHANNEL RUN COMPLETE")
        print("="*75)
        print(f"• Sequence Reconstruction Channel   | AUROC: {metrics_recon['auroc']:.4f} | F1: {metrics_recon['f1']:.4f} | Recall: {metrics_recon['recall']:.4f}")
        print(f"• Manifold ANN Distance Channel     | AUROC: {metrics_ann['auroc']:.4f}   | F1: {metrics_ann['f1']:.4f} | Recall: {metrics_ann['recall']:.4f}")
        print("-"*75)
        print(f"• Strategy: Combined Decision Score | AUROC: {combined_decision_score_continuous_auroc:.4f}")
        print(f"• Strategy: Discrete [OR] Gate      | F1: {metrics_or_gate['f1']:.4f} | Precision: {metrics_or_gate['precision']:.4f} | Recall: {metrics_or_gate['recall']:.4f}")
        print(f"• Strategy: Discrete [AND] Gate     | F1: {metrics_and_gate['f1']:.4f} | Precision: {metrics_and_gate['precision']:.4f} | Recall: {metrics_and_gate['recall']:.4f}")
        print(f"• Strategy: Discrete [JOINT] Gate   | F1: {metrics_joint_contour['f1']:.4f} | Precision: {metrics_joint_contour['precision']:.4f} | Recall: {metrics_joint_contour['recall']:.4f}")
        print("="*75 + "\n")
    else:
        logging.error("❌ Execution terminated with zero processed track units.")


if __name__ == "__main__":
    main()