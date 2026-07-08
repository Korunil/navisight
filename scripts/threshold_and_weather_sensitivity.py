# scripts/threshold_and_weather_sensitivity.py
"""
NAVISIGHT SENSITIVITY STUDY Pipeline
-----------------------------------------------
1. Task 1: Inference-Time Environmental Feature Sensitivity Analysis
   (CRITICAL METHODOLOGICAL NOTE: This is an Inference-Time Sensitivity Analysis, 
    NOT a true structural training ablation study.)
2. Task 2: Threshold Sensitivity Analysis & Operational Frontier Mapping
"""

import os
import sys
import json
import logging
import hashlib
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NaviSight.ReviewerStudies")

# Set high-fidelity visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

def load_canonical_benchmarks():
    """Loads existing baseline thresholds and normalization constants from disk."""
    ledger_path = os.path.join(PROJECT_ROOT, "configs", "evaluation_results.json")
    if not os.path.exists(ledger_path):
        raise FileNotFoundError(f"Missing existing benchmark artifacts ledger at: {ledger_path}")
        
    with open(ledger_path, "r") as f:
        data = json.load(f)
        
    meta = data["meta_config_constants"]
    baselines = data["calibration_baselines"]
    
    return {
        "ann_p50": meta["ann_p50"],
        "ann_p99": meta["ann_p99"],
        "thresh_recon": baselines["sequence_reconstruction"]["threshold"],
        "thresh_ann": baselines["manifold_ann"]["threshold"],
        "joint_factor": meta["joint_warning_factor_multiplier"]
    }

def load_global_weather_centers():
    """Natively extracts the un-normalized calibration center values from global_stats.json."""
    stats_path = os.path.join(PROJECT_ROOT, "configs", "global_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Critical configuration asset missing: {stats_path}")
        
    with open(stats_path, "r") as f:
        stats_space = json.load(f)
        
    return stats_space.get("weather_meso_scales", {})

def main():
    logger.info("🛰️ INITIATING NAVISIGHT CHRONOLOGICALLY ANCHORED REVIEWER ENGINE")
    
    # 1. Reuse existing benchmark configuration properties
    constants = load_canonical_benchmarks()
    weather_meso_scales = load_global_weather_centers()
    
    # 2. Re-instantiate model layers matching eval_pipeline.py specifications
    checkpoint_directory = "models/checkpoints/"
    global_stats_json_path = "configs/global_stats.json"
    index_directory = "models/state/index"
    
    # Defined clean target destination directory path variable
    output_dir = "configs"
    
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
        best_checkpoint = sorted([os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")])[-1]
        
    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    initial_profile = RollingBehaviorProfileEngine(alpha=0.1)
    detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, ann_index, initial_profile)
    injector = ManifoldAwareAnomalyPerturbationEngine()
    
    if hasattr(detector, "model") and detector.model is not None:
        detector.model.eval()

    # 3. Discover weather vectors inside registry features
    weather_features = [f for f in REGISTRY.weather]
    logger.info(f"Identified {len(weather_features)} environmental weather dimensions for sensitivity testing.")

    # 4. Ingest Trajectories over Benchmark Splits (Locked strictly to Month 11)
    processed_lakehouse_root = "data/processed/"
    
    benchmark_months = [11]
    ANOMALY_MODES = ["dead_reckoning", "coastal_creep", "loitering"]
    MAX_WINDOW_SIZE = 200
    STRIDE = 100
    
    evaluation_cache = []
    TIMESTAMP_MAX_LIMIT = 1573862399  # Nov 15, 2019 11:59:59 PM

    for m_int in benchmark_months:
        month_path = os.path.join(processed_lakehouse_root, "year=2019", f"month={m_int:02d}")
        if not os.path.exists(month_path): 
            continue
        
        for root, _, files in os.walk(month_path):
            for file in sorted(files):
                if file.endswith(".parquet"):
                    df_vessel = pl.read_parquet(os.path.join(root, file))
                    if df_vessel.height < MAX_WINDOW_SIZE: 
                        continue
                    
                    v_id = int(df_vessel["vessel_id_int"][0])
                    context_meta = {
                        "timestamp_sec": int(df_vessel["timestamp_sec"][0]), 
                        "trip_id": str(df_vessel["trip_id"][0]),
                        "superclass_id": int(df_vessel["vessel_superclass_id"][0])
                    }
                    
                    detector.profile_engine = RollingBehaviorProfileEngine(alpha=0.1)
                    
                    for start_idx in range(0, df_vessel.height - MAX_WINDOW_SIZE + 1, STRIDE * 2):
                        df_slice = df_vessel.slice(start_idx, MAX_WINDOW_SIZE)
                        
                        # Validate window sequence context explicitly against the timestamp cap
                        current_window_time = int(df_slice["timestamp_sec"][0])
                        if current_window_time > TIMESTAMP_MAX_LIMIT:
                            # Escape the window loop for this specific chronological file, 
                            # but allow the outer loops to keep parsing other vessel subdirectories
                            logging.info(f"Time stamp exceed for {v_id}, moving to next vessel")
                            break
                        
                        # Sequential evaluation pairing across normal and anomalous trajectories
                        for is_anomaly in [0, 1]:
                            if is_anomaly == 0:
                                df_target = df_slice
                            else:
                                seed_hash = int(hashlib.md5(f"{v_id}_{start_idx}".encode()).hexdigest(), 16)
                                rng = np.random.default_rng(seed_hash & 0xFFFFFFFF)
                                anomaly_mode = ANOMALY_MODES[seed_hash % len(ANOMALY_MODES)]
                                severity = float(np.clip(rng.normal(0.5, 0.2), 0.1, 0.9))
                                df_target = injector.inject_counterfactual_scenario(df_slice, anomaly_mode, severity)
                                
                            # --- CONFIGURATION A: PRODUCTION BASELINE ---
                            res_a = detector.evaluate_live_sequence_anomaly_score(df_target, v_id, context_meta, mode="production")
                            
                            # --- CONFIGURATION B: INFERENCE-TIME ENVIRONMENTAL NEUTRALIZATION ---
                            df_neutralized = df_target.clone()
                            
                            # Overriding raw parameters using their exact un-normalized calibration center values
                            for wf in weather_features:
                                if wf in df_neutralized.columns:
                                    center_val = weather_meso_scales.get(wf, {"center": 0.0})["center"]
                                    df_neutralized = df_neutralized.with_columns(pl.lit(center_val).alias(wf))
                                    
                            res_b = detector.evaluate_live_sequence_anomaly_score(df_neutralized, v_id, context_meta, mode="production")
                            
                            evaluation_cache.append({
                                "is_anomaly": is_anomaly,
                                "recon_a": res_a["reconstruction_mse"],
                                "ann_a": float(np.maximum(0.0, (res_a["ood_score"] - constants["ann_p50"]) / (constants["ann_p99"] - constants["ann_p50"] + 1e-8))),
                                "recon_b": res_b["reconstruction_mse"],
                                "ann_b": float(np.maximum(0.0, (res_b["ood_score"] - constants["ann_p50"]) / (constants["ann_p99"] - constants["ann_p50"] + 1e-8)))
                            })

    # ==========================================================================
    # TASK 1: ENVIRONMENTAL FEATURE SENSITIVITY CALCULATION
    # ==========================================================================
    logger.info("📊 PROCESSING TASK 1: ENVIRONMENTAL SENSITIVITY PROFILE")
    y_true = np.array([w["is_anomaly"] for w in evaluation_cache])
    
    def extract_channel_metrics(y_t, scores, thresh):
        preds = (scores >= thresh).astype(int)
        return {
            "auroc": round(float(roc_auc_score(y_t, scores)), 4),
            "precision": round(float(precision_score(y_t, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_t, preds, zero_division=0)), 4),
            "auprc": round(float(average_precision_score(y_t, scores)), 4),
            "f1": round(float(f1_score(y_t, preds, zero_division=0)), 4)
        }

    # Extract continuous scores across combinations
    rec_a = np.array([w["recon_a"] for w in evaluation_cache])
    ann_a = np.array([w["ann_a"] for w in evaluation_cache])
    rec_b = np.array([w["recon_b"] for w in evaluation_cache])
    ann_b = np.array([w["ann_b"] for w in evaluation_cache])
    
    # Compute combinatorial OR-gate tracking arrays
    or_preds_a = ((rec_a >= constants["thresh_recon"]) | (ann_a >= constants["thresh_ann"])).astype(int)
    or_preds_b = ((rec_b >= constants["thresh_recon"]) | (ann_b >= constants["thresh_ann"])).astype(int)

    weather_report = {
        "Configuration_A_Baseline": {
            "reconstruction": extract_channel_metrics(y_true, rec_a, constants["thresh_recon"]),
            "manifold": extract_channel_metrics(y_true, ann_a, constants["thresh_ann"]),
            "or_fusion": {
                "precision": round(float(precision_score(y_true, or_preds_a, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, or_preds_a, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, or_preds_a, zero_division=0)), 4)
            }
        },
        "Configuration_B_Weather_Neutralized": {
            "reconstruction": extract_channel_metrics(y_true, rec_b, constants["thresh_recon"]),
            "manifold": extract_channel_metrics(y_true, ann_b, constants["thresh_ann"]),
            "or_fusion": {
                "precision": round(float(precision_score(y_true, or_preds_b, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, or_preds_b, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, or_preds_b, zero_division=0)), 4)
            }
        }
    }

    # FIXED: Re-mapped JSON output path to output_dir
    with open(os.path.join(output_dir, "weather/weather_sensitivity_report.json"), "w") as f:
        json.dump(weather_report, f, indent=4)
        
    df_weather_csv = pl.DataFrame([
        {
            "Configuration": "Configuration A (Baseline)",
            "Recon AUROC": weather_report["Configuration_A_Baseline"]["reconstruction"]["auroc"],
            "Manifold AUROC": weather_report["Configuration_A_Baseline"]["manifold"]["auroc"],
            "Fusion F1": weather_report["Configuration_A_Baseline"]["or_fusion"]["f1"]
        },
        {
            "Configuration": "Configuration B (Weather Neutralized)",
            "Recon AUROC": weather_report["Configuration_B_Weather_Neutralized"]["reconstruction"]["auroc"],
            "Manifold AUROC": weather_report["Configuration_B_Weather_Neutralized"]["manifold"]["auroc"],
            "Fusion F1": weather_report["Configuration_B_Weather_Neutralized"]["or_fusion"]["f1"]
        }
    ])
    # FIXED: Re-mapped CSV output path to output_dir
    df_weather_csv.write_csv(os.path.join(output_dir, "weather/weather_sensitivity_matrix.csv"))

    # Plot Task 1 Distribution Overlaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(rec_a[y_true == 0], bins=25, alpha=0.5, density=True, label="Normal (Base)", color="#10B981")
    axes[0].hist(rec_a[y_true == 1], bins=25, alpha=0.5, density=True, label="Anomalous (Base)", color="#EF4444")
    axes[0].hist(rec_b[y_true == 1], bins=25, alpha=0.4, density=True, label="Anomalous (Neutralized)", color="#3B82F6", histtype="step", linewidth=2, linestyle="--")
    axes[0].set_title("Reconstruction Channel Profile Density Slices")
    axes[0].set_xlabel("Reconstruction MSE Score")
    axes[0].legend()

    axes[1].hist(ann_a[y_true == 0], bins=25, alpha=0.5, density=True, label="Normal (Base)", color="#10B981")
    axes[1].hist(ann_a[y_true == 1], bins=25, alpha=0.5, density=True, label="Anomalous (Base)", color="#EF4444")
    axes[1].hist(ann_b[y_true == 1], bins=25, alpha=0.4, density=True, label="Anomalous (Neutralized)", color="#3B82F6", histtype="step", linewidth=2, linestyle="--")
    axes[1].set_title("Manifold ANN Channel Profile Density Slices")
    axes[1].set_xlabel("Calibrated Latent OOD Distance")
    axes[1].legend()
    
    plt.suptitle("Inference-Time Environmental Feature Sensitivity: Density Displacements", y=0.98, fontweight="bold")
    plt.tight_layout()
    # FIXED: Re-mapped PNG layout destination to output_dir
    plt.savefig(os.path.join(output_dir, "weather/weather_distribution_degradation.png"), dpi=300)
    plt.close()

    # ==========================================================================
    # TASK 2: OPERATIONAL THRESHOLD SENSITIVITY FRONTIER
    # ==========================================================================
    logger.info("📊 PROCESSING TASK 2: OPERATIONAL THRESHOLD FRONTIER SENSITIVITY")
    percentiles = [90.0, 92.5, 95.0, 97.5, 98.0, 99.0, 99.5, 99.7, 99.9]
    
    # Isolate normal continuous background profiles to map the calibration cuts
    norm_rec_pool = rec_a[y_true == 0]
    norm_ann_pool = ann_a[y_true == 0]
    
    recon_frontier, manifold_frontier, fusion_frontier = [], [], []

    for p in percentiles:
        t_r = float(np.percentile(norm_rec_pool, p))
        t_a = float(np.percentile(norm_ann_pool, p))
        
        p_recon = (rec_a >= t_r).astype(int)
        p_ann = (ann_a >= t_a).astype(int)
        p_fuse = (p_recon | p_ann).astype(int)
        
        recon_frontier.append({"Percentile": p, "Threshold": t_r, "Precision": precision_score(y_true, p_recon, zero_division=0), "Recall": recall_score(y_true, p_recon, zero_division=0), "F1": f1_score(y_true, p_recon, zero_division=0)})
        manifold_frontier.append({"Percentile": p, "Threshold": t_a, "Precision": precision_score(y_true, p_ann, zero_division=0), "Recall": recall_score(y_true, p_ann, zero_division=0), "F1": f1_score(y_true, p_ann, zero_division=0)})
        fusion_frontier.append({"Percentile": p, "Precision": precision_score(y_true, p_fuse, zero_division=0), "Recall": recall_score(y_true, p_fuse, zero_division=0), "F1": f1_score(y_true, p_fuse, zero_division=0)})

    # FIXED: Re-mapped Task 2 CSV/JSON storage targets to output_dir
    for key, data in [("reconstruction", recon_frontier), ("manifold", manifold_frontier), ("or_fusion", fusion_frontier)]:
        pl.DataFrame(data).write_csv(os.path.join(output_dir, f"/threshold/{key}_threshold_sensitivity.csv"))
        with open(os.path.join(output_dir, f"/threshold/{key}_threshold_sensitivity.json"), "w") as f:
            json.dump(data, f, indent=4)

    # Plot individual parameters (Figures 1-3)
    metrics_keys = ["Precision", "Recall", "F1"]
    colors = {"reconstruction": "#F59E0B", "manifold": "#6366F1", "or_fusion": "#10B981"}
    
    for mk in metrics_keys:
        plt.figure(figsize=(7, 4.5))
        plt.plot(percentiles, [w[mk] for w in recon_frontier], marker='o', color=colors["reconstruction"], label="Reconstruction Channel", linewidth=1.8)
        plt.plot(percentiles, [w[mk] for w in manifold_frontier], marker='s', color=colors["manifold"], label="Manifold ANN Channel", linewidth=1.8)
        plt.plot(percentiles, [w[mk] for w in fusion_frontier], marker='^', color=colors["or_fusion"], label="OR-Gate Fusion", linewidth=2)
        plt.xlabel("Calibration Cutoff Percentile (%)")
        plt.ylabel(f"Operational {mk}")
        plt.title(f"Operational Boundary Response: {mk} Scaling Metrics")
        plt.legend()
        plt.tight_layout()
        # FIXED: Re-mapped PNG layout destination to output_dir
        plt.savefig(os.path.join(output_dir, f"/threshold/threshold_vs_{mk.lower()}.png"), dpi=300)
        plt.close()

    # Figure 4: Master Unified Combined Plot for OR-Gate Fusion
    plt.figure(figsize=(9, 5.5))
    plt.plot(percentiles, [w["Precision"] for w in fusion_frontier], marker='o', color="#10B981", label="Operational Precision", linewidth=2.5)
    plt.plot(percentiles, [w["Recall"] for w in fusion_frontier], marker='s', color="#EF4444", label="Operational Recall (Coverage)", linewidth=2.5)
    plt.plot(percentiles, [w["F1"] for w in fusion_frontier], marker='^', color="#3B82F6", label="Operational F1-Score Balance Point", linewidth=2.5)
    plt.axvline(x=99.5, color="#8892b0", linestyle="--", alpha=0.8, label="Production Control Cutoff Baseline (99.5%)")
    plt.xlabel("Calibration Cutoff Percentile (%)", fontweight="bold")
    plt.ylabel("Operational Performance Profile Metric", fontweight="bold")
    plt.title("Multi-Channel Decision Space: OR-Gate Operational Tradeoff Frontier", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    # FIXED: Re-mapped PNG layout destination to output_dir
    plt.savefig(os.path.join(output_dir, "/threshold/combined_fusion_frontier.png"), dpi=300)
    plt.close()

    logger.info("  ALL SENSITIVITY STUDY METRICS AND GRAPHIC ARTIFACTS COMPILED SUCCESSFULLY")

if __name__ == "__main__":
    main()