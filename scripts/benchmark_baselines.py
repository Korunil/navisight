# scripts/benchmark_baselines.py
import os
import sys
import logging
import json
import random
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION MATCHING PRIMARY PIPELINE ---
GLOBAL_SEED = 42
CALIBRATION_MONTHS = [7, 8, 9]
BENCHMARK_MONTHS = [10, 11, 12]
MAX_WINDOW_SIZE = 200
STRIDE = 100
FEATURE_DIM = len(REGISTRY.all_features)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional strict determinism flag (warn_only prevents crashing on non-deterministic operations)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(GLOBAL_SEED)

# ==========================================================================
# TIMESERIES RECURRENT BASELINE: LSTM AUTOENCODER MODEL
# ==========================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hidden, cell) = self.encoder(x)
        dec_in = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        dec_out, _ = self.decoder(dec_in, (hidden, cell))
        return self.output_layer(dec_out)


def prepare_statistical_window(df_slice):
    """Generates the high-fidelity 246-dimensional statistical feature vector."""
    raw_array = df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32)
    v_mean = np.mean(raw_array, axis=0)
    v_std  = np.std(raw_array, axis=0)
    v_min  = np.min(raw_array, axis=0)
    v_max  = np.max(raw_array, axis=0)
    v_p25  = np.percentile(raw_array, 25, axis=0)
    v_p75  = np.percentile(raw_array, 75, axis=0)
    return np.concatenate([v_mean, v_std, v_min, v_max, v_p25, v_p75])


def find_optimal_f1_threshold(y_true, y_scores):
    """Identifies the validation threshold maximizing operational F1 performance."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    best_f1 = 0.0
    best_threshold = 0.0
    
    # Sweep over 200 distributed percentile points across the calculated score space
    threshold_grid = np.percentile(y_scores, np.linspace(1, 99, 200))
    for thresh in threshold_grid:
        preds = (y_scores >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
            
    return best_threshold


def main():
    logging.info("🛰️ INITIATING SCIENTIFICALLY HARDENED OPTIMIZED BASELINE SUITE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processed_lakehouse_root = "data/processed/"
    output_baselines_json = "configs/external_baselines_results_legacy.json"
    injector = ManifoldAwareAnomalyPerturbationEngine()

    # Store lists of lightweight window slices grouped by vessel
    vessel_slice_groups = []
    vessels_processed_cal = 0
    
    # ==========================================================================
    # PHASE A: INGEST & VESSEL GROUP POOLING (MONTHS 7-9)
    # ==========================================================================
    logging.info("📥 Harvesting calibration sequence slices partitioned by vessel...")
    for m_int in CALIBRATION_MONTHS:
        partition_path = os.path.join(processed_lakehouse_root, "year=2019", f"month={m_int:02d}")
        if not os.path.exists(partition_path): continue
        for root, _, files in os.walk(partition_path):
            for file in files:
                if file.endswith(".parquet"):
                    df_vessel = pl.read_parquet(os.path.join(root, file))
                    if df_vessel.height < MAX_WINDOW_SIZE: continue
                    
                    v_slices = []
                    for start_idx in range(0, df_vessel.height - MAX_WINDOW_SIZE + 1, STRIDE * 2):
                        v_slices.append(df_vessel.slice(start_idx, MAX_WINDOW_SIZE))
                        
                    if v_slices:
                        vessel_slice_groups.append(v_slices)
                        vessels_processed_cal += 1
                    if vessels_processed_cal >= 150: break
            if vessels_processed_cal >= 150: break

    # Vessel-level group partition logic (80% Train, 20% Val)
    num_vessels = len(vessel_slice_groups)
    vessel_indices = np.arange(num_vessels)
    np.random.shuffle(vessel_indices)
    
    train_vessel_idx = vessel_indices[:int(num_vessels * 0.8)]
    val_vessel_idx = vessel_indices[int(num_vessels * 0.8):]
    
    # Execute feature extraction precisely once after splitting is finalized
    raw_train_flat, X_train_seq = [], []
    for idx in train_vessel_idx:
        for df_slice in vessel_slice_groups[idx]:
            raw_train_flat.append(prepare_statistical_window(df_slice))
            X_train_seq.append(df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32))
            
    raw_val_flat, X_val_seq = [], []
    for idx in val_vessel_idx:
        for df_slice in vessel_slice_groups[idx]:
            raw_val_flat.append(prepare_statistical_window(df_slice))
            X_val_seq.append(df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32))

    # Fit scaling transformations strictly on the training partition to eliminate leakage
    logging.info("⚖️ Fitting Standard Scalers over clean training partition boundaries...")
    flat_scaler = StandardScaler()
    X_train_flat = flat_scaler.fit_transform(np.array(raw_train_flat))
    X_val_flat = flat_scaler.transform(np.array(raw_val_flat))
    
    seq_scaler = StandardScaler()
    seq_scaler.fit(np.concatenate(X_train_seq, axis=0))
    X_train_seq = [seq_scaler.transform(s) for s in X_train_seq]
    X_val_seq = [seq_scaler.transform(s) for s in X_val_seq]

    # Initialize traditional unsupervised models (Point 4: Contamination defaults cleanly)
    logging.info("🔧 Optimizing Classical Baseline configurations...")
    iforest = IsolationForest(n_estimators=300, random_state=GLOBAL_SEED, n_jobs=-1)
    iforest.fit(X_train_flat)

    oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    oc_svm.fit(X_train_flat)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    lof_model.fit(X_train_flat)

    # Train sequential LSTM Autoencoder
    logging.info("🔧 Training sequential LSTM Autoencoder network...")
    lstm_ae = LSTMAutoencoder(feature_dim=FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(lstm_ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_limit = 5
    batch_size = 32

    for epoch in range(50):
        lstm_ae.train()
        train_epoch_loss = 0.0
        
        # Minor Quality Fix: Maintain order indices instead of in-place list modification
        perm = np.random.permutation(len(X_train_seq))
        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_samples = [X_train_seq[k] for _, k in enumerate(batch_idx)]
            
            optimizer.zero_grad()
            batch_tensor = torch.from_numpy(np.array(batch_samples, dtype=np.float32)).to(device)
            reconstructed = lstm_ae(batch_tensor)
            loss = criterion(reconstructed, batch_tensor)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item() * len(batch_samples)
            
        # Efficient batched processing for validation metrics
        lstm_ae.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for j in range(0, len(X_val_seq), batch_size):
                val_batch_samples = X_val_seq[j:j+batch_size]
                val_tensor = torch.from_numpy(np.array(val_batch_samples, dtype=np.float32)).to(device)
                val_rec = lstm_ae(val_tensor)
                val_epoch_loss += criterion(val_rec, val_tensor).item() * len(val_batch_samples)
                
        mean_train = train_epoch_loss / len(X_train_seq)
        mean_val = val_epoch_loss / len(X_val_seq)
        
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            epochs_no_improve = 0
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save(lstm_ae.state_dict(), "models/checkpoints/baseline_lstm_ae_best.pt")
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"   --> Epoch {epoch+1:02d}/50 | Train MSE: {mean_train:.6f} | Val MSE: {mean_val:.6f}")
        if epochs_no_improve >= patience_limit:
            lstm_ae.load_state_dict(torch.load("models/checkpoints/baseline_lstm_ae_best.pt", map_location=device))
            break

    # ==========================================================================
    # PHASE B: EVALUATING THRESHOLDS ON VALIDATION GROUPS
    # ==========================================================================
    logging.info("📊 Calibrating frozen operational thresholds over validation splits...")
    lstm_ae.eval() # Explicit evaluation mode lock
    
    val_y_true, val_scores_if, val_scores_svm, val_scores_lof, val_scores_lstm = [], [], [], [], []
    v_anom_counter = 0
    
    for idx in val_vessel_idx:
        for df_slice in vessel_slice_groups[idx]:
            # 1. Normal Sequence Track Profiling
            val_y_true.append(0)
            f_norm = flat_scaler.transform(prepare_statistical_window(df_slice).reshape(1, -1))
            val_scores_if.append(float(-iforest.score_samples(f_norm)[0]))
            val_scores_svm.append(float(-oc_svm.score_samples(f_norm)[0]))
            val_scores_lof.append(float(-lof_model.score_samples(f_norm)[0]))
            
            with torch.no_grad():
                s_norm = seq_scaler.transform(df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32))
                s_tensor = torch.from_numpy(s_norm).unsqueeze(0).float().to(device)
                val_scores_lstm.append(float(criterion(lstm_ae(s_tensor), s_tensor).item()))
                
            # 2. Anomalous Sequence Track Profiling
            val_y_true.append(1)
            mode = ["dead_reckoning", "coastal_creep", "loitering"][v_anom_counter % 3]
            df_perturbed = injector.inject_counterfactual_scenario(df_vessel_voyage=df_slice, mode=mode, severity=0.85)
            
            f_anom = flat_scaler.transform(prepare_statistical_window(df_perturbed).reshape(1, -1))
            val_scores_if.append(float(-iforest.score_samples(f_anom)[0]))
            val_scores_svm.append(float(-oc_svm.score_samples(f_anom)[0]))
            val_scores_lof.append(float(-lof_model.score_samples(f_anom)[0]))
            
            with torch.no_grad():
                s_anom = seq_scaler.transform(df_perturbed.select(REGISTRY.all_features).to_numpy().astype(np.float32))
                s_tensor_a = torch.from_numpy(s_anom).unsqueeze(0).float().to(device)
                val_scores_lstm.append(float(criterion(lstm_ae(s_tensor_a), s_tensor_a).item()))
                
            v_anom_counter += 1

    # Freeze validation thresholds to preserve test-set metrics integrity
    frozen_thresh_if   = find_optimal_f1_threshold(val_y_true, val_scores_if)
    frozen_thresh_svm  = find_optimal_f1_threshold(val_y_true, val_scores_svm)
    frozen_thresh_lof  = find_optimal_f1_threshold(val_y_true, val_scores_lof)
    frozen_thresh_lstm = find_optimal_f1_threshold(val_y_true, val_scores_lstm)

    # ==========================================================================
    # PHASE C: FINAL BENCHMARK TESTING (MONTHS 10-12)
    # ==========================================================================
    logging.info("🏁 Ingesting held-out test split partitions to calculate final metrics...")
    y_true = []
    scores_iforest, y_scores_ocsvm, scores_lof, scores_lstmae = [], [], [], []
    bench_count = 0

    for m_int in BENCHMARK_MONTHS:
        partition_path = os.path.join(processed_lakehouse_root, "year=2019", f"month={m_int:02d}")
        if not os.path.exists(partition_path): continue
        for root, _, files in os.walk(partition_path):
            for file in files:
                if file.endswith(".parquet"):
                    df_vessel = pl.read_parquet(os.path.join(root, file))
                    if df_vessel.height < MAX_WINDOW_SIZE: continue
                    
                    for start_idx in range(0, df_vessel.height - MAX_WINDOW_SIZE + 1, STRIDE * 2):
                        df_slice = df_vessel.slice(start_idx, MAX_WINDOW_SIZE)
                        
                        # --- Normal Testing Data ---
                        y_true.append(0)
                        flat_raw_n = prepare_statistical_window(df_slice).reshape(1, -1)
                        flat_norm = flat_scaler.transform(flat_raw_n)
                        
                        scores_iforest.append(float(-iforest.score_samples(flat_norm)[0]))
                        y_scores_ocsvm.append(float(-oc_svm.score_samples(flat_norm)[0]))
                        scores_lof.append(float(-lof_model.score_samples(flat_norm)[0]))

                        with torch.no_grad():
                            seq_raw_n = df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32)
                            seq_norm = seq_scaler.transform(seq_raw_n)
                            seq_tensor = torch.from_numpy(seq_norm).unsqueeze(0).float().to(device)
                            scores_lstmae.append(float(criterion(lstm_ae(seq_tensor), seq_tensor).item()))

                        # --- Anomalous Testing Data ---
                        y_true.append(1)
                        mode = ["dead_reckoning", "coastal_creep", "loitering"][bench_count % 3]
                        df_perturbed = injector.inject_counterfactual_scenario(df_vessel_voyage=df_slice, mode=mode, severity=0.85)
                        
                        flat_raw_a = prepare_statistical_window(df_perturbed).reshape(1, -1)
                        flat_anom = flat_scaler.transform(flat_raw_a)
                        
                        scores_iforest.append(float(-iforest.score_samples(flat_anom)[0]))
                        y_scores_ocsvm.append(float(-oc_svm.score_samples(flat_anom)[0]))
                        scores_lof.append(float(-lof_model.score_samples(flat_anom)[0]))

                        with torch.no_grad():
                            seq_raw_a = df_perturbed.select(REGISTRY.all_features).to_numpy().astype(np.float32)
                            seq_anom = seq_scaler.transform(seq_raw_a)
                            seq_tensor_a = torch.from_numpy(seq_anom).unsqueeze(0).float().to(device)
                            scores_lstmae.append(float(criterion(lstm_ae(seq_tensor_a), seq_tensor_a).item()))
                        
                        bench_count += 1

    # ==========================================================================
    # PHASE D: METRIC GENERATION USING FROZEN THRESHOLDS
    # ==========================================================================
    y_true_arr = np.array(y_true)
    
    def generate_final_metrics_record(y_t, y_s, frozen_threshold):
        return {
            "auroc": round(float(roc_auc_score(y_t, y_s)), 4),
            "auprc": round(float(average_precision_score(y_t, y_s)), 4),
            "frozen_test_f1": round(float(f1_score(y_t, (y_s >= frozen_threshold).astype(int), zero_division=0)), 4)
        }

    metrics_report = {
        "isolation_forest": generate_final_metrics_record(y_true_arr, np.array(scores_iforest), frozen_thresh_if),
        "one_class_svm": generate_final_metrics_record(y_true_arr, np.array(y_scores_ocsvm), frozen_thresh_svm),
        "local_outlier_factor": generate_final_metrics_record(y_true_arr, np.array(scores_lof), frozen_thresh_lof),
        "lstm_autoencoder": generate_final_metrics_record(y_true_arr, np.array(scores_lstmae), frozen_thresh_lstm)
    }

    os.makedirs(os.path.dirname(output_baselines_json), exist_ok=True)
    with open(output_baselines_json, "w") as f:
        json.dump(metrics_report, f, indent=4)

    print("\n" + "="*75)
    print("📊 EXTERNAL COMPARATIVE BASELINE EVALUATION COMPLETE")
    print("="*75)
    for model_name, path in [("Isolation Forest Baseline", "isolation_forest"), 
                             ("One-Class SVM Baseline   ", "one_class_svm"), 
                             ("Local Outlier Factor     ", "local_outlier_factor"), 
                             ("Sequential LSTM AE       ", "lstm_autoencoder")]:
        print(f"• {model_name} | AUROC: {metrics_report[path]['auroc']:.4f} | AUPRC: {metrics_report[path]['auprc']:.4f} | Frozen F1: {metrics_report[path]['frozen_test_f1']:.4f}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
