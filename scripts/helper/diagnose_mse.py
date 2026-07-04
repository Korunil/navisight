# scripts/diagnose_mse.py
import os
import sys
import torch
import polars as pl
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY, SUPERCLASS_VOCAB
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector

def main():
    processed_lakehouse_root    = "data/processed"
    checkpoint_directory        = "models/checkpoints/"
    global_stats_json_path      = "configs/global_stats.json"
    index_directory             = "models/state/index"
    
    best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = [os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")][-1]

    # Initialize components
    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    profile_engine = RollingBehaviorProfileEngine(alpha=0.1)
    detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, ann_index, profile_engine)

    # Automatically grab the very first parquet file from December 2019
    mock_live_path = os.path.join(processed_lakehouse_root, "year=2019", "month=12")
    target_file = None
    for root, _, files in os.walk(mock_live_path):
        for file in files:
            if file.endswith(".parquet"):
                target_file = os.path.join(root, file)
                break
        if target_file: break

    if not target_file:
        print("❌ Error: No validation parquet shards discovered inside December 2019 partition.")
        return

    print(f"📦 Target Diagnostic Shard: {target_file}")
    df_vessel_chunk = pl.read_parquet(target_file)
    
    # Isolate a single baseline context tracking window
    df_slice = df_vessel_chunk.slice(0, 60)
    feature_cols = df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32)
    feature_tensor = torch.from_numpy(feature_cols).unsqueeze(0)

    # Force a case-insensitive lookup map of your feature vocabulary registry
    vocab_clean = {str(k).lower().strip(): v for k, v in SUPERCLASS_VOCAB.items()}
    
    # Tier 1: Try reading the explicit integer ID column from your schema
    sc_id_raw = 0
    if "vessel_superclass_id" in df_vessel_chunk.columns and df_vessel_chunk["vessel_superclass_id"][0] is not None:
        val = df_vessel_chunk["vessel_superclass_id"][0]
        if float(val) > 0:
            sc_id_raw = int(val)

    # Tier 2: Fallback to cross-referencing the raw text shiptype string if ID is missing/0
    if sc_id_raw == 0 and "shiptype" in df_vessel_chunk.columns and df_vessel_chunk["shiptype"][0] is not None:
        raw_shiptype_str = str(df_vessel_chunk["shiptype"][0]).lower().strip()
        sc_id_raw = vocab_clean.get(raw_shiptype_str, 0)
    
    # Process live normalization pass
    norm_tensor = detector._normalize_tensor_live(feature_tensor, sc_id_raw)
    normalized_tensor = norm_tensor.to(detector.device).contiguous()

    # Fire a single model forward check to intercept raw errors
    detector.model.eval()
    with torch.no_grad():
        reconstructed, _, _, _ = detector.model(normalized_tensor, None, mask_ratio=0.40)
        
    print("\n" + "="*95)
    print("🎯 NAVISIGHT METRIC SPACE: DETAILED FEATURE-BY-FEATURE RECONSTRUCTION MSE BREAKDOWN")
    print("="*95)
    print(f"{'Maskable Feature Name':<30} | {'Raw Mean':<12} | {'Normalized Mean':<16} | {'Recon MSE':<15}")
    print("-" * 95)
    
    # Audit only the specific fields evaluated by your loss masking configuration
    for feat in REGISTRY.maskable_for_loss:
        idx = REGISTRY.feature_index[feat]
        
        raw_mean = feature_tensor[0, :, idx].mean().item()
        norm_mean = normalized_tensor[0, :, idx].mean().item()
        
        pred = reconstructed[0, :, idx]
        actual = normalized_tensor[0, :, idx]
        individual_mse = torch.nn.functional.mse_loss(pred, actual).item()
        
        print(f"{feat:<30} | {raw_mean:<12.4f} | {norm_mean:<16.4f} | {individual_mse:<15.4f}")
    print("="*95)

if __name__ == "__main__":
    main()