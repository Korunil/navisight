# scripts/eval_pipeline.py
import os
import sys
import logging
import polars as pl
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY
from navisight.evaluation.extract_embeddings import ZeroCopyExtractionPipeline
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MONTH_MAP = {7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}

def main():
    logging.info("🚨 RUNNING PHASE 2: LATENT EMBEDDING EXTRACTION & LIVE SIMULATION LOOP")
    
    processed_lakehouse_root    = "data/processed/year=2019"
    manifest_output_json        = "configs/production_manifest.json"
    checkpoint_directory        = "models/checkpoints/"
    global_stats_json_path      = "configs/global_stats.json"
    embedding_destination_dir   = "data/embeddings"
    index_directory             = "models/state/index"
    
    best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = [os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")][-1]

    logging.info(f"Using Master Checkpoint Asset: {best_checkpoint}")

    # STAGE 4: EMBEDDING EXTRACTION
    # Check directory state to see if valid extracted embedding shards are already present
    existing_shards = []
    if os.path.exists(embedding_destination_dir):
        existing_shards = sorted([f for f in os.listdir(embedding_destination_dir) if f.endswith(".parquet")])

    skip_extraction = False
    if existing_shards:
        print("\n" + "="*70)
        print(f"🛰️  FOUND {len(existing_shards)} EXISTING EMBEDDING SHARDS IN: {embedding_destination_dir}")
        print("="*70)
        user_choice = input(">> Skip model extraction pass and rebuild index from existing files? (y/n): ").strip().lower()
        print("="*70 + "\n")
        
        if user_choice in ['y', 'yes']:
            logging.info("⏩ User confirmed: Skipping model extraction pass. Rebuilding index graph only.")
            skip_extraction = True
            
            # Reset metadata database to prevent unique key constraint collisions on recovery
            db_path = os.path.join(index_directory, "ann_metadata_registry.db")
            if os.path.exists(db_path):
                os.remove(db_path)
                logging.info("Cleared old database registry for a clean index recovery pass.")
        else:
            logging.info("🔄 WARNING: Purging of existing shards requested.")
            user_input = input(">> Are you sure? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                logging.info("🔄 User declined: Purging existing shards and initiating a fresh model extraction run.")
                for shard in existing_shards:
                    os.remove(os.path.join(embedding_destination_dir, shard))

    # ── EMBEDDING EXTRACTION (RUN ONLY IF TARGET CHUNKS ARE NOT SKIPPED) ──
    if not skip_extraction:
        logging.info("Extracting embeddings from calibration shards...")
        extractor = ZeroCopyExtractionPipeline(best_checkpoint, embedding_destination_dir)
        extractor.feature_cols = REGISTRY.all_features
        extractor.extract_and_serialize_latent_manifold(data_source_partitions_dir=processed_lakehouse_root, config_json=manifest_output_json)
        logging.info("🎉 Extraction pipeline complete. Chunks saved safely to disk.")

    # ── REBUILD COHORT-GATED HNSW GRAPHS & SQL LEDGER ──
    logging.info("Initializing HNSW Index Engine Subsystems...")
    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    
    # Re-read active file directory mappings
    shards_to_index = sorted([f for f in os.listdir(embedding_destination_dir) if f.endswith(".parquet")])
    if not shards_to_index:
        logging.error(f"Execution terminated: No embedding shards discovered inside {embedding_destination_dir}")
        return

    logging.info(f"Indexing {len(shards_to_index)} vector source fragments into HNSW...")
    for idx, shard in enumerate(shards_to_index):
        shard_path = os.path.join(embedding_destination_dir, shard)
        logging.info(f"[{idx+1}/{len(shards_to_index)}] Indexing shard: {shard}")
        ann_index.index_parquet_shard(shard_path)

    logging.info("🎉 SUCCESS! Phase 2 processing finalized safely with full metadata tracking.")
    
    # STAGE 5: LIVE SIMULATION
    profile_engine = RollingBehaviorProfileEngine(alpha=0.1)
    detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, ann_index, profile_engine)
    injector = ManifoldAwareAnomalyPerturbationEngine()

    logging.info("Entering Live Production Simulation Mode (July - December 2019)...")
    live_schedule = [(2019, m) for m in range(7, 13)]
    
    for year, m_int in live_schedule:
        m_str = MONTH_MAP[m_int]
        logging.info(f"🚨 [STREAMFEED] Streaming live data for: {m_str} {year}")
        mock_live_path = os.path.join(processed_lakehouse_root, f"year={year}", f"month={m_int}")
        if not os.path.exists(mock_live_path): continue

        for root, _, files in os.walk(mock_live_path):
            for file in files:
                if file.endswith(".parquet"):
                    df_vessel_chunk = pl.read_parquet(os.path.join(root, file))
                    
                    # Core Structural Contract Constants
                    MAX_WINDOW_SIZE = 200  # Must match max_window_size of temporal encoder
                    total_rows = df_vessel_chunk.height
                    
                    if total_rows < 65: 
                        continue
                    
                    # Extract vessel meta context safely
                    v_id = int(df_vessel_chunk["vessel_id_int"][0])
                    context_meta = {
                        "timestamp_sec": int(df_vessel_chunk["timestamp_sec"][0]), 
                        "trip_id": str(df_vessel_chunk["trip_id"][0]),
                        "superclass_id": int(df_vessel_chunk["superclass_id"][0])
                    }

                    # Storage buffers for calculating metrics across tracking chunks
                    normal_scores = []
                    anomalous_scores = []

                    # ── FIXED: DYNAMIC CONTEXT-WINDOW SLIDING ITERATOR ──
                    # Slides down the 50k dataset using max 200-row intervals to keep the sequences
                    # completely aligned with the absolute position embedding boundaries.
                    for start_idx in range(0, total_rows, MAX_WINDOW_SIZE):
                        end_idx = min(start_idx + MAX_WINDOW_SIZE, total_rows)
                        df_slice = df_vessel_chunk.slice(start_idx, end_idx - start_idx)
                        
                        # Guard against processing tiny trailing residual fragments
                        if df_slice.height < 10: 
                            continue

                        # 1. Evaluate Normal Tracking Window
                        feature_cols = df_slice.select(REGISTRY.all_features).to_numpy().astype(np.float32)
                        feature_tensor = torch.from_numpy(feature_cols)
                        res_normal = detector.evaluate_live_sequence_anomaly_score(feature_tensor, v_id, context_meta)
                        normal_scores.append(res_normal['fused_risk_score'])
                        
                        # 2. Inject Anomaly & Evaluate Perturbed Window
                        df_perturbed = injector.inject_loitering_smuggling_drift(df_vessel_voyage=df_slice, severity=0.85)
                        perturbed_cols = df_perturbed.select(REGISTRY.all_features).to_numpy().astype(np.float32)
                        perturbed_tensor = torch.from_numpy(perturbed_cols)
                        res_anomalous = detector.evaluate_live_sequence_anomaly_score(perturbed_tensor, v_id, context_meta)
                        anomalous_scores.append(res_anomalous['fused_risk_score'])

                    # Aggregate window tracking metrics to output an authoritative track assessment
                    avg_normal_score = float(np.mean(normal_scores)) if normal_scores else 0.0
                    avg_anomalous_score = float(np.mean(anomalous_scores)) if anomalous_scores else 0.0

                    logging.info(f"Vessel {v_id} | Tracks Processed: {len(normal_scores)} windows")
                    logging.info(f"Vessel {v_id} | Aggregated Normal Track Risk Score: {avg_normal_score:.4f} -> {'Anomaly Detected' if avg_normal_score > 0.75 else 'Normal Operations'}")
                    logging.info(f"Vessel {v_id} | Aggregated Perturbed Smuggling Risk Score: {avg_anomalous_score:.4f} -> {'Anomaly Detected' if avg_anomalous_score > 0.75 else 'Normal Operations'}")

    logging.info("🏁 Evaluation sandbox simulation run completed successfully.")

if __name__ == "__main__":
    main()