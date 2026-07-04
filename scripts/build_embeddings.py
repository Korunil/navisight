# scripts/build_embeddings.py
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY
from navisight.evaluation.extract_embeddings import ZeroCopyExtractionPipeline
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# HISTORICAL BASELINE CONFIGURATION
# ==========================================
TRAINING_BASE_YEAR = 2019
MAX_INDEX_MONTH = 6  # Exclusively index January - June to prevent target evaluation leakage


def main():
    logging.info("🛰️ STARTING OFFLINE EMBEDDING EXTRACTION & HNSW REBUILD PASSTHROUGH")
    
    processed_lakehouse_root    = "data/processed/"
    manifest_output_json        = "configs/production_manifest.json"
    checkpoint_directory        = "models/checkpoints/"
    embedding_destination_dir   = "data/embeddings"
    index_directory             = "models/state/index"
    
    best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = [
            os.path.join(checkpoint_directory, f) 
            for f in os.listdir(checkpoint_directory) 
            if f.endswith(".pt")
        ][-1]

    logging.info(f"Targeting Weights Checkpoint Asset:{best_checkpoint}")

    # Ensure clean state environments exist safely
    os.makedirs(embedding_destination_dir, exist_ok=True)
    os.makedirs(index_directory, exist_ok=True)

    # 1. Purge Old Database State Files to prevent Unique Constraint Collision Exceptions
    db_path = os.path.join(index_directory, "ann_metadata_registry.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        logging.info("Cleared prior metadata database ledger registry to guarantee data consistency.")

    # 2. Extract Latent Embeddings (Constrained historical scan)
    logging.info(f"Running zero-copy extractor loop over baseline fragments (<= Month {MAX_INDEX_MONTH})...")
    extractor = ZeroCopyExtractionPipeline(best_checkpoint, embedding_destination_dir)
    extractor.feature_cols = REGISTRY.all_features
    
    processed_dirs = set()
    total_shards_extracted = 0
    for root, _, files in os.walk(processed_lakehouse_root):
        # Path parsing maps partition context keys dynamically
        parts = root.replace("\\", "/").split("/")
        try:
            y_val = [int(p.split("=")[1]) for p in parts if "year=" in p][0]
            m_val = [int(p.split("=")[1]) for p in parts if "month=" in p][0]
        except IndexError:
            # Skip paths that don't match standard hive partitioned schemas
            continue
        
        # Structural check: Only ingest historical base frames to completely avoid data leakage
        if not (y_val < TRAINING_BASE_YEAR or 
                (y_val == TRAINING_BASE_YEAR and m_val <= MAX_INDEX_MONTH)):
            continue
            
        # IMPORTANT: detect leaf-level folders only (vessel_bucket level)
        if not any("vessel_bucket=" in p for p in parts):
            continue
        
        # Ensure folder is processed only once
        if root in processed_dirs:
            continue
        processed_dirs.add(root)
        
        # Only process folders that actually contain parquet files
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            continue

        extractor.extract_and_serialize_latent_manifold(
            data_source_partitions_dir=root, 
            config_json=manifest_output_json
        )
        total_shards_extracted += 1

    logging.info(f"🎉 Core latent space data serialization complete. {total_shards_extracted} historical partitions built.")

    # 3. Build & Populate Cohort-Gated HNSW Structural Graphs
    logging.info("Initializing HNSW Index Space Structural Topologies...")
    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    
    shards_to_index = sorted([f for f in os.listdir(embedding_destination_dir) if f.endswith(".parquet")])
    if not shards_to_index:
        logging.error(f"Execution Terminated: No raw vector files populated inside {embedding_destination_dir}")
        return

    logging.info(f"Indexing {len(shards_to_index)} localized matrix files into HNSW graph...")
    for idx, shard in enumerate(shards_to_index):
        shard_path = os.path.join(embedding_destination_dir, shard)
        logging.info(f"[{idx+1}/{len(shards_to_index)}] Committing vector node: {shard}")
        ann_index.index_parquet_shard(shard_path)

    logging.info("🎉 SUCCESS! Historical Baseline Vector Artifacts committed cleanly to disk.")


if __name__ == "__main__":
    main()