# navisight/evaluation/embedding_store.py
import os
import polars as pl
import pyarrow as pa
import numpy as np

class AppendOnlyEmbeddingStore:
    """
    Saves high-throughput latent representations to disk as high-precision float32 data arrays,
    preserving fine-grained variations between behavioral patterns.
    """
    def __init__(self, base_dir: str = "data/embeddings", max_shard_rows: int = 50000):
        self.base_dir = base_dir
        self.max_shard_rows = max_shard_rows
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize dictionary structures to avoid structured array stride leakage
        self.clear_buffer()
        self.shard_sequence_id = 0

    def clear_buffer(self):
        self.current_rows = 0
        self.meta_store = {
            "embedding_id": [], "vessel_id_int": [], "timestamp_sec": [],
            "trip_id": [], "superclass_id": [], "reliability": []
        }
        self.embedding_vectors = []

    def append_latent_vectors(self, batch_meta: dict, embeddings: np.ndarray):
        num_rows = len(embeddings)
        self.embedding_vectors.append(embeddings.astype(np.float32))
        
        for k in self.meta_store.keys():
            self.meta_store[k].extend(batch_meta[k])
            
        self.current_rows += num_rows
        if self.current_rows >= self.max_shard_rows:
            self.flush_buffer_to_shard()

    def flush_buffer_to_shard(self):
        if self.current_rows == 0:
            return
            
        shard_path = os.path.join(self.base_dir, f"embedding_shard_{self.shard_sequence_id:04d}.parquet")
        tmp_path = shard_path + ".tmp"
        
        flat_embeddings = np.concatenate(self.embedding_vectors, axis=0).reshape(-1)
        arrow_flat_array = pa.array(flat_embeddings, type=pa.float32())
        
        # FIXED: trip_id is preserved as a string array, matching upstream formats exactly
        arrow_arrays = {
            "embedding_id": pa.array(self.meta_store["embedding_id"], type=pa.int64()),
            "vessel_id_int": pa.array(self.meta_store["vessel_id_int"], type=pa.int64()),
            "timestamp_sec": pa.array(self.meta_store["timestamp_sec"], type=pa.int64()),
            "trip_id": pa.array(self.meta_store["trip_id"], type=pa.string()), 
            "superclass_id": pa.array(self.meta_store["superclass_id"], type=pa.int16()),
            "reliability": pa.array(self.meta_store["reliability"], type=pa.float32()),
            "embedding": pa.FixedSizeListArray.from_arrays(arrow_flat_array, list_size=128)
        }
        
        df_shard = pl.from_arrow(pa.Table.from_pydict(arrow_arrays))
        df_shard.write_parquet(tmp_path, compression="snappy")
        os.replace(tmp_path, shard_path)
        
        self.shard_sequence_id += 1
        self.clear_buffer()
    
    def close(self):
        self.flush_buffer_to_shard()