# navisight/evaluation/ann_index.py
import hnswlib
import numpy as np
import polars as pl
import sqlite3
import os
import logging

class HierarchicalNavigableNetworkIndex:
    """Manages separate HNSW graphs grouped by vessel taxonomy."""
    def __init__(self, index_dir: str = "models/state/index", dimension: int = 128, initial_capacity: int = 100000):
        self.index_dir = index_dir
        self.dim = dimension
        self.capacity = initial_capacity
        self.db_path = os.path.join(index_dir, "ann_metadata_registry.db")
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Dictionary of HNSW sub-graphs: superclass_id -> hnswlib.Index
        self.sub_indices = {}
        
        self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS ann_node_registry (
                embedding_id INTEGER, vessel_id_int INTEGER, timestamp_sec INTEGER, superclass_id INTEGER, PRIMARY KEY(embedding_id, superclass_id)
            );
        """)
        self.db_conn.commit()

    def _get_or_init_index(self, superclass_id: int) -> hnswlib.Index:
        if superclass_id not in self.sub_indices:
            idx = hnswlib.Index(space='cosine', dim=self.dim)
            idx.init_index(max_elements=self.capacity, ef_construction=200, M=16)
            idx.set_ef(50)
            self.sub_indices[superclass_id] = idx
        return self.sub_indices[superclass_id]

    def index_parquet_shard(self, shard_parquet_path: str):
        df = pl.read_parquet(shard_parquet_path)
        if df.is_empty(): return
        
        # Group incoming vectors by class identity
        for cohort_df in df.partition_by("superclass_id", include_key=True):
            sc_id = int(cohort_df["superclass_id"][0])
            idx = self._get_or_init_index(sc_id)
            
            ids = cohort_df["embedding_id"].to_numpy().astype(np.int64)
            vessel_ids = cohort_df["vessel_id_int"].to_numpy().astype(np.int64)
            timestamps = cohort_df["timestamp_sec"].to_numpy().astype(np.int64)
            embeddings = np.vstack(cohort_df["embedding"].to_numpy()).astype(np.float32)
            
            # Normalize step preserves cosine topology bounds
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            embeddings_norm = embeddings / norms
            
            # ── CAPACITY CAP ESCALATION SECURITY PROTECTION ──
            current_count = idx.get_current_count()
            incoming_count = len(ids)
            max_capacity = idx.get_max_elements()

            if current_count + incoming_count > max_capacity:
                new_capacity = max(max_capacity * 2, current_count + incoming_count + 10000)
                idx.resize_index(new_capacity)
            
            # Append vector space coordinates into the HNSW sub-graph
            idx.add_items(embeddings_norm, ids)
            
            self.db_conn.executemany("""
                INSERT OR REPLACE INTO ann_node_registry VALUES (?, ?, ?, ?);
            """, zip(ids.tolist(), vessel_ids.tolist(), timestamps.tolist(), [sc_id]*len(ids)))
        self.db_conn.commit()

    def query_behavior_neighborhood(self, query_embedding: np.ndarray, superclass_id: int, k: int = 3) -> list:
        idx = self._get_or_init_index(superclass_id)
        if idx.get_current_count() == 0: return []
        
        norm_val = max(np.linalg.norm(query_embedding), 1e-8)
        emb_norm = (query_embedding / norm_val).astype(np.float32).reshape(1, -1)
        
        labels, distances = idx.knn_query(emb_norm, k=min(k, idx.get_current_count()))
        match_ids = [int(mid) for mid in labels[0]]
        placeholders = ",".join(["?"] * len(match_ids))
        
        cursor = self.db_conn.execute(f"""
            SELECT embedding_id, vessel_id_int, timestamp_sec FROM ann_node_registry 
            WHERE superclass_id = ? AND embedding_id IN ({placeholders});
        """, [superclass_id] + match_ids)
        
        meta_lookup = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        
        return [
            {"embedding_id": m_id, "vessel_id_int": meta_lookup[m_id][0], "timestamp_sec": meta_lookup[m_id][1], "distance": float(distances[0][i])}
            for i, m_id in enumerate(match_ids) if m_id in meta_lookup
        ]