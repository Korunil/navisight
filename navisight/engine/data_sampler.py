# navisight/pipeline/data_sampler.py
import os
import polars as pl
from navisight.pipeline.feature_registry import REGISTRY

def compute_phase_weights_from_parquet(partition_root: str) -> dict:
    """Runs a pre-scan pass across Parquet shards to compute inverse phase frequency weights."""
    phase_counts = {'anchoring': 0, 'cruising': 0, 'departure': 0, 'maneuvering': 0}
    phase_cols = ['voyage_phase_anchoring', 'voyage_phase_cruising', 'voyage_phase_departure', 'voyage_phase_maneuvering']
    
    for root, _, files in os.walk(partition_root):
        for f in files:
            if not f.endswith('.parquet'): continue
            df = pl.read_parquet(os.path.join(root, f), columns=[c for c in phase_cols if c in pl.read_parquet_schema(os.path.join(root, f))])
            for col in phase_cols:
                if col in df.columns:
                    phase_key = col.replace('voyage_phase_', '')
                    phase_counts[phase_key] += int((df[col] > 0.5).sum())

    total = max(sum(phase_counts.values()), 1)
    return {phase: float(total / (count + 1e-5)) for phase, count in phase_counts.items()}