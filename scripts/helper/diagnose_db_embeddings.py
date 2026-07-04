import os, sqlite3, polars as pl

shard_dir = "data/embeddings"
total_shard_rows = sum(
    pl.read_parquet(os.path.join(shard_dir, f)).height
    for f in os.listdir(shard_dir) if f.endswith(".parquet")
)

conn = sqlite3.connect("models/state/index/ann_metadata_registry.db")
db_rows = conn.execute("SELECT COUNT(*) FROM ann_node_registry").fetchone()[0]

print(f"Total rows across all shard parquet files: {total_shard_rows}")
print(f"Rows actually present in ann_node_registry: {db_rows}")
print(f"Missing/overwritten: {total_shard_rows - db_rows}")