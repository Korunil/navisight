# scripts/find_december_shards.py
import os
import polars as pl

EMBEDDING_DIR = "data/embeddings"
START_TS = 1561939201  # 2019-07-01 00:00:01 UTC
# START_TS = 1575158400  # 2019-12-01 00:00:00 UTC

print("🔍 Scanning embedding shards for Data...")
found_shards = False

if not os.path.exists(EMBEDDING_DIR):
    print(f"❌ Directory missing: {EMBEDDING_DIR}")
    exit(1)

for file in sorted(os.listdir(EMBEDDING_DIR)):
    if file.endswith(".parquet"):
        file_path = os.path.join(EMBEDDING_DIR, file)
        
        # Read only the timestamp column to keep execution fast
        df = pl.read_parquet(file_path, columns=["timestamp_sec"])
        max_ts = df["timestamp_sec"].max()
        
        if max_ts is not None and max_ts >= START_TS:
            print(f"❌ DELETE: {file} (Contains timestamps up to {max_ts})")
            found_shards = True

if not found_shards:
    print("✅ Clean split! No data found in any shards.")