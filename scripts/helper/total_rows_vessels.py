import pyarrow.dataset as ds

base_path = "data/processed/"

dataset = ds.dataset(base_path, format="parquet")

total_rows = 0
unique_vessels = set()

for batch in dataset.to_batches(columns=["mmsi_str"]):
    total_rows += batch.num_rows
    unique_vessels.update(batch.column("mmsi_str").to_pylist())

print("Total rows:", total_rows)
print("Unique vessels:", len(unique_vessels))