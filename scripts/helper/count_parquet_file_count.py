from pathlib import Path

processed_lakehouse_root = "data/processed/"
root = Path(processed_lakehouse_root)

TRAINING_BASE_YEAR = 2019
MAX_INDEX_MONTH = 6

expected_files = 0
selected_paths = []

for year_dir in root.glob("year=*"):
    year = int(year_dir.name.split("=")[1])

    if year < 2018:
        continue

    for month_dir in year_dir.glob("month=*"):
        month = int(month_dir.name.split("=")[1])

        # apply cutoff rule
        if year == 2019 and month > MAX_INDEX_MONTH:
            continue

        # count parquet files in all leaf folders
        for pq_file in month_dir.rglob("*.parquet"):
            expected_files += 1
            selected_paths.append(pq_file)

print("Expected parquet files:", expected_files)
print("Sample files:", selected_paths[:10])