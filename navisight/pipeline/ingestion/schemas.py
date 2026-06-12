# navisight/pipeline/ingestion/schemas.py
import polars as pl

# CANONICAL STRATIFIED SCHEMA CONTRACT
# Rigid data types guarantee structural immutability across Polars query plans
CANONICAL_TRACK_SCHEMA = {
    "global_lsn": pl.Int64,
    "timestamp": pl.Int64,
    "mmsi_str": pl.Utf8,
    "lon_raw": pl.Float64,
    "lat_raw": pl.Float64,
    "heading_raw": pl.Float64,
    "speed_raw": pl.Float64,
    "course_raw": pl.Float64,
    "vessel_id_int": pl.Int64,
    "vessel_bucket_id": pl.Int32,
    "timestamp_sec": pl.Int64,
    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    "flag_null_course": pl.Int32,
    "course_sin": pl.Float64,
    "course_cos": pl.Float64,
    "shiptype": pl.Int64,
    "vessel_superclass_id": pl.Int16,
    "vessel_behavior_id": pl.Int16,
    # ── NEW SCHEMA ENTRANCES ──
    "distance_to_coast_nm": pl.Float64,
    "distance_to_terminal_nm": pl.Float64,
    "harbor_basin_proximity": pl.Float64,
    # ── PRODUCTION AUGMENTATIONS ──
    "turn_rate_speed_ratio": pl.Float64,
    "distance_to_land_raster": pl.Float64,
    "flag_land_ingress": pl.Int32
}

CSV_INGESTION_INPUT_SCHEMA = {
    "timestamp": pl.Utf8,
    "mmsi_str": pl.Utf8,
    "lon_raw": pl.Float64,
    "lat_raw": pl.Float64,
    "heading_raw": pl.Float64,
    "speed_raw": pl.Float64,
    "course_raw": pl.Float64,
}