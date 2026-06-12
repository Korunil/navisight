# scripts/compute_global_stats.py
import os
import sys
import json
import logging
import argparse
import zipfile
import io
import numpy as np
import polars as pl
from dbfread import DBF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from navisight.pipeline.feature_registry import REGISTRY, SUPERCLASS_VOCAB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical Geographic Bounding Box Limits for the Piraeus/Saronikos Gulf Region
PIRAEUS_LON_MIN, PIRAEUS_LON_MAX = 23.300000, 23.900000
PIRAEUS_LAT_MIN, PIRAEUS_LAT_MAX = 37.600000, 38.100000

def run_feature_engineering_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    """
    Core vectorised engineering compiler. Assures full extraction of kinematics
    and circular spatio-temporal features BEFORE evaluating global metrics.
    """
    if df.is_empty():
        return df

    # Enforce strict chronological order over individual vessel voyages
    df_sorted = df.sort(["vessel_id_int", "timestamp_sec"])
    partition_scope = ["vessel_id_int"]

    # Calculate actual temporal differences
    df_sorted = df_sorted.with_columns([
        pl.col("timestamp_sec").diff().over(partition_scope).fill_null(10).cast(pl.Int64).alias("dt")
    ]).with_columns([
        pl.when(pl.col("dt") < 1).then(1).otherwise(pl.col("dt")).alias("dt_clamped")
    ])

    # True Geodesic Displacements using Harvesine Equations
    prev_lat = pl.col("lat_raw").shift(1).over(partition_scope)
    prev_lon = pl.col("lon_raw").shift(1).over(partition_scope)
    
    lat_diff_scaled = (pl.col("lat_raw") * 10000000.0).cast(pl.Int64).diff().over(partition_scope).fill_null(0)
    lon_diff_scaled = (pl.col("lon_raw") * 10000000.0).cast(pl.Int64).diff().over(partition_scope).fill_null(0)
    speed_diff_scaled = (pl.col("speed_raw") * 10000.0).cast(pl.Int64).diff().over(partition_scope).fill_null(0)
    course_diff_scaled = (pl.col("course_raw") * 10000.0).cast(pl.Int64).diff().over(partition_scope).fill_null(0)

    # Convert the precision-isolated deltas back to standard float ranges
    delta_lat = lat_diff_scaled / 10000000.0
    delta_lon = lon_diff_scaled / 10000000.0
    delta_speed = speed_diff_scaled / 10000.0
    delta_course = course_diff_scaled / 10000.0
    
    corrected_course_change = ((delta_course + 180.0) % 360.0) - 180.0

    # High-precision Geodesic Haversine Calculations
    dlat_rad = delta_lat.radians()
    dlon_rad = delta_lon.radians()
    lat1_rad = prev_lat.radians()
    lat2_rad = pl.col("lat_raw").radians()

    haversine_a = (dlat_rad / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon_rad / 2).sin().pow(2)
    haversine_clamped = haversine_a.clip(1e-9, 1.0 - 1e-9)
    geodetic_distance_nm = pl.when(prev_lat.is_null()).then(0.0).otherwise(2 * haversine_clamped.sqrt().arcsin() * 3440.065)
    
    # Authoritative reference anchor coordinates for the main destination hub (Port of Piraeus)
    PIRAEUS_PORT_LAT = 37.93757531153567
    PIRAEUS_PORT_LON = 23.620630391770646

    # Convert geographic points to radians for geodetic tracking
    lat_vessel_rad = pl.col("lat_raw").radians()
    lon_vessel_rad = pl.col("lon_raw").radians()
    lat_port_rad  = pl.lit(PIRAEUS_PORT_LAT).radians()
    lon_port_rad  = pl.lit(PIRAEUS_PORT_LON).radians()

    # Compute high-precision dynamic Haversine distance to port
    dlat_port = lat_port_rad - lat_vessel_rad
    dlon_port = lon_port_rad - lon_vessel_rad
    
    haversine_a_port = (dlat_port / 2).sin().pow(2) + lat_vessel_rad.cos() * lat_port_rad.cos() * (dlon_port / 2).sin().pow(2)
    haversine_clamped_port = haversine_a_port.clip(1e-9, 1.0 - 1e-9)
    derived_distance_to_port = 2 * haversine_clamped_port.sqrt().arcsin() * 3440.065 # Output in Nautical Miles (nm)

    # 2. Compute dynamic orthodromic forward bearing to port
    y_bearing = dlon_port.sin() * lat_port_rad.cos()
    x_bearing = lat_vessel_rad.cos() * lat_port_rad.sin() - lat_vessel_rad.sin() * lat_port_rad.cos() * dlon_port.cos()
    
    # Calculate initial bearing in radians and map cleanly to compass degrees [0, 360]
    derived_bearing_to_port = (pl.arctan2(y_bearing, x_bearing) * (180.0 / np.pi) + 360.0) % 360.0

    # Derive continuous physical feature transformations
    df_kinematics = df_sorted.with_columns([
        pl.col("speed_raw").alias("actual_speed_knots"),
        corrected_course_change.alias("course_change"),
        (delta_speed / pl.col("dt_clamped")).alias("acceleration"),
        (delta_lon * pl.col("lat_raw").radians().cos() * 60.0).alias("delta_x_nm"),
        (delta_lat * 60.0).alias("delta_y_nm"),
        geodetic_distance_nm.alias("geodetic_displacement_nm"),
        geodetic_distance_nm.alias("geodetic_distance_nm"),
        (pl.col("course_raw") * (np.pi / 180.0)).sin().alias("course_sin"),
        (pl.col("course_raw") * (np.pi / 180.0)).cos().alias("course_cos"),
        pl.col("dt_clamped").log().alias("log_time_diff"),
        derived_distance_to_port.alias("distance_to_port_nm"),
        derived_bearing_to_port.alias("bearing_to_port"),
    ]).with_columns([
        (pl.col("acceleration").diff().over(partition_scope).fill_null(0.0) / pl.col("dt_clamped")).alias("jerk"),
        (pl.col("course_change") / pl.col("dt_clamped")).alias("turn_rate")
    ])
    
    return df_kinematics

def compute_stats(zip_path: str, weather_root_dir: str = None, sample_rows: int = 750_000) -> dict:
    logger.info(f"Extracting sample stream from: {zip_path}...")
    rows_collected = []
    rows_read = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        csv_members = [m for m in archive.namelist() if m.endswith(".csv") and not m.startswith("__MACOSX")]
        if not csv_members:
            raise FileNotFoundError(f"No valid CSV members inside {zip_path}")

        for member in csv_members:
            with archive.open(member, "r") as stream:
                carry = b""
                while rows_read < sample_rows:
                    raw = stream.read(8 * 1024 * 1024)
                    if not raw: break
                    nl = raw.rfind(b"\n")
                    if nl == -1:
                        carry += raw
                        continue
                    chunk = carry + raw[: nl + 1]
                    carry = raw[nl + 1:]
                    
                    try:                     
                        # Read the chunk uniformly as an unheaded table first 
                        # to protect against floating slice positions
                        df = pl.read_csv(io.BytesIO(chunk), has_header=False)
                        
                        chunk_str = chunk.decode('utf-8', errors='ignore')
                        first_line = chunk_str.split('\n', 1)[0]

                        # 2. FIXED: Dynamically map columns based on schema discovery.
                        # If a known column string exists in the first row, re-label with row 0.
                        if "timestamp" in first_line or "mmsi" in first_line or "vessel_id" in first_line:
                            # Extract true text labels from the first row record
                            header_names = [str(df.row(0)[i]) for i in range(df.width)]
                            df = df.slice(1) # Drop the text header row from records
                            df.columns = header_names
                            
                            # Standardize column naming variations cleanly
                            rename_map = {"mmsi": "mmsi_str", "vessel_id": "mmsi_str", "timestamp": "timestamp_sec"}
                            df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
                        else:
                            # If it is a completely headless data segment, apply the canonical index mapping
                            # Matching: timestamp, mmsi, lon, lat, heading, speed, course
                            df.columns = ["timestamp_sec", "mmsi_str", "lon_raw", "lat_raw", "heading_raw", "speed_raw", "course_raw"][:df.width]
                    except Exception as e:
                        logger.info(f"Unable to read {chunk}, skipping...")
                        continue
                    
                    # Convert object identifiers to string keys cleanly
                    df = df.with_columns(pl.col("mmsi_str").cast(pl.Utf8))
                    
                    is_pure_dec = df["mmsi_str"].str.contains(r"^\d+$").fill_null(False)
                    df = df.with_columns([
                        pl.when(is_pure_dec)
                        .then(pl.col("mmsi_str").cast(pl.Int64, strict=False))
                        .otherwise(pl.lit(999999999).cast(pl.Int64))
                        .alias("vessel_id_int")
                    ])

                    required = ["timestamp_sec", "vessel_id_int", "speed_raw", "course_raw", "lon_raw", "lat_raw"]
                    if all(c in df.columns for c in required):
                        rows_collected.append(df.select(required))
                        rows_read += len(df)

            if rows_read >= sample_rows: break

    df_raw_pool = pl.concat(rows_collected).head(sample_rows)
    logger.info("Running vector feature calculations across the sample pool...")
    df_engineered = run_feature_engineering_pipeline(df_raw_pool)

    # Compile the final statistics manifest structure
    manifest_space = {
        "spatial_bounds": {
            "lon_min": PIRAEUS_LON_MIN, "lon_max": PIRAEUS_LON_MAX,
            "lat_min": PIRAEUS_LAT_MIN, "lat_max": PIRAEUS_LAT_MAX
        },
        "global_cohort_scales": {},
        "weather_meso_scales": {}
    }

    # Generate Vessel-Class specific normalization layers
    for class_label, class_idx in SUPERCLASS_VOCAB.items():
        manifest_space["global_cohort_scales"][str(class_idx)] = {}
        
        # Approximate features distribution using a fallback uniform split if cohort data is sparse
        for feat in REGISTRY.continuous_scaled:
            if feat in df_engineered.columns:
                col_data = df_engineered[feat].drop_nulls().drop_nans().to_numpy()
                if len(col_data) > 100:
                    med = float(np.median(col_data))
                    q25, q75 = np.percentile(col_data, [25, 75])
                    iqr = float(q75 - q25) if (q75 - q25) > 1e-4 else 1.0
                else:
                    med, iqr = 0.0, 1.0
                
                manifest_space["global_cohort_scales"][str(class_idx)][feat] = {"center": med, "scale": iqr}

    # Generate Monthly/Regional weather scales
    if weather_root_dir and os.path.exists(weather_root_dir):
        logger.info(f"Scanning weather directories under: {weather_root_dir}...")
        w_files = []
        for r, _, files in os.walk(weather_root_dir):
            for f in files:
                if f.endswith(".dbf") and ("2018" in r or "2019" in r):
                    w_files.append(os.path.join(r, f))

        if w_files:
            try:
                weather_records = []
                for p in sorted(w_files)[:6]:
                    dbf_file = DBF(p, load=True)
                    weather_records.extend(dbf_file.records)
                
                if weather_records:
                    df_w = pl.DataFrame(weather_records)
                    for feat in REGISTRY.continuous_scaled:
                        if feat in df_w.columns:
                            w_arr = df_w[feat].drop_nulls().drop_nans().to_numpy()
                            w_med = float(np.median(w_arr)) if len(w_arr) > 0 else 0.0
                            w_q25, w_q75 = np.percentile(w_arr, [25, 75]) if len(w_arr) > 0 else (0.0, 1.0)
                            w_iqr = float(w_q75 - w_q25) if (w_q75 - w_q25) > 1e-4 else 1.0
                            manifest_space["weather_meso_scales"][feat] = {"center": w_med, "scale": w_iqr}
                            logger.info(f"  [WEATHER] Calculated {feat}: center={w_med:.4f}, scale={w_iqr:.4f}")
            except Exception as e:
                logger.warning(f"Weather database scanner encountered a disruption: {e}")

    # Inject baseline fallback profiles for any missing weather parameters
    weather_defaults = {
        "wind_speed": {"center": 4.5, "scale": 3.0}, "wind_direction": {"center": 180.0, "scale": 90.0},
        "temperature": {"center": 293.15, "scale": 8.0}, "visibility": {"center": 10000.0, "scale": 2000.0},
        "gust_factor": {"center": 6.0, "scale": 4.0}, "pressure": {"center": 101325.0, "scale": 800.0},
        "humidity": {"center": 75.0, "scale": 15.0}, "headwind_component": {"center": 0.0, "scale": 4.0},
        "crosswind_component": {"center": 0.0, "scale": 4.0}
    }
    for k, v in weather_defaults.items():
        if k not in manifest_space["weather_meso_scales"]:
            manifest_space["weather_meso_scales"][k] = v

    return manifest_space

def compute_and_save_global_stats(parquet_root: str, output_path: str):
    """Bridge function programmatically reads your written lakehouse shards."""
    parquet_files = [os.path.join(r, f) for r, _, files in os.walk(parquet_root) for f in files if f.endswith(".parquet")]
    
    if not parquet_files:
        default_zip = "data/raw/unipi_ais_dynamic_2018.zip"
        stats = compute_stats(default_zip, weather_root_dir="data/raw/noaa_weather", sample_rows=500_000)
    else:
        # Read the unscaled data shapes directly out of your processed shards
        lazy_frames = [pl.scan_parquet(p) for p in parquet_files[:20]]
        df_lakehouse = pl.concat(lazy_frames).collect().head(300_000)
        
        manifest_space = {
            "spatial_bounds": {
                "lon_min": PIRAEUS_LON_MIN, "lon_max": PIRAEUS_LON_MAX,
                "lat_min": PIRAEUS_LAT_MIN, "lat_max": PIRAEUS_LAT_MAX
            },
            "global_cohort_scales": {}, 
            "weather_meso_scales": {}
        }
        
        # Ensure the "global_cohort_scales" profile maps cleanly
        manifest_space["global_cohort_scales"]["0"] = {}
        for feat in REGISTRY.continuous_scaled:
            if feat in df_lakehouse.columns:
                col_data = df_lakehouse[feat].drop_nulls().drop_nans().to_numpy()
                med = float(np.median(col_data)) if len(col_data) > 0 else 0.0
                q25, q75 = np.percentile(col_data, [25, 75]) if len(col_data) > 0 else (0.0, 1.0)
                iqr = float(q75 - q25) if (q75 - q25) > 1e-4 else 1.0
                manifest_space["global_cohort_scales"]["0"][feat] = {"center": med, "scale": iqr}
                
        # Re-run weather initialization loop dynamically for shard scans
        default_weather = "data/raw/noaa_weather"
        if os.path.exists(default_weather):
            w_files = []
            for r, _, files in os.walk(default_weather):
                for f in files:
                    if f.endswith(".dbf") and ("2018" in r or "2019" in r):
                        w_files.append(os.path.join(r, f))
            if w_files:
                try:
                    weather_records = []
                    for p in sorted(w_files)[:6]:
                        dbf_file = DBF(p, load=True)
                        weather_records.extend(dbf_file.records)
                    if weather_records:
                        df_w = pl.DataFrame(weather_records)
                        for feat in REGISTRY.continuous_scaled:
                            if feat in df_w.columns:
                                w_arr = df_w[feat].drop_nulls().drop_nans().to_numpy()
                                w_med = float(np.median(w_arr)) if len(w_arr) > 0 else 0.0
                                w_q25, w_q75 = np.percentile(w_arr, [25, 75]) if len(w_arr) > 0 else (0.0, 1.0)
                                w_iqr = float(w_q75 - w_q25) if (w_q75 - w_q25) > 1e-4 else 1.0
                                manifest_space["weather_meso_scales"][feat] = {"center": w_med, "scale": w_iqr}
                except Exception:
                    pass

        weather_defaults = {
            "wind_speed": {"center": 4.5, "scale": 3.0}, "wind_direction": {"center": 180.0, "scale": 90.0},
            "temperature": {"center": 293.15, "scale": 8.0}, "visibility": {"center": 10000.0, "scale": 2000.0},
            "gust_factor": {"center": 6.0, "scale": 4.0}, "pressure": {"center": 101325.0, "scale": 800.0},
            "humidity": {"center": 75.0, "scale": 15.0}, "headwind_component": {"center": 0.0, "scale": 4.0},
            "crosswind_component": {"center": 0.0, "scale": 4.0}
        }
        for k, v in weather_defaults.items():
            if k not in manifest_space["weather_meso_scales"]:
                manifest_space["weather_meso_scales"][k] = v
                
        stats = manifest_space
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Multi-scope normalization manifest saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_zip", default="data/raw/unipi_ais_dynamic_2018.zip")
    parser.add_argument("--weather_dir", default="data/raw/noaa_weather/noaa_weather")
    parser.add_argument("--output", default="configs/global_stats.json")
    args = parser.parse_args()
    
    stats = compute_stats(args.raw_zip, args.weather_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()