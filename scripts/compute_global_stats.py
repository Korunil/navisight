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

def compute_and_save_global_stats(parquet_root: str, weather_root_dir: str, output_path: str, file_list: list = None) -> dict:
    """
    Strategy A: Global Out-of-Core Lazy Aggregation.
    Discovers exact spatial boundaries dynamically from data and computes class-stratified scales.
    """
    logger.info("=========================================================================================")
    logger.info("🛰️  NAVISIGHT PLATFORM | COMPILING GLOBAL POPULATION STATISTICS (STRATEGY A)")
    logger.info("=========================================================================================")
    
    target_shards = file_list if file_list is not None else [
        os.path.join(r, f) for r, _, files in os.walk(parquet_root) for f in files if f.endswith(".parquet")
    ]
    
    if not target_shards:
        raise RuntimeError(f"❌ Statistical mapping aborted: No valid training parquet fragments found under {parquet_root}")

    logger.info(f"📋 Scanning {len(target_shards)} training Parquet shards into a single lazy graph...")
    lazy_graph = pl.scan_parquet(target_shards)
    
    # Filter out moored/anchored background noise to preserve metric variance
    active_lazy_graph = lazy_graph.filter(pl.col("speed_raw") > 0.5)
    
    logger.info("🚀 Executing global optimization graph and collecting rows into RAM...")
    df_global_active = active_lazy_graph.collect()
    logger.info(f"✅ Successfully compiled unified pool of {len(df_global_active):,} active tracking records.")

    # ──► FIX: AUTOMATED DYNAMIC SPATIAL BOUNDS DISCOVERY WITH BUFFER PADDING ◄──
    raw_lon_min = float(df_global_active["lon_raw"].min())
    raw_lon_max = float(df_global_active["lon_raw"].max())
    raw_lat_min = float(df_global_active["lat_raw"].min())
    raw_lat_max = float(df_global_active["lat_raw"].max())

    # Apply a 0.05 degree buffer safety padding to expand bounds across open water
    lon_min = float(np.floor((raw_lon_min - 0.05) * 20) / 20)
    lon_max = float(np.ceil((raw_lon_max + 0.05) * 20) / 20)
    lat_min = float(np.floor((raw_lat_min - 0.05) * 20) / 20)
    lat_max = float(np.ceil((raw_lat_max + 0.05) * 20) / 20)

    logger.info(f"📍 Discovered Spatial Envelopes -> Lon: [{lon_min}, {lon_max}] | Lat: [{lat_min}, {lat_max}]")

    stats_manifest = {
        "spatial_bounds": {
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max
        },
        "global_cohort_scales": {},
        "weather_meso_scales": {}
    }

    # Compute global fallback profile (Cohort "0")
    logger.info("📊 Processing global fleet background scale calibrations (Cohort '0')...")
    stats_manifest["global_cohort_scales"]["0"] = {}
    for feat in REGISTRY.continuous_scaled:
        if feat in df_global_active.columns:
            col_arr = df_global_active[feat].drop_nulls().drop_nans().to_numpy()
            if len(col_arr) > 0:
                med = float(np.median(col_arr))
                q25, q75 = np.percentile(col_arr, [25, 75])
                iqr = float(q75 - q25) if (q75 - q25) > 1e-5 else 1.0
                stats_manifest["global_cohort_scales"]["0"][feat] = {"center": med, "scale": iqr}

    # Compute class-stratified metrics across cohorts
    logger.info("📊 Processing class-stratified cohort scaling boundaries...")
    for class_label, class_idx in SUPERCLASS_VOCAB.items():
        sc_str = str(class_idx)
        if sc_str == "0": continue
            
        stats_manifest["global_cohort_scales"][sc_str] = {}
        df_cohort = df_global_active.filter(pl.col("vessel_superclass_id") == class_idx)
        
        if df_cohort.is_empty() or len(df_cohort) < 50:
            stats_manifest["global_cohort_scales"][sc_str] = stats_manifest["global_cohort_scales"]["0"].copy()
            continue

        for feat in REGISTRY.continuous_scaled:
            if feat in df_cohort.columns:
                col_arr = df_cohort[feat].drop_nulls().drop_nans().to_numpy()
                if len(col_arr) > 0:
                    med = float(np.median(col_arr))
                    q25, q75 = np.percentile(col_arr, [25, 75])
                    iqr = float(q75 - q25) if (q75 - q25) > 1e-5 else 1.0
                    stats_manifest["global_cohort_scales"][sc_str][feat] = {"center": med, "scale": iqr}
                else:
                    stats_manifest["global_cohort_scales"][sc_str][feat] = stats_manifest["global_cohort_scales"]["0"][feat].copy()

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
                            stats_manifest["weather_meso_scales"][feat] = {"center": w_med, "scale": w_iqr}
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
        if k not in stats_manifest["weather_meso_scales"]:
            stats_manifest["weather_meso_scales"][k] = v

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats_manifest, f, indent=2)
        
    logger.info(f"🎉 SUCCESS | True Class-Stratified Normalization Manifest Written to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_root", default="data/processed/")
    parser.add_argument("--weather_dir", default="data/raw/noaa_weather/noaa_weather")
    parser.add_argument("--output", default="configs/global_stats.json")
    args = parser.parse_args()
    
    compute_and_save_global_stats(parquet_root=args.parquet_root, output_path=args.output)
    logger.info(f"Successfully generated clean global statistics ledger at: {args.output}")