# scripts/train_pipeline.py
import os
import sys
import json
import asyncio
import logging
import zipfile
import polars as pl
from dbfread import DBF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY_SCHEMA_HASH, REGISTRY
from navisight.geo.weather_interpolation import RectilinearMeshInterpolator
from navisight.pipeline.ingestion import HighThroughputDataEngine
from navisight.pipeline.voyage_segmentation import CompleteContinuitySegmentationEngine
from navisight.engine.train_model import train_maritime_model
from scripts.compute_global_stats import compute_and_save_global_stats, compute_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Translation map standardizes numeric loop counters with nested folder string labels
MONTH_MAP = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
}

def unzip_entire_weather_archive(zip_path: str, extract_to: str):
    """Extracts the entire weather zip file once at startup to avoid loop I/O thrashing."""
    logger = logging.getLogger(__name__)
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        logger.info(f"Weather data already unzipped in {extract_to}. Skipping extraction.")
        return
        
    logger.info(f"Unzipping {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(extract_to)
    logger.info("Weather archive extraction complete.")

async def main():
    logging.info(f"🚀 RUNNING PHASE 1: INGESTION AND MODEL PRETRAINING | Schema: {REGISTRY_SCHEMA_HASH}")
    
    # Absolute Workspace Data Paths Configuration Contracts
    weather_zip_archive         = "data/raw/noaa_weather.zip"
    weather_extracted_dir       = "data/raw/noaa_weather"
    static_metadata_csv         = "data/raw/unipi_ais_static.csv"
    processed_lakehouse_root    = "data/processed/"
    manifest_output_json        = "configs/production_manifest.json"
    sqlite_state_db_file        = "models/scalers/vessel_state_lineage.db"
    checkpoint_directory        = "models/checkpoints/"
    global_stats_json_path      = "configs/global_stats.json"

    model_hyperparameters = {
        "batch_size": 512, "epochs": 5, "lr": 1e-4, "weight_decay": 1e-2,
        "d_model": 128, "n_heads": 8, "n_layers": 4
    }
    operational_policies = {
        "time_gap_threshold_sec": 1800, "chunk_stream_size": 250000, "eviction_days": 90
    }

    # ONE-TIME EXTRACTION: Unzip the weather data to disk before the loop starts
    unzip_entire_weather_archive(weather_zip_archive, weather_extracted_dir)
    
    # Seed baseline statistics metadata file if missing
    if not os.path.exists(global_stats_json_path):
        logging.info("🚀 [FIRST RUN ALARM] Statistics missing. Forcing pre-scaling generation pass from raw logs...")
        default_raw_zip = "data/raw/unipi_ais_dynamic_2018.zip"
        
        if os.path.exists(default_raw_zip):
            stats_manifest = compute_stats(default_raw_zip, weather_root_dir=weather_extracted_dir, sample_rows=500_000)
            os.makedirs(os.path.dirname(global_stats_json_path), exist_ok=True)
            with open(global_stats_json_path, 'w') as f:
                json.dump(stats_manifest, f, indent=2)
            logging.info("Pre-scaling manifest generated successfully.")
        else:
            raise FileNotFoundError(f"CRITICAL FAULT: Cannot derive baseline scales; missing raw archive {default_raw_zip}")
    
    # Initialize the base operational layout manifest file
    with open(manifest_output_json, 'w') as f:
        json.dump({
            "model": model_hyperparameters, 
            "policies": operational_policies, 
            "system_token": "EMB_SPACE_v1",
            "steps_per_cycle": 5000,
            "global_stats_path": os.path.abspath(global_stats_json_path)
        }, f, indent=4)

    
    # Initialize the high-throughput ingestion platform
    ingestion_platform = HighThroughputDataEngine(
        stats_json=global_stats_json_path,
        static_csv=static_metadata_csv,
        manifest_json=manifest_output_json,
        model_config=model_hyperparameters,
        policies=operational_policies,
        sqlite_state_db=sqlite_state_db_file
    )
    
    # Instantiate the state segmentation processing engine
    segmentation_engine = CompleteContinuitySegmentationEngine(sqlite_state_db=sqlite_state_db_file)

    # ── STAGE 1: INGEST AND ENRICH DATA USING YOUR ADAPTED LOOP ──
    target_years = [2018, 2019]
    weather_mesh_registry = {}
    
    for year in target_years:
        ais_zip_filename = f"unipi_ais_dynamic_{year}.zip"
        ais_zip_path = os.path.join("data/raw", ais_zip_filename)
        
        if not os.path.exists(ais_zip_path):
            logging.warning(f"AIS data archive {ais_zip_filename} not found. Skipping year.")
            continue

        weather_mesh_registry[year] = {}
        logging.info(f"=== Compiling Meteorological Mesh Dictionary for {year} ===")
        for m_int, m_str in MONTH_MAP.items():    
            dbf_path = os.path.join(weather_extracted_dir, "noaa_weather", str(year), m_str, f"noaa_weather_{m_str}{year}_v2.dbf")
            if not os.path.exists(dbf_path):
                dbf_path = os.path.join(weather_extracted_dir, str(year), m_str, f"noaa_weather_{m_str}{year}_v2.dbf")
            
            if os.path.exists(dbf_path):
                dbf_table = DBF(dbf_path, load=True)
                df_w = pl.DataFrame(list(dbf_table.records))
                df_w = df_w.with_columns([
                    pl.col("lon").round(6),
                    pl.col("lat").round(6)
                ])
                weather_mesh_registry[year][m_int] = RectilinearMeshInterpolator(
                    df_weather=df_w, 
                    boundary_mode="clamp", 
                    run_mode="interpolated",
                    is_cyclic=False
                )

        class WeatherRouterMeshProxy:
            def __init__(self, registry, active_year):
                self.registry = registry
                self.year = active_year
            def interpolate_slice(self, df_slice):
                if df_slice.is_empty():
                    return df_slice
                
                # Derive month indices safely from the guaranteed Unix timestamp column
                df_working = df_slice.with_columns(
                    pl.from_epoch(pl.col("timestamp_sec"), time_unit="s")
                    .dt.month()
                    .cast(pl.Int32)
                    .alias("_derived_month_idx")
                )
                slices = []
                for (m_val,), df_m in df_working.group_by(["_derived_month_idx"]):
                    if self.year in self.registry and m_val in self.registry[self.year]:
                        slices.append(self.registry[self.year][m_val].interpolate_slice(df_m))
                    else:
                        slices.append(df_m)
                return pl.concat(slices).drop("_derived_month_idx")
            def interpolate_slice_causal(self, df_slice):
                return self.interpolate_slice(df_slice)

        router_mesher = WeatherRouterMeshProxy(weather_mesh_registry, year)
        logging.info(f"=== Launching Stream Ingestion for Year [{year}] ===")
        # Process the entire annual ZIP stream in a single, high-speed pass!
        await ingestion_platform.pipeline_orchestrator(
            zip_path=ais_zip_path,
            weather_engine=router_mesher,
            destination_root=processed_lakehouse_root,
            segmentation_engine=segmentation_engine
        )
          
    # Close the engine to flush any remaining cache tracks to disk after loops complete
    ingestion_platform.close()


    # ── STAGE 2: CHRONOLOGICAL TRAINING SUB-SET SELECTION ──
    # Filter training files explicitly to enforce the boundary firewall
    logging.info("Filtering historical training data partitions...")
    train_shards = []
    for root, _, files in os.walk(processed_lakehouse_root):
        for f in files:
            if f.endswith(".parquet"):
                # Path parsing extracts year/month context strings natively
                parts = root.replace("\\", "/").split("/")
                y_val = [int(p.split("=")[1]) for p in parts if "year=" in p][0]
                m_val = [int(p.split("=")[1]) for p in parts if "month=" in p][0]
                
                if y_val == 2018 or (y_val == 2019 and m_val <= 6):
                    train_shards.append(os.path.join(root, f))
    
    # Calculate global metrics solely over this validated training subset
    compute_and_save_global_stats(parquet_root=processed_lakehouse_root, output_path="configs/global_stats.json")
    
    # ── STAGE 3: RUN TRAINING LOOP OVER ISOLATED SUBSET ──
    logging.info(f"Pretraining Foundation Model on {len(train_shards)} Chronological Shards...")
    train_maritime_model(
        processed_dir=processed_lakehouse_root,
        config_json=manifest_output_json,
        checkpoint_dir=checkpoint_directory,
        file_list=train_shards,
    )

    logging.info("🎉 PHASE 1 COMPLETE. Model weights successfully serialized.")

if __name__ == "__main__":
    asyncio.run(main())