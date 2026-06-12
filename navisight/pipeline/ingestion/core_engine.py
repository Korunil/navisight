# navisight/pipeline/ingestion/core_engine.py
import polars as pl
import numpy as np
import zipfile
import io
import os
import time
import struct
import zlib
import logging
import asyncio      
import threading    
from queue import Queue, Full, Empty

from navisight.pipeline.feature_registry import compile_complete_system_manifest, resolve_hierarchical_taxonomy
from navisight.pipeline.quality_assurance import compute_kinematic_reliability_weights

from navisight.pipeline.ingestion.schemas import CANONICAL_TRACK_SCHEMA, CSV_INGESTION_INPUT_SCHEMA
from navisight.pipeline.ingestion.database import LineageDatabaseBroker
from navisight.pipeline.ingestion.cache import PureFunctionalProjectionCache
from navisight.pipeline.ingestion.kinematics import compute_kinematic_features

logger = logging.getLogger(__name__)

class HighThroughputDataEngine:
    def __init__(self, stats_json: str, static_csv: str, manifest_json: str, 
                 model_config: dict, policies: dict, sqlite_state_db: str, max_vessels: int = 250000):
        self.static_csv = static_csv
        self.manifest_path = manifest_json
        self.destination_root = ""
        self.tx_log_path = os.path.join(os.path.dirname(manifest_json), "canonical_event_log.wal")
        self.tx_spill_path = self.tx_log_path + ".spill"
        self.retry_log_path = sqlite_state_db + ".retry.wal"
        
        self.global_system_sequence_counter = 0
        
        self.db = LineageDatabaseBroker(sqlite_state_db)
        self.cache = PureFunctionalProjectionCache()
        self.system_token = compile_complete_system_manifest(model_config, policies, "EMB_SPACE_v1")
        self.chunk_processed_count = 0
        self.consumer_worker_count = 1 

        self.df_taxonomy = (
            pl.read_csv(self.static_csv)
            .rename({"vessel_id": "mmsi_str"})
            .select(["mmsi_str", "shiptype"])
            .with_columns(pl.col("shiptype").fill_null(0))
        )

        self.journal_queue = Queue(maxsize=10000)
        self.journal_thread = threading.Thread(target=self._background_journal_writer_worker, daemon=True)
        self.journal_thread.start()

        self.adaptive_backpressure_delay_sec = 0.0
        self.execute_startup_crash_recovery_reconciliation()

    def _background_journal_writer_worker(self):
        staged_lines = []
        last_flush = time.time()
        
        while True:
            line = self.journal_queue.get()
            if line is None:
                break
            staged_lines.append(line)
                
            now = time.time()
            if staged_lines and (len(staged_lines) >= 100 or (now - last_flush) > 1.0):
                try:
                    with open(self.tx_log_path, "a", encoding="utf-8") as f:
                        f.writelines(staged_lines)
                        f.flush()
                        os.fsync(f.fileno())
                    staged_lines = []
                    last_flush = now
                except Exception as e:
                    logger.error(f"Background transaction journal append failed: {e}")
                    
        if staged_lines:
            try:
                with open(self.tx_log_path, "a", encoding="utf-8") as f:
                    f.writelines(staged_lines)
            except Exception:
                pass

    def execute_startup_crash_recovery_reconciliation(self):
        if not os.path.exists(self.tx_log_path):
            return
            
        logger.info("Initializing system log snapshot recovery checking sequences...")
        max_seq_id = -1
        
        try:
            with open(self.tx_log_path, "rb") as f:
                while True:
                    curr = f.read(10)
                    if len(curr) < 10: break
                    magic, p_len, crc = struct.unpack(">2sII", curr)
                    if magic == b"EV":
                        payload = f.read(p_len)
                        if len(payload) < p_len or zlib.crc32(payload) & 0xFFFFFFFF != crc: continue
                        rec = json.loads(payload.decode('utf-8'))
                        max_seq_id = max(max_seq_id, rec.get("seq_id", -1))
                        
                        self.cache.apply_mutation_record(
                            int(rec["v_id"]), int(rec["ts"]), int(rec["seq_id"]), str(rec["mmsi"]), rec["data"]
                        )
            if max_seq_id >= 0:
                self.global_system_sequence_counter = max_seq_id + 1
            logger.info(f"System boot log reconstruction complete. Sequence clock advanced to: {self.global_system_sequence_counter}")
        except Exception as e:
            logger.error(f"Crash recovery reconciliation pipeline pass dropped: {e}")

    def _transform_bytes_to_dataframe(self, clean_bytes: bytes) -> pl.DataFrame:
        # Keep this function active for baseline byte filtering passes
        first_line_bytes = clean_bytes.split(b"\n", 1)[0]
        header_probe = first_line_bytes.lstrip(b"\xef\xbb\xbf").lower()
        
        if header_probe.startswith(b"timestamp") or header_probe.startswith(b"vessel_id"):
            df_slice = pl.read_csv(io.BytesIO(clean_bytes), has_header=True, schema=CSV_INGESTION_INPUT_SCHEMA)
        else:
            df_slice = pl.read_csv(io.BytesIO(clean_bytes), has_header=False, schema=CSV_INGESTION_INPUT_SCHEMA)

        if df_slice.is_empty(): return pl.DataFrame()

        is_pure_decimal = pl.col("mmsi_str").str.contains(r"^\d+$")
        hex_expr = (
            pl.col("mmsi_str").str.slice(-15).str.to_integer(base=16, strict=False)
            .fill_null(0).cast(pl.Int64) & pl.lit(0x7FFF_FFFF_FFFF_FFFF, dtype=pl.Int64)
        )
        dec_expr = pl.col("mmsi_str").str.to_integer(base=10, strict=False).fill_null(0).cast(pl.Int64)

        df_slice = df_slice.with_columns([
            pl.when(is_pure_decimal).then(dec_expr).otherwise(hex_expr).alias("vessel_id_int")
        ]).with_columns([
            (pl.col("vessel_id_int") % 100).cast(pl.Int32).alias("vessel_bucket_id")
        ])

        ts = pl.col("timestamp").cast(pl.Int64, strict=False)
        normalized_ts = (
            pl.when(ts > 1_000_000_000_000_000_000).then(ts // 1_000_000_000)
            .when(ts > 1_000_000_000_000_000).then(ts // 1_000_000)
            .when(ts > 1_000_000_000_000).then(ts // 1000)
            .otherwise(ts)
        )

        df_slice = df_slice.with_columns([
            normalized_ts.alias("timestamp_sec"),
        ]).filter(pl.col("timestamp_sec").is_not_null())
        
        if df_slice.is_empty(): return pl.DataFrame()
        df_slice = df_slice.with_columns(pl.from_epoch("timestamp_sec", time_unit="s").dt.replace_time_zone("UTC").alias("datetime"))

        df_slice = df_slice.filter(pl.col("lon_raw").is_not_null() & pl.col("lon_raw").is_not_nan() &
                                   pl.col("lat_raw").is_not_null() & pl.col("lat_raw").is_not_nan())
        if df_slice.is_empty(): return pl.DataFrame()

        invalid_speed = pl.col("speed_raw").is_null() | pl.col("speed_raw").is_nan()
        invalid_course = pl.col("course_raw").is_null() | pl.col("course_raw").is_nan()
        
        course_expr = pl.when(invalid_course).then(0.0).otherwise(pl.col("course_raw"))
        course_rad = course_expr * (np.pi / 180.0)
        
        df_slice = df_slice.with_columns([
            invalid_course.cast(pl.Int32).alias("flag_null_course"),
            course_expr.alias("course_raw"),
            pl.when(invalid_speed).then(0.0).otherwise(pl.col("speed_raw")).alias("speed_raw"),
            course_rad.sin().alias("course_sin"),
            course_rad.cos().alias("course_cos"),
        ]).join(self.df_taxonomy, on="mmsi_str", how="left")

        raw_shiptypes = df_slice["shiptype"].to_numpy()
        superclasses = np.zeros(len(df_slice), dtype=np.int16)
        behaviors = np.zeros(len(df_slice), dtype=np.int16)
        for i, st in enumerate(raw_shiptypes):
            sc, bh = resolve_hierarchical_taxonomy(st)
            superclasses[i] = sc
            behaviors[i] = bh

        return df_slice.with_columns([
            pl.Series("vessel_superclass_id", superclasses, dtype=pl.Int16),
            pl.Series("vessel_behavior_id", behaviors, dtype=pl.Int16),
        ])

    async def pipeline_orchestrator(self, zip_path: str, weather_engine, destination_root: str, segmentation_engine):
        """Unified processing pipeline utilizing streaming zip extraction mappings natively via Polars."""
        self.destination_root = destination_root
        ingest_seq_id = 0
        chunk_trackers = {}

        with zipfile.ZipFile(zip_path, "r") as archive:
            # FIXED: Normalize Windows backslash delimiters to standard forward slashes to discover file members
            csv_members = [
                m for m in archive.namelist() 
                if m.replace("\\", "/").endswith(".csv") and not m.replace("\\", "/").startswith("__MACOSX")
            ]
            
            if not csv_members:
                logger.error(f"CRITICAL FAULT: No readable CSV track logs discovered inside archive: {zip_path}")
                return

            for file_idx, csv_member in enumerate(csv_members):
                logger.info(f"Processing {csv_member} file")
                chunk_trackers[file_idx] = chunk_trackers.get(file_idx, 0) + 1
                self.chunk_processed_count += 1
                ingest_seq_id += 1

                # FIXED: Streams archive handle directly into Polars, bypassing Python buffer block constraints
                with archive.open(csv_member, "r") as stream:
                    df_raw_stream = pl.read_csv(stream, schema=CSV_INGESTION_INPUT_SCHEMA)
                
                if df_raw_stream.is_empty():
                    continue

                # Run raw dataset parsing structures over the incoming data frames
                # Converting via Bytes array mappings optimizes execution speeds
                clean_bytes = df_raw_stream.write_csv().encode('utf-8')
                df_slice = self._transform_bytes_to_dataframe(clean_bytes)
                if df_slice.is_empty(): continue

                # Map Ingress Sequential Identity Counters
                slice_len = len(df_slice)
                sequence_array = np.arange(self.global_system_sequence_counter, self.global_system_sequence_counter + slice_len, dtype=np.int64)
                self.global_system_sequence_counter += slice_len
                
                df_slice = df_slice.with_columns([
                    pl.Series("global_lsn", sequence_array, dtype=pl.Int64),
                    pl.lit(int(self.chunk_processed_count)).cast(pl.Int64).alias("ingest_sequence_id")
                ])

                # ── CONSTRUCT GHOST CONTINUITY CHANNELS FROM AUTHORITATIVE LOG MEMTABLE SCHEMAS ──
                unique_vessels = df_slice["vessel_id_int"].unique().to_numpy()
                ghost_records = []
                max_ghost_gap_seconds = 3600
                chunk_min_ts = df_slice["timestamp_sec"].min()
                
                for v_id in unique_vessels:
                    v_id_int = int(v_id)
                    cache_snapshot = self.cache.query_vessel_state_read_only(v_id_int)
                    
                    if cache_snapshot is not None:
                        if 0 < (chunk_min_ts - cache_snapshot.last_timestamp) <= max_ghost_gap_seconds:
                            ghost_records.append((
                                v_id_int, int(cache_snapshot.last_timestamp), float(cache_snapshot.last_speed), 
                                float(cache_snapshot.last_lon), float(cache_snapshot.last_lat), 0.0, 0.0, 1, 
                                str(cache_snapshot.mmsi_str), int(cache_snapshot.global_lsn)
                            ))

                if ghost_records:
                    df_ghost_payload = pl.DataFrame(ghost_records, schema={
                        "vessel_id_int": pl.Int64, "timestamp_sec": pl.Int64, "speed_raw": pl.Float64,
                        "lon_raw": pl.Float64, "lat_raw": pl.Float64, "course_raw": pl.Float64,
                        "heading_raw": pl.Float64, "is_ghost": pl.Int32, "mmsi_str": pl.Utf8, "global_lsn": pl.Int64
                    }).with_columns([
                        pl.from_epoch("timestamp_sec", time_unit="s").dt.replace_time_zone("UTC").alias("datetime"),
                        pl.lit(0).cast(pl.Int64).alias("timestamp"), pl.lit(0).cast(pl.Int64).alias("ingest_sequence_id")
                    ])
                    df_ghost = df_ghost_payload.select([pl.col(c).cast(CANONICAL_TRACK_SCHEMA[c]) for c in CANONICAL_TRACK_SCHEMA])
                    df_slice_aligned = df_slice.with_columns(pl.lit(0).cast(pl.Int32).alias("is_ghost")).select(df_ghost.columns)
                    df_working = pl.concat([df_slice_aligned, df_ghost], how="vertical")
                else:
                    df_working = df_slice.with_columns(pl.lit(0).cast(pl.Int32).alias("is_ghost"))

                df_enriched = weather_engine.interpolate_slice_causal(df_working)
                df_enriched = compute_kinematic_features(df_enriched)
                df_enriched = segmentation_engine.assign_voyage_phases_and_trips(df_enriched, self)
                df_final = compute_kinematic_reliability_weights(df_enriched)

                df_final = df_final.filter(pl.col("is_ghost") == 0).drop("is_ghost")
                if df_final.is_empty(): continue

                valid_dt = df_final["datetime"].drop_nulls()
                if len(valid_dt) == 0: continue
                sample_dt = valid_dt[0]

                df_tails = df_final.sort("global_lsn").unique(subset=["vessel_id_int"], keep="last")
                v_ids, ts_vals, speeds = df_tails["vessel_id_int"].to_numpy(), df_tails["timestamp_sec"].to_numpy(), df_tails["speed_raw"].to_numpy()
                lons, lats, strings_mmsi = df_tails["lon_raw"].to_numpy(), df_tails["lat_raw"].to_numpy(), df_tails["mmsi_str"].to_numpy()
                seq_ids_arr = df_tails["global_lsn"].to_numpy()
                is_static_arr = df_tails["is_static"].to_numpy() if "is_static" in df_tails.columns else np.zeros(len(df_tails))

                db_records_to_append = []
                for i in range(len(df_tails)):
                    v_id_int = int(v_ids[i])
                    self.cache.apply_mutation_record(
                        v_id_int, int(ts_vals[i]), int(seq_ids_arr[i]), str(strings_mmsi[i]),
                        {"last_timestamp": int(ts_vals[i]), "last_speed": float(speeds[i]), "last_lon": float(lons[i]), 
                         "last_lat": float(lats[i]), "last_is_static": int(is_static_arr[i])}
                    )
                        
                    db_records_to_append.append((
                        v_id_int, int(ts_vals[i]), float(speeds[i]), float(lons[i]), float(lats[i]),
                        int(is_static_arr[i]), 0, 0, int(ts_vals[i]), int(seq_ids_arr[i])
                    ))

                if db_records_to_append:
                    await asyncio.get_running_loop().run_in_executor(None, self.db.apply_log_records, db_records_to_append)

                year_str, month_str = f"{sample_dt.year:04d}", f"{sample_dt.month:02d}"

                for df_bucket in df_final.partition_by("vessel_bucket_id", maintain_order=False):
                    if df_bucket.is_empty(): continue
                    bucket_int = int(df_bucket["vessel_bucket_id"][0])
                    
                    partition_path = os.path.join(destination_root, f"year={year_str}", f"month={month_str}", f"vessel_bucket={bucket_int:02d}")
                    os.makedirs(partition_path, exist_ok=True)
                    
                    out_file = os.path.join(partition_path, f"shard_{bucket_int}.parquet")
                    df_bucket.write_parquet(out_file + ".tmp", use_pyarrow=True)
                    os.replace(out_file + ".tmp", out_file)

                log_payload = f"COMMIT|{self.system_token}|{year_str}/{month_str}|chunks={ingest_seq_id}\n"
                try:
                    self.journal_queue.put_nowait(log_payload)
                    self.adaptive_backpressure_delay_sec = max(0.0, self.adaptive_backpressure_delay_sec - 0.005)
                except Full:
                    self.adaptive_backpressure_delay_sec = min(0.2, self.adaptive_backpressure_delay_sec + 0.02)
                    with open(self.tx_spill_path, "ab") as sf:
                        sf.write(log_payload.encode('utf-8'))

                if self.chunk_processed_count % 500 == 0:
                    await asyncio.get_running_loop().run_in_executor(None, self.db.prune_historical_lineage)

    @property
    def vessel_index_map(self) -> dict:
        """
        Backward-compatible interface proxy property hook.
        Maps external segmentation checks safely down into the pure log projection cache keys.
        """
        # Exposes the active cache dictionary key footprint to satisfy voyage_segmentation.py
        return self.cache.state_map

    @property
    def state_cache_matrix(self):
        """
        Backward-compatible interface proxy property hook.
        Intercepts raw index mutations from voyage_segmentation.py and 
        applies them safely to the pure log projection cache view map.
        """
        class VirtualStateCacheMatrix:
            def __init__(self, engine_ref):
                self.engine_ref = engine_ref

            def __getitem__(self, slot_key):
                # Returns a proxy handler that intercepts key-based updates (like matrix[slot]["field"] = val)
                class RowMutationProxy:
                    def __init__(self, cache_ref, vessel_id):
                        self.cache_ref = cache_ref
                        self.vessel_id = vessel_id

                    def __setitem__(self, field_key, value):
                        # Intercept the direct field update and channel it through the mutation gate
                        snapshot = self.cache_ref.query_vessel_state_read_only(self.vessel_id)
                        if snapshot is not None:
                            # Re-apply record parameters back into the functional projection state map
                            self.cache_ref.apply_mutation_record(
                                vessel_id_int=self.vessel_id,
                                timestamp_sec=snapshot.last_timestamp,
                                global_lsn=snapshot.global_lsn,
                                mmsi_str=snapshot.mmsi_str,
                                record_data={
                                    "last_speed": snapshot.last_speed,
                                    "last_lon": snapshot.last_lon,
                                    "last_lat": snapshot.last_lat,
                                    field_key: value  # Apply the updated attribute dynamically (e.g., trip_id_offset)
                                }
                            )

                # Resolve the vessel ID associated with this key slot
                vessel_id = None
                if hasattr(slot_key, 'vessel_id_int'):
                    vessel_id = slot_key.vessel_id_int
                elif isinstance(slot_key, dict) and 'vessel_id_int' in slot_key:
                    vessel_id = slot_key['vessel_id_int']
                elif isinstance(slot_key, int):
                    # Fallback lookup: assume the integer is a vessel_id_int directly
                    vessel_id = slot_key

                if vessel_id is not None:
                    return RowMutationProxy(self.engine_ref.cache, vessel_id)
                    
                # Safe fallback if key typing wanders out of bounds
                class EmptyProxy:
                    def __setitem__(self, k, v): pass
                return EmptyProxy()

        return VirtualStateCacheMatrix(self)


    @property
    def slot_generation_vec(self):
        """
        Backward-compatible interface proxy property hook.
        Synthesizes a virtual generation sequence track to satisfy voyage_segmentation.py lookups.
        """
        class VirtualGenerationVector:
            def __init__(self, cache_ref):
                self.cache_ref = cache_ref

            def __getitem__(self, slot_key):
                # Maps slot tracking identifiers to their respective global_lsn sequence value.
                # If the key is missing or is a VesselStateValue object, extracts the structural sequence ID attribute cleanly.
                if hasattr(slot_key, 'global_lsn'):
                    return slot_key.global_lsn
                elif isinstance(slot_key, int) and slot_key in self.cache_ref.state_map:
                    return self.cache_ref.state_map[slot_key].global_lsn
                return 0

        return VirtualGenerationVector(self.cache)


    def get_vessel_state(self, vessel_id_int: int, expected_gen_id: int) -> np.ndarray:
        """
        Retrieves or rehydrates authoritative vessel state coordinates safely.
        Aligns the functional cache output format to match downstream matrix requirements.
        """
        # Step 1: Query the immutable functional cache view representation
        snapshot = self.cache.query_vessel_state_read_only(vessel_id_int)
        
        # Step 2: Cold-start fallback path. If missing from cache, pull directly from the database view
        if snapshot is None:
            row = self.db.fetch_vessel_history(vessel_id_int)
            if row is None:
                return None
                
            # Populate the cache view asynchronously to accelerate subsequent lookups
            self.cache.apply_mutation_record(
                vessel_id_int=vessel_id_int,
                timestamp_sec=int(row[7]),
                global_lsn=int(row[8]),
                mmsi_str="",
                record_data={
                    "last_speed": float(row[1]),
                    "last_lon": float(row[2]),
                    "last_lat": float(row[3])
                }
            )
            snapshot = self.cache.query_vessel_state_read_only(vessel_id_int)

        from navisight.pipeline.feature_registry import VESSEL_STATE_DTYPE
        
        # Step 3: Package the values into a structured numpy row profile to keep voyage_segmentation.py intact
        vessel_matrix_row = np.zeros((), dtype=VESSEL_STATE_DTYPE)
        vessel_matrix_row["vessel_id_int"] = snapshot.vessel_id_int
        vessel_matrix_row["last_timestamp"] = snapshot.last_timestamp
        vessel_matrix_row["last_speed"] = snapshot.last_speed
        vessel_matrix_row["last_lon"] = snapshot.last_lon
        vessel_matrix_row["last_lat"] = snapshot.last_lat
        vessel_matrix_row["generation_counter"] = snapshot.global_lsn
        
        return vessel_matrix_row

    def close(self):
        logger.info("Initiating structural data pipeline shutdown sequences...")
        try:
            # self.db.force_transaction_commit()
            for _ in range(self.consumer_worker_count):
                self.journal_queue.put(None)
            self.journal_thread.join(timeout=5.0)
            self.db.close()
            logger.info("Subsystem infrastructure layers finalized successfully.")
        except Exception as e:
            logger.warning(f"Shutdown pass encountered a disruption context: {e}")