# navisight/pipeline/voyage_segmentation.py
"""
Voyage segmentation for Navisight AI.
 
Segments raw AIS trajectory streams into discrete voyage windows.
A voyage boundary is triggered by either:
  1. A time gap exceeding the class-specific anchor limit
  2. A vessel transitioning from underway to static (at-anchor) state
 
State is persisted in SQLite so that boundaries at chunk edges are handled
correctly even when processing large multi-file archives in serial chunks.
 
NOTE: This module operates on RAW (unscaled) speed_raw values in knots.
      Ingestion must NOT apply z-scoring before this step.
      (Original bug: ingestion scaled speed before segmentation ran, making
       the velocity_floor comparisons meaningless.)
"""

import polars as pl
import sqlite3
import numpy as np
import logging
 
logger = logging.getLogger(__name__)

class CompleteContinuitySegmentationEngine:

    """
    Vectorised voyage segmenter.
 
    Vessel-class-specific parameters:
        velocity_floor — minimum SOG (knots) below which vessel is considered static
        anchor_limit   — maximum static duration (seconds) before a new voyage starts
 
    These reflect realistic operational patterns:
        Fishing vessels (class 3)  : low speed, long anchoring periods
        Tug/Towing (class 4)       : slow transit, moderate anchor limit
        Cargo/Tanker (class 11/12) : high speed, short in-port dwell
        Default                    : moderate parameters
    """
 
    # (velocity_floor knots, anchor_limit seconds)
    _CLASS_PARAMS = {
        3:  (0.2, 28_800),   # Fishing
        4:  (0.4, 14_400),   # TugTowing
        11: (1.0,  7_200),   # Cargo
        12: (1.0,  7_200),   # Tanker
    }
    _DEFAULT_PARAMS = (0.5, 14_400)
 
    def __init__(
        self, 
        sqlite_state_db: str, 
        time_gap_threshold_sec: int = 1800
    ):
        self.time_gap_limit = time_gap_threshold_sec
        self.db_path = sqlite_state_db
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voyage_lineage_metadata (
                    vessel_id_int   INTEGER PRIMARY KEY,
                    last_timestamp  INTEGER, 
                    last_is_static  INTEGER,
                    static_duration INTEGER, 
                    trip_offset     INTEGER
                );
            """)
            conn.commit()

    def segment_voyage_chunks_pure_vectorized(
        self, df_chunk: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Assign trip_id to every row in df_chunk.
 
        Requires columns: vessel_id_int, timestamp_sec, speed_raw,
                          vessel_superclass_id (from taxonomy join).
        Returns df_chunk with trip_id column added and internal intermediates
        dropped.
        """
        df_sorted = df_chunk.sort(["vessel_id_int", "timestamp_sec"])
        unique_vessels = df_sorted["vessel_id_int"].unique().to_numpy()
        
        # ── Load persisted trip offsets ───────────────────────────────────────
        vessel_offsets = []
        with sqlite3.connect(self.db_path) as conn:
            for v_id in unique_vessels:
                cursor = conn.execute("SELECT trip_offset FROM voyage_lineage_metadata WHERE vessel_id_int = ?;", (int(v_id),))
                row = cursor.fetchone()
                offset = row[0] if row else 0
                vessel_offsets.append((v_id, offset))
                
        df_offsets = pl.DataFrame(vessel_offsets, schema=["vessel_id_int", "trip_offset"])
        
        # ── Assign per-vessel class parameters ───────────────────────────────
        df_sorted = df_sorted.with_columns([
            pl.col("timestamp_sec")
              .diff()
              .over("vessel_id_int")
              .fill_null(0)
              .alias("_time_delta"),
        ])
 
        # Build velocity_floor and anchor_limit per row from superclass_id
        # Using when/then chains for Polars compatibility
        vel_expr = (
            pl.when(pl.col("vessel_superclass_id") == 3).then(0.2)
              .when(pl.col("vessel_superclass_id") == 4).then(0.4)
              .when(pl.col("vessel_superclass_id").is_in([11, 12])).then(1.0)
              .otherwise(0.5)
              .alias("_velocity_floor")
        )
        anchor_expr = (
            pl.when(pl.col("vessel_superclass_id") == 3).then(28_800)
              .when(pl.col("vessel_superclass_id") == 4).then(14_400)
              .when(pl.col("vessel_superclass_id").is_in([11, 12])).then(7_200)
              .otherwise(14_400)
              .alias("_anchor_limit")
        )
 
        df_sorted = df_sorted.with_columns([vel_expr, anchor_expr])
 
        # ── is_static flag ────────────────────────────────────────────────────
        # speed_raw is RAW knots here — safe because ingestion no longer scales
        df_sorted = df_sorted.with_columns(
            (pl.col("speed_raw") < pl.col("_velocity_floor"))
            .alias("is_static")
        )
 
        # ── Static duration accumulator ───────────────────────────────────────
        df_sorted = df_sorted.with_columns(
            (pl.col("is_static") != pl.col("is_static").shift(1).over("vessel_id_int"))
              .fill_null(True)
              .alias("_state_shifted")
        ).with_columns(
            pl.col("_state_shifted")
              .cast(pl.Int32)
              .cum_sum()
              .over("vessel_id_int")
              .alias("_state_run_id")
        ).with_columns(
            pl.when(pl.col("is_static"))
              .then(
                  pl.col("_time_delta")
                    .cum_sum()
                    .over(["vessel_id_int", "_state_run_id"])
              )
              .otherwise(0)
              .alias("_active_static_duration")
        )
 
        # ── Break detection ───────────────────────────────────────────────────
        df_sorted = df_sorted.with_columns(
            (
                (pl.col("_time_delta") > self.time_gap_limit) |
                (pl.col("_active_static_duration") > pl.col("_anchor_limit"))
            )
            .cast(pl.Int32)
            .alias("_raw_break_flag")
        )
 
        # ── Join persisted offsets and compute trip_id ────────────────────────
        df_out = (
            df_sorted
            .join(df_offsets, on="vessel_id_int", how="left")
            .with_columns(
                (
                    pl.col("_raw_break_flag")
                      .cum_sum()
                      .over("vessel_id_int")
                    + pl.col("trip_offset")
                ).alias("trip_id")
            )
        )
 
        # ── Persist updated offsets ───────────────────────────────────────────
        df_lasts = df_out.unique(subset=["vessel_id_int"], keep="last")
        with sqlite3.connect(self.db_path) as conn:
            for row in df_lasts.iter_rows(named=True):
                conn.execute(
                    "INSERT OR REPLACE INTO voyage_lineage_metadata VALUES (?,?,?,?,?);",
                    (
                        int(row["vessel_id_int"]),
                        int(row["timestamp_sec"]),
                        int(row["is_static"]),
                        int(row["_active_static_duration"]),
                        int(row["trip_id"]),
                    ),
                )
            conn.commit()
 
        # ── Drop internal columns ─────────────────────────────────────────────
        internal = [
            "_time_delta", "_velocity_floor", "_anchor_limit",
            "_state_shifted", "_state_run_id", "_active_static_duration",
            "_raw_break_flag", "trip_offset",
        ]
        return df_out.drop([c for c in internal if c in df_out.columns])
        
    def assign_voyage_phases_and_trips(self, df: pl.DataFrame, ingestion_engine) -> pl.DataFrame:
        """
        Highly optimized, vectorized stream segmentation engine.
        Computes within-chunk transitions natively in Polars and hooks to cache handles.
        """
        if df.is_empty():
            return df

        # 1. Vectorized intra-chunk transition profiling via native expressions
        df_sorted = df.sort(["vessel_id_int", "timestamp_sec"])
        
        # Track time deltas and speed states within the chunk boundaries
        dt = pl.col("timestamp_sec").diff().over("vessel_id_int").fill_null(0)
        is_static_col = pl.col("speed_raw") < 1.0
        was_static_col = is_static_col.shift(1).over("vessel_id_int")
        
        # Identify row indexes where a trip increment is physically triggered
        gap_trigger = dt > 1800
        state_trigger = (was_static_col.is_not_null()) & (is_static_col != was_static_col)
        intra_chunk_increments = (gap_trigger | state_trigger).cast(pl.Int32)
        
        # Compute base relative increments inside this slice frame
        df_expr = df_sorted.with_columns([
            is_static_col.alias("is_static"),
            intra_chunk_increments.cum_sum().over("vessel_id_int").alias("slice_trip_increment")
        ])

        # 2. Extract boundaries to reconcile chunk offsets against cache history
        df_boundaries = df_expr.group_by("vessel_id_int", maintain_order=True).agg([
            pl.col("timestamp_sec").first().alias("first_ts"),
            pl.col("speed_raw").first().alias("first_speed")
        ])

        v_ids = df_boundaries["vessel_id_int"].to_numpy()
        first_ts_arr = df_boundaries["first_ts"].to_numpy()
        first_speed_arr = df_boundaries["first_speed"].to_numpy()
        
        # Pre-allocate dictionary arrays to map absolute vessel trip adjustments
        trip_offsets = {}

        for i in range(len(df_boundaries)):
            v_id = int(v_ids[i])
            
            # Look up generation metrics from the ingestion matrix registry
            if v_id in ingestion_engine.vessel_index_map:
                slot = ingestion_engine.vessel_index_map[v_id]
                gen_id = int(ingestion_engine.slot_generation_vec[slot])
                v_state = ingestion_engine.get_vessel_state(v_id, gen_id)
            else:
                v_state = None

            if v_state is not None:
                current_offset = int(v_state["trip_id_offset"])
                last_ts = int(v_state["last_timestamp"])
                last_speed = float(v_state["last_speed"])
                
                # Check if a boundary break occurred between the prior chunk tail and this chunk head
                boundary_gap = (int(first_ts_arr[i]) - last_ts) > 1800
                boundary_state = (float(first_speed_arr[i]) < 1.0) != (last_speed < 1.0)
                
                if boundary_gap or boundary_state:
                    current_offset += 1
                    
                trip_offsets[v_id] = current_offset
            else:
                trip_offsets[v_id] = 0

        # 3. Apply the compiled trip offsets to vector groups using a mapping link
        # If a vessel wasn't cached, default its baseline offset to 0
        offset_series = df_expr["vessel_id_int"].replace(trip_offsets, default=0).cast(pl.Int32)
        
        # Calculate final global trip tokens and update tracking states
        df_final = df_expr.with_columns([
            (offset_series + pl.col("slice_trip_increment")).alias("absolute_trip_counter")
        ]).with_columns([
            (pl.col("vessel_id_int").cast(pl.String) + "_" + pl.col("absolute_trip_counter").cast(pl.String)).alias("trip_id")
        ]).drop(["slice_trip_increment", "absolute_trip_counter"])

        # 4. Synchronize active offsets back to our caching registry matrix
        df_tails = df_final.sort(["vessel_id_int", "timestamp_sec"]).unique(subset=["vessel_id_int"], keep="last")
        tail_v_ids = df_tails["vessel_id_int"].to_numpy()
        tail_trip_ids = df_tails["trip_id"].to_numpy()

        for i in range(len(df_tails)):
            v_id = int(tail_v_ids[i])
            if v_id in ingestion_engine.vessel_index_map:
                slot = ingestion_engine.vessel_index_map[v_id]
                # Parse out the tail counter index from the composite string handle "MMSI_OFFSET"
                final_offset = int(tail_trip_ids[i].split("_")[-1])
                ingestion_engine.state_cache_matrix[slot]["trip_id_offset"] = final_offset

        return df_final