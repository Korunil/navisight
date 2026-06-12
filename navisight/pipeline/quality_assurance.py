# navisight/pipeline/quality_assurance.py
"""
Kinematic reliability scoring for Navisight AI.
 
Produces per-observation quality flags and a composite reliability score [0, 1].
The reliability score is used as a per-token loss weight in NumericallyStableMaskedLoss
so that physically impossible observations contribute less to the MAE gradient.
 
Flags:
    flag_speed_anomaly    — SOG > 45 kn (surface vessel physical limit)
    flag_frozen_position  — lat/lon unchanged but SOG > 2 kn (sensor fault)
    flag_position_jump    — implied SOG > 90 kn OR acceleration > 15 kn/s
    flag_null_course      — course was NaN in source data; imputed to 0.0
    reliability_score     — 1.0 - sum(penalties), clipped to [0, 1]
"""
 
import polars as pl
 
 
def compute_kinematic_reliability_weights(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute quality flags and reliability score for every row.
 
    Expects df to have columns:
        actual_speed_knots, acceleration, lon_raw, lat_raw,
        speed_raw, vessel_id_int, log_time_diff, flag_null_course
 
    Returns df with five new columns matching QUALITY_FEATURES in feature_registry.py.
    """
 
    df_validated = df.with_columns([
 
        # ── flag_speed_anomaly ────────────────────────────────────────────────
        # SOG > 45 knots is physically impossible for any surface vessel.
        # This threshold is generous — fastest patrol boats top out ~50 kn.
        (pl.col("actual_speed_knots") > 45.0)
        .cast(pl.Int32)
        .alias("flag_speed_anomaly"),
 
        # ── flag_position_jump ────────────────────────────────────────────────
        # Implied SOG > 90 kn or acceleration > 15 kn/s indicates a
        # position teleport — GPS spoof or corrupt record.
        (
            (pl.col("actual_speed_knots") > 90.0) |
            (pl.col("acceleration").abs() > 15.0)
        )
        .fill_null(False)
        .cast(pl.Int32)
        .alias("flag_position_jump"),
 
        # ── flag_frozen_position ─────────────────────────────────────────────
        # Position is identical to previous observation, yet SOG > 2 kn.
        # Indicates AIS transponder sending stale GPS fix.
        (
            (pl.col("lon_raw") == pl.col("lon_raw").shift(1).over("vessel_id_int")) &
            (pl.col("lat_raw") == pl.col("lat_raw").shift(1).over("vessel_id_int")) &
            (pl.col("speed_raw") > 2.0)
        )
        .fill_null(False)
        .cast(pl.Int32)
        .alias("flag_frozen_position"),
    ])
 
    # ── Composite penalty — additive, summed before clipping ─────────────────
    # Penalty weights chosen so that a single isolated anomaly does not fully
    # zero out a record (model still needs to learn from context).
    penalty_expr = (
        pl.when(pl.col("flag_speed_anomaly")   == 1).then(-0.30).otherwise(0.0) +
        pl.when(pl.col("flag_position_jump")   == 1).then(-0.50).otherwise(0.0) +
        pl.when(pl.col("flag_frozen_position") == 1).then(-0.20).otherwise(0.0) +
        pl.when(pl.col("flag_null_course")     == 1).then(-0.05).otherwise(0.0) +
        # Long AIS gaps (log_time_diff > 7.5 ≈ ~1800 s = 30 min) reduce trust
        pl.when(pl.col("log_time_diff")       > 7.5).then(-0.10).otherwise(0.0) +
        pl.when(pl.col("flag_land_ingress")    == 1).then(-0.80).otherwise(0.0) # Down-weights land ingress artifacts
    )
 
    return df_validated.with_columns(
        (pl.lit(1.0) + penalty_expr).clip(0.0, 1.0).alias("reliability_score")
    )
 