# navisight/pipeline/feature_registry.py
"""
Feature schema registry for Navisight AI.
 
This module is the single source of truth for:
- Feature names, groups, and ordering (ALL downstream code reads from REGISTRY)
- AIS vessel type taxonomy (superclass + behaviour vocabularies)
- Numpy dtypes for cache and embedding storage
- Schema hashes used to validate checkpoint compatibility
 
IMPORTANT: Any change to ALL_FEATURES ordering or content increments the schema
version and invalidates existing checkpoints and parquet files.
"""
import json
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Schema version — bump whenever ALL_FEATURES order/content changes
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_SCHEMA_VERSION = "v3.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Feature groups
# Each group maps to one projection branch in MultimodalProjector.
# Order within each group is fixed — do not reorder without bumping version.
# ──────────────────────────────────────────────────────────────────────────────

PHYSICS_FEATURES: List[str] = [
    # Raw kinematics
    "speed_raw",             # SOG in knots (raw, unscaled)
    "actual_speed_knots",    # SOG after validity filtering
    "course_change",         # delta COG between consecutive points (degrees)
    "acceleration",          # delta SOG / delta_t (knots/s)
    "turn_rate",             # course_change / delta_t (deg/s)
    "jerk",                  # delta acceleration / delta_t
    # Spatial displacement
    "delta_x_nm",            # eastward displacement (nautical miles)
    "delta_y_nm",            # northward displacement (nautical miles)
    # Circular-encoded course
    "course_sin",            # sin(COG radians) — handles 0/360 wrap
    "course_cos",            # cos(COG radians)
    # Temporal gap
    "log_time_diff",         # log(seconds since last AIS message)
    # Port proximity
    "distance_to_port_nm",   # great-circle distance to nearest port (nm)
    "bearing_to_port",       # bearing to nearest port (degrees)
    "turn_rate_speed_ratio", #  interactive signature capturing uncoordinated turns
]  # dim = 14

WEATHER_FEATURES: List[str] = [
    "wind_speed",          # WSPD — scalar wind speed (m/s)
    "wind_direction",      # WDIRMET — met convention (degrees)
    "temperature",         # TMP — sea-level temperature (Kelvin) ← was silently dropped before
    "visibility",          # VIS (metres)
    "gust_factor",         # GUST (m/s)
    "pressure",            # PRMSL (Pa)
    "humidity",            # RH (%)
    "headwind_component",  # computed from UGRD/VGRD · ship heading vector
    "crosswind_component", # computed from UGRD/VGRD ⊥ ship heading vector
]  # dim = 9

CONTEXT_FEATURES: List[str] = [
    # Circular-encoded time-of-day (hour)
    "hour_sin",
    "hour_cos",
    # Circular-encoded day-of-week
    "day_sin",
    "day_cos",
    # Voyage phase one-hot (mutually exclusive)
    "voyage_phase_anchoring",   # speed < class velocity floor
    "voyage_phase_cruising",    # underway, stable heading
    "voyage_phase_departure",   # leaving port, accelerating
    "voyage_phase_maneuvering", # low speed, high turn rate
    # Distance based features
    "distance_to_coast_nm",     # Vectorized proximity to nearest topographic landmass
    "distance_to_terminal_nm",  # Proximity to high-density Piraeus cargo/container berths
    "harbor_basin_proximity",   # Continuous inverse-distance scaling field for harbor boundaries
    "distance_to_land_raster",  # New high-resolution O(1) static matrix distance layer
]  # dim = 12

QUALITY_FEATURES: List[str] = [
    "flag_speed_anomaly",    # SOG > 45 kn (physically impossible for surface vessel)
    "flag_frozen_position",  # position unchanged but SOG > 2 kn (GPS/AIS fault)
    "flag_position_jump",    # implied speed > 90 kn or accel > 15 kn/s (teleport)
    "flag_null_course",      # COG was NaN in source data — imputed to 0.0
    "reliability_score",     # composite quality score [0.0, 1.0]
    "flag_land_ingress",     # quality guard flag tripping on invalid land coordinate records
]  # dim = 6

# Canonical concatenation order — never change without bumping FEATURE_SCHEMA_VERSION
ALL_FEATURES: List[str] = PHYSICS_FEATURES + WEATHER_FEATURES + CONTEXT_FEATURES + QUALITY_FEATURES
# Total dim = 14 + 9 + 12 + 6 = 41
 
# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary tables
# ──────────────────────────────────────────────────────────────────────────────

SUPERCLASS_VOCAB: Dict[str, int] = {
    "Unknown":       0,   # code 0 or unresolved
    "Reserved":      1,   # codes 1-19
    "WIG":           2,   # codes 20-29 (Wing in Ground)
    "Fishing":       3,   # code 30
    "TugTowing":     4,   # codes 31-32
    "SpecialOps":    5,   # codes 33-34 (dredging, diving)
    "Military":      6,   # code 35
    "PleasureSailing": 7, # codes 36-37
    "HighSpeed":     8,   # codes 40-49
    "ServiceVessel": 9,   # codes 50-59
    "Passenger":    10,   # codes 60-69
    "Cargo":        11,   # codes 70-79
    "Tanker":       12,   # codes 80-89
    "Other":        13,   # codes 90-99
}

BEHAVIOR_VOCAB: Dict[str, int] = {
    "Generic":          0,
    "Hazardous":        1,  # hazardous category variants (A/B/C/D)
    "Trawling":         2,
    "StandardTug":      3,
    "LargeTug":         4,  # code 32 — tow exceeds 200m
    "Dredging":         5,
    "Diving":           6,
    "Tactical":         7,
    "Sailing":          8,  # code 36
    "Leisure":          9,  # code 37
    "HighSpeedTransit": 10,
    "SAR":              11, # code 51
    "PilotVessel":      12, # code 50
    "PortService":      13, # codes 53-59
    "TransitFerry":     14, # codes 60-69 passenger
    "DeepSea":          15, # codes 70-79 cargo
    "LiquidBulk":       16, # codes 80-89 tanker
    "Standard":         17,
}

# ──────────────────────────────────────────────────────────────────────────────
# Numpy structured dtypes
# ──────────────────────────────────────────────────────────────────────────────
 
VESSEL_STATE_DTYPE = np.dtype([
    ("vessel_id_int",        "<i8"),
    ("last_timestamp",       "<i8"),
    ("last_speed",           "<f4"),
    ("last_lon",             "<f4"),
    ("last_lat",             "<f4"),
    ("last_is_static",       "<i4"),
    ("active_static_duration","<i8"),
    ("trip_id_offset",       "<i4"),
    ("last_seen_epoch",      "<i8"),
    ("generation_counter",   "<i8"),
])
 
EMBEDDING_SHARD_DTYPE = np.dtype([
    ("embedding_id",   np.int64),
    ("vessel_id_int",  np.int64),
    ("timestamp_sec",  np.int64),
    ("trip_id",        np.int32),
    ("superclass_id",  np.int16),
    ("behavior_id",    np.int16),
    ("reliability",    np.float32),
    ("embedding",      np.float16, (128,)),
])

def resolve_hierarchical_taxonomy(shiptype_raw) -> tuple:
    """
    Maps a raw AIS shiptype integer to (superclass_id, behavior_id).
 
    Handles:
    - None / NaN float inputs safely
    - All 100 AIS type codes (0-99) per ITU-R M.1371
    - Hazardous subcategory variants (A/B/C/D) within each class
    - Reserved codes mapped to nearest meaningful parent class
    """
    # Guard against None and NaN (shiptype is float64 in the static CSV)
    if shiptype_raw is None: 
        return SUPERCLASS_VOCAB["Unknown"], BEHAVIOR_VOCAB["Generic"]
    try: 
        f = float(shiptype_raw)
    except (TypeError, ValueError): 
        return SUPERCLASS_VOCAB["Unknown"], BEHAVIOR_VOCAB["Generic"]
    if f != f: return SUPERCLASS_VOCAB["Unknown"], BEHAVIOR_VOCAB["Generic"]
    code = int(f)
    if code == 0: return SUPERCLASS_VOCAB["Unknown"], BEHAVIOR_VOCAB["Generic"]
    if 1 <= code <= 19: return SUPERCLASS_VOCAB["Reserved"], BEHAVIOR_VOCAB["Generic"]
    if code == 30: return SUPERCLASS_VOCAB["Fishing"], BEHAVIOR_VOCAB["Trawling"]
    if code in (31, 52): return SUPERCLASS_VOCAB["TugTowing"], BEHAVIOR_VOCAB["StandardTug"]
    if code == 32: return SUPERCLASS_VOCAB["TugTowing"], BEHAVIOR_VOCAB["LargeTug"]
    if code == 33: return SUPERCLASS_VOCAB["SpecialOps"], BEHAVIOR_VOCAB["Dredging"]
    if code == 34: return SUPERCLASS_VOCAB["SpecialOps"], BEHAVIOR_VOCAB["Diving"]
    if code == 35: return SUPERCLASS_VOCAB["Military"], BEHAVIOR_VOCAB["Tactical"]
    if code == 36: return SUPERCLASS_VOCAB["PleasureSailing"], BEHAVIOR_VOCAB["Sailing"]
    if code == 37: return SUPERCLASS_VOCAB["PleasureSailing"], BEHAVIOR_VOCAB["Leisure"]
    if code in (40, 45, 46, 47, 48, 49): return SUPERCLASS_VOCAB["HighSpeed"], BEHAVIOR_VOCAB["HighSpeedTransit"]
    if code == 50: return SUPERCLASS_VOCAB["ServiceVessel"], BEHAVIOR_VOCAB["PilotVessel"]
    if code == 51: return SUPERCLASS_VOCAB["ServiceVessel"], BEHAVIOR_VOCAB["SAR"]
    if 53 <= code <= 59: return SUPERCLASS_VOCAB["ServiceVessel"], BEHAVIOR_VOCAB["PortService"]
    if code in (60, 65, 66, 67, 68, 69): return SUPERCLASS_VOCAB["Passenger"], BEHAVIOR_VOCAB["TransitFerry"]
    if code in (70, 75, 76, 77, 78, 79): return SUPERCLASS_VOCAB["Cargo"], BEHAVIOR_VOCAB["DeepSea"]
    if code in (80, 85, 86, 87, 88, 89): return SUPERCLASS_VOCAB["Tanker"], BEHAVIOR_VOCAB["LiquidBulk"]
    if code in (21, 22, 23, 24, 41, 42, 43, 44, 61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 84, 91, 92, 93, 94):
        for k, v in SUPERCLASS_VOCAB.items():
            if code // 10 == v: return v, BEHAVIOR_VOCAB["Hazardous"]
    # Fallback for any code outside 0-99
    return SUPERCLASS_VOCAB["Unknown"], BEHAVIOR_VOCAB["Generic"]


# ──────────────────────────────────────────────────────────────────────────────
# Feature schema dataclass — REGISTRY is the single import point for all modules
# ──────────────────────────────────────────────────────────────────────────────
 
@dataclass(slots=True)
class FeatureSchema:
    version:          str
    physics:          List[str]
    weather:          List[str]
    context:          List[str]
    quality:          List[str]
    continuous_scaled: List[str]   # columns that get z-scored (physics + weather)
    maskable_for_loss: List[str]   # columns included in MAE reconstruction loss
    all_features:     List[str]
    feature_index:    Dict[str, int]
    # Group slice boundaries into all_features (for projector slicing)
    physics_slice:    tuple         # (start, end)
    weather_slice:    tuple
    context_slice:    tuple
    quality_slice:    tuple
 
 
def _build_registry() -> FeatureSchema:
    p_end = len(PHYSICS_FEATURES)
    w_end = p_end + len(WEATHER_FEATURES)
    c_end = w_end + len(CONTEXT_FEATURES)
    q_end = c_end + len(QUALITY_FEATURES)
 
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        physics=PHYSICS_FEATURES,
        weather=WEATHER_FEATURES,
        context=CONTEXT_FEATURES,
        quality=QUALITY_FEATURES,
        # Scaling is mandated for geographic features to properly scale their dynamic range alongside raw physics
        continuous_scaled=PHYSICS_FEATURES + WEATHER_FEATURES+ [
            "distance_to_coast_nm", 
            "distance_to_terminal_nm", 
            "harbor_basin_proximity", 
            "distance_to_land_raster", 
            "turn_rate_speed_ratio",
        ],
        maskable_for_loss=[
            # Physics features we reconstruct (exclude derived quality flags)
            "speed_raw", 
            "course_change", 
            "acceleration", 
            "turn_rate", 
            "jerk",
            "delta_x_nm", 
            "delta_y_nm",
            "turn_rate_speed_ratio",
            # Weather features (model should learn weather-behaviour relationship)
            "wind_speed", 
            "visibility", 
            "headwind_component",
        ],
        all_features=ALL_FEATURES,
        feature_index={feat: idx for idx, feat in enumerate(ALL_FEATURES)},
        physics_slice=(0,     p_end),
        weather_slice=(p_end, w_end),
        context_slice=(w_end, c_end),
        quality_slice=(c_end, q_end),
    )
 
 
REGISTRY: FeatureSchema = _build_registry()
 
# ──────────────────────────────────────────────────────────────────────────────
# Schema hashes — used by checkpoints to detect feature drift
# ──────────────────────────────────────────────────────────────────────────────
 
# Stable hash of the feature list — invalidates checkpoints when features change
FEATURE_SCHEMA_HASH: str = hashlib.md5(
    ",".join(ALL_FEATURES).encode("utf-8")
).hexdigest()
 
# Alias used by run_pipeline.py bootstrap logging
REGISTRY_SCHEMA_HASH: str = FEATURE_SCHEMA_HASH
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Manifest helpers — used by train_model.py and train_mae.py
# ──────────────────────────────────────────────────────────────────────────────
 
def compile_complete_system_manifest(
    model_config: dict,
    operational_policies: dict,
    embedding_space_uuid: str,
) -> str:
    """Returns a deterministic SHA-256 hash of the full system contract."""
    manifest = {
        "schema_version":    FEATURE_SCHEMA_VERSION,
        "registry_hash":     FEATURE_SCHEMA_HASH,
        "superclass_vocab":  SUPERCLASS_VOCAB,
        "behavior_vocab":    BEHAVIOR_VOCAB,
        "model_architecture": model_config,
        "operational_policies": operational_policies,
        "embedding_space_uuid": embedding_space_uuid,
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
 
 
def generate_production_manifest_hash(model_config: dict, policies: dict) -> str:
    """Convenience alias — imported by train_model.py."""
    return compile_complete_system_manifest(model_config, policies, "EMB_SPACE_v1")
 