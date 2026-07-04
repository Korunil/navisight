# view_validation_dashboard.py
import os
import sys
import time
import logging
import json
import streamlit as st
import polars as pl
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY, SUPERCLASS_VOCAB
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

from dashboard.services.detector_service import DetectorService
from dashboard.utils.schema_resolution import resolve_superclass_id
from dashboard.visualizations.plots import render_tactical_map, render_risk_gauge

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

st.set_page_config(page_title="NaviSight Tactical Intelligence Hub", page_icon="🛰️", layout="wide")

# ==========================================================================
# HARDENED INLINE CSS: DYNAMIC THEME VARIATION HOOKS & COMPACT LAYOUTS
# ==========================================================================
def inject_tactical_css():
    st.markdown("""
    <style>
        /* FIX 1: Make header transparent so the 3-dot options menu floats over content */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            background: transparent !important;
            height: 2.5rem !important;
        }
        
        /* FIX 2: Pull workspace to the top edge while leaving room for floating options icon */
        .block-container {
            padding-top: 2.0rem !important;
            padding-bottom: 1.0rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            max-width: 98% !important;
        }
        
        /* Hide the Deploy Button wrapper specifically, keeping only the 3-dot options menu visible */
        div[data-testid="stActionButton"] {
            display: none !important;
        }
        
        /* FIX 3: Glassmorphism panels linked to dynamic Streamlit theme colors */
        .intel-card {
            background: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.18) !important;
            border-radius: 12px !important;
            padding: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
            backdrop-filter: blur(12px) !important;
            margin-bottom: 16px !important;
            color: var(--text-color) !important;
        }
        
        /* Theme adaptive labels and metric typography */
        .card-title {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: var(--text-color) !important;
            opacity: 0.65 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.2px !important;
            margin-bottom: 6px !important;
        }
        .card-value { 
            font-size: 1.8rem !important; 
            font-weight: 800 !important;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: -0.5px !important;
        }
        
        .card-title-accent {
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            color: #3b82f6 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            margin-bottom: 14px !important;
            border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
            padding-bottom: 6px !important;
        }
        
        .profile-row {
            display: flex !important;
            justify-content: space-between !important;
            padding: 7px 0 !important;
            border-bottom: 1px solid rgba(128, 128, 128, 0.12) !important;
            color: var(--text-color) !important;
        }
        .profile-row:last-child { border-bottom: none !important; }
        .profile-label { font-size: 0.82rem !important; font-weight: 500 !important; opacity: 0.75 !important; }
        .profile-value { font-size: 0.82rem !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }
        
        .production-table { width: 100% !important; border-collapse: collapse !important; margin-top: 4px !important; }
        .production-table th {
            text-align: left !important; padding: 10px !important; font-size: 0.78rem !important; color: var(--text-color) !important; opacity: 0.65 !important;
            text-transform: uppercase !important; border-bottom: 1px solid rgba(128, 128, 128, 0.25) !important;
        }
        .production-table td { padding: 10px !important; font-size: 0.85rem !important; border-bottom: 1px solid rgba(128, 128, 128, 0.12) !important; color: var(--text-color) !important; }
        
        /* Threat Active Pulsing Highlights */
        @keyframes status-pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        @keyframes status-pulse-warning {
            0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        @keyframes status-pulse-nominal {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes status-pulse-joint {
            0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }
        
        .pulse-badge-critical { animation: status-pulse 2s infinite; }
        .pulse-badge-warning { animation: status-pulse-warning 2s infinite; }
        .pulse-badge-nominal { animation: status-pulse-nominal 2s infinite; }
        .pulse-badge-joint { animation: status-pulse-joint 2s infinite; }
    </style>
    """, unsafe_allow_html=True)

inject_tactical_css()

@st.cache_resource
def get_detector_service():
    return DetectorService()

# ==========================================================================
# DYNAMIC LEDGER LOADER: DECOUPLING CONFIGURATIONS FROM HARDCODED VALUES
# ==========================================================================
@st.cache_data
def load_calibrated_constants():
    """Natively resolves thresholds and normalization parameters directly from evaluation_results.json."""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "configs", "evaluation_results.json"),
        os.path.join(PROJECT_ROOT, "evaluation_results.json"),
        "evaluation_results.json"
    ]
    
    json_path = None
    for p in possible_paths:
        if os.path.exists(p):
            json_path = p
            break
            
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            meta = data.get("meta_config_constants", {})
            baselines = data.get("calibration_baselines", {})
            
            # Extract thresholds from configuration file
            threshold_recon = baselines.get("sequence_reconstruction", {}).get("threshold", 0.9575)
            threshold_ann = baselines.get("manifold_ann", {}).get("threshold", 1.0490)
            joint_factor = meta.get("joint_warning_factor_multiplier", 0.95)
            
            # Extract dynamic baseline calibration parameters
            ann_p50 = meta.get("ann_p50", 0.1467)
            ann_p99 = meta.get("ann_p99", 0.4984)
            
            return threshold_recon, threshold_ann, joint_factor, ann_p50, ann_p99, json_path
        except Exception as e:
            logger.error(f"Failed parsing evaluation results ledger: {e}")
            
    return 0.9575, 1.0490, 0.95, 0.1467, 0.4984, "Fallback Context"

THRESHOLD_RECON, THRESHOLD_ANN, JOINT_WARNING_FACTOR, ANN_P50, ANN_P99, active_ledger_source = load_calibrated_constants()

ANOMALY_MODES = ["dead_reckoning", "coastal_creep", "loitering"]

@st.cache_data
def load_raw_telemetry_window(file_path: str, start_row: int, max_window_size: int):
    df = pl.read_parquet(file_path)
    actual_end = min(start_row + max_window_size, df.height)
    return df.slice(start_row, actual_end - start_row)

detector_service = get_detector_service()
injector = ManifoldAwareAnomalyPerturbationEngine()

if "animation_frame" not in st.session_state: st.session_state.animation_frame = 0
if "is_playing" not in st.session_state: st.session_state.is_playing = False

# --- MISSION CONTROL PANEL LAYOUT ---
with st.sidebar:
    st.markdown("## MISSION CONTROL PANEL")
    st.caption(f"Active Calibration Profile: **{os.path.basename(active_ledger_source)}**")
    
    col_play, col_pause, col_reset = st.columns(3)
    if col_play.button("▶️ Play"): st.session_state.is_playing = True
    if col_pause.button("⏸️ Pause"): st.session_state.is_playing = False
    if col_reset.button("🔄 Reset"):
        st.session_state.animation_frame = 0
        st.session_state.is_playing = False

    playback_speed = st.slider("Playback Delta Delay (s)", 0.02, 0.5, 0.1)
    show_counterfactual = st.checkbox("Show Adversarial Scenario Paths", value=False)
    selected_hypothesis = st.selectbox("Adversarial Behavioral Hypothesis", options=ANOMALY_MODES, index=0)
    
    processed_lakehouse_root = "data/processed/year=2019"
    BENCHMARK_MONTHS = [10, 11, 12]
    available_files = []
    
    if os.path.exists(processed_lakehouse_root):
        for m_int in BENCHMARK_MONTHS:
            month_partition_path = os.path.join(processed_lakehouse_root, f"month={m_int:02d}")
            if os.path.exists(month_partition_path):
                for root, _, files in os.walk(month_partition_path):
                    for f in files:
                        if f.endswith(".parquet"):
                            available_files.append(os.path.join(root, f))
                            
    if not available_files:
        st.error("❌ Fatal: No valid timeline benchmark split files (Months 10-12) discovered in local lakehouse paths.")
        st.stop()
    
    selected_file_path = st.selectbox("Target Vessel Stream File", sorted(available_files), index=0)
    
    meta_df = pl.scan_parquet(selected_file_path).select(pl.len()).collect()
    total_file_rows = meta_df[0, 0]
    
    max_window_size = st.slider("Window Length Context", 50, 200, 119)
    if st.session_state.is_playing:
        if st.session_state.animation_frame + max_window_size < total_file_rows:
            st.session_state.animation_frame += 2
        else:
            st.session_state.is_playing = False

    start_row_idx = int(st.session_state.animation_frame)
    # map_style_options = [
    #     "carto-darkmatter", "satellite", "satellite-streets", "open-street-map",
    #     "basic", "carto-darkmatter-nolabels", "carto-positron", "carto-positron-nolabels",
    #     "carto-voyager", "carto-voyager-nolabels", "dark", "light", "outdoors", "streets"
    # ]

    # ==========================================================================
    # FIX: INTEGRATED NATIVE THEME INTERCEPTION LAYER FOR THE MAP ELEMENT
    # ==========================================================================
    try:
        active_theme_type = st.context.theme.type
    except Exception:
        active_theme_type = "dark"

    # Initialize tracking nodes inside session state on cold boot
    if "last_seen_theme" not in st.session_state:
        st.session_state.last_seen_theme = active_theme_type
        st.session_state.forced_map_style = "carto-darkmatter" if active_theme_type == "dark" else "carto-voyager"

    # CRITICAL: Intercept theme flips and dynamically force style variables to update[cite: 2]
    if st.session_state.last_seen_theme != active_theme_type:
        st.session_state.last_seen_theme = active_theme_type
        st.session_state.forced_map_style = "carto-darkmatter" if active_theme_type == "dark" else "carto-voyager"

    # Sort array layouts to place high-contrast matching targets at the top[cite: 2]
    if active_theme_type == "dark":
        map_style_options = [
            "carto-darkmatter", "dark", "carto-darkmatter-nolabels", "satellite", "satellite-streets",
            "carto-voyager", "open-street-map", "carto-positron", "basic", "streets", "outdoors"
        ]
    else:
        map_style_options = [
            "carto-voyager", "light", "open-street-map", "carto-positron", "basic", "streets", "outdoors",
            "carto-darkmatter", "dark", "satellite", "satellite-streets"
        ]

    # Dynamically find the exact index location matching our active state tracking variable
    if st.session_state.forced_map_style in map_style_options:
        current_style_index = map_style_options.index(st.session_state.forced_map_style)
    else:
        current_style_index = 0

    selected_map_style = st.selectbox(
        "Base Map Style Layer", 
        options=map_style_options, 
        index=current_style_index
    )
    
    # Save any manual selectbox overrides back to the tracking variable to preserve user choice
    st.session_state.forced_map_style = selected_map_style
    
    # selected_map_style = st.selectbox("Base Map Style Layer", options=map_style_options, index=1)

# --- WORKSPACE EVALUATION PIPELINE ---
df_window = load_raw_telemetry_window(selected_file_path, start_row_idx, max_window_size)
vessel_id = int(df_window["vessel_id_int"][0])
sc_id = resolve_superclass_id(df_window)

context_meta = {
    "timestamp_sec": int(df_window["timestamp_sec"][0]), 
    "trip_id": str(df_window["trip_id"][0]), 
    "superclass_id": sc_id
}

missing = [c for c in REGISTRY.all_features if c not in df_window.columns]
if missing:
    st.error(f"Missing required features: {missing}")
    st.stop()

# 1. Execute full-sequence production pass over observed track
res_normal = detector_service.evaluate(df_window, vessel_id, context_meta, mode="production")

raw_recon = res_normal.get("reconstruction_mse", 0.0)
raw_ann_dist = res_normal.get("ood_score", 0.0)
calibrated_ann_risk = float(np.maximum(0.0, (raw_ann_dist - ANN_P50) / (ANN_P99 - ANN_P50 + 1e-8)))

# 2. Evaluate Counterfactual Adversarial Scenario Paths
res_anomalous = None
df_perturbed = None
calibrated_anomaly_ann_risk = 0.0
anom_recon = 0.0

if show_counterfactual:
    df_perturbed = injector.inject_counterfactual_scenario(df_vessel_voyage=df_window, mode=selected_hypothesis, severity=0.85)
    res_anomalous = detector_service.evaluate(df_perturbed, vessel_id, context_meta, mode="production")
    
    anom_recon = res_anomalous.get("reconstruction_mse", 0.0)
    anom_ann_dist = res_anomalous.get("ood_score", 0.0)
    calibrated_anomaly_ann_risk = float(np.maximum(0.0, (anom_ann_dist - ANN_P50) / (ANN_P99 - ANN_P50 + 1e-8)))

# Determine active values depending on counterfactual checkbox toggle state
active_recon = anom_recon if show_counterfactual and res_anomalous is not None else raw_recon
active_ann = calibrated_anomaly_ann_risk if show_counterfactual and res_anomalous is not None else calibrated_ann_risk

# Determine Discrete Channel Alert Status
alert_recon = active_recon >= THRESHOLD_RECON
alert_ann = active_ann >= THRESHOLD_ANN
joint_warning = (active_recon >= JOINT_WARNING_FACTOR * THRESHOLD_RECON) and (active_ann >= JOINT_WARNING_FACTOR * THRESHOLD_ANN)

# Dynamic Description Generator using clean HTML styling tags to avoid raw Markdown text rendering bugs
if not alert_recon and not alert_ann:
    max_ratio = max(active_recon / (THRESHOLD_RECON + 1e-8), active_ann / (THRESHOLD_ANN + 1e-8))
    combined_decision_score = float(np.clip(max_ratio * 0.49, 0.0, 0.49))
    status = "NOMINAL OPERATIONS"
    status_color = "#10B981"
    pulse_class = "pulse-badge-nominal"
    status_desc = "<strong>System State Clear:</strong> Vessel trajectory parameters conform securely to normal bounds. Local kinematics and macro routing patterns align perfectly with historical peer tracks."
elif alert_recon and alert_ann:
    max_ratio = max(active_recon / (THRESHOLD_RECON + 1e-8), active_ann / (THRESHOLD_ANN + 1e-8))
    combined_decision_score = float(np.clip(0.75 + (max_ratio - 1.0) * 0.25, 0.75, 1.0))
    status = "CRITICAL DETECTION [AND]"
    status_color = "#EF4444"
    pulse_class = "pulse-badge-critical"
    status_desc = f"<strong>🚨 Dual-Channel Breach Triggered:</strong> Correlated high-severity anomaly detected. Both short-term velocity kinematics (MSE: {active_recon:.4f} &gt;= {THRESHOLD_RECON}) and geographic route selection (ANN Risk: {active_ann:.4f} &gt;= {THRESHOLD_ANN}) have simultaneously breached calibrated safety bounds."
else:
    if alert_recon:
        breached_ratio = active_recon / THRESHOLD_RECON
        status_desc = f"<strong>⚠️ Kinematic Maneuver Alert (Channel 1 Trigger):</strong> Local trajectory dynamics exhibit severe deviation. The sequence reconstruction error (MSE: {active_recon:.4f} &gt;= {THRESHOLD_RECON}) indicates abnormal speed patterns or uncharacteristic course alterations, though geographic corridor limits remain safe."
    else:
        breached_ratio = active_ann / THRESHOLD_ANN
        status_desc = f"<strong>⚠️ Topological Spatial Alert (Channel 2 Trigger):</strong> Irregular geographic routing transit profile detected. While current speed and pacing values are normal, the position vector (ANN Risk: {active_ann:.4f} &gt;= {THRESHOLD_ANN}) has drifted far outside standard route boundaries."
        
    combined_decision_score = float(np.clip(0.50 + (breached_ratio - 1.0) * 0.24, 0.50, 0.74))
    status = "WARNING RISK EVENT [OR]"
    status_color = "#F59E0B"
    pulse_class = "pulse-badge-warning"

if joint_warning and not (alert_recon or alert_ann):
    status = "JOINT CONTOUR DETECTED"
    status_color = "#3B82F6"
    pulse_class = "pulse-badge-joint"
    combined_decision_score = float(np.clip(JOINT_WARNING_FACTOR * 0.55, 0.50, 0.74))
    status_desc = f"<strong>⚡ Co-occurring Channel Elevation Alert:</strong> Neither tracking indicator has broken past its independent 99.5th percentile threshold boundary, but both channels are simultaneously elevated above <strong>{JOINT_WARNING_FACTOR:.0%}</strong> of their maximum normal capacity. This flags a high-probability tactical anomaly."

# Extract and normalize continuous spatial timelines for mapping layers
actual_model_timeline = []
orig_timeline = res_normal.get("risk_timeline", [])
if orig_timeline:
    for chunk in orig_timeline:
        c_recon = chunk.get("reconstruction_component", 0.0)
        c_ann = chunk.get("drift_component", 0.0)
        c_ann_scaled = float(np.maximum(0.0, (c_ann - ANN_P50) / (ANN_P99 - ANN_P50 + 1e-8)))
        
        t_alert_recon = c_recon >= THRESHOLD_RECON
        t_alert_ann = c_ann_scaled >= THRESHOLD_ANN
        if not t_alert_recon and not t_alert_ann:
            t_score = max(c_recon / THRESHOLD_RECON, c_ann_scaled / THRESHOLD_ANN) * 0.49
        elif t_alert_recon and t_alert_ann:
            t_score = 0.75 + (max(c_recon / THRESHOLD_RECON, c_ann_scaled / THRESHOLD_ANN) - 1.0) * 0.25
        else:
            t_br = (c_recon / THRESHOLD_RECON) if t_alert_recon else (c_ann_scaled / THRESHOLD_ANN)
            t_score = 0.50 + (t_br - 1.0) * 0.24
            
        actual_model_timeline.append(float(np.clip(t_score, 0.0, 1.0)))

# Calculate baseline vs anomaly score variation
if show_counterfactual and res_anomalous is not None:
    base_ratio = max(raw_recon / (THRESHOLD_RECON + 1e-8), calibrated_ann_risk / (THRESHOLD_ANN + 1e-8))
    base_norm = base_ratio * 0.49 if (raw_recon < THRESHOLD_RECON and calibrated_ann_risk < THRESHOLD_ANN) else (0.50 + (base_ratio - 1.0) * 0.24)
    scenario_delta = f"{combined_decision_score - base_norm:+.3f}"
else:
    scenario_delta = "Inactive"

inv_vocab = {v: k for k, v in SUPERCLASS_VOCAB.items()}
class_label = inv_vocab.get(sc_id, "Unknown").upper()

# --- UI WORKSPACE ASSEMBLY WITH COMPACT LAYOUTS & FLOATING OPTIONS MENU ---
st.markdown(f"""
<div style="padding:16px 20px; border-radius:12px; background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.15); box-shadow: 0 4px 20px rgba(0,0,0,0.15); display:flex; justify-content:space-between; align-items:center; width:100%; color: var(--text-color);">
    <div>
        <h1 style="margin:0; font-size:2.05rem; font-weight:850; color: var(--text-color); letter-spacing:-0.15px;">NaviSight Tactical Intelligence Hub</h1>
        <div style="display:flex; gap:24px; font-family:'JetBrains Mono', monospace; font-size:0.78rem; margin-top:4px; opacity:0.8;">
            <span>VESSEL HULL ID: <span style="color:#3B82F6; font-weight:700;">{vessel_id}</span></span>
            <span>CLASS IDENTIFIER: <span style="color:#3B82F6; font-weight:700;">{class_label}</span></span>
        </div>
    </div>
    <div style="margin-right: 45px;">
        <span class="{pulse_class}" style="background:{status_color}; padding:10px 24px; border-radius:6px; font-weight:820; font-family:'JetBrains Mono', monospace; color:white; font-size:1.25rem; letter-spacing:0.55px; box-shadow: 0 0 12px {status_color}30; display: inline-block;">
            {status}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# State-tinted, context-aware notification banner built with pure HTML tags to ensure correct theme-responsive text rendering
st.markdown(f"""
<div style="margin-top: 12px; margin-bottom: 18px; padding: 14px 18px; border-left: 5px solid {status_color}; background: var(--secondary-background-color); border-top: 1px solid rgba(128,128,128,0.1); border-right: 1px solid rgba(128,128,128,0.1); border-bottom: 1px solid rgba(128,128,128,0.1); font-size: 0.92rem; border-radius: 0 6px 6px 0; color: var(--text-color); line-height: 1.55; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
    {status_desc}
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: 
    st.markdown(f'<div class="intel-card"><div class="card-title">Combined Decision Score</div><div class="card-value" style="color:{status_color};">{combined_decision_score:.4f}</div></div>', unsafe_allow_html=True)
with m2: 
    st.markdown(f'<div class="intel-card"><div class="card-title">Channel 1: Reconstruction MSE</div><div class="card-value" style="color:#F59E0B;">{active_recon:.4f}</div></div>', unsafe_allow_html=True)
with m3: 
    st.markdown(f'<div class="intel-card"><div class="card-title">Channel 2: Calibrated ANN Risk</div><div class="card-value" style="color:#6366F1;">{active_ann:.4f}</div></div>', unsafe_allow_html=True)
with m4: 
    st.markdown(f'<div class="intel-card"><div class="card-title">Prioritization Index (Ranking)</div><div class="card-value" style="color: var(--text-color); opacity: 0.85;">{res_normal.get("fused_risk_score", 0.0):.4f}</div></div>', unsafe_allow_html=True)

map_col, intel_col = st.columns([3.8, 1.2])
with map_col:
    st.subheader("Tactical Localization View")
    map_fig = render_tactical_map(
        df_window["lat_raw"].to_numpy(), 
        df_window["lon_raw"].to_numpy(), 
        actual_model_timeline if actual_model_timeline else [combined_decision_score], 
        selected_map_style, 
        df_perturbed, 
        show_counterfactual
    )
    st.plotly_chart(map_fig, width="stretch")

with intel_col:
    st.subheader("Threat Matrix")
    st.plotly_chart(render_risk_gauge(combined_decision_score), width="stretch")
    
    st.markdown(f"""
    <div class="intel-card">
        <div class="card-title-accent">Contextual Factors Matrix</div>
        <div class="profile-row"><span class="profile-label">Coast Distance</span><span class="profile-value">{df_window['distance_to_coast_nm'].mean():.2f} nm</span></div>
        <div class="profile-row"><span class="profile-label">Terminal Distance</span><span class="profile-value">{df_window['distance_to_terminal_nm'].mean():.2f} nm</span></div>
        <div class="profile-row"><span class="profile-label">Scenario Risk Delta</span><span class="profile-value" style="color:#6366F1; font-weight:700;">{scenario_delta}</span></div>
    </div>
    """, unsafe_allow_html=True)

# --- COHORT NEIGHBORHOOD EXPLAINABILITY ---
st.markdown("### Deep Behavior Neighborhood Profiles (HNSW Verification)")

cls_embedding = res_normal.get("cls_embedding")
peers = []

if cls_embedding is not None:
    try:
        raw_neighborhood = detector_service.query_neighbors(cls_embedding, sc_id, k=30)
        
        seen_vessels = set([vessel_id])
        unique_peers = []
        
        for match in raw_neighborhood:
            peer_id = match.get("vessel_id_int")
            if peer_id not in seen_vessels:
                seen_vessels.add(peer_id)
                unique_peers.append(match)
            if len(unique_peers) >= 3:
                break
                
        peers = unique_peers
    except Exception as exc:
        logger.warning("Neighborhood analytics lookup threw an execution anomaly: %s", exc)

if cls_embedding is not None:
    if peers:
        rows_html = "".join(
            f'<tr>'
            f'<td>Vessel #{r}</td>'
            f'<td>Vessel Key ID {m["vessel_id_int"]}</td>'
            f'<td>{m["distance"]:.4f}</td>'
            f'<td><span style="color:#10B981; font-weight:600;">{1.0 - m["distance"]:.2%}</span></td>'
            f'</tr>' 
            for r, m in enumerate(peers, 1)
        )
        st.markdown(
            f'<div class="intel-card" style="padding:4px;">'
            f'<table class="production-table">'
            f'<thead><tr><th>Nearest Vessels</th><th>Instance Neighbor</th><th>Vector Distance</th><th>Confidence Alignment</th></tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>'
            f'</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="intel-card"><div style="opacity:0.6; font-size:0.85rem; text-align:center; padding:15px;">No active baseline neighborhoods clustered inside latent HNSW parameters.</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="intel-card"><div style="opacity:0.6; font-size:0.85rem; text-align:center; padding:15px; color:#EF4444;">No latents available for neighborhood analysis on current checkpoint payload.</div></div>', unsafe_allow_html=True)

if st.session_state.is_playing:
    time.sleep(playback_speed)
    st.rerun()