# scripts/view_validation_dashboard.py
import os
import sys
import streamlit as st
import polars as pl
import torch
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from navisight.pipeline.feature_registry import REGISTRY, SUPERCLASS_VOCAB
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector
from navisight.evaluation.synthetic_anomaly_injector import ManifoldAwareAnomalyPerturbationEngine

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NaviSight Tactical Intelligence Hub",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME-AGNOSTIC GLASSMORPHISM STYLING MATRIX ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        max-width: 1700px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Dynamic Glassmorphic Card Container Elements */
    .intel-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 16px;
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    .card-title-accent {
        font-size: 0.85rem;
        font-weight: 600;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 14px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 6px;
    }
    
    .profile-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .profile-row:last-child {
        border-bottom: none;
    }
    .profile-label { font-size: 0.85rem; font-weight: 500; opacity: 0.8; }
    .profile-value { font-size: 0.85rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    
    .production-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
    }
    .production-table th {
        text-align: left;
        padding: 10px;
        font-size: 0.8rem;
        color: #8892b0;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
    }
    .production-table td {
        padding: 12px 10px;
        font-size: 0.85rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# --- 3. CACHED RESOURCE INITIALIZATION ---
@st.cache_resource
def initialize_detection_engine():
    checkpoint_directory = "models/checkpoints/"
    global_stats_json_path = "configs/global_stats.json"
    index_directory = "models/state/index"
    embedding_destination_dir = "data/embeddings"
    
    best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
    if not os.path.exists(best_checkpoint):
        checkpoints = [os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")]
        if checkpoints:
            best_checkpoint = checkpoints[-1]

    if not best_checkpoint or not os.path.exists(best_checkpoint):
        st.error("❌ Critical System Error: Best Model state checkpoint ('best_model.pt') could not be resolved.")
        st.stop()

    ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
    if os.path.exists(embedding_destination_dir):
        shards = [f for f in os.listdir(embedding_destination_dir) if f.endswith(".parquet")]
        if shards:
            for shard in sorted(shards):
                ann_index.index_parquet_shard(os.path.join(embedding_destination_dir, shard))

    profile_engine = RollingBehaviorProfileEngine(alpha=0.1)
    detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, ann_index, profile_engine)
    injector = ManifoldAwareAnomalyPerturbationEngine()
    return detector, injector

# Integrated UI loading spinner context wraps engine boot routines
with st.spinner("🔮 Loading Maritime MAE Engine Weights & Spatial Registries..."):
    detector, injector = initialize_detection_engine()

# --- 4. FLUID COMMAND CONTROL PANEL ---
with st.sidebar:
    st.markdown("## MISSION CONTROL PANEL")
    st.markdown("---")
    
    page_selection = st.radio("Navigate Workspace Views", ["Live Tactical Stream", "Read Me - Operational Manual"])
    st.markdown("---")
    
    st.markdown("### 🎭 SCENARIO SIMULATION ENGINE")
    selected_hypothesis = st.selectbox(
        "Adversarial Behavioral Hypothesis", 
        options=["loitering", "dead_reckoning", "coastal_creep"],
        format_func=lambda x: {
            "loitering": "Covert Loitering Circle",
            "dead_reckoning": "AR(1) Deception Drift",
            "coastal_creep": "Constrained Topographic Creep"
        }.get(x, x)
    )
    
    processed_lakehouse_root = "data/processed/year=2019"
    available_files = []
    if os.path.exists(processed_lakehouse_root):
        for root, _, files in os.walk(processed_lakehouse_root):
            for f in files:
                if f.endswith(".parquet"):
                    available_files.append(os.path.join(root, f))

    if not available_files:
        st.error("❌ No processed telemetry shards discovered inside active Parquet lakehouse roots.")
        st.stop()

    available_files.sort()
    selected_file_path = st.selectbox("Target Vessel Stream File", available_files)

    df_raw = pl.read_parquet(selected_file_path)
    vessel_id = int(df_raw["vessel_id_int"][0])
    total_rows = df_raw.height

    max_window_size = st.slider("Window Length Context", 50, 200, 119)
    start_row = st.number_input("Start Row Index Offset", min_value=0, max_value=max(0, total_rows - 10), value=0)
    end_row = min(start_row + max_window_size, total_rows)
    df_window = df_raw.slice(start_row, end_row - start_row)

    # scripts/view_validation_dashboard.py (At the end of Section 4 inside the sidebar block)
    # st.markdown("---")
    # st.markdown("### 🔍 METADATA SCHEMA DIAGNOSTIC")
    # st.sidebar.write("Active Columns found in file:", list(df_window.columns))
    
    st.markdown("---")
    st.markdown("### MAP CANVAS CONFIGURATION")
    map_style_options = [
        "carto-darkmatter",             #0 index
        "satellite",                    #1 index
        "satellite-streets",            #2 index
        "open-street-map",              #3 index
        "basic",                        #4 index
        "carto-darkmatter-nolabels",    #5 index
        "carto-positron",               #6 index
        "carto-positron-nolabels",      #7 index
        "carto-voyager",                #8 index
        "carto-voyager-nolabels",       #9 index
        "dark",                         #10 index
        "light",                        #11 index
        "outdoors",                     #12 index
        "streets",                      #13 index
    ]
    selected_map_style = st.selectbox("Base Map Style Layer", options=map_style_options, index=9)


# --- 5. TECHNICAL DOCUMENTATION OVERLAY ---
if page_selection == "Read Me - Operational Manual":
    st.title("🛰️ NaviSight Professional Operations Manual")
    st.caption("Technical Specification Ledger & Core Algorithmic System Documentation")
    st.divider()

    st.header("1. Mathematical Core Architecture")

    st.markdown("""
    The NaviSight trajectory platform processes raw AIS telemetric lines using a unified
    **Continuous-Time Stochastic Hybrid System** mapped inside local tangent planes.

    Instead of relying on static coordinate manipulation, the system represents routing
    decisions via an active state vector:
    """)

    st.latex(r"x_t = [p_x, p_y, v, \theta, \mathbf{z}]^T")
    
    st.write(r"""
    Where 
    - $p$ represents the metric East-North-Up (ENU) positioning matrix,    
    - $v$ tracks forward physical velocity, 
    - $\theta$ isolates true manifold heading,    
    - $\mathbf{z}$ governs a 3-dimensional continuous latent regime logit matrix.

    The foundational equation of motion follows a stochastic differential setup:
    """)

    st.latex(r"dx_t = f(x_t,t)\,dt + g(x_t,t)\,dW_t")

    st.markdown("""
    Where $dW_t$ is a multivariate Brownian noise vector.
    """)
    
    st.divider()
    st.header("2. Algorithmic Behavioral Modes")
    st.markdown("""
    The simulation panel features three specialized adversarial behavioral generators:""")
    st.markdown("""
    - **Covert Loitering Circle (`loitering`)**
      - Executes continuous circular orbital paths inside metric ENU space.
      - Evaluates topographical land barriers before adding spatial variations to enforce kinematic tracking integrity.

    - **AR(1) Deception Drift (`dead_reckoning`)**
      - Evaluates a continuous random walk across active operational tracks.
      - Balances target headings against localized coast repulsion forces.

    - **Constrained Topographic Creep (`coastal_creep`)**
      - Couples target route tracking with continuous navigation potential fields.
      - Steers the vessel cleanly inside a fixed buffer zone near shore boundaries.
    """)
    st.divider()
    st.header("3. Metric Potential-Field Repulsion Stencil")
    st.markdown("""
    To avoid coastlines realistically, the engine drops greedy binary look-aheads and evaluates an isotropic finite-difference gradient directly inside local metric space. Spatial grids are evaluated using local geodetic scaling coefficients $100 m$ spatial offsets):
    """)
    
    st.latex(
        r"""
    \nabla \mathcal{D}(\mathbf{p}) =
    \left[
    \frac{\mathcal{D}(p_x+\Delta s,p_y)-\mathcal{D}(p_x-\Delta s,p_y)}
    {2\Delta s},
    \frac{\mathcal{D}(p_x,p_y+\Delta s)-\mathcal{D}(p_x,p_y-\Delta s)}
    {2\Delta s}
    \right]^T
    """
    )
    
    st.markdown("""
    This gradient is mapped through a sigmoid-style soft saturation function to derive the metric velocity repulsion vector:
    """)
    
    st.latex(
        r"""
    \mathbf{v}_{\mathrm{repulse}}
    =
    \frac{\nabla \mathcal{D}}
    {\|\nabla \mathcal{D}\|}
    \cdot
    \frac{v_{\mathrm{mps}}}
    {1+\exp\left(15(\mathcal{D}_{\mathrm{nm}}-0.10)\right)}
    """
    )
    
    st.divider()
    st.header("4. Cohort-Gated Vector Index Isolation")
    st.markdown("""
    Vessel tracking trajectories are compressed into a 128-dimensional latent space using a deep Masked Autoencoder (MAE) Transformer backbone. To verify anomalous behaviors against historical baselines, vectors are evaluated against isolated, cohort-gated **Hierarchical Navigable Small World (HNSW)** sub-graphs. 
    
    Unique indices utilize a split-key architecture. Every rolling sequence is assigned an incremental, unique `embedding_id`, which maps directly to raw coordinates inside a thread-safe SQLite metadata register (`ann_metadata_registry.db`), tracking vessel identifier keys (`vessel_id_int`) and temporal windows (`timestamp_sec`).
    """)
    
    st.divider()
    st.header("5. Color Convention Ledger")
    st.markdown("""
        * <span style="color:#3B82F6;">🔵 **Primary Blue**</span>: Identifies active normal tracking operations vectors.  
        * <span style="color:#10B981;">🟢 **Nominal Green**</span>: Represents safe conditions and trajectory tracking baseline limits.  
        * <span style="color:#F59E0B;">🟡 **Warning Amber**</span>: Identifies low-speed maneuvering patterns and port proximity approach horizons.  
        * <span style="color:#EF4444;">🔴 **Anomaly Red**</span>: Flags critical breach vectors, land ingress events, and high reconstruction anomalies. 
        """, unsafe_allow_html=True)
    st.stop()


# --- 6. AUTONOMOUS DETECTION PIPELINE & SCHEMA RESOLUTION ---

# Tier 1: Try reading the explicit integer ID column
sc_id_raw = 0
if "vessel_superclass_id" in df_window.columns and df_window["vessel_superclass_id"][0] is not None:
    sc_id_raw = int(df_window["vessel_superclass_id"][0])

# Tier 2: Fallback to cross-referencing the raw text shiptype string if ID is missing/0
if sc_id_raw == 0 and "shiptype" in df_window.columns and df_window["shiptype"][0] is not None:
    raw_shiptype_str = str(df_window["shiptype"][0]).lower().strip()
    sc_id_raw = SUPERCLASS_VOCAB.get(raw_shiptype_str, 0)

context_meta = {
    "timestamp_sec": int(df_window["timestamp_sec"][0]), 
    "trip_id": str(df_window["trip_id"][0]),
    "superclass_id": sc_id_raw
}

vessel_lats = df_window["lat_raw"].to_numpy().astype(np.float64)
vessel_lons = df_window["lon_raw"].to_numpy().astype(np.float64)

# Execute primary model estimation over visible features
feature_cols = df_window.select(REGISTRY.all_features).to_numpy().astype(np.float32)
res_normal = detector.evaluate_live_sequence_anomaly_score(torch.from_numpy(feature_cols), vessel_id, context_meta)

# Inject topographically bound spatial variations
df_perturbed = injector.inject_counterfactual_scenario(df_vessel_voyage=df_window, mode=selected_hypothesis, severity=0.85)
perturbed_cols = df_perturbed.select(REGISTRY.all_features).to_numpy().astype(np.float32)
res_anomalous = detector.evaluate_live_sequence_anomaly_score(torch.from_numpy(perturbed_cols), vessel_id, context_meta)

score_norm = res_normal["fused_risk_score"]
score_anom = res_anomalous["fused_risk_score"]

inv_vocab = {v: k for k, v in SUPERCLASS_VOCAB.items()}
# sc_id_raw = int(df_window["superclass_id"][0]) if "superclass_id" in df_window.columns else 0
superclass_label = inv_vocab.get(sc_id_raw, "Unknown")

status = "CRITICAL" if score_norm > 0.75 else "NOMINAL"
status_color = "#EF4444" if score_norm > 0.75 else "#10B981"

last_dt = df_window["datetime"].to_list()[-1]
try:
    last_timestamp_str = last_dt.strftime("%H:%M:%S UTC")
except AttributeError:
    last_timestamp_str = str(last_dt)


# --- 7. INFORMATION-DENSE HERO OPERATIONS BANNER ---
st.markdown(f"""
<div style="padding:24px; border-radius:20px; background: rgba(128, 128, 128, 0.08); border:1px solid rgba(128, 128, 128, 0.25); display:flex; justify-content:space-between; align-items:center; width:100%;">
    <div>
        <h1 style="margin-bottom:0; font-size:2.1rem; font-weight:700; letter-spacing:-0.5px;">NaviSight Intelligence Platform</h1>
        <p style="margin-top:4px; margin-bottom:12px; opacity:0.7; font-size:15px; font-weight:500;">Real-Time Maritime Behavioral Analytics Core</p>
        <div style="display:flex; gap:24px; font-family:'JetBrains Mono', monospace; font-size:0.85rem; opacity:0.85; font-weight:600;">
            <span>VESSEL ID: <span style="color:#3B82F6;">{vessel_id}</span></span>
            <span>WINDOW SIZE: <span style="color:#3B82F6;">{len(df_window)} TOKENS</span></span>
            <span>LIVE RISK: <span style="color:{status_color};">{score_norm:.4f}</span></span>
            <span>CLASS SCHEMA: <span style="color:#3B82F6;">{superclass_label.upper()}</span></span>
        </div>
    </div>
    <div style="text-align:right; flex-shrink:0;">
        <span style="background:{status_color}; padding:12px 28px; border-radius:12px; font-weight:700; color:white; font-size:14px; letter-spacing:1px; box-shadow:0 0 30px {status_color}35; border: 1px solid rgba(255,255,255,0.15);">
            {status}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- 8. INDUSTRIAL MINI INTELLIGENCE CARD ROW ---
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="intel-card" style="margin-bottom:0;"><div class="card-title">Risk Score</div><div class="card-value" style="color:{status_color};">{score_norm:.3f}</div></div>', unsafe_allow_html=True)
with k2:
    # Drop font size slightly to handle 64-bit long integer tracking keys without text wrapping breaks
    vessel_font_size = "1.2rem" if len(str(vessel_id)) > 10 else "1.75rem"
    st.markdown(
        f'<div class="intel-card" style="margin-bottom:0;">'
        f'<div class="card-title">Vessel ID</div>'
        f'<div class="card-value" style="color:#3B82F6; font-size:{vessel_font_size};">{vessel_id}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )
with k3:
    st.markdown(f'<div class="intel-card" style="margin-bottom:0;"><div class="card-title">Partition Records</div><div class="card-value">{total_rows:,}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="intel-card" style="margin-bottom:0;"><div class="card-title">Track Length</div><div class="card-value">{len(df_window)}</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="intel-card" style="margin-bottom:0;"><div class="card-title">Risk Delta</div><div class="card-value" style="color:#6366F1;">{score_anom - score_norm:+.3f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --- 9. ASYMMETRIC TACTICAL ROW (4:1 SCALE VIEWPORT) ---
map_col, intel_col = st.columns([4.0, 1.0])

with map_col:
    st.subheader("Tactical Navigation View")
    
    if "lon_raw" in df_window.columns and "lat_raw" in df_window.columns:
        fig = go.Figure()
        
        lons_anom = df_perturbed["lon_raw"].to_numpy()
        lats_anom = df_perturbed["lat_raw"].to_numpy()

        fig.add_trace(go.Scattermap(
            lon=vessel_lons, lat=vessel_lats,
            mode="lines", line=dict(color="#3B82F6", width=5),
            name="Observed Route"
        ))
        
        fig.add_trace(go.Scattermap(
            lat=lats_anom,
            lon=lons_anom,
            mode="lines+markers",
            line=dict(width=3, color="#FF3B30"),
            marker=dict(size=4, color="#FF3B30"),
            name="Counterfactual Hypothesis Path"
        ))
        
        fig.add_trace(go.Scattermap(
            lon=[vessel_lons[0]], lat=[vessel_lats[0]],
            mode="markers+text", 
            marker=dict(size=14, color="#10B981"),
            text=["START"], textposition="top center", name="Origin Anchor"
        ))
        
        fig.add_trace(go.Scattermap(
            lon=[vessel_lons[-1]], lat=[vessel_lats[-1]],
            mode="markers+text", 
            marker=dict(size=20, color="#F59E0B"),
            text=["CURRENT POSITION"], textposition="bottom center", name="Target Head"
        ))

        lat_range = np.max(vessel_lats) - np.min(vessel_lats)
        lon_range = np.max(vessel_lons) - np.min(vessel_lons)
        max_bound = max(lat_range, lon_range)
        calculated_zoom = 13 if max_bound == 0 else max(1, min(12, int(11.5 - np.log2(max_bound))))

        # Adjusted paper and plot backgrounds to transparent to support native theme parameters
        fig.update_layout(
            map=dict(
                style=selected_map_style,
                center=dict(lat=float(np.mean(vessel_lats)), lon=float(np.mean(vessel_lons))),
                zoom=calculated_zoom
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=800, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(128, 128, 128, 0.15)")
        )
        st.plotly_chart(fig, width="stretch")

with intel_col:
    st.subheader("Threat Assessment")
    
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score_norm),
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": "#3B82F6", "thickness": 0.24},
            "steps": [
                {"range": [0, 0.50], "color": "rgba(16, 185, 129, 0.08)"},
                {"range": [0.50, 0.75], "color": "rgba(245, 158, 11, 0.08)"},
                {"range": [0.75, 1.0], "color": "rgba(239, 68, 68, 0.12)"}
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 4},
                "thickness": 0.8,
                "value": 0.75
            }
        }
    ))
    gauge.update_layout(
        height=200, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11), margin=dict(l=10, r=10, t=25, b=10)
    )
    st.plotly_chart(gauge, width="stretch")
    
    # REVISED VISUAL PROFILER LABELING
    inv_vocab = {v: k for k, v in SUPERCLASS_VOCAB.items()}
    
    superclass_label = inv_vocab.get(sc_id_raw, "Unknown").upper()

    # Ultimate Safety Net: If vocab translation fails, display the raw text string from the file
    if superclass_label == "UNKNOWN" and "shiptype" in df_window.columns and df_window["shiptype"][0] is not None:
        superclass_label = str(df_window["shiptype"][0]).upper().strip()
    
    # Built structural model-driven risk drivers attribution parameters
    rows = "".join(
        f'<div class="profile-row"><span class="profile-label">{feat_name}</span>'
        f'<span class="profile-value" style="color:#EF4444;"> {error_val:.1%}</span></div>'
        for feat_name, error_val in res_normal.get("top_contributors", [])[:3]
    )
    st.markdown(f"""
    <div class="intel-card">
        <div class="card-title-accent">Risk Drivers</div>
        {rows if rows else '<div style="font-size:0.8rem; opacity:0.6;">No exceptional variations tracked.</div>'}
    </div>
    """, unsafe_allow_html=True)

    # Vessel Metadata Profile Card
    st.markdown(f"""
    <div class="intel-card">
        <div class="card-title-accent">Vessel Profile</div>
        <div class="profile-row"><span class="profile-label">Vessel ID</span><span class="profile-value">{vessel_id}</span></div>
        <div class="profile-row"><span class="profile-label">Classification</span><span class="profile-value">{superclass_label}</span></div>
        <div class="profile-row"><span class="profile-label">Track Points</span><span class="profile-value">{len(df_window)}</span></div>
        <div class="profile-row"><span class="profile-label">Last Sync</span><span class="profile-value">{last_timestamp_str}</span></div>
    </div>
    """, unsafe_allow_html=True)


# --- 10. KINEMATIC TREND ANALYTICS PANEL ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Behavioral Analytics", unsafe_allow_html=True)

left, right = st.columns(2)

time_axis = df_window["datetime"].to_list() if "datetime" in df_window.columns else list(range(len(df_window)))

with left:
    if "speed_raw" in df_window.columns:
        speed_fig = go.Figure()
        speed_fig.add_trace(go.Scatter(
            x=time_axis, y=df_window["speed_raw"].to_list(),
            mode="lines", line=dict(color="#3B82F6", width=3),
            fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.08)"
        ))
        speed_fig.update_layout(
            title=dict(text="Speed Profile (knots)", font=dict(size=13)),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)"),
            yaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)")
        )
        st.plotly_chart(speed_fig, width="stretch")

with right:
    if "course_raw" in df_window.columns:
        course_fig = go.Figure()
        course_fig.add_trace(go.Scatter(
            x=time_axis, y=df_window["course_raw"].to_list(),
            mode="lines", line=dict(color="#10B981", width=3)
        ))
        course_fig.update_layout(
            title=dict(text="Course Over Ground Profile (degrees)", font=dict(size=13)),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)"),
            yaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)")
        )
        st.plotly_chart(course_fig, width="stretch")


# --- 11. NEAREST HISTORICAL MATCHES EXPLAINABILITY ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Nearest Vessel Matches", unsafe_allow_html=True)

true_model_latent_embedding = res_normal["cls_embedding"]
peer_neighborhood = detector.ann_index.query_behavior_neighborhood(
    true_model_latent_embedding, superclass_id=context_meta["superclass_id"], k=3
)

if peer_neighborhood:
    rows_html = "".join(
        f"""
    <tr>
        <td>#{rank}</td>
        <td>Vessel ID {match["vessel_id_int"]}</td>
        <td style="font-family: 'JetBrains Mono';">{match["distance"]:.4f}</td>
        <td><span style="color:#10B981; font-weight:600;">{1.0 - match["distance"]:.2%}</span></td>
    </tr>"""
        for rank, match in enumerate(peer_neighborhood, 1)
    )
    html = f"""
    <div class="intel-card" style="padding: 10px;">
        <table class="production-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Vessel Match</th>
                    <th>Distance</th>
                    <th>Similarity</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="intel-card">
        <div style="opacity:0.6; font-size:0.85rem; text-align:center; padding: 20px 0;">
            No historical peer tracking matches isolated inside current HNSW cluster bounds.
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 12. ADVANCED MODEL VALIDATION ACCORDION ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Advanced Model Validation Metrics"):
    st.markdown(
        f"""
<div class="intel-card" style="padding: 10px; margin-bottom: 0;">
    <table class="production-table">
        <thead>
            <tr>
                <th>Original Evaluation Stream</th>
                <th>Risk Score</th>
                <th>Classification</th>
                <th>Risk Delta</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="color:#3B82F6; font-weight:600;">Observed Route (Clean)</td>
                <td>{score_norm:.4f}</td>
                <td>{res_normal["anomaly_classification"]}</td>
                <td>—</td>
            </tr>
            <tr>
                <td style="color:#EF4444; font-weight:600;">Synthetic Divergence (Perturbed)</td>
                <td>{score_anom:.4f}</td>
                <td>{res_anomalous["anomaly_classification"]}</td>
                <td style="color:#6366F1; font-weight:600;">{score_anom - score_norm:+.4f}</td>
            </tr>
        </tbody>
    </table>
</div>
""", 
        unsafe_allow_html=True
    )


# --- 13. CENTRALIZED ENVIRONMENTAL GEOSPATIAL CONTEXT ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Environmental Context", unsafe_allow_html=True)
e1, e2, e3 = st.columns(3)

if "distance_to_coast_nm" in df_window.columns:
    e1.metric("Coast Distance", f"{df_window['distance_to_coast_nm'].mean():.2f} nm")

if "distance_to_terminal_nm" in df_window.columns:
    e2.metric("Terminal Distance", f"{df_window['distance_to_terminal_nm'].mean():.2f} nm")

if "harbor_basin_proximity" in df_window.columns:
    e3.metric("Harbor Proximity", f"{df_window['harbor_basin_proximity'].mean():.3f}")