
███╗   ██╗ █████╗ ██╗   ██╗██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗
████╗  ██║██╔══██╗██║   ██║██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝
██╔██╗ ██║███████║██║   ██║██║███████╗██║██║  ███╗███████║   ██║
██║╚██╗██║██╔══██║╚██╗ ██╔╝██║╚════██║██║██║   ██║██╔══██║   ██║
██║ ╚████║██║  ██║ ╚████╔╝ ██║███████║██║╚██████╔╝██║  ██║   ██║
╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝


**Contextual Spatiotemporal Foundation Model Platform for Maritime Anomaly Intelligence** 
*Piraeus Operational Corridor Dataset · 2017–2019 · Out-of-Core Processing Stream*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Polars](https://img.shields.io/badge/Polars-Streaming-CD792C?style=flat-square)](https://pola.rs)
[![Database](https://img.shields.io/badge/Storage-Parquet%20%7C%20SQLite%20WAL-orange?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

</div>

---

## 🛰️ Platform Overview

**NaviSight AI** is a research-grade production spatiotemporal platform and deep foundation model architecture engineered for continuous, out-of-core maritime trajectory representation learning. Traditional tracking frameworks rely on rigid, rule-based speed thresholds or shallow clustering techniques that evaluate movement patterns in isolation. This leads to massive false-alarm spikes caused by environmental variations (e.g., a cargo vessel slowing down to navigate a narrow strait or weather system).

NaviSight AI fixes this limitation by building a self-supervised **Trajectory Masked Autoencoder (Traj-MAE)**. The system conditions physical kinematic vectors on three synchronized metadata domains: **hierarchical vessel taxonomy profiles, local geodetic geography structures, and continuous spatial weather fields**. By analyzing how these elements interact across years of historical tracking logs from the Piraeus coastal corridor, the architecture learns to world-model normal operations, suppressing environmental false positives while identifying genuine behavioral anomalies down to exact physical root causes.

---

## 📐 Core Mathematical & Geospatial Modeling

### 1. Continuous-Time Stochastic System Formulations
Vessel trajectory dynamics are represented as a **Continuous-Time Stochastic Hybrid System** operating inside local tangent planes. Instead of manipulating un-normalized latitude and longitude coordinates directly, the model evaluates state dynamics via an active state vector:

$$x_t = [p_x, p_y, v, \theta, \mathbf{z}]^T$$

Where $p$ represents metric East-North-Up (ENU) position offsets, $v$ tracks forward physical velocity, $\theta$ isolates true manifold heading, and $\mathbf{z}$ governs an underlying continuous latent regime logit matrix. Trajectory state transitions follow a multi-variate stochastic differential equation (SDE):

$$dx_t = f(x_t, t)dt + g(x_t, t)dW_t$$

Where $dW_t$ represents a multivariate Brownian noise vector tracking environmental disturbances.

### 2. Shortest-Path Geodesic Unwrapping Lookups
Standard Euclidean distance metrics generate non-physical data jumps near the international dateline ($\pm180^\circ$). NaviSight AI resolves longitudinal transitions natively using **Modular Angular Distance** unwrapping equations:

$$\Delta \lambda = ((\lambda_2 - \lambda_1 + 180) \bmod 360) - 180$$

This guarantees a continuous $C^2$ spatial manifold wrapper across coordinate boundaries. 

### 3. Multi-Branch Positional Trajectory Encodings
To prevent token permutation errors within the self-attention mechanism, temporal steps are passed into three parallel tracking branches:
* **Absolute Position Index Branch:** Maps sequence indices within the rolling window context.
* **Continuous Local Delta Branch:** Tracks localized velocity variation by encoding step time-deltas ($\Delta t$).
* **Continuous Cumulative Timeline Branch:** Captures long-term journey progression via an active trip integration sum ($t_{cum} = \sum \Delta t$).

---

## 🛠️ Data Schema Contract

The ingestion framework enforces strict type safety, parsing 62 explicit features out-of-core. Columns are categorized across four semantic groups:

| Modality Namespace | Selected Target Fields | Technical Description | Transformation Rule |
| :--- | :--- | :--- | :--- |
| **PHYSICS** | `speed_raw`, `actual_speed_knots`, `course_change`, `turn_rate`, `jerk`, `delta_x_nm`, `delta_y_nm` | Kinematic spatial derivatives and local tangent coordinate plane shifts. | Cohort-Stratified Z-Score Normalization |
| **WEATHER** | `wind_speed`, `temperature`, `pressure`, `headwind_component`, `crosswind_component` | Continuous meteorological variables from NOAA GFS spatial grids. | Robust Centered Scaling |
| **CONTEXT** | `hour_sin`, `hour_cos`, `voyage_phase_cruising`, `vessel_superclass_id`, `shiptype` | Cyclic temporal values, one-hot operational status, and taxonomy details. | Passthrough / Int-Mapping |
| **QUALITY** | `flag_speed_anomaly`, `flag_position_jump`, `reliability_score` | Binary quality filters and a continuous asset reliability index `[0.0, 1.0]`. | Dynamic Penalty Deduction |

---

## 📋 The Forensic Debugging Story

NaviSight’s core framework was stabilized following a forensic debugging campaign that traced a **15-bug causal chain** responsible for complete training collapse. The structural breakdown and subsequent architectural fixes included:

1. **The String Membership Vulnerability (`in` vs `==`):** The initial preprocessing loop checked feature names using Python's `in` operator (e.g., `if 'lon_raw' in feat_name`). This triggered accidental substring matches on features like `acceleration` and `log_time_diff`, routing raw physics columns to a geographic bounding-box scaler instead of their cohort statistics. This delivered near-zero feature variance to the encoder. **Fix:** Replaced with explicit `==` equality constraints.
2. **The Loss Target Shifting Bug:** The tracking attribution collector utilized a zip expression: `zip(REGISTRY.maskable_for_loss, mean_feature_errors)`. The error array contained the full 41 model dimensions, while `maskable_for_loss` contained only 21 tracking features. This length mismatch caused an index shift, forcing features like `turn_rate` to inherit the massive un-normalized errors of spatial displacement fields. **Fix:** Array lengths are explicitly mapped against the full registry before filtering down to maskable subsets.
3. **The Constant Variance NaN Generator:** The departure phase column was hardcoded to a literal zero (`voyage_phase_departure = pl.lit(0)`). Z-scoring a constant column divided values by zero variance, injecting `NaN` fields throughout the tensor matrices. **Fix:** Implemented automated non-zero variance checks with an epsilon floor handler.
4. **The Streamlit Memory Caching Trap:** The dashboard initialization routines utilized a resource cache wrapper (`@st.cache_resource`). When downstream model optimization code changed, Streamlit froze the old model instance in system memory, leading to persistent layout anomalies until the cache was manually invalidated.

---

## 🚀 Pipeline Processing Architecture

[ Raw ZIP Telemetry Stream ] ──► [ Async Line-Buffer Backpressure ] ──► [ Persistent xxhash Index Mapping ]
│
▼
[ Lakehouse Parquet Dataset ] ◄── [ Atomic os.replace() Swap ] ◄── [ Geodesic Bilinear Weather Mesh ]
│
▼
[ Continuity Dataset Loader ] ──► [ MaritimeMAE Transformer ] ──► [ L2 Stabilized CLS Embedding ]
│
▼
[ Dual-Channel Alerts Panel ] ◄── [ HNSW Neighbor Graph Match ] ◄── [ Append-Only PyArrow Shard Store ]


---

## ⚡ Dual-Channel Anomaly Intelligence

Once pre-training completes, incoming evaluation sequences are scored simultaneously across two isolated analytical channels to separate telemetry faults from suspicious intentions:

* **Channel 1 (Kinematic Reconstruction Error):** Tracks the Mean Squared Error ($MSE$) between raw trajectory features and the model's decoded reconstructions. Sudden spikes flag raw point anomalies like sensor tampering or GPS position manipulation.
* **Channel 2 (Manifold Behavioral Drift):** Extracts the normalized high-dimensional `[CLS]` token and measures its cosine distance against the vessel's historical rolling Exponential Moving Average ($EMA$) behavior profile. Drift beyond adaptive statistical thresholds flags systematic deviations like unauthorized routing updates, loitering loops, or illicit offshore rendezvous.

---

## 🛠️ Setup & Execution

### Prerequisites
Ensure your local system has `CUDA` compilation layers initialized. Install system dependencies natively:
```bash
pip install polars pyarrow xxhash hnswlib torch streamlit plotly-express
Execution Lifecycles
Bash
# 1. Ingest raw AIS logs, compile weather overlays, and generate parquet shards
python scripts/run_pipeline.py

# 2. Extract cohort-stratified normalization statistics 
python scripts/compute_global_stats.py

# 3. Launch the self-supervised pre-training loop
python -m navisight.engine.train_model

# 4. Spin up the tactical real-time UI dashboard console
streamlit run scripts/view_validation_dashboard.py
