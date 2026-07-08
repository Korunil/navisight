# 🛰️ NaviSight

<div align="center">

### Geospatial Behavioral Intelligence Platform for Maritime Anomaly Detection

Self-Supervised Learning • Behavioral Representation Learning • HNSW Similarity Search • Explainable AI • Counterfactual Threat Simulation • Maritime Intelligence

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![Polars](https://img.shields.io/badge/Polars-Analytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![HNSW](https://img.shields.io/badge/HNSW-ANN_Search-green)
![Transformers](https://img.shields.io/badge/Transformer-Masked_Autoencoder-blueviolet)
![AIS](https://img.shields.io/badge/AIS-Maritime_Intelligence-navy)
![Research Prototype](https://img.shields.io/badge/Research-Prototype-blue)
![Self-Supervised Learning](https://img.shields.io/badge/SSL-Transformer-green)
![Behavioral AI](https://img.shields.io/badge/Behavioral-Intelligence-orange)
![Maritime AI](https://img.shields.io/badge/Maritime-AI-blueviolet)
![Explainable AI](https://img.shields.io/badge/XAI-Enabled-red)
![License](https://img.shields.io/badge/License-Apache_2.0-red)

</div>

---

# Table of Contents

- [Executive Summary](#executive-summary)
- Research Results
- Associated Publication
- Mission
- Overview
- Why NaviSight Matters
- Dataset & Data Engineering
  - AIS Maritime Telemetry Corpus
  - Geospatial Intelligence Layer
  - Meteorological Intelligence Layer
- Feature Engineering Pipeline
- Training Corpus Construction
- Key Capabilities
- Architecture
- Research Contributions
- NAVISIGHT Methodology
- Model Card
- Evaluation Framework
- Mathematical Foundations
- Engineering Challenges Solved
- Behavioral Embedding Space
- Similarity Search Engine
- Counterfactual Threat Simulation
- Dashboard Preview
- Repository Structure
- System Metrics
- Technology Stack
- Quick Start
- Engineering Tradeoffs
- Design Decisions
- Lessons Learned
- Future Work
- Contributing
- License
- Author

---
# Executive Summary

NaviSight is a behavioral intelligence platform for maritime anomaly detection built around self-supervised representation learning.

The system transforms raw AIS telemetry into latent behavioral embeddings using a Masked Autoencoder Transformer and detects anomalous activity through cohort-aware similarity search, behavioral profiling, and contextual risk fusion.

The architecture was designed to answer a fundamental question:

> Can vessel behavior itself become the primary signal for maritime threat detection?

Rather than relying on manually engineered rules, NaviSight learns behavioral manifolds directly from historical trajectories and identifies deviations in latent space.

The platform combines:

- Self-Supervised Transformers
- Geospatial Feature Engineering
- Behavioral Embedding Learning
- HNSW Similarity Search
- Rolling Profile Modeling
- Counterfactual Threat Simulation
- Explainable Risk Attribution

into a unified operational intelligence system.

The result is a framework capable of detecting subtle behavioral anomalies that would remain invisible to threshold-based monitoring systems.

---
# Research Results

NAVISIGHT was evaluated on approximately 192 million AIS observations
from the Aegean Sea.

Key results:

| Detection Channel | AUROC |
|-------------------|--------|
| Isolation Forest | 0.6252 |
| One-Class SVM | 0.5007 |
| LOF | 0.5431 |
| LSTM AutoEncoder | 0.5617 |
| Reconstruction | 0.8737 |
| Behavioural Manifold | 0.9309 |
| OR Fusion | 0.9247 |

Evaluation followed a strictly chronological train-calibration-test protocol.

---

# Mission

NaviSight exists to transform maritime monitoring from rule-based alerting into behavioral intelligence.

The long-term objective is to build systems capable of understanding how vessels behave, not simply where they are.

By learning latent behavioral patterns directly from telemetry, NaviSight aims to enable earlier detection of emerging maritime threats while reducing analyst workload.

---
## Overview

NaviSight is an end-to-end maritime behavioral intelligence platform that learns latent vessel behavior directly from AIS telemetry and detects anomalous activity using self-supervised representation learning, cohort-aware similarity search, rolling behavioral profiling, and contextual risk fusion.

Unlike traditional maritime monitoring systems that depend on manually crafted rules and thresholds, NaviSight learns behavioral manifolds from historical vessel movement patterns and identifies deviations through representation-space reasoning.

The platform combines:

- Self-Supervised Masked Autoencoder Transformers
- Behavioral Embedding Learning
- Cohort-Gated HNSW Similarity Search
- Rolling Vessel Behavior Profiling
- Dual-Channel Contextual Risk Fusion
- Counterfactual Threat Simulation
- Explainable Anomaly Analysis Dashboard

to create a modern behavioral intelligence framework for maritime anomaly detection.

---

# Why NaviSight Matters

NaviSight demonstrates how modern self-supervised learning,
approximate nearest-neighbour search, geospatial analytics,
and maritime domain knowledge can be combined into a unified
anomaly detection framework.

The project was developed as both a research contribution and
an exploration of scalable behavioural AI for real-world
maritime surveillance systems.

More than 80% of global trade travels by sea.

Every day, millions of AIS transmissions are generated by vessels operating across:

- international shipping lanes
- territorial waters
- ports
- coastal regions
- offshore facilities

Most monitoring systems still rely on:

- static rules
- speed thresholds
- geofencing alerts
- manually engineered heuristics

These approaches struggle to detect:

- covert loitering
- route manipulation
- smuggling activity
- AIS spoofing
- deceptive navigation
- slow behavioral drift

Traditional systems ask:

> Did the vessel violate a predefined rule?

NaviSight asks:

> Does the vessel still behave like vessels that belong to its behavioral cohort?

This shifts anomaly detection from rule matching to behavioral intelligence.

Rather than relying on fixed thresholds, NaviSight learns normal vessel behavior directly from historical trajectories and identifies subtle deviations that may indicate emerging threats.

---

# 📊 Dataset & Data Engineering

## AIS Maritime Telemetry Corpus

NaviSight is trained on large-scale Automatic Identification System (AIS) telemetry collected from commercial maritime traffic operating within the Eastern Mediterranean maritime domain.

The dataset captures vessel movement behavior at scale and serves as the foundation for self-supervised behavioral representation learning.

---

## Exploratory Dataset Statistics (Month Sample - September 2018)

The following statistics correspond to a one-month AIS telemetry snapshot used for exploratory analysis, feature validation, and data quality assessment.

| Attribute | Value |
|------------|---------|
| AIS Observations | **~192 Million** |
| Calibration Windows | 309,268 |
| Evaluation Windows | 262,287 |
| Geographic Region | Aegean Sea |
| Coverage Area | Piraeus Maritime Domain |
| Data Type | Vessel Telemetry |
| Learning Paradigm | Self-Supervised |
| Sequence Length | 60 Steps |
| Engineered Features | 35 |
| Embedding Dimension | 128 |
| ANN Backend | HNSW |
| Storage Format | Partitioned Parquet |
| Inference Backend | PyTorch |

> Dataset statistics generated from exploratory profiling and validation pipelines.

---

## AIS Signals

Raw vessel telemetry includes:

- Latitude
- Longitude
- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading
- Navigation Status
- Timestamp
- Vessel Metadata
- Voyage Context

These signals are transformed into a high-dimensional behavioral representation through extensive feature engineering.

---

## Geospatial Intelligence Layer

NaviSight augments vessel telemetry with operational maritime context.

Derived geospatial features include:

- Distance to Coast
- Distance to Port
- Distance to Terminal
- Harbor Proximity
- Land Ingress Detection
- Voyage Phase Indicators
- Spatial Density Features
- Regional Context Encoding

This enables the model to reason not only about vessel motion but also about environmental and operational conditions.

---

## Meteorological Intelligence Layer

To model vessel behavior under realistic operating conditions, AIS trajectories are fused with spatiotemporal weather data.

---

### Weather Dataset Statistics

| Attribute | Value |
|------------|---------|
| Records | 10,800 |
| Temporal Coverage | September 2018 |
| Time Resolution | 6-Hour Intervals |
| Spatial Representation | Rectilinear Grid |
| Weather Variables | 18 |
| Interpolation Strategy | Trilinear Spatiotemporal Interpolation |

---

### Weather Variables

NaviSight incorporates:

| Variable | Description |
|-----------|-------------|
| TMP | Temperature |
| RH | Relative Humidity |
| PRMSL | Mean Sea Level Pressure |
| VIS | Visibility |
| WSPD | Wind Speed |
| GUST | Wind Gust |
| UGRD | Zonal Wind Component |
| VGRD | Meridional Wind Component |
| DPT | Dew Point |
| APCP | Accumulated Precipitation |

---

### Environmental Feature Derivation

Weather observations are transformed into vessel-centric behavioral features:

- Headwind Component
- Crosswind Component
- Environmental Resistance
- Visibility Exposure
- Weather Severity Indicators
- Atmospheric Stability Signals

This allows the behavioral encoder to distinguish operational behavior from weather-driven motion changes.

---

# 🧠 Feature Engineering Pipeline

```text
Raw AIS Telemetry
        │
        ▼

Trajectory Cleaning
        │
        ▼

Voyage Segmentation
        │
        ▼

Kinematic Feature Extraction
        │
        ├── Speed
        ├── Acceleration
        ├── Turn Rate
        ├── Jerk
        └── Heading Dynamics

        ▼

Geospatial Enrichment
        │
        ├── Port Distance
        ├── Coast Distance
        ├── Harbor Features
        └── Terminal Context

        ▼

Weather Fusion
        │
        ├── Wind Speed
        ├── Gusts
        ├── Visibility
        ├── Pressure
        └── Humidity

        ▼

Temporal Context Encoding
        │
        ├── Hour-of-Day
        ├── Day-of-Week
        └── Voyage Phase

        ▼

41-Dimensional Feature Space
```

---

# 📈 Training Corpus Construction

```text
192M+ AIS Observations
          │
          ▼

Trajectory Segmentation
          │
          ▼

Window Generation
          │
          ▼

60-Step Sequences
          │
          ▼

Masked Token Corruption
          │
          ▼

Self-Supervised Training
          │
          ▼

Transformer Encoder
          │
          ▼

128-D Behavioral Embeddings
```

---

## Feature Registry Summary

NaviSight's feature registry currently spans four primary domains:

| Category | Purpose |
|-----------|---------|
| Kinematic Features | Vessel motion dynamics |
| Geospatial Features | Environmental context |
| Weather Features | Operational conditions |
| Quality-Control Features | Data reliability signals |

This multimodal representation enables the system to learn behavioral manifolds rather than relying solely on trajectory geometry.

---

## Why the Dataset Matters

Most maritime anomaly detection systems focus exclusively on:

- positional tracks
- speed thresholds
- geofencing rules

NaviSight instead learns from a richer behavioral state space combining:

```text
Movement
      +
Environment
      +
Weather
      +
Context
      +
Historical Behavior
```

This enables detection of subtle anomalies that may remain invisible to traditional rule-based monitoring systems.

---
# Key Capabilities

### Behavioral Representation Learning

Learn compact latent representations of vessel behavior.

### Similarity-Based Intelligence

Compare vessels against historical behavioral peers rather than global populations.

### Explainable Risk Assessment

Surface risk drivers and nearest historical behavioral matches.

### Counterfactual Threat Simulation

Generate realistic adversarial vessel trajectories.

### Operational Dashboard

Interactive intelligence hub for monitoring, validation, and analysis.

---

# Architecture

## High-Level System Architecture

![Architecture Diagram](assets/navisight_architecture.png)

---

## System Flow

```text
Raw AIS Telemetry
        │
        ▼
Geospatial Feature Engineering
        │
        ▼
Sequence Construction
        │
        ▼
Masked Autoencoder Transformer
        │
        ▼
128-D Behavioral Embeddings
        │
 ┌──────┴──────────────┐
 ▼                     ▼

HNSW Index       Behavior Profiles
 │                     │
 └──────────┬──────────┘
            ▼

Dual-Channel Detector
            │
            ▼

Counterfactual Simulator
            │
            ▼

Interactive Maritime Analytics Dashboard
```

---

# Mermaid Architecture Diagram

```mermaid
flowchart TD

A[AIS Telemetry]

A --> B[Feature Engineering]

B --> C[Sequence Builder]

C --> D[Masked Autoencoder Transformer]

D --> E[128-D Behavioral Embeddings]

E --> F[HNSW Similarity Search]

E --> G[Rolling Behavioral Profiles]

F --> H[Dual Channel Detector]

G --> H[Risk Fusion Engine]

H --> I[Counterfactual Threat Simulator]

I --> J[Tactical Intelligence Dashboard]
```
---

# NaviSight Methodology

Data Engineering
     ↓
Multimodal Feature Construction
     ↓
Maritime Masked Autoencoder
     ↓
128-D Embeddings
     ↓
HNSW Behavioral Memory
     ↓
Reconstruction Channel
+
Behavioral Drift Channel
     ↓
Fusion
     ↓
Anomaly Detection

---

# Research Contributions

NaviSight introduces several architectural concepts inspired by modern representation learning and behavioral intelligence systems.

---

## 1. Self-Supervised Maritime Representation Learning

Rather than training directly on anomaly labels, NaviSight learns vessel behavior through reconstruction objectives.

Benefits:

- reduced labeling requirements
- improved generalization
- richer latent representations

---

## 2. Cohort-Gated Similarity Search

Instead of comparing all vessels globally, NaviSight isolates comparisons to behaviorally similar vessel classes.

Benefits:

- lower false positive rates
- improved contextual relevance
- more meaningful peer comparisons

---

## 3. Rolling Behavioral Profiles

Historical behavior is continuously tracked using exponentially weighted profile updates.

Benefits:

- adaptive baselines
- temporal awareness
- behavior drift detection

---

## 4. Counterfactual Threat Simulation

The system can inject realistic behavioral perturbations.

Examples:

- Covert Loitering
- Dead Reckoning Drift
- Coastal Creep

Benefits:

- validation
- robustness testing
- explainability

---

# Model Card v1.0

## Model Name

NaviSight Maritime Behavioral Encoder

---

## Version

v1.0

---

## Model Type

Masked Autoencoder Transformer

---

## Objective

Learn vessel behavioral representations from AIS trajectories.

---

## Model Overview

| Attribute | Value |
|------------|---------|
| Model Type | Masked Autoencoder Transformer |
| Learning Paradigm | Self-Supervised |
| Domain | Maritime AIS |
| Embedding Dimension | 128 |
| Input Type | Sequential Telemetry |
| Output | Behavioral Embedding |

---

## Inputs

119-step telemetry sequences.

Features include:

- position
- velocity
- heading
- turn rate
- harbor proximity
- coast distance
- terminal distance
- derived motion features

---

## Outputs

### Behavioral Embedding

```text
128-dimensional latent representation
```

### Risk Score

```text
0.0 → 1.0
```

### Threat Classification

```text
NOMINAL
WARNING
CRITICAL
```

---

## Downstream Tasks

- Anomaly Detection
- Vessel Similarity Search
- Threat Assessment
- Behavioral Clustering
- Operational Monitoring

---

## Intended Use

Operational maritime intelligence.

---

## Limitations

- dependent on AIS quality
- vulnerable to missing transmissions
- limited environmental context

---

# Mathematical Foundations

The platform models vessel state as:

```math
x_t = [p_x, p_y, v, \theta, z]^T
```

where:

- position
- velocity
- heading
- latent behavioral regime

evolve through a stochastic dynamical process.

---

## State Evolution

```math
dx_t = f(x_t,t)dt + g(x_t,t)dW_t
```

where:

- f = deterministic dynamics
- g = stochastic dynamics
- W = Brownian process

---

# Engineering Challenges Solved

---

## Scaling Massive AIS Datasets

### Challenge

Global AIS telemetry contains hundreds of millions of records.

### Solution

Partitioned Parquet Lakehouse architecture.

Benefits:

- efficient storage
- lazy loading
- parallel processing

---

## Fast Behavioral Similarity Search

### Challenge

Exhaustive pairwise vessel comparison is computationally infeasible.

### Solution

Hierarchical Navigable Small World (HNSW) graphs.

Benefits:

- logarithmic search complexity
- real-time retrieval
- scalable indexing

---

## Reducing False Positives

### Challenge

Different vessel classes naturally exhibit different behaviors.

### Solution

Cohort-Gated ANN Search.

Benefits:

- class-aware comparisons
- stronger contextual relevance

---

## Explainability

### Challenge

Deep anomaly models are often opaque.

### Solution

Risk attribution engine.

Provides:

- nearest historical matches
- feature contributions
- counterfactual validation

---

# Behavioral Embedding Space

Every trajectory window is transformed into:

```text
Trajectory
      ↓
Transformer Encoder
      ↓
CLS Token
      ↓
128-D Embedding
```

The embedding captures:

- movement style
- route characteristics
- maneuvering behavior
- environmental interaction

---

# Similarity Search Engine

## ANN Backend

```text
HNSW
```

## Stored Metadata

Each embedding tracks:

```text
embedding_id
vessel_id
timestamp
trip_id
superclass
```

allowing traceability from latent vectors back to raw trajectories.

---

# Counterfactual Threat Simulation

NaviSight contains a dedicated adversarial simulation engine.

---

## Covert Loitering

Simulates circular holding patterns.

Use cases:

- surveillance
- rendezvous behavior
- illegal waiting

---

## Dead Reckoning Drift

Simulates deceptive heading drift.

Use cases:

- AIS spoofing
- route manipulation

---

## Coastal Creep

Simulates constrained coastal movement.

Use cases:

- smuggling
- illegal fishing
- shoreline monitoring

---

# Tactical Intelligence Hub

The platform includes a professional operational dashboard.

---

## Threat Assessment

Features:

- live risk scoring
- anomaly classification
- behavioral diagnostics

---

## Tactical Navigation View

Interactive geospatial intelligence map.

Displays:

- observed route
- simulated route
- vessel status
- environmental context

---

## Behavioral Analytics

Trend visualization:

- speed
- heading
- course
- maneuvering dynamics

---

## Similarity Intelligence

Displays:

- nearest behavioral neighbors
- embedding distance
- similarity confidence

---

# Dashboard Preview

```text
assets/
├── dashboard_overview.png
├── dashboard_clc.png                # Coverted Loitering Circle
├── dashboard_add.png                # AR(1) Deception Drift
├── dashboard_ctc.png                # Constrained Topological Creep
├── dashboard_analytics.png
├── navisight_architecture.png
```
---
# Tactical Intelligence Hub

![Dashboard Demo](assets/dashboard_overview.png)

---

# Repository Structure

```text
navisight/

├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── models/
│   ├── checkpoints/
│   └── state/
│
├── navisight/
│
│   ├── pipeline/
│   │
│   ├── training/
│   │
│   ├── evaluation/
│   │
│   ├── feature_registry/
│   │
│   ├── geospatial/
│   │
│   ├── embeddings/
│   │
│   └── inference/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_embeddings.py
│   └── view_validation_dashboard.py
│
├── assets/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# System Metrics

| Metric | Value |
|----------|---------|
| Embedding Size | 128 |
| Sequence Length | 119 |
| ANN Backend | HNSW |
| Storage Format | Parquet |
| Inference Framework | PyTorch |
| Dashboard | Streamlit |
| Analytics Engine | Polars |
| Search Complexity | O(log N) |

---

# Technology Stack

| Layer | Technology |
|---------|------------|
| Deep Learning | PyTorch |
| Dashboard | Streamlit |
| Data Processing | Polars |
| Visualization | Plotly |
| Similarity Search | HNSW |
| Storage | Parquet |
| Numerical Computing | NumPy |
| Language | Python |

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/yourusername/navisight.git

cd navisight
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Launch Dashboard

```bash
streamlit run dashboard/view_validation_dashboard.py
```

---
# Engineering Tradeoffs

## Why HNSW Instead of FAISS IVF?

### HNSW Advantages

- lower retrieval latency
- excellent recall
- dynamic insertions

### Tradeoff

Higher memory consumption.

Decision:

Prioritized retrieval quality and operational latency.

---

## Why Self-Supervised Learning?

### Advantages

- minimal labeling requirements
- better scalability

### Tradeoff

Interpretability challenges.

Decision:

Added explainability layers through similarity retrieval and feature attribution.

---

# Design Decisions

## Decision #1

Behavior First, Rules Second

Traditional systems:

Rule → Alert

NaviSight:

Behavior → Embedding → Risk

---

## Decision #2

Cohort-Aware Similarity

Rejected:

Global vessel comparisons.

Implemented:

Superclass-isolated HNSW indices.

Reason:

Different vessel classes exhibit fundamentally different movement patterns.

---

## Decision #3

Explainability as a First-Class Citizen

Every anomaly score must be traceable through:

- nearest neighbors
- feature attribution
- counterfactual simulation


---
# Lessons Learned Building NaviSight

### 1. Data Engineering Matters More Than Models

Most gains came from improving trajectory quality rather than increasing model complexity.

---

### 2. Similarity Search Is Surprisingly Powerful

Many anomalies become obvious when viewed through behavioral neighbors.

---

### 3. Explainability Cannot Be Added Later

Operational users require trust before they accept anomaly alerts.

---

### 4. Synthetic Scenarios Are Essential

Counterfactual simulation revealed failure modes that traditional evaluation never exposed.

---

### 5. Maritime Behavior Is Highly Contextual

The same trajectory may be normal for one vessel class and anomalous for another.

---
# Example Workflow

```text
AIS Data
    ↓

Feature Engineering
    ↓

Embedding Generation
    ↓

ANN Indexing
    ↓

Behavior Profiling
    ↓

Threat Detection
    ↓

Dashboard Visualization
```

---

# Future Work

### Planned Improvements

- Graph Neural Networks
- Multi-Vessel Interaction Modeling
- Online Learning
- Satellite Data Fusion
- Global Traffic Forecasting
- Streaming Inference
- Distributed HNSW Infrastructure
- Explainable Attention Maps
- Real-Time Alerting

### Target use cases:

- Maritime Security
- Illegal Fishing Detection
- Port Intelligence
- Offshore Asset Monitoring
- Trade Route Analysis

---

# Academic Inspiration

Relevant fields:

- Representation Learning
- Self-Supervised Learning
- Geospatial AI
- Maritime Informatics
- Anomaly Detection
- Approximate Nearest Neighbor Search
- Explainable AI

---

# Contributing

Contributions are welcome.

Areas of interest:

- anomaly detection
- geospatial analytics
- transformer architectures
- behavioral modeling
- visualization
- performance optimization

---

# License

Licensed under the Apache 2.0 License.

---

# Acknowledgements

Built using open-source technologies from:

- PyTorch
- Streamlit
- Plotly
- Polars
- NumPy

and inspired by research across:

- Self-Supervised Learning
- Geospatial Intelligence
- Maritime Analytics
- Behavioral Modeling

---

# Author

### Anil Kumar Korupoju

AI Engineer • Machine Learning Engineer • Applied AI Research Enthusiast

Focused on:

- Representation Learning
- Self-Supervised Sequence Systems
- RAG Systems
- Agentic AI
- Geospatial Intelligence
- Large Language Models
- Anomaly Detection

---

If you found NaviSight useful:

⭐ Star the repository

🍴 Fork the project

🛰️ Build the future of maritime intelligence
