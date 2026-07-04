import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Define file paths
results_json_path = "configs/evaluation_results.json"
output_plot_path = "multi_channel_percentiles.png"

if not os.path.exists(results_json_path):
    raise FileNotFoundError(f"Could not locate the file: {results_json_path}")

# Load the audited ledger data
with open(results_json_path, "r") as f:
    data = json.load(f)

profiles = data["score_distribution_percentiles"]
channels = ["sequence_reconstruction", "manifold_ann_distance", "prioritization_fusion"]
channel_display_names = {
    "sequence_reconstruction": "Sequence Reconstruction MSE",
    "manifold_ann_distance": "Manifold ANN Distance (OOD)",
    "prioritization_fusion": "Prioritization Fusion Index"
}

# Configure plot typography and aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle("NaviSight Performance Audit: Score Distribution Shifts", fontsize=16, fontweight='bold', y=1.02)

percentile_labels = ["p50", "p90", "p95", "p99"]
x_indexes = np.arange(len(percentile_labels))
bar_width = 0.35

for idx, channel in enumerate(channels):
    ax = axes[idx]
    
    # Extract baseline points from ledger
    normal_vals = [profiles[channel]["normal_profile"][p] for p in percentile_labels]
    anom_vals = [profiles[channel]["anomaly_profile"][p] for p in percentile_labels]
    
    # Retrieve independent threshold line from calibration block
    threshold_key = "manifold_ann" if channel == "manifold_ann_distance" else channel
    fitted_threshold = data["calibration_baselines"][threshold_key]["threshold"]
    
    # Render tracking data bars
    rects_norm = ax.bar(x_indexes - bar_width/2, normal_vals, bar_width, label='Normal Profile', color='#2b5c8f', alpha=0.85)
    rects_anom = ax.bar(x_indexes + bar_width/2, anom_vals, bar_width, label='Anomaly Profile', color='#c94c4c', alpha=0.85)
    
    # Draw the strict operational threshold cutoff line
    thresh_line = ax.axhline(y=fitted_threshold, color='#d97706', linestyle='--', linewidth=2, 
                             label=f'Threshold (p{data["meta_config_constants"]["calibration_percentile_cutoff"]})')
    
    # Add text labels on top of the bars to reveal exact values
    ax.bar_label(rects_norm, padding=3, fmt='%.2f', fontsize=9)
    ax.bar_label(rects_anom, padding=3, fmt='%.2f', fontsize=9)
    
    # Format individual channel subplots
    ax.set_title(channel_display_names[channel], fontsize=12, fontweight='bold')
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(["Median (p50)", "p90 Target", "p95 Target", "Extreme (p99)"], fontsize=10)
    ax.set_xlabel("Distribution Percentiles", fontsize=11)
    
    if idx == 0:
        ax.set_ylabel("Aggregated Profile Anomaly Score", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f" [SUCCESS] High-fidelity visualization generated and saved to: {output_plot_path}")
plt.show()