import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. AUTHORITATIVE DATA ARRAYS (AUDITED AGAINST VERIFIED SPECIFICATIONS)
# ----------------------------------------------------------------------
epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

loss = np.array([0.01236, 0.00948, 0.00716, 0.00594, 0.00535, 0.00481, 0.00446, 
                  0.00427, 0.00405, 0.00393, 0.00380, 0.00383, 0.00358, 0.00350, 0.00347])

eff_rank = np.array([67.5, 66.2, 65.1, 63.1, 62.0, 61.1, 60.2, 59.7, 59.2, 58.2, 
                     57.2, 56.6, 56.1, 55.8, 55.0])

mean_cos = np.array([0.1750, 0.1950, 0.2123, 0.2100, 0.1900, 0.1837, 0.1824, 
                     0.1821, 0.1739, 0.1702, 0.1665, 0.1627, 0.1568, 0.1516, 0.1577])

latent_var = np.array([0.0064, 0.0063, 0.0061, 0.0062, 0.0063, 0.0064, 0.0064, 
                       0.0064, 0.0065, 0.0065, 0.0065, 0.0065, 0.0066, 0.0066, 0.0066])

# ----------------------------------------------------------------------
# 2. JOURNAL GRAPHICS AESTHETICS AND CONFIGURATION (OCEAN ENGINEERING)
# ----------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 13,
    'text.usetex': False  # Set to True if local system has an active LaTeX engine
})

fig, (ax_a, ax_b1) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=False)
fig.subplots_adjust(wspace=0.30)  # Expanded padding to accommodate custom multi-scale axes margins

# ----------------------------------------------------------------------
# 3. PANEL (A) GENERATION: MULTI-TASK CONVERGENCE LOSS
# ----------------------------------------------------------------------
color_loss = '#1e293b'  # Academic Deep Charcoal Slate
ax_a.plot(epochs, loss, color=color_loss, linestyle='-', marker='o', 
          markersize=5, linewidth=1.75, label='Multi-Task Pre-training Loss')
ax_a.set_xlabel('Epoch', fontweight='bold', labelpad=8)
ax_a.set_ylabel('Training Loss (Multi-task)', color=color_loss, fontweight='bold', labelpad=8)
ax_a.set_xlim(0.5, 15.5)
ax_a.set_ylim(0.002, 0.013)
ax_a.set_xticks(np.arange(1, 16, 1))
ax_a.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
ax_a.set_title('(a) Foundation Model Convergence Profile', loc='left', pad=10, fontweight='bold')
ax_a.tick_params(axis='y', labelcolor=color_loss)

# ----------------------------------------------------------------------
# 4. PANEL (B) GENERATION: HIDDEN MANIFOLD CAPACITY DIAGNOSTICS
# ----------------------------------------------------------------------
# Axis B1: Centered Effective Rank (Primary Left Axis)
color_rank = '#0f766e'  # Academic Muted Teal
ax_b1.plot(epochs, eff_rank, color=color_rank, linestyle='-', marker='s', 
           markersize=5, linewidth=1.5, label='Effective Rank ($R_{\\mathrm{eff}}$)')
ax_b1.set_xlabel('Epoch', fontweight='bold', labelpad=8)
ax_b1.set_ylabel('Effective Rank ($R_{\\mathrm{eff}}$)', color=color_rank, fontweight='bold', labelpad=8)
ax_b1.set_xlim(0.5, 15.5)
ax_b1.set_ylim(40, 100)
ax_b1.set_xticks(np.arange(1, 16, 1))
ax_b1.tick_params(axis='y', labelcolor=color_rank)
ax_b1.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
ax_b1.set_title('(b) Latent Manifold Structural Evolution', loc='left', pad=10, fontweight='bold')

# Axis B2: Mean Pairwise Cosine Similarity (Secondary Inner Right Axis)
color_cos = '#b45309'  # Academic Muted Amber
ax_b2 = ax_b1.twinx()
ax_b2.plot(epochs, mean_cos, color=color_cos, linestyle='--', marker='^', 
           markersize=5, linewidth=1.5, label='Mean Cosine Similarity')
ax_b2.set_ylabel('Mean Cosine Similarity', color=color_cos, fontweight='bold', labelpad=8)
ax_b2.set_ylim(0.00, 0.30)
ax_b2.tick_params(axis='y', labelcolor=color_cos)

# Axis B3: Latent Variance (Independent Outer Right Axis)
color_var = '#1d4ed8'  # Academic Cobalt Blue
ax_b3 = ax_b1.twinx()
ax_b3.spines['right'].set_position(('outward', 65))  # Explicit physical displacement to eliminate line collision
ax_b3.plot(epochs, latent_var, color=color_var, linestyle='-.', marker='d', 
           markersize=5, linewidth=1.5, label='Latent Manifold Variance')
ax_b3.set_ylabel('Latent Representation Variance', color=color_var, fontweight='bold', labelpad=8)
ax_b3.set_ylim(0.004, 0.010)
ax_b3.tick_params(axis='y', labelcolor=color_var)

# Consolidate individual sub-axes descriptors into a unified, clean legend window
lines_1, labels_1 = ax_b1.get_legend_handles_labels()
lines_2, labels_2 = ax_b2.get_legend_handles_labels()
lines_3, labels_3 = ax_b3.get_legend_handles_labels()
ax_b1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, 
             loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#e2e8f0', fontsize=9)

# Export assets using clean vector formats (PDF preserves transparency seamlessly)
plt.savefig('navisight_structural_diagnostics.png', dpi=300, bbox_inches='tight')
plt.savefig('navisight_structural_diagnostics.eps', format='eps', bbox_inches='tight')
plt.savefig('navisight_structural_diagnostics.pdf', format='pdf', bbox_inches='tight')
print("Successfully initialized revised vector maps with requested y-axis scale windows.")