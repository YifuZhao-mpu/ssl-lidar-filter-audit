"""Fig 2 — D2 mechanism: inter-camera support correlation.
Side-by-side 6×6 heatmaps for real teacher vs perturbed-GT pilot.
Pooled means: real r=0.42, pilot r=0.03.
"""
import sys
import pathlib
# Import paper_plot_style from this script's own directory rather than a
# hardcoded absolute path, so the generator runs from any working copy
# and from any cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_plot_style import *
import numpy as np

corr = load_correlation()
pooled = corr['pooled']
real_mat = np.array(pooled['real_corr_matrix'])
pilot_mat = np.array(pooled['pilot_corr_matrix'])
cameras = pooled['cameras']
short_names = [c.replace('CAM_', '') for c in cameras]

# wspace: panel (b)'s y-tick labels are long (FRONT_RIGHT, BACK_RIGHT,
# BACK_LEFT, FRONT_LEFT) and at the matplotlib default spacing they ran left
# into panel (a)'s plotting area, covering the (BACK_LEFT, FRONT_LEFT) cell
# and the (FRONT_LEFT, FRONT_LEFT) diagonal. Reserve the gap explicitly.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4),
                          gridspec_kw={'width_ratios': [1, 1.07],
                                       'wspace': 0.55})

vmin, vmax = -0.2, 1.0
for ax, mat, title, mean_r in [
    (axes[0], real_mat, f'(a) Real teacher (mean off-diag $r$={pooled["real_offdiag_mean"]:.2f})', pooled['real_offdiag_mean']),
    (axes[1], pilot_mat, f'(b) Perturbed-GT pilot (mean off-diag $r$={pooled["pilot_offdiag_mean"]:.2f})', pooled['pilot_offdiag_mean']),
]:
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='equal')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=FONT_SIZE - 2)
    ax.set_yticklabels(short_names, fontsize=FONT_SIZE - 2)
    ax.set_title(title, fontsize=FONT_SIZE)
    # Annotate cells
    for i in range(6):
        for j in range(6):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=FONT_SIZE - 2,
                        color='#aaa')
            else:
                color = 'white' if abs(v) > 0.5 else '#222'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=FONT_SIZE - 2, color=color)
    # Restore the spines that the global style hid
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

# Single colorbar to the right of axes[1]
cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.04)
cbar.set_label('Pearson $r$', fontsize=FONT_SIZE - 1)

save_fig(fig, 'fig2_correlation_pair')
plt.close(fig)
print(f'Fig 2 done. Real r mean={pooled["real_offdiag_mean"]:.4f}; '
      f'pilot r mean={pooled["pilot_offdiag_mean"]:.4f}; '
      f'Δ={pooled["delta_mean_real_minus_pilot"]:.4f}')
