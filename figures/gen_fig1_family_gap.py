"""Fig 1 - D2 family-gap visualization at 5% labels (single panel hero).

THREE-SEED version (Package B, 2026-07-18). Supersedes the two-seed figure
generated 2026-05-11; the paper text and Table 1 are three-seed, so the hero
figure must be too.

Score-based filters cluster ~0.41-0.46; camera-support filters ~0.30-0.38.
The class-aware positive-only hybrid (2 seeds) sits inside the score band.
Grey band = same-init labeled-only fine-tune reference across the three seeds.
"""
import sys
sys.path.insert(0, '/workspace/BMVC2026/figures')
from paper_plot_style import *
import numpy as np

# (display label, s0 tag, s1 tag, s2 tag); None = run not in the protocol
SCORE_VARIANTS = [
    ('single-thr', 'R6', 'R6_s1', 'R6_ssl_5p_s2_single_thr'),
    ('perclass+\nboost', 'R7', 'R7_s1', 'R7_ssl_5p_s2_perclass_boost'),
    ('perclass-\nonly', 'R7b', 'R7b_s1', 'R7b_ssl_5p_s2_perclass_only'),
]
CAMERA_VARIANTS = [
    ('best-view', 'R8b', 'R8b_s1', 'R8b_ssl_5p_s2_best_view'),
    ('any-view', 'R8c', 'R8c_s1', 'R8c_ssl_5p_s2_any_view'),
    ('max-cam\nK=5', 'R8d', 'R8d_s1', 'R8d_ssl_5p_s2_max_cam'),
    ('consensus', 'R7c', 'R7c_s1', 'R7c_ssl_5p_s2_consensus'),
]
HYBRID_VARIANTS = [
    ('hybrid\npos-only', 'R18_ssl_5p_s0_hybrid_posonly', None,
     'R18_ssl_5p_s2_hybrid_posonly'),
]

LABEL_ONLY = {
    's0': load_run('R4b')['NDS'],
    's1': load_run('R4_s1')['NDS'],
    's2': load_run('R4_s2')['NDS'],
}

# Per-seed horizontal offsets and marker alpha (s0 solid -> s2 lightest)
SEED_DX = {'s0': -0.20, 's1': 0.00, 's2': 0.20}
SEED_SIZE = {'s0': 78, 's1': 62, 's2': 62}

# Sized for a full-text-width (figure*) slot so that 9pt labels render ~1:1
# and need no downscaling: 24 data points do not fit legibly in one column.
fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.0))

# Horizontal background bands marking the two families' empirical envelopes
ax.axhspan(0.405, 0.465, color=BAND_SCORE, zorder=0)
ax.axhspan(0.295, 0.390, color=BAND_CAMERA, zorder=0)

# Labeled-only reference: a grey band spanning the three seeds (0.4471-0.4515)
lo, hi = min(LABEL_ONLY.values()), max(LABEL_ONLY.values())
ax.axhspan(lo, hi, color='#bbbbbb', alpha=0.55, zorder=1)
ax.axhline(y=lo, color=COLOR_BASELINE, linestyle='--', linewidth=0.8,
           alpha=0.8, zorder=2)
ax.axhline(y=hi, color=COLOR_BASELINE, linestyle='--', linewidth=0.8,
           alpha=0.8, zorder=2)


def plot_family(variants, x0, marker, colors, legend_prefix):
    """Scatter one filter family, three seeds per variant. Returns x positions."""
    xs = list(range(x0, x0 + len(variants)))
    for x, (_, t0, t1, t2) in zip(xs, variants):
        for seed, tag in (('s0', t0), ('s1', t1), ('s2', t2)):
            if tag is None:
                continue
            r = load_run(tag)
            if r is None:
                print(f'  WARNING: missing eval for {tag}')
                continue
            ax.scatter([x + SEED_DX[seed]], [r['NDS']], s=SEED_SIZE[seed],
                       marker=marker, c=colors[seed], edgecolors='white',
                       linewidths=0.9, zorder=4)
    # legend proxies, one per seed
    for seed in ('s0', 's1', 's2'):
        ax.scatter([], [], s=SEED_SIZE[seed], marker=marker, c=colors[seed],
                   edgecolors='white', linewidths=0.9,
                   label=f'{legend_prefix} (seed {seed[1]})')
    return xs


SCORE_COLORS = {'s0': COLOR_SCORE_S0, 's1': '#5b9bf8', 's2': COLOR_SCORE_S1}
CAMERA_COLORS = {'s0': COLOR_CAMERA_S0, 's1': '#ec7b7b', 's2': COLOR_CAMERA_S1}
HYBRID_COLORS = {'s0': '#7b3fa0', 's1': '#a97cc4', 's2': '#c9a8dc'}

xs_score = plot_family(SCORE_VARIANTS, 0, 's', SCORE_COLORS, 'Score-based')
xs_camera = plot_family(CAMERA_VARIANTS, len(xs_score), 'D', CAMERA_COLORS,
                        'Camera-support')
xs_hybrid = plot_family(HYBRID_VARIANTS, len(xs_score) + len(xs_camera), 'o',
                        HYBRID_COLORS, 'Hybrid pos-only')

all_xs = xs_score + xs_camera + xs_hybrid
all_labels = ([v[0] for v in SCORE_VARIANTS] + [v[0] for v in CAMERA_VARIANTS]
              + [v[0] for v in HYBRID_VARIANTS])
ax.set_xticks(all_xs)
ax.set_xticklabels(all_labels, rotation=0, fontsize=FONT_SIZE - 1)

# Vertical separators between sections
sep_sc = (max(xs_score) + min(xs_camera)) / 2
sep_ch = (max(xs_camera) + min(xs_hybrid)) / 2
for sx in (sep_sc, sep_ch):
    ax.axvline(x=sx, color='#888', linestyle=':', linewidth=0.6, zorder=1)

# Family-gap arrow + label (worst-case .. best-case over the three seeds)
ax.annotate('', xy=(sep_sc + 0.05, 0.315), xytext=(sep_sc + 0.05, 0.408),
            arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2), zorder=5)
ax.text(sep_sc + 0.18, 0.362, '+6.9 to +11.1 NDS\nfamily gap (3 seeds)',
        fontsize=FONT_SIZE - 1, ha='left', va='center', color='#222',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#888', lw=0.5),
        zorder=6)

# Section headers
ax.text(np.mean(xs_score), 0.478, 'score-based', ha='center', va='top',
        fontsize=FONT_SIZE, color=COLOR_SCORE_S0, fontweight='bold')
ax.text(np.mean(xs_camera), 0.478, 'camera-support', ha='center', va='top',
        fontsize=FONT_SIZE, color=COLOR_CAMERA_S0, fontweight='bold')
ax.text(np.mean(xs_hybrid), 0.478, 'hybrid', ha='center', va='top',
        fontsize=FONT_SIZE, color=HYBRID_COLORS['s0'], fontweight='bold')

# The labeled-only band goes in the legend rather than as inline text, which
# would collide with the hybrid seed-2 point sitting at 0.4546.
from matplotlib.patches import Patch
labeled_proxy = Patch(facecolor='#bbbbbb', alpha=0.55, edgecolor=COLOR_BASELINE,
                      linestyle='--', linewidth=0.8,
                      label=f'Labeled-only, 3 seeds ({lo:.4f}-{hi:.4f})')

ax.set_ylabel('NDS (val_full=6019, ep10)')
ax.set_xlabel('Pseudo-label filter variant @ 5% nuScenes labels')
# Bottom headroom is reserved for the legend so it never overlaps the lowest
# camera-support points (max-cam K=5 s0 = 0.3041).
ax.set_ylim(0.255, 0.492)
ax.set_xlim(-0.6, len(all_xs) - 0.4)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [labeled_proxy], labels + [labeled_proxy.get_label()],
          loc='lower center', frameon=False, fontsize=FONT_SIZE - 3,
          ncol=3, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.2)

save_fig(fig, 'fig1_hero_family_gap')
plt.close(fig)

# Provenance echo so the numbers in the caption can be checked against the log
print('\nPlotted NDS values:')
for name, group in (('score', SCORE_VARIANTS), ('camera', CAMERA_VARIANTS),
                    ('hybrid', HYBRID_VARIANTS)):
    for label, t0, t1, t2 in group:
        vals = {s: (load_run(t)['NDS'] if t and load_run(t) else None)
                for s, t in (('s0', t0), ('s1', t1), ('s2', t2))}
        print(f'  [{name}] {label.replace(chr(10), " "):<16} {vals}')
print(f'  labeled-only: {LABEL_ONLY}')
print('Fig 1 (3-seed) done.')
