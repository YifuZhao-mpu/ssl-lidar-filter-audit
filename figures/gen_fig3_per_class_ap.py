"""Fig 3 — Per-class AP at 5% labels.
Three filter variants: R4b (labeled-only), R7b (score-based SSL), R8b (camera-support SSL).
Highlights bicycle / construction_vehicle collapse for camera-support.
"""
import sys
import pathlib
# Import paper_plot_style from this script's own directory rather than a
# hardcoded absolute path, so the generator runs from any working copy
# and from any cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_plot_style import *
import numpy as np

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
SHORT_LABELS = ['car', 'truck', 'CV', 'bus', 'trailer',
                'barrier', 'motor', 'bicycle', 'ped', 'cone']

R4b = load_run('R4b')['per_class_ap']
R7b = load_run('R7b')['per_class_ap']
R8b = load_run('R8b')['per_class_ap']

x = np.arange(len(CLASSES))
w = 0.27

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.0))

bars1 = ax.bar(x - w, [R4b[c] for c in CLASSES], w,
               color=COLOR_BASELINE, edgecolor='white', linewidth=0.5,
               label='Labeled-only (no SSL)')
bars2 = ax.bar(x,     [R7b[c] for c in CLASSES], w,
               color=COLOR_SCORE_S0, edgecolor='white', linewidth=0.5,
               label='Score-based SSL (perclass-only)')
bars3 = ax.bar(x + w, [R8b[c] for c in CLASSES], w,
               color=COLOR_CAMERA_S0, edgecolor='white', linewidth=0.5,
               label='Camera-support SSL (best-view)')

# Highlight tail-class collapse: bicycle, construction_vehicle for R8b
for ci in (CLASSES.index('bicycle'), CLASSES.index('construction_vehicle')):
    bar = bars3[ci]
    ax.annotate('AP=0', xy=(bar.get_x() + bar.get_width()/2, 0.005),
                xytext=(bar.get_x() + bar.get_width()/2, 0.10),
                fontsize=FONT_SIZE - 2, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_CAMERA_S0,
                                lw=0.7),
                color=COLOR_CAMERA_S0)

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, rotation=20, ha='right', fontsize=FONT_SIZE - 1)
ax.set_ylabel('per-class AP (mean over 0.5/1.0/2.0/4.0 m)')
ax.set_xlabel('nuScenes class @ 5% labels')
ax.set_ylim(0, 0.85)
ax.legend(loc='upper right', frameon=False, fontsize=FONT_SIZE - 1)

save_fig(fig, 'fig3_per_class_ap')
plt.close(fig)
print('Fig 3 done.')
