"""Shared matplotlib style for BMVC 2026 paper figures.
Use: from paper_plot_style import *
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import json
import os
from pathlib import Path

FONT_SIZE = 9
DPI = 300
# Project root. Every path below used to be hardcoded to the original
# root-owned checkout (/root/BMVC2026), which made the generators unrunnable
# from any other working copy -- including the one the paper is built from.
# Derive it from this file's own location instead; ARIS_PROJECT_ROOT overrides.
PROJECT_ROOT = Path(os.environ.get(
    'ARIS_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
FIG_DIR = Path(os.environ.get('ARIS_FIG_DIR', PROJECT_ROOT / 'figures'))
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,  # TrueType (editable)
    'ps.fonttype': 42,
})

# Colorblind-safe palette
COLOR_BASELINE = '#222222'      # near-black for labeled-only reference
COLOR_SCORE_S0 = '#1f6feb'      # strong blue (score-based, seed 0)
COLOR_SCORE_S1 = '#86b6f6'      # faded blue (score-based, seed 1)
COLOR_CAMERA_S0 = '#d62728'     # red diamond (camera-support, seed 0)
COLOR_CAMERA_S1 = '#f5a8a8'     # faded red (camera-support, seed 1)
BAND_SCORE = '#deeaff'           # very light blue band
BAND_CAMERA = '#ffe0e0'          # very light red band


def save_fig(fig, name, formats=('pdf', 'png')):
    """Save figure to FIG_DIR in multiple formats."""
    for fmt in formats:
        path = FIG_DIR / f'{name}.{fmt}'
        fig.savefig(path)
        print(f'  saved: {path}')


def parse_eval_log(path):
    """Extract NDS, mAP, mTP family + per-class AP from an mmdet3d eval.log."""
    if not Path(path).exists():
        return None
    text = Path(path).read_text(errors='ignore')
    out = {}
    for key in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        m = re.search(rf'^{re.escape(key)}: (0\.[0-9]+)', text, re.M)
        if m:
            out[key] = float(m.group(1))
    # Per-class AP: extract directly from individual key/value matches in the
    # final result dict line, sidestepping eval() problems.
    classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    out['per_class_ap'] = {}
    for cls in classes:
        aps = []
        for thr in ('0.5', '1.0', '2.0', '4.0'):
            mm = re.search(
                rf"'pts_bbox_NuScenes/{re.escape(cls)}_AP_dist_{re.escape(thr)}': ([0-9.]+)",
                text)
            if mm:
                aps.append(float(mm.group(1)))
        if aps:
            out['per_class_ap'][cls] = sum(aps) / len(aps)
    return out


def load_run(tag):
    """Convenience: load an external eval log by run tag."""
    p = PROJECT_ROOT / 'work_dirs' / f'{tag}_external_eval' / 'eval.log'
    return parse_eval_log(p)


def load_correlation():
    """Load D2 correlation analysis JSON outputs."""
    pooled_path = PROJECT_ROOT / 'cache' / 'd2_correlation_analysis.json'
    perclass_path = PROJECT_ROOT / 'cache' / 'd2_correlation_perclass.json'
    out = {}
    if pooled_path.exists():
        out['pooled'] = json.loads(pooled_path.read_text())
    if perclass_path.exists():
        out['perclass'] = json.loads(perclass_path.read_text())
    return out
