#!/usr/bin/env python
"""
Scene-level (block) bootstrap confidence intervals for nuScenes detection NDS.

Motivation
----------
The paper reports NDS point estimates for 7 pseudo-label filters x up to 3 seeds and
claims that a within-family spread of <= 0.5 NDS is "seed noise" while the +6.9..+11.1
NDS score-vs-camera family gap is real.  Neither claim is currently accompanied by any
uncertainty estimate.  This driver supplies one *evaluation* uncertainty estimate (it
does NOT estimate training/seed variance -- see CAVEAT at the bottom of this docstring)
by non-parametrically resampling the val set.

Design
------
1. RESAMPLING UNIT = SCENE (block bootstrap).  The nuScenes val split is 150 scenes /
   6019 keyframes.  Keyframes inside a scene are ~2 Hz samples of the same traffic
   situation with the same sensor calibration, weather and object instances, so they are
   very far from i.i.d.  A per-keyframe bootstrap would treat 6019 dependent units as
   independent and produce dishonestly narrow intervals.  We therefore resample the 150
   scenes with replacement and take the union (with multiplicity) of their keyframes.

2. OFFICIAL DEVKIT.  Metrics are computed by the unmodified nuScenes devkit:
   `nuscenes.eval.detection.evaluate.DetectionEval.evaluate()` (which calls the official
   `accumulate`, `calc_ap`, `calc_tp` and `DetectionMetrics`).  We subclass DetectionEval
   only to bypass its `__init__` (which hard-codes loading from disk and from the
   predefined split) and inject already-loaded, already-filtered `gt_boxes` / `pred_boxes`.
   `add_center_dist` and `filter_eval_boxes` (distance / zero-point / bike-rack filtering)
   are the official loader functions and are applied ONCE per run on the full val split,
   before any resampling.  This is safe because all three filters are per-box and
   independent of which other samples are in the eval set.

   INJECTION POINT / CONSISTENCY OF THE RESTRICTION.  `DetectionEval.evaluate()` reads
   exactly three things: `self.gt_boxes`, `self.pred_boxes` and `self.cfg`.  Restriction
   therefore has to happen on BOTH EvalBoxes objects, and it has to be the *same* token
   multiset, otherwise `npos` (from `gt_boxes.all`) and the prediction list (from
   `pred_boxes.all`) refer to different data and the metric is silently wrong.  We build
   both from the same scene draw in `build_resampled_boxes()` and assert key equality.

3. MULTIPLICITY -- RE-KEYING, NOT WEIGHTING.  `EvalBoxes` is a dict keyed by
   sample_token, so a scene drawn m times would collide into a single key and be counted
   once.  We instead RE-KEY: the j-th (j>=1) copy of sample token T is stored under
   "T#j", in BOTH the GT and the prediction EvalBoxes, and the prediction boxes of that
   copy are shallow-copied with `sample_token` rewritten to "T#j" (the devkit matches via
   `gt_boxes[pred_box.sample_token]` and de-duplicates matched GT with a
   `(sample_token, gt_idx)` set, so a copy must carry its own token to get its own GT).
   GT box objects are shared by reference (the devkit never mutates them during
   `evaluate()`); only the list keys are duplicated.

   Effect on the variance estimate: re-keying reproduces the resampled dataset EXACTLY
   as if those keyframes had physically occurred m times.  In particular `npos` is
   multiplied by m and each copy is matched greedily against its OWN copy of the GT, so a
   detector cannot "reuse" one GT box for two copies.  The cheaper alternative -- keeping
   one key and giving it weight m -- would give the same npos but would NOT reproduce the
   duplicated greedy matching nor the duplicated entries in the confidence-sorted TP/FP
   accumulation used by the 101-point interpolated PR curve and by `calc_tp`'s
   recall-indexed error curves; it would slightly *understate* the variance because the
   duplicated FP/TP runs would be collapsed.  Re-keying is the honest choice and is what
   we do.

4. PAIRED REPLICATES.  Replicate b uses `numpy.random.default_rng(b)` over the sorted
   list of the 150 val scene names.  This depends on b ONLY, so every run in the study
   sees the *identical* resampled val set in replicate b.  Differences between filters
   are therefore paired, and the CI on a difference is a CI on the difference of paired
   quantities (much tighter and much more relevant than differencing two marginal CIs).

5. PARALLELISM.  Prep (JSON load + center-dist + filter, needs the NuScenes DB) happens
   once per run in a small pool and is cached to disk as a pickle.  The bootstrap pool
   then needs NO NuScenes DB at all: workers only hold the filtered GT (~0.3 GB) plus one
   run's filtered predictions (~0.7-1.7 GB) and evaluate replicates.

CAVEAT (state this in the paper).  This is an *evaluation-set* bootstrap.  It answers
"how much of the observed NDS gap is attributable to which 150 scenes we happened to
evaluate on?".  It does NOT answer "how much would NDS move if we retrained with another
seed?".  Training-seed variance is estimated separately by the 3 seeds actually run; the
two sources are reported side by side, never conflated.

Outputs (default /workspace/BMVC2026/cache/bootstrap_ci.json):
  - per run:   point NDS/mAP on full val (no resampling), bootstrap mean, SE,
               percentile 95% CI, B completed
  - per (seed, score filter, camera filter): paired difference distribution, mean,
               95% CI, sign-flip fraction
  - per seed:  the family statistic the paper actually claims, min(score) - max(camera),
               with bootstrap CI
  - within-family D1 check: perclass_only - perclass_boost per seed with CI, plus
               two-sided sign test / Wilcoxon signed-rank across seeds.

CLI
---
  python tools/g2_bootstrap_ci.py --smoke
  python tools/g2_bootstrap_ci.py --runs main5p --B 1000 --workers 64
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import pickle
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------

REPO = '/workspace/BMVC2026'
WORK_DIRS = os.path.join(REPO, 'work_dirs')
DEFAULT_DATAROOT = '/datasets/data/nuscenes'
DEFAULT_VERSION = 'v1.0-trainval'
DEFAULT_CACHE = os.path.join(REPO, 'cache')
DEFAULT_PREP_DIR = os.path.join(DEFAULT_CACHE, 'bootstrap_prep')
DEFAULT_OUT = os.path.join(DEFAULT_CACHE, 'bootstrap_ci.json')
EVAL_SET = 'val'          # 150 scenes / 6019 keyframes ("val_full" in the paper)
N_VAL_SAMPLES = 6019
N_VAL_SCENES = 150

RESULTS_REL = os.path.join('results', 'pts_bbox', 'results_nusc.json')
METRICS_REL = os.path.join('results', 'pts_bbox', 'metrics_summary.json')

# --------------------------------------------------------------------------------------
# Run registry.  Seed-suffix conventions on disk are inconsistent, so they are spelled out
# explicitly here rather than inferred.  `expected_nds` is the value the paper reports and
# is cross-checked against the cached metrics_summary.json AND against our own
# no-resampling recomputation.
# --------------------------------------------------------------------------------------

def _R(key, tag, seed, family, label, d, expected):
    return dict(key=key, tag=tag, seed=seed, family=family, label=label, dir=d,
                expected_nds=expected)

RUN_REGISTRY: "OrderedDict[str, dict]" = OrderedDict()
for _r in [
    # ---- reference: labeled-only teacher, no SSL -------------------------------------
    _R('R4b_s0', 'R4b', 0, 'labeled_only', 'labeled-only (5%)', 'R4b_external_eval', 0.4471),
    _R('R4_s1',  'R4',  1, 'labeled_only', 'labeled-only (5%)', 'R4_s1_external_eval', 0.4482),
    _R('R4_s2',  'R4',  2, 'labeled_only', 'labeled-only (5%)', 'R4_s2_external_eval', 0.4515),
    # ---- score-based family ----------------------------------------------------------
    _R('R6_s0',  'R6',  0, 'score',  'single-thr',      'R6_external_eval', 0.4108),
    _R('R6_s1',  'R6',  1, 'score',  'single-thr',      'R6_s1_external_eval', 0.4513),
    _R('R6_s2',  'R6',  2, 'score',  'single-thr',      'R6_ssl_5p_s2_single_thr_external_eval', 0.4534),
    _R('R7_s0',  'R7',  0, 'score',  'perclass+boost',  'R7_external_eval', 0.4129),
    _R('R7_s1',  'R7',  1, 'score',  'perclass+boost',  'R7_s1_external_eval', 0.4558),
    _R('R7_s2',  'R7',  2, 'score',  'perclass+boost',  'R7_ssl_5p_s2_perclass_boost_external_eval', 0.4556),
    _R('R7b_s0', 'R7b', 0, 'score',  'perclass-only',   'R7b_external_eval', 0.4150),
    _R('R7b_s1', 'R7b', 1, 'score',  'perclass-only',   'R7b_s1_external_eval', 0.4609),
    _R('R7b_s2', 'R7b', 2, 'score',  'perclass-only',   'R7b_ssl_5p_s2_perclass_only_external_eval', 0.4535),
    # ---- camera-support family -------------------------------------------------------
    _R('R8b_s0', 'R8b', 0, 'camera', 'best-view',       'R8b_external_eval', 0.3193),
    _R('R8b_s1', 'R8b', 1, 'camera', 'best-view',       'R8b_s1_external_eval', 0.3790),
    _R('R8b_s2', 'R8b', 2, 'camera', 'best-view',       'R8b_ssl_5p_s2_best_view_external_eval', 0.3844),
    _R('R8c_s0', 'R8c', 0, 'camera', 'any-view',        'R8c_external_eval', 0.3176),
    _R('R8c_s1', 'R8c', 1, 'camera', 'any-view',        'R8c_s1_external_eval', 0.3774),
    _R('R8c_s2', 'R8c', 2, 'camera', 'any-view',        'R8c_ssl_5p_s2_any_view_external_eval', 0.3835),
    _R('R8d_s0', 'R8d', 0, 'camera', 'max-cam K=5',     'R8d_external_eval', 0.3041),
    _R('R8d_s1', 'R8d', 1, 'camera', 'max-cam K=5',     'R8d_s1_external_eval', 0.3558),
    _R('R8d_s2', 'R8d', 2, 'camera', 'max-cam K=5',     'R8d_ssl_5p_s2_max_cam_external_eval', 0.3570),
    _R('R7c_s0', 'R7c', 0, 'camera', '6-cam consensus', 'R7c_external_eval', 0.3059),
    _R('R7c_s1', 'R7c', 1, 'camera', '6-cam consensus', 'R7c_s1_external_eval', 0.3655),
    _R('R7c_s2', 'R7c', 2, 'camera', '6-cam consensus', 'R7c_ssl_5p_s2_consensus_external_eval', 0.3661),
    # ---- hybrid ----------------------------------------------------------------------
    _R('R18_s0', 'R18', 0, 'hybrid', 'class-aware pos-only', 'R18_ssl_5p_s0_hybrid_posonly_external_eval', 0.4124),
    _R('R18_s2', 'R18', 2, 'hybrid', 'class-aware pos-only', 'R18_ssl_5p_s2_hybrid_posonly_external_eval', 0.4546),
    # ---- shuffle interventions (s0 only) ---------------------------------------------
    _R('R7d_s0', 'R7d', 0, 'intervention', 'consensus-shuffled',  'R7d_external_eval', 0.3196),
    _R('R7e_s0', 'R7e', 0, 'intervention', 'consensus-decorr',    'R7e_external_eval', 0.3279),
    # ---- label-fraction side studies -------------------------------------------------
    _R('R10a_s0', 'R10a', 0, 'labeled_only_10p', 'labeled-only (10%)', 'R10a_external_eval', 0.4590),
    _R('R10a_s1', 'R10a', 1, 'labeled_only_10p', 'labeled-only (10%)', 'R10a_s1_external_eval', 0.4640),
    _R('R10b_s0', 'R10b', 0, 'score_10p', 'perclass-only (10%)', 'R10b_external_eval', 0.4704),
    _R('R10b_s1', 'R10b', 1, 'score_10p', 'perclass-only (10%)', 'R10b_s1_external_eval', 0.4657),
    _R('R17a_s0', 'R17a', 0, 'labeled_only_1p', 'labeled-only (1%)', 'R17a_external_eval', 0.4235),
    _R('R17b_s0', 'R17b', 0, 'score_1p', 'perclass-only (1%)', 'R17b_external_eval', 0.4264),
]:
    RUN_REGISTRY[_r['key']] = _r

SCORE_TAGS = ['R6', 'R7', 'R7b']
CAMERA_TAGS = ['R8b', 'R8c', 'R8d', 'R7c']

GROUPS = {
    'all': list(RUN_REGISTRY),
    'main5p': [k for k, v in RUN_REGISTRY.items()
               if v['family'] in ('labeled_only', 'score', 'camera', 'hybrid')],
    'score': [k for k, v in RUN_REGISTRY.items() if v['family'] == 'score'],
    'camera': [k for k, v in RUN_REGISTRY.items() if v['family'] == 'camera'],
    'labeled': [k for k, v in RUN_REGISTRY.items() if v['family'] == 'labeled_only'],
    'hybrid': [k for k, v in RUN_REGISTRY.items() if v['family'] == 'hybrid'],
    'families': [k for k, v in RUN_REGISTRY.items() if v['family'] in ('score', 'camera')],
    'smoke': ['R6_s0', 'R8b_s0', 'R4b_s0'],
}


def resolve_runs(spec: str) -> List[str]:
    out: List[str] = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok in GROUPS:
            for k in GROUPS[tok]:
                if k not in out:
                    out.append(k)
        elif tok in RUN_REGISTRY:
            if tok not in out:
                out.append(tok)
        else:
            raise SystemExit('Unknown run/group %r. Known runs: %s ; groups: %s'
                             % (tok, ','.join(RUN_REGISTRY), ','.join(GROUPS)))
    return out


# --------------------------------------------------------------------------------------
# Devkit injection
# --------------------------------------------------------------------------------------

def _devkit():
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import DetectionEval
    return config_factory, DetectionBox, DetectionEval


class InjectedDetectionEval:
    """Factory returning a DetectionEval subclass whose __init__ takes pre-built boxes.

    We deliberately do NOT reimplement `evaluate()`: it is inherited verbatim from the
    official devkit, so `accumulate` / `calc_ap` / `calc_tp` / `DetectionMetrics` are the
    official ones.  We only bypass the disk-loading `__init__`.
    """
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            _, _, DetectionEval = _devkit()

            class _E(DetectionEval):
                def __init__(self, cfg, gt_boxes, pred_boxes, meta=None):
                    self.cfg = cfg
                    self.gt_boxes = gt_boxes
                    self.pred_boxes = pred_boxes
                    self.meta = meta or {}
                    self.verbose = False
                    self.output_dir = None
                    self.eval_set = EVAL_SET
                    self.result_path = None
                    self.nusc = None
                    # Same consistency contract the official __init__ asserts.
                    assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
                        'Samples in resampled split do not match samples in predictions.'
                    self.sample_tokens = self.gt_boxes.sample_tokens

            cls._cls = _E
        return cls._cls


def evaluate_boxes(cfg, gt_boxes, pred_boxes) -> Tuple[float, float]:
    """Run the official evaluation on an already-restricted, already-filtered box pair."""
    E = InjectedDetectionEval.get()
    metrics, _ = E(cfg, gt_boxes, pred_boxes).evaluate()
    s = metrics.serialize()
    return float(s['nd_score']), float(s['mean_ap'])


# --------------------------------------------------------------------------------------
# Prep phase (needs the NuScenes DB; runs once, cached to disk)
# --------------------------------------------------------------------------------------

_PREP_NUSC = None
_PREP_ARGS: dict = {}


def _prep_init(dataroot: str, version: str, prep_dir: str):
    global _PREP_ARGS
    _PREP_ARGS = dict(dataroot=dataroot, version=version, prep_dir=prep_dir)


def _get_nusc():
    global _PREP_NUSC
    if _PREP_NUSC is None:
        from nuscenes import NuScenes
        t = time.time()
        _PREP_NUSC = NuScenes(version=_PREP_ARGS['version'],
                              dataroot=_PREP_ARGS['dataroot'], verbose=False)
        sys.stderr.write('[prep pid %d] NuScenes DB loaded in %.1fs\n' % (os.getpid(), time.time() - t))
    return _PREP_NUSC


def gt_cache_path(prep_dir: str) -> str:
    return os.path.join(prep_dir, 'gt_val_filtered.pkl')


def run_cache_path(prep_dir: str, key: str) -> str:
    return os.path.join(prep_dir, 'pred_%s.pkl' % key)


def _prep_gt(_=None) -> str:
    """Filtered GT for the full val split + scene -> sample_token map."""
    from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
    from nuscenes.utils.splits import create_splits_scenes
    config_factory, DetectionBox, _ = _devkit()

    out = gt_cache_path(_PREP_ARGS['prep_dir'])
    if os.path.exists(out):
        return out
    nusc = _get_nusc()
    cfg = config_factory('detection_cvpr_2019')
    t = time.time()
    gt = load_gt(nusc, EVAL_SET, DetectionBox, verbose=False)
    gt = add_center_dist(nusc, gt)
    gt = filter_eval_boxes(nusc, gt, cfg.class_range, verbose=False)
    assert len(gt.sample_tokens) == N_VAL_SAMPLES, \
        'Expected %d val keyframes, got %d' % (N_VAL_SAMPLES, len(gt.sample_tokens))

    val_scenes = set(create_splits_scenes()[EVAL_SET])
    scene2tokens: "OrderedDict[str, List[str]]" = OrderedDict()
    for tok in gt.sample_tokens:
        sc = nusc.get('scene', nusc.get('sample', tok)['scene_token'])['name']
        scene2tokens.setdefault(sc, []).append(tok)
    assert set(scene2tokens) == val_scenes, 'Scene set mismatch vs official val split'
    assert len(scene2tokens) == N_VAL_SCENES, 'Expected %d val scenes' % N_VAL_SCENES
    # Deterministic ordering: scene names sorted, tokens in DB order within a scene.
    scene2tokens = OrderedDict((s, scene2tokens[s]) for s in sorted(scene2tokens))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'wb') as f:
        pickle.dump({'gt': gt, 'scene2tokens': scene2tokens,
                     'sample_tokens': list(gt.sample_tokens)}, f, protocol=5)
    os.replace(out + '.tmp', out)
    sys.stderr.write('[prep] GT cached (%d boxes, %.1fs) -> %s\n'
                     % (len(gt.all), time.time() - t, out))
    return out


def _prep_run(key: str) -> Tuple[str, str, float, int]:
    from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
    config_factory, DetectionBox, _ = _devkit()

    out = run_cache_path(_PREP_ARGS['prep_dir'], key)
    if os.path.exists(out):
        return key, out, 0.0, -1
    info = RUN_REGISTRY[key]
    rp = os.path.join(WORK_DIRS, info['dir'], RESULTS_REL)
    if not os.path.exists(rp):
        raise FileNotFoundError('Missing results json for %s: %s' % (key, rp))
    nusc = _get_nusc()
    cfg = config_factory('detection_cvpr_2019')
    t = time.time()
    pred, meta = load_prediction(rp, cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    pred = add_center_dist(nusc, pred)
    pred = filter_eval_boxes(nusc, pred, cfg.class_range, verbose=False)
    nbox = len(pred.all)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'wb') as f:
        pickle.dump({'pred': pred, 'meta': meta, 'key': key,
                     'sample_tokens': list(pred.sample_tokens)}, f, protocol=5)
    os.replace(out + '.tmp', out)
    dt = time.time() - t
    sys.stderr.write('[prep] %-10s %d boxes / %d samples in %.1fs\n'
                     % (key, nbox, len(pred.sample_tokens), dt))
    return key, out, dt, nbox


def run_prep(keys: Sequence[str], prep_dir: str, dataroot: str, version: str,
             prep_workers: int) -> None:
    todo_gt = not os.path.exists(gt_cache_path(prep_dir))
    todo_runs = [k for k in keys if not os.path.exists(run_cache_path(prep_dir, k))]
    if not todo_gt and not todo_runs:
        print('[prep] all caches present, skipping prep (NuScenes DB not loaded)')
        return
    os.makedirs(prep_dir, exist_ok=True)
    print('[prep] need GT=%s, %d run(s): %s' % (todo_gt, len(todo_runs), ','.join(todo_runs)))
    tasks = (['__GT__'] if todo_gt else []) + list(todo_runs)
    nw = max(1, min(prep_workers, len(tasks)))
    t0 = time.time()
    if nw == 1:
        _prep_init(dataroot, version, prep_dir)
        for tk in tasks:
            _prep_gt() if tk == '__GT__' else _prep_run(tk)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(nw, initializer=_prep_init, initargs=(dataroot, version, prep_dir)) as pool:
            for _ in pool.imap_unordered(_prep_dispatch, tasks):
                pass
    print('[prep] done in %.1fs' % (time.time() - t0))


def _prep_dispatch(tk: str):
    return _prep_gt() if tk == '__GT__' else _prep_run(tk)


# --------------------------------------------------------------------------------------
# Scene resampling
# --------------------------------------------------------------------------------------

def scene_draw_counts(b: int, n_scenes: int) -> np.ndarray:
    """Multiplicity vector for replicate b.

    Depends on b ONLY (not on the run), which is what makes replicates paired across runs.
    """
    rng = np.random.default_rng(b)
    idx = rng.integers(0, n_scenes, size=n_scenes)
    return np.bincount(idx, minlength=n_scenes)


def build_resampled_boxes(gt, pred, scene_names: List[str], scene2tokens: Dict[str, List[str]],
                          counts: np.ndarray):
    """Build the (gt, pred) EvalBoxes pair for one bootstrap replicate.

    Multiplicity is realised by RE-KEYING (see module docstring, point 3):
      copy 0 of token T keeps key T and shares the original box lists (no copying);
      copy j>=1 gets key "T#j"; GT boxes are shared by reference, prediction boxes are
      shallow-copied with `sample_token` rewritten so the devkit's
      `gt_boxes[pred.sample_token]` lookup and its `(sample_token, gt_idx)` "taken" set
      resolve to the correct copy.
    """
    from nuscenes.eval.common.data_classes import EvalBoxes
    g = EvalBoxes()
    p = EvalBoxes()
    gb, pb = g.boxes, p.boxes
    gsrc, psrc = gt.boxes, pred.boxes
    for si in range(len(scene_names)):
        m = int(counts[si])
        if m == 0:
            continue
        toks = scene2tokens[scene_names[si]]
        for j in range(m):
            if j == 0:
                for tok in toks:
                    gb[tok] = gsrc[tok]
                    pb[tok] = psrc[tok]
            else:
                suf = '#%d' % j
                for tok in toks:
                    k = tok + suf
                    gb[k] = gsrc[tok]
                    new = []
                    for box in psrc[tok]:
                        c = copy.copy(box)
                        c.sample_token = k
                        new.append(c)
                    pb[k] = new
    return g, p


# --------------------------------------------------------------------------------------
# Bootstrap worker
# --------------------------------------------------------------------------------------

_W: dict = {}


def _boot_init(prep_dir: str):
    """Load the (small) shared state once per worker: config, GT, scene map."""
    config_factory, _, _ = _devkit()
    with open(gt_cache_path(prep_dir), 'rb') as f:
        d = pickle.load(f)
    _W['cfg'] = config_factory('detection_cvpr_2019')
    _W['gt'] = d['gt']
    _W['scene2tokens'] = d['scene2tokens']
    _W['scene_names'] = list(d['scene2tokens'].keys())
    _W['prep_dir'] = prep_dir
    _W['pred_key'] = None
    _W['pred'] = None


def _get_pred(key: str):
    """LRU(1) cache of one run's filtered predictions (largest object in the worker)."""
    if _W['pred_key'] == key:
        return _W['pred']
    _W['pred'] = None
    _W['pred_key'] = None
    gc.collect()
    with open(run_cache_path(_W['prep_dir'], key), 'rb') as f:
        d = pickle.load(f)
    _W['pred'] = d['pred']
    _W['pred_key'] = key
    return _W['pred']


def _task(args) -> dict:
    """One task = (run key, list of replicate indices).  b == -1 means 'no resampling'."""
    key, bs = args
    cfg = _W['cfg']
    gt = _W['gt']
    pred = _get_pred(key)
    scene_names = _W['scene_names']
    scene2tokens = _W['scene2tokens']
    n_scenes = len(scene_names)

    res = {'key': key, 'nds': {}, 'map': {}, 'secs': {}, 'ntok': {}}
    for b in bs:
        t0 = time.time()
        if b < 0:
            # Point estimate: full val, every scene exactly once, no re-keying.
            g, p = gt, pred
            ntok = len(g.boxes)
        else:
            counts = scene_draw_counts(b, n_scenes)
            g, p = build_resampled_boxes(gt, pred, scene_names, scene2tokens, counts)
            ntok = len(g.boxes)
        nds, mp = evaluate_boxes(cfg, g, p)
        res['nds'][b] = nds
        res['map'][b] = mp
        res['secs'][b] = time.time() - t0
        res['ntok'][b] = ntok
        if b >= 0:
            del g, p
    return res


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def pct_ci(x: np.ndarray, alpha: float = 0.05) -> List[float]:
    if x.size == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(x, 100 * alpha / 2)),
            float(np.percentile(x, 100 * (1 - alpha / 2)))]


def summarize(x: np.ndarray) -> dict:
    return {
        'boot_mean': float(np.mean(x)) if x.size else float('nan'),
        'boot_se': float(np.std(x, ddof=1)) if x.size > 1 else float('nan'),
        'ci95': pct_ci(x),
        'B': int(x.size),
    }


def two_sided_bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap 'sign' p-value: 2*min(P(d<=0), P(d>=0)), clipped to [1/B, 1]."""
    if diffs.size == 0:
        return float('nan')
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return float(min(1.0, max(p, 1.0 / diffs.size)))


def exact_sign_test(diffs: Sequence[float]) -> dict:
    """Exact two-sided sign test on paired differences (zeros dropped)."""
    d = [x for x in diffs if x != 0 and not math.isnan(x)]
    n = len(d)
    if n == 0:
        return {'n': 0, 'n_pos': 0, 'p_two_sided': float('nan'),
                'min_attainable_p': float('nan')}
    k = sum(1 for x in d if x > 0)

    def cdf(j):
        return sum(math.comb(n, i) for i in range(0, j + 1)) / (2.0 ** n)
    p = 2.0 * min(cdf(k), 1.0 - cdf(k - 1) if k > 0 else 1.0)
    return {'n': n, 'n_pos': k, 'p_two_sided': float(min(1.0, p)),
            'min_attainable_p': float(2.0 / (2.0 ** n))}


def wilcoxon_test(diffs: Sequence[float]) -> dict:
    try:
        from scipy.stats import wilcoxon
    except Exception as e:                                  # pragma: no cover
        return {'available': False, 'reason': repr(e)}
    d = [float(x) for x in diffs if not math.isnan(x)]
    if len(d) < 1:
        return {'available': False, 'reason': 'no data'}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            st, p = wilcoxon(d, alternative='two-sided', zero_method='wilcox',
                             mode='exact' if len(d) <= 25 else 'auto')
        return {'available': True, 'n': len(d), 'statistic': float(st),
                'p_two_sided': float(p),
                'min_attainable_p': float(2.0 / (2.0 ** len(d)))}
    except Exception as e:
        return {'available': False, 'reason': repr(e)}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def _json_safe(o):
    """NaN/Inf -> null so the output is strict, parser-agnostic JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def make_tasks(keys: Sequence[str], B: int, chunk: int) -> List[Tuple[str, List[int]]]:
    tasks: List[Tuple[str, List[int]]] = []
    for k in keys:
        tasks.append((k, [-1]))                       # point estimate
        for s in range(0, B, chunk):
            tasks.append((k, list(range(s, min(s + chunk, B)))))
    # Interleave runs so that all workers do not hammer the same pickle at once and so
    # that a worker's LRU(1) prediction cache is reused within a run's chunks.
    tasks.sort(key=lambda t: (t[1][0], t[0]))
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Scene-level block bootstrap CIs for nuScenes NDS (cached results only).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--runs', default='main5p',
                    help='Comma list of run keys and/or groups: %s' % ', '.join(GROUPS))
    ap.add_argument('--B', type=int, default=1000, help='Number of bootstrap replicates')
    ap.add_argument('--workers', type=int, default=64, help='Bootstrap processes (<=96)')
    ap.add_argument('--prep-workers', type=int, default=4,
                    help='Processes for the prep phase (each holds a ~8 GB NuScenes DB)')
    ap.add_argument('--chunk', type=int, default=0,
                    help='Replicates per task (0 = auto: ~3 tasks per worker)')
    ap.add_argument('--out', default=DEFAULT_OUT, help='Output JSON path')
    ap.add_argument('--prep-dir', default=DEFAULT_PREP_DIR, help='Prep cache directory')
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--version', default=DEFAULT_VERSION)
    ap.add_argument('--smoke', action='store_true',
                    help='Fast path: B=20, runs=R6_s0,R8b_s0,R4b_s0, out=cache/bootstrap_ci_smoke.json')
    ap.add_argument('--prep-only', action='store_true', help='Build caches and exit')
    ap.add_argument('--tolerance', type=float, default=5e-4,
                    help='Max |recomputed point NDS - paper NDS| tolerated')
    args = ap.parse_args(argv)

    if args.smoke:
        if args.runs == 'main5p':
            args.runs = 'smoke'
        if args.B == 1000:
            args.B = 20
        if args.out == DEFAULT_OUT:
            args.out = os.path.join(DEFAULT_CACHE, 'bootstrap_ci_smoke.json')
        if args.workers == 64:
            args.workers = min(24, os.cpu_count() or 8)

    args.workers = max(1, min(96, args.workers))
    keys = resolve_runs(args.runs)
    print('=' * 88)
    print('RUN MAPPING (%d runs)' % len(keys))
    print('%-10s %-6s %-5s %-18s %-45s %s' % ('key', 'tag', 'seed', 'family', 'dir', 'paper NDS'))
    for k in keys:
        v = RUN_REGISTRY[k]
        print('%-10s %-6s %-5d %-18s %-45s %.4f' % (k, v['tag'], v['seed'], v['family'],
                                                    v['dir'], v['expected_nds']))
    print('=' * 88)

    # -- cross-check the registry against the cached metrics_summary.json ---------------
    print('Cross-check vs cached metrics_summary.json:')
    for k in keys:
        v = RUN_REGISTRY[k]
        mp_ = os.path.join(WORK_DIRS, v['dir'], METRICS_REL)
        if os.path.exists(mp_):
            with open(mp_) as f:
                nds = json.load(f)['nd_score']
            ok = abs(nds - v['expected_nds']) < 1e-4
            print('  %-10s cached %.6f  paper %.4f  %s' % (k, nds, v['expected_nds'],
                                                           'OK' if ok else '*** MISMATCH ***'))
            if not ok:
                raise SystemExit('Registry/dir mapping is wrong for %s' % k)
        else:
            print('  %-10s (no cached metrics_summary.json)' % k)

    run_prep(keys, args.prep_dir, args.dataroot, args.version, args.prep_workers)
    if args.prep_only:
        return 0

    chunk = args.chunk or max(1, min(args.B, math.ceil(args.B * len(keys) / max(1, 3 * args.workers))))
    tasks = make_tasks(keys, args.B, chunk)
    print('Bootstrap: B=%d, %d runs, chunk=%d, %d tasks, %d workers'
          % (args.B, len(keys), chunk, len(tasks), args.workers))

    import multiprocessing as mp
    ctx = mp.get_context('fork')
    nds = {k: {} for k in keys}
    maps = {k: {} for k in keys}
    secs = {k: [] for k in keys}
    ntok = {}
    t0 = time.time()
    done = 0
    with ctx.Pool(min(args.workers, len(tasks)), initializer=_boot_init,
                  initargs=(args.prep_dir,)) as pool:
        for r in pool.imap_unordered(_task, tasks):
            k = r['key']
            nds[k].update(r['nds'])
            maps[k].update(r['map'])
            secs[k].extend(v for b, v in r['secs'].items() if b >= 0)
            ntok.update(r['ntok'])
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print('  ... %d/%d tasks, %.1fs elapsed' % (done, len(tasks), time.time() - t0),
                      flush=True)
    wall = time.time() - t0
    print('Bootstrap wall clock: %.1fs' % wall)

    # ---------------------------------------------------------------------------------
    # Point-estimate verification (THE critical check)
    # ---------------------------------------------------------------------------------
    print('\nPOINT ESTIMATE VERIFICATION (no resampling, full val=%d)' % N_VAL_SAMPLES)
    print('%-10s %-14s %-14s %-12s %s' % ('key', 'recomputed', 'paper', 'abs diff', 'status'))
    verify = {}
    bad = []
    for k in keys:
        got = nds[k].get(-1, float('nan'))
        exp = RUN_REGISTRY[k]['expected_nds']
        d = abs(got - exp)
        ok = d <= args.tolerance
        verify[k] = {'recomputed_nds': got, 'paper_nds': exp, 'abs_diff': d, 'ok': bool(ok)}
        print('%-10s %-14.10f %-14.4f %-12.2e %s' % (k, got, exp, d, 'OK' if ok else '*** FAIL ***'))
        if not ok:
            bad.append(k)
    if bad:
        print('\n!!! POINT ESTIMATES DO NOT REPRODUCE for: %s' % ','.join(bad))
        print('!!! The token restriction is wrong -- do NOT trust the CIs below.')

    # ---------------------------------------------------------------------------------
    # Assemble results
    # ---------------------------------------------------------------------------------
    bs_all = list(range(args.B))

    def arr(k):
        return np.array([nds[k][b] for b in bs_all if b in nds[k]], dtype=float)

    def arr_common(k1, k2):
        common = [b for b in bs_all if b in nds[k1] and b in nds[k2]]
        return (np.array([nds[k1][b] for b in common]),
                np.array([nds[k2][b] for b in common]), common)

    out: dict = {
        'meta': {
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'script': os.path.abspath(__file__),
            'eval_set': EVAL_SET,
            'n_val_samples': N_VAL_SAMPLES,
            'n_val_scenes': N_VAL_SCENES,
            'B_requested': args.B,
            'runs': keys,
            'workers': args.workers,
            'chunk': chunk,
            'wall_clock_s': wall,
            'resampling_unit': 'scene (block bootstrap over the 150 nuScenes val scenes)',
            'multiplicity': 're-keyed ("T#j"); duplicated scenes are physically duplicated '
                            'in both GT and predictions, not weighted',
            'pairing': 'numpy.random.default_rng(b) over sorted val scene names; identical '
                       'resample across all runs within replicate b',
            'metric_backend': 'nuscenes.eval.detection.evaluate.DetectionEval.evaluate() '
                              '(official devkit, unmodified)',
            'caveat': 'Evaluation-set uncertainty only. Does NOT estimate training-seed '
                      'variance; the 3 training seeds are reported separately.',
        },
        'point_estimate_verification': verify,
        'runs': {},
        'replicate_nds': {k: {str(b): nds[k][b] for b in sorted(nds[k])} for k in keys},
        'pairs': [],
        'family_gap': {},
        'within_family': {},
        'timing': {},
    }

    for k in keys:
        v = RUN_REGISTRY[k]
        x = arr(k)
        s = summarize(x)
        out['runs'][k] = {
            'tag': v['tag'], 'seed': v['seed'], 'family': v['family'], 'label': v['label'],
            'dir': v['dir'],
            'point_nds': nds[k].get(-1, float('nan')),
            'point_map': maps[k].get(-1, float('nan')),
            'paper_nds': v['expected_nds'],
            'boot_mean_nds': s['boot_mean'], 'boot_se_nds': s['boot_se'],
            'ci95_nds': s['ci95'], 'B_completed': s['B'],
            'bias_boot_mean_minus_point': s['boot_mean'] - nds[k].get(-1, float('nan')),
        }
        if secs[k]:
            out['timing'][k] = {
                'sec_per_replicate_mean': float(np.mean(secs[k])),
                'sec_per_replicate_median': float(np.median(secs[k])),
                'n_timed': len(secs[k]),
            }

    by_tag_seed = {(v['tag'], v['seed']): k for k, v in RUN_REGISTRY.items() if k in keys}

    # -- (seed, score, camera) paired differences -------------------------------------
    seeds = sorted({RUN_REGISTRY[k]['seed'] for k in keys})
    for seed in seeds:
        for st in SCORE_TAGS:
            ks = by_tag_seed.get((st, seed))
            if ks is None:
                continue
            for ct in CAMERA_TAGS:
                kc = by_tag_seed.get((ct, seed))
                if kc is None:
                    continue
                a, b_, common = arr_common(ks, kc)
                d = a - b_
                pt = nds[ks].get(-1, float('nan')) - nds[kc].get(-1, float('nan'))
                s = summarize(d)
                flip = float(np.mean(np.sign(d) != np.sign(pt))) if d.size else float('nan')
                out['pairs'].append({
                    'seed': seed, 'score_run': ks, 'camera_run': kc,
                    'score_tag': st, 'camera_tag': ct,
                    'score_label': RUN_REGISTRY[ks]['label'],
                    'camera_label': RUN_REGISTRY[kc]['label'],
                    'point_diff_nds': pt,
                    'boot_mean_diff': s['boot_mean'], 'boot_se_diff': s['boot_se'],
                    'ci95_diff': s['ci95'], 'B': s['B'],
                    'sign_flip_frac': flip,
                    'frac_diff_le_0': float(np.mean(d <= 0)) if d.size else float('nan'),
                    'two_sided_bootstrap_p': two_sided_bootstrap_p(d),
                })

    # -- family-level statistic actually claimed: min(score) - max(camera) --------------
    for seed in seeds:
        sk = [by_tag_seed[(t, seed)] for t in SCORE_TAGS if (t, seed) in by_tag_seed]
        ck = [by_tag_seed[(t, seed)] for t in CAMERA_TAGS if (t, seed) in by_tag_seed]
        if not sk or not ck:
            continue
        common = [b for b in bs_all if all(b in nds[k] for k in sk + ck)]
        if not common:
            continue
        S = np.array([[nds[k][b] for b in common] for k in sk])   # (nscore, B)
        C = np.array([[nds[k][b] for b in common] for k in ck])
        gap = S.min(axis=0) - C.max(axis=0)
        pt = (min(nds[k][-1] for k in sk) - max(nds[k][-1] for k in ck))
        s = summarize(gap)
        out['family_gap'][str(seed)] = {
            'statistic': 'min_NDS(score-based) - max_NDS(camera-support)',
            'score_runs': sk, 'camera_runs': ck,
            'point_gap': pt,
            'boot_mean_gap': s['boot_mean'], 'boot_se_gap': s['boot_se'],
            'ci95_gap': s['ci95'], 'B': s['B'],
            'frac_gap_le_0': float(np.mean(gap <= 0)),
            'two_sided_bootstrap_p': two_sided_bootstrap_p(gap),
            'argmin_score_point': min(sk, key=lambda k: nds[k][-1]),
            'argmax_camera_point': max(ck, key=lambda k: nds[k][-1]),
        }

    # -- within-family D1 check: perclass_only (R7b) - perclass_boost (R7) --------------
    wf: dict = {'comparison': 'R7b(perclass-only) - R7(perclass+boost)', 'per_seed': {}}
    per_seed_point = []
    for seed in seeds:
        k1 = by_tag_seed.get(('R7b', seed))
        k2 = by_tag_seed.get(('R7', seed))
        if k1 is None or k2 is None:
            continue
        a, b_, common = arr_common(k1, k2)
        d = a - b_
        pt = nds[k1].get(-1, float('nan')) - nds[k2].get(-1, float('nan'))
        per_seed_point.append(pt)
        s = summarize(d)
        wf['per_seed'][str(seed)] = {
            'run_a': k1, 'run_b': k2,
            'point_diff_nds': pt,
            'boot_mean_diff': s['boot_mean'], 'boot_se_diff': s['boot_se'],
            'ci95_diff': s['ci95'], 'B': s['B'],
            'ci_contains_zero': bool(s['ci95'][0] <= 0.0 <= s['ci95'][1]),
            'sign_flip_frac': float(np.mean(np.sign(d) != np.sign(pt))) if d.size else float('nan'),
            'two_sided_bootstrap_p': two_sided_bootstrap_p(d),
        }
    wf['across_seeds'] = {
        'point_diffs': per_seed_point,
        'sign_test': exact_sign_test(per_seed_point),
        'wilcoxon_signed_rank': wilcoxon_test(per_seed_point),
        'note': 'With only 3 seeds the smallest attainable two-sided p is 0.25, so a '
                'non-significant result here is uninformative about the null; the '
                'per-seed bootstrap CIs are the usable evidence for D1.',
    }
    # Also give the other two within-score-family contrasts for completeness.
    wf['other_within_score_contrasts'] = {}
    for ta, tb in [('R7b', 'R6'), ('R7', 'R6')]:
        rows = {}
        for seed in seeds:
            k1, k2 = by_tag_seed.get((ta, seed)), by_tag_seed.get((tb, seed))
            if k1 is None or k2 is None:
                continue
            a, b_, _ = arr_common(k1, k2)
            d = a - b_
            s = summarize(d)
            rows[str(seed)] = {
                'point_diff_nds': nds[k1].get(-1, float('nan')) - nds[k2].get(-1, float('nan')),
                'boot_mean_diff': s['boot_mean'], 'ci95_diff': s['ci95'],
                'ci_contains_zero': bool(s['ci95'][0] <= 0.0 <= s['ci95'][1]),
                'two_sided_bootstrap_p': two_sided_bootstrap_p(d),
            }
        wf['other_within_score_contrasts']['%s_minus_%s' % (ta, tb)] = rows
    out['within_family'] = wf

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(_json_safe(out), f, indent=2)
    print('\nWrote %s' % args.out)

    # ---------------------------------------------------------------------------------
    # Console summary + budget projection
    # ---------------------------------------------------------------------------------
    print('\nPER-RUN BOOTSTRAP SUMMARY')
    print('%-10s %-9s %-9s %-9s %-22s %-5s %s' %
          ('key', 'point', 'bootmean', 'SE', '95% CI', 'B', 's/rep'))
    for k in keys:
        r = out['runs'][k]
        sp = out['timing'].get(k, {}).get('sec_per_replicate_mean', float('nan'))
        print('%-10s %-9.4f %-9.4f %-9.4f [%7.4f, %7.4f]      %-5d %.1f' %
              (k, r['point_nds'], r['boot_mean_nds'], r['boot_se_nds'],
               r['ci95_nds'][0], r['ci95_nds'][1], r['B_completed'], sp))

    if out['pairs']:
        print('\nPAIRED score-vs-camera DIFFERENCES (NDS)')
        print('%-5s %-9s %-9s %-9s %-9s %-22s %-9s %s' %
              ('seed', 'score', 'camera', 'point', 'bootmean', '95% CI', 'flip%', 'p'))
        for pr in out['pairs']:
            print('%-5d %-9s %-9s %-9.4f %-9.4f [%7.4f, %7.4f]      %-9.3f %.4f' %
                  (pr['seed'], pr['score_run'], pr['camera_run'], pr['point_diff_nds'],
                   pr['boot_mean_diff'], pr['ci95_diff'][0], pr['ci95_diff'][1],
                   100 * pr['sign_flip_frac'], pr['two_sided_bootstrap_p']))

    if out['family_gap']:
        print('\nFAMILY GAP  min(score) - max(camera)')
        for seed, g in sorted(out['family_gap'].items()):
            print('  seed %s: point %.4f  boot mean %.4f  95%% CI [%.4f, %.4f]  '
                  'frac<=0 %.4f  p %.4f'
                  % (seed, g['point_gap'], g['boot_mean_gap'], g['ci95_gap'][0],
                     g['ci95_gap'][1], g['frac_gap_le_0'], g['two_sided_bootstrap_p']))

    if out['within_family']['per_seed']:
        print('\nWITHIN-FAMILY D1  %s' % out['within_family']['comparison'])
        for seed, g in sorted(out['within_family']['per_seed'].items()):
            print('  seed %s: point %+.4f  boot mean %+.4f  95%% CI [%+.4f, %+.4f]  '
                  'contains 0: %s  p %.4f'
                  % (seed, g['point_diff_nds'], g['boot_mean_diff'], g['ci95_diff'][0],
                     g['ci95_diff'][1], g['ci_contains_zero'], g['two_sided_bootstrap_p']))
        st = out['within_family']['across_seeds']['sign_test']
        wx = out['within_family']['across_seeds']['wilcoxon_signed_rank']
        print('  across seeds: sign test n=%s p=%s | wilcoxon %s'
              % (st['n'], st['p_two_sided'], wx.get('p_two_sided', wx.get('reason'))))

    all_secs = [s for k in keys for s in secs[k]]
    if all_secs:
        mean_s = float(np.mean(all_secs))
        print('\nTIMING: %.2f s per (run, replicate) averaged over %d timed replicates'
              % (mean_s, len(all_secs)))
        for nruns, lab in [(len(keys), 'this run set'), (26, 'all 26 5%-label runs'),
                           (34, 'all 34 runs')]:
            core_s = 1000 * nruns * mean_s
            print('  projection B=1000 x %2d runs (%s): %.0f core-s = %.1f core-h '
                  '-> %.1f h at %d workers' %
                  (nruns, lab, core_s, core_s / 3600, core_s / 3600 / args.workers, args.workers))
    return 0


if __name__ == '__main__':
    sys.exit(main())
