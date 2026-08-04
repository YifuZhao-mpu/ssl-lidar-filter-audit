#!/usr/bin/env python3
"""Compute pilot cocoF1 for score-based filters at seed 1, matching the
seed-0 protocol (Phase B). Outputs JSON with per-filter cocoF1 / cocoP /
cocoR / per-class breakdown.

cocoF1 = harmonic mean of precision and recall computed *only* over the 8
COCO-detectable nuScenes classes (car, truck, construction_vehicle, bus,
trailer, motorcycle, bicycle, pedestrian). barrier and traffic_cone
excluded since COCO 2D detector cannot evaluate them upstream.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, compute_pr, box_supports, CAMERAS, NUSC_TO_COCO,
)

# COCO-detectable nuScenes classes for cocoF1
COCO_DETECTABLE = [c for c in NUSC_CLASSES if c in NUSC_TO_COCO]
print(f'COCO-detectable classes for cocoF1: {COCO_DETECTABLE}')
print(f'Excluded (non-COCO): {[c for c in NUSC_CLASSES if c not in NUSC_TO_COCO]}')

# Per-class thresholds (same as seed 0; tuned on train-GT held-out)
PER_CLASS_THR = {
    'car': 0.35, 'truck': 0.40, 'construction_vehicle': 0.45,
    'bus': 0.30, 'trailer': 0.30, 'barrier': 0.30,
    'motorcycle': 0.40, 'bicycle': 0.40, 'pedestrian': 0.50,
    'traffic_cone': 0.35,
}
BOOST_FLOOR = 0.25
BOOST_THR = 0.10


def load_data(seed=1):
    raw = pickle.load(open(f'/root/BMVC2026/cache/pseudo_5p_s{seed}_raw_full.pkl', 'rb'))
    src = pickle.load(open(f'/root/BMVC2026/infos/nuscenes_infos_train_5p_s{seed}_unlabeled.pkl', 'rb'))
    det_cache = pickle.load(open('/root/BMVC2026/cache/fasterrcnn_v2_train.pkl', 'rb'))
    return raw['pseudo'], src['infos'], det_cache['dets'], det_cache['meta']['coco_categories']


def filter_mask_factory(pseudo, infos, det_cache, coco_cats, variant):
    """Return a function tok -> bool mask of len(pseudo[tok]['boxes_lidar'])."""
    if variant == 'single_thr':
        thr = 0.40
        def mask(tok):
            return pseudo[tok]['scores'] >= thr
        return mask
    # For perclass / boost we need camera support
    info_by_tok = {i['token']: i for i in infos}
    sup_cache = {}
    print(f'  computing camera support for {variant}...')
    proc = 0
    for tok, pb in pseudo.items():
        info = info_by_tok.get(tok); dets = det_cache.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dets is None:
            sup_cache[tok] = (np.zeros((N, 6)), np.zeros((N, 6), dtype=bool))
            continue
        sup_cache[tok] = box_supports(pb, info, dets, coco_cats, 0.30, 0.30)
        proc += 1
        if proc % 5000 == 0:
            print(f'    {proc}/{len(pseudo)}')
    def mask(tok):
        pb = pseudo[tok]
        N = len(pb['boxes_lidar'])
        if N == 0:
            return np.zeros((0,), dtype=bool)
        scores = pb['scores']; labels = pb['labels']
        sup, valid = sup_cache[tok]
        n_valid = valid.sum(axis=1)
        sum_sup = sup.sum(axis=1)
        mean_sup = np.where(n_valid > 0, sum_sup / np.clip(n_valid, 1, None), 0.0)
        keep = np.zeros(N, dtype=bool)
        for i in range(N):
            cls = NUSC_CLASSES[int(labels[i])]
            thr = PER_CLASS_THR[cls]
            passes_pc = scores[i] >= thr
            passes_boost = (scores[i] >= BOOST_FLOOR
                            and scores[i] < thr
                            and mean_sup[i] >= BOOST_THR)
            if variant == 'perclass_only':
                keep[i] = passes_pc
            elif variant == 'perclass_boost':
                keep[i] = passes_pc or passes_boost
        return keep
    return mask


def coco_f1(per_class):
    """Compute cocoF1 from per-class tp/fp/fn dict, averaging F1 over
    COCO-detectable classes only."""
    f1s = []
    for cls in COCO_DETECTABLE:
        if cls not in per_class:
            continue
        d = per_class[cls]
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        f1s.append((cls, p, r, f1))
    macro_p = np.mean([x[1] for x in f1s]) if f1s else 0
    macro_r = np.mean([x[2] for x in f1s]) if f1s else 0
    macro_f1 = 2 * macro_p * macro_r / max(macro_p + macro_r, 1e-6)
    return macro_p, macro_r, macro_f1, f1s


def main():
    print('Loading seed 1 data...')
    pseudo, infos, det_cache, coco_cats = load_data(seed=1)
    print(f'  {len(pseudo)} pseudo samples, {len(infos)} train infos')

    results = {}
    for variant in ['single_thr', 'perclass_only', 'perclass_boost']:
        print(f'\n=== {variant} ===')
        mask_fn = filter_mask_factory(pseudo, infos, det_cache, coco_cats, variant)
        P, R, F1, tp, fp, fn, per_class = compute_pr(pseudo, infos, mask_fn)
        cocoP, cocoR, cocoF1, per_class_breakdown = coco_f1(per_class)
        total_kept = tp + fp
        print(f'  kept={total_kept}  P={P*100:.2f}  R={R*100:.2f}  F1={F1*100:.2f}')
        print(f'  cocoP={cocoP*100:.2f}  cocoR={cocoR*100:.2f}  cocoF1={cocoF1*100:.2f}')
        results[variant] = {
            'kept': total_kept,
            'P': P*100, 'R': R*100, 'F1': F1*100,
            'cocoP': cocoP*100, 'cocoR': cocoR*100, 'cocoF1': cocoF1*100,
            'per_class': {c: {'P': p*100, 'R': r*100, 'F1': f*100} for c,p,r,f in per_class_breakdown},
        }

    out = Path('/root/BMVC2026/cache/pilot_cocoF1_seed1.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n→ {out}')

    print('\nSummary (seed 1):')
    print(f'  {"filter":<20s} {"kept":>8s} {"P":>6s} {"R":>6s} {"F1":>6s} {"cocoF1":>7s}')
    for v, r in results.items():
        print(f'  {v:<20s} {r["kept"]:>8d} {r["P"]:>6.2f} {r["R"]:>6.2f} {r["F1"]:>6.2f} {r["cocoF1"]:>7.2f}')


if __name__ == '__main__':
    main()
