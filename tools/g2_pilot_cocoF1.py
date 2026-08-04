#!/usr/bin/env python3
"""Pilot cocoF1 for the score-based filter family, parameterized by seed.

Generalizes g2_pilot_cocoF1_seed{0_full,1_v2}.py, which were seed-hardcoded.
Table 3 (D1) needs all three seeds: the paper's seed-2 result reverses the
pilot-vs-downstream inversion seen on seeds 0 and 1, and that reversal is the
basis of the demoted D1 claim ("within-family pilot selection is seed-fragile").

Usage:  python tools/g2_pilot_cocoF1.py <seed>       # e.g. 2
Output: cache/pilot_cocoF1_seed<seed>_full.json
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/BMVC2026')
sys.path.insert(0, str(ROOT))
from tools.g2_filter_pseudo import (  # noqa: E402
    NUSC_CLASSES, box_supports, NUSC_TO_COCO,
)

COCO_DETECTABLE = [c for c in NUSC_CLASSES if NUSC_TO_COCO[c]]

# Frozen from the paper (sec:filters); identical across seeds by protocol.
PER_CLASS_THR = {
    'car': 0.35, 'truck': 0.40, 'construction_vehicle': 0.45,
    'bus': 0.30, 'trailer': 0.30, 'barrier': 0.30,
    'motorcycle': 0.40, 'bicycle': 0.40, 'pedestrian': 0.50,
    'traffic_cone': 0.35,
}
SINGLE_THR = 0.40
BOOST_FLOOR = 0.25
BOOST_THR = 0.10
NUSC_CLASS_IDX = {c: i for i, c in enumerate(NUSC_CLASSES)}


def compute_pr_vectorized(pseudo, infos, keep_mask_dict):
    """Greedy same-class BEV matching at 2.0 m, kept-pseudo vs train GT."""
    tp_total = fp_total = fn_total = 0
    per_class = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in NUSC_CLASSES}
    info_by_tok = {i['token']: i for i in infos}
    proc, t0 = 0, time.time()
    for tok, pb in pseudo.items():
        info = info_by_tok.get(tok)
        if info is None:
            continue
        keep = keep_mask_dict.get(tok)
        if keep is None or keep.sum() == 0:
            for gn in info.get('gt_names', []):
                if gn in NUSC_CLASS_IDX:
                    fn_total += 1
                    per_class[gn]['fn'] += 1
            continue
        kept_boxes = pb['boxes_lidar'][keep]
        kept_labels = pb['labels'][keep]
        gt_boxes, gt_names = info.get('gt_boxes'), info.get('gt_names')
        if gt_boxes is None or gt_names is None or len(gt_names) == 0:
            fp_total += len(kept_labels)
            for li in kept_labels:
                per_class[NUSC_CLASSES[int(li)]]['fp'] += 1
            continue

        gt_cls_idx = np.array([NUSC_CLASS_IDX.get(g, -1) for g in gt_names])
        diff = kept_boxes[:, None, :2] - gt_boxes[None, :, :2]
        dist = np.sqrt((diff ** 2).sum(-1))
        same = kept_labels[:, None] == gt_cls_idx[None, :]
        dist = np.where(same, dist, 1e9)

        matched = np.zeros(len(gt_names), dtype=bool)
        for i in range(len(kept_boxes)):
            cand = np.where(~matched & (dist[i] <= 2.0))[0]
            if cand.size > 0:
                matched[cand[np.argmin(dist[i, cand])]] = True
                tp_total += 1
                per_class[NUSC_CLASSES[int(kept_labels[i])]]['tp'] += 1
            else:
                fp_total += 1
                per_class[NUSC_CLASSES[int(kept_labels[i])]]['fp'] += 1
        for j, gn in enumerate(gt_names):
            if not matched[j] and gn in NUSC_CLASS_IDX:
                fn_total += 1
                per_class[gn]['fn'] += 1
        proc += 1
        if proc % 5000 == 0:
            print(f'    {proc}/{len(pseudo)} ({time.time()-t0:.0f}s)', flush=True)

    P = tp_total / max(tp_total + fp_total, 1)
    R = tp_total / max(tp_total + fn_total, 1)
    F = 2 * P * R / max(P + R, 1e-6)
    return P, R, F, tp_total, fp_total, fn_total, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seed', type=int)
    ap.add_argument('--frac', default='5p')
    args = ap.parse_args()
    s, frac = args.seed, args.frac

    print(f'=== pilot cocoF1, {frac} seed {s} ===', flush=True)
    print(f'COCO-detectable: {COCO_DETECTABLE}', flush=True)
    print(f'Excluded: {[c for c in NUSC_CLASSES if not NUSC_TO_COCO[c]]}', flush=True)

    raw = pickle.load(open(ROOT / f'cache/pseudo_{frac}_s{s}_raw_full.pkl', 'rb'))
    src = pickle.load(open(ROOT / f'infos/nuscenes_infos_train_{frac}_s{s}_unlabeled.pkl', 'rb'))
    pseudo, infos = raw['pseudo'], src['infos']
    print(f'  {len(pseudo)} pseudo, {len(infos)} infos', flush=True)

    print('Caching camera support (shared across perclass variants)...', flush=True)
    det_cache = pickle.load(open(ROOT / 'cache/fasterrcnn_v2_train.pkl', 'rb'))
    dets, coco_cats = det_cache['dets'], det_cache['meta']['coco_categories']
    info_by_tok = {i['token']: i for i in infos}
    sup_cache, t0 = {}, time.time()
    for k, (tok, pb) in enumerate(pseudo.items()):
        info, dr = info_by_tok.get(tok), dets.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dr is None or N == 0:
            sup_cache[tok] = (np.zeros((N, 6)), np.zeros((N, 6), dtype=bool))
            continue
        sup_cache[tok] = box_supports(pb, info, dr, coco_cats, 0.30, 0.30)
        if (k + 1) % 5000 == 0:
            print(f'    support {k+1}/{len(pseudo)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  supports cached in {time.time()-t0:.0f}s', flush=True)

    print('Building keep masks...', flush=True)
    masks = {}
    for variant in ['single_thr', 'perclass_only', 'perclass_boost']:
        masks[variant] = {}
        for tok, pb in pseudo.items():
            N = len(pb['boxes_lidar'])
            if N == 0:
                masks[variant][tok] = np.zeros((0,), dtype=bool)
                continue
            scores, labels = pb['scores'], pb['labels']
            if variant == 'single_thr':
                masks[variant][tok] = scores >= SINGLE_THR
                continue
            sup, valid = sup_cache[tok]
            n_valid = valid.sum(axis=1)
            mean_sup = np.where(n_valid > 0,
                                sup.sum(axis=1) / np.clip(n_valid, 1, None), 0.0)
            cls_arr = np.array([NUSC_CLASSES[int(l)] for l in labels])
            thr_arr = np.array([PER_CLASS_THR[c] for c in cls_arr])
            passes_pc = scores >= thr_arr
            passes_boost = ((scores >= BOOST_FLOOR) & (scores < thr_arr)
                            & (mean_sup >= BOOST_THR))
            masks[variant][tok] = passes_pc if variant == 'perclass_only' \
                else (passes_pc | passes_boost)
        print(f'  {variant}: {sum(int(m.sum()) for m in masks[variant].values())} kept',
              flush=True)

    results = {}
    for variant, keep_dict in masks.items():
        print(f'\n=== {variant} ===', flush=True)
        t0 = time.time()
        P, R, F, tp, fp, fn, pc = compute_pr_vectorized(pseudo, infos, keep_dict)
        print(f'  done in {time.time()-t0:.0f}s; P={P*100:.2f} R={R*100:.2f} F1={F*100:.2f}',
              flush=True)
        f1s = []
        for c in COCO_DETECTABLE:
            d = pc.get(c, {})
            t, fpc, fnc = d.get('tp', 0), d.get('fp', 0), d.get('fn', 0)
            p = t / max(t + fpc, 1)
            r = t / max(t + fnc, 1)
            f1s.append((c, p, r, 2 * p * r / max(p + r, 1e-6)))
        cocoP = float(np.mean([x[1] for x in f1s]))
        cocoR = float(np.mean([x[2] for x in f1s]))
        cocoF1 = 2 * cocoP * cocoR / max(cocoP + cocoR, 1e-6)
        print(f'  cocoP={cocoP*100:.2f} cocoR={cocoR*100:.2f} cocoF1={cocoF1*100:.2f}',
              flush=True)
        results[variant] = {
            'kept': tp + fp, 'P': P * 100, 'R': R * 100, 'F1': F * 100,
            'cocoP': cocoP * 100, 'cocoR': cocoR * 100, 'cocoF1': cocoF1 * 100,
            'per_class': {c: {'P': p * 100, 'R': r * 100, 'F1': f * 100}
                          for c, p, r, f in f1s},
        }

    out = ROOT / f'cache/pilot_cocoF1_seed{s}_full.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\n-> {out}', flush=True)
    print(f'\nSummary (seed {s}):', flush=True)
    print(f'  {"filter":<20s} {"kept":>9s} {"P":>6s} {"R":>6s} {"F1":>6s} {"cocoF1":>7s}')
    for v, r in results.items():
        print(f'  {v:<20s} {r["kept"]:>9d} {r["P"]:>6.2f} {r["R"]:>6.2f} '
              f'{r["F1"]:>6.2f} {r["cocoF1"]:>7.2f}', flush=True)


if __name__ == '__main__':
    main()
