#!/usr/bin/env python3
"""Optimized seed-1 cocoF1 — numpy-vectorized matching + cached supports."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, box_supports, NUSC_TO_COCO,
)

COCO_DETECTABLE = [c for c in NUSC_CLASSES if NUSC_TO_COCO[c]]
print(f'COCO-detectable: {COCO_DETECTABLE}', flush=True)
print(f'Excluded: {[c for c in NUSC_CLASSES if c not in NUSC_TO_COCO]}', flush=True)

PER_CLASS_THR = {
    'car': 0.35, 'truck': 0.40, 'construction_vehicle': 0.45,
    'bus': 0.30, 'trailer': 0.30, 'barrier': 0.30,
    'motorcycle': 0.40, 'bicycle': 0.40, 'pedestrian': 0.50,
    'traffic_cone': 0.35,
}
BOOST_FLOOR = 0.25
BOOST_THR = 0.10
NUSC_CLASS_IDX = {c: i for i, c in enumerate(NUSC_CLASSES)}


def compute_pr_vectorized(pseudo, infos, keep_mask_dict):
    """Vectorized matching: for each token, build cdist matrix between
    kept pseudo and GT, greedy match by distance <= 2.0 m + same class."""
    tp_total = 0; fp_total = 0; fn_total = 0
    per_class = {c: {'tp':0,'fp':0,'fn':0} for c in NUSC_CLASSES}
    info_by_tok = {i['token']: i for i in infos}
    proc = 0
    t0 = time.time()
    for tok, pb in pseudo.items():
        info = info_by_tok.get(tok)
        if info is None:
            continue
        keep = keep_mask_dict.get(tok)
        if keep is None or keep.sum() == 0:
            # Still count GT as fn
            gt_names = info.get('gt_names', [])
            for gn in gt_names:
                if gn in NUSC_CLASS_IDX:
                    fn_total += 1
                    per_class[gn]['fn'] += 1
            continue
        kept_boxes = pb['boxes_lidar'][keep]
        kept_labels = pb['labels'][keep]
        gt_boxes = info.get('gt_boxes')
        gt_names = info.get('gt_names')
        if gt_boxes is None or gt_names is None or len(gt_names) == 0:
            fp_total += len(kept_labels)
            for li in kept_labels:
                per_class[NUSC_CLASSES[int(li)]]['fp'] += 1
            continue

        # Build GT class array
        gt_cls_idx = np.array([NUSC_CLASS_IDX.get(g, -1) for g in gt_names])

        # cdist BEV (x,y) between kept_boxes (N,2) and gt_boxes (M,2)
        pc = kept_boxes[:, :2]
        gc = gt_boxes[:, :2]
        # (N, M) distance matrix
        diff = pc[:, None, :] - gc[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        # Mask: same class required
        same = kept_labels[:, None] == gt_cls_idx[None, :]
        dist = np.where(same, dist, 1e9)

        matched = np.zeros(len(gt_names), dtype=bool)
        # Greedy: iterate kept boxes in order, pick best unmatched GT
        for i in range(len(kept_boxes)):
            j_candidates = np.where(~matched & (dist[i] <= 2.0))[0]
            if j_candidates.size > 0:
                j = j_candidates[np.argmin(dist[i, j_candidates])]
                matched[j] = True
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
            print(f'    {proc}/{len(pseudo)} ({time.time()-t0:.0f}s elapsed)', flush=True)

    P = tp_total / max(tp_total + fp_total, 1)
    R = tp_total / max(tp_total + fn_total, 1)
    F = 2*P*R / max(P+R, 1e-6)
    return P, R, F, tp_total, fp_total, fn_total, per_class


def main():
    print('Loading seed 1 data...', flush=True)
    raw = pickle.load(open('/root/BMVC2026/cache/pseudo_5p_s1_raw_full.pkl','rb'))
    src = pickle.load(open('/root/BMVC2026/infos/nuscenes_infos_train_5p_s1_unlabeled.pkl','rb'))
    pseudo = raw['pseudo']
    infos = src['infos']
    print(f'  {len(pseudo)} pseudo, {len(infos)} infos', flush=True)

    # Cache supports once (used by perclass_only, perclass+boost)
    print('Computing camera support once (shared across perclass variants)...', flush=True)
    det_cache = pickle.load(open('/root/BMVC2026/cache/fasterrcnn_v2_train.pkl','rb'))
    dets = det_cache['dets']
    coco_cats = det_cache['meta']['coco_categories']
    info_by_tok = {i['token']: i for i in infos}
    sup_cache = {}
    t0 = time.time()
    for k, (tok, pb) in enumerate(pseudo.items()):
        info = info_by_tok.get(tok); dr = dets.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dr is None or N == 0:
            sup_cache[tok] = (np.zeros((N, 6)), np.zeros((N, 6), dtype=bool))
            continue
        sup_cache[tok] = box_supports(pb, info, dr, coco_cats, 0.30, 0.30)
        if (k+1) % 5000 == 0:
            print(f'    support {k+1}/{len(pseudo)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  supports cached in {time.time()-t0:.0f}s', flush=True)

    # Build keep masks per variant
    print('Building keep masks...', flush=True)
    masks = {}
    for variant in ['single_thr', 'perclass_only', 'perclass_boost']:
        masks[variant] = {}
        for tok, pb in pseudo.items():
            N = len(pb['boxes_lidar'])
            if N == 0:
                masks[variant][tok] = np.zeros((0,), dtype=bool); continue
            scores = pb['scores']; labels = pb['labels']
            if variant == 'single_thr':
                masks[variant][tok] = scores >= 0.40; continue
            sup, valid = sup_cache[tok]
            n_valid = valid.sum(axis=1)
            sum_sup = sup.sum(axis=1)
            mean_sup = np.where(n_valid > 0, sum_sup / np.clip(n_valid, 1, None), 0.0)
            keep = np.zeros(N, dtype=bool)
            cls_arr = np.array([NUSC_CLASSES[int(l)] for l in labels])
            thr_arr = np.array([PER_CLASS_THR[c] for c in cls_arr])
            passes_pc = scores >= thr_arr
            passes_boost = (scores >= BOOST_FLOOR) & (scores < thr_arr) & (mean_sup >= BOOST_THR)
            if variant == 'perclass_only':
                keep = passes_pc
            elif variant == 'perclass_boost':
                keep = passes_pc | passes_boost
            masks[variant][tok] = keep
        kept_total = sum(int(m.sum()) for m in masks[variant].values())
        print(f'  {variant}: {kept_total} kept', flush=True)

    # Compute per-variant metrics
    results = {}
    for variant, keep_dict in masks.items():
        print(f'\n=== {variant} (vectorized matching) ===', flush=True)
        t0 = time.time()
        P, R, F, tp, fp, fn, pc = compute_pr_vectorized(pseudo, infos, keep_dict)
        print(f'  done in {time.time()-t0:.0f}s; P={P*100:.2f} R={R*100:.2f} F1={F*100:.2f}', flush=True)
        # cocoF1 over COCO-detectable classes
        f1s = []
        for c in COCO_DETECTABLE:
            d = pc.get(c, {})
            t = d.get('tp', 0); fpc = d.get('fp', 0); fnc = d.get('fn', 0)
            p = t / max(t+fpc, 1); r = t / max(t+fnc, 1); f1 = 2*p*r/max(p+r,1e-6)
            f1s.append((c, p, r, f1))
        cocoP = np.mean([x[1] for x in f1s])
        cocoR = np.mean([x[2] for x in f1s])
        cocoF1 = 2*cocoP*cocoR/max(cocoP+cocoR,1e-6)
        print(f'  cocoP={cocoP*100:.2f} cocoR={cocoR*100:.2f} cocoF1={cocoF1*100:.2f}', flush=True)
        results[variant] = {
            'kept': tp+fp, 'P': P*100, 'R': R*100, 'F1': F*100,
            'cocoP': cocoP*100, 'cocoR': cocoR*100, 'cocoF1': cocoF1*100,
            'per_class': {c: {'P':p*100,'R':r*100,'F1':f*100} for c,p,r,f in f1s},
        }

    out = Path('/root/BMVC2026/cache/pilot_cocoF1_seed1.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\n→ {out}', flush=True)
    print('\nSummary (seed 1):', flush=True)
    print(f'  {"filter":<20s} {"kept":>8s} {"P":>6s} {"R":>6s} {"F1":>6s} {"cocoF1":>7s}')
    for v, r in results.items():
        print(f'  {v:<20s} {r["kept"]:>8d} {r["P"]:>6.2f} {r["R"]:>6.2f} {r["F1"]:>6.2f} {r["cocoF1"]:>7.2f}', flush=True)


if __name__ == '__main__':
    main()
