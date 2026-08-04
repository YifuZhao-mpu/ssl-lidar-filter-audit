#!/usr/bin/env python3
"""Per-class kept-box / P / R analysis for R7b, R7c, R7d, R7e on 5% s0.

Output: cache/perclass_kept_pr_5p_s0.json + paper/figures/table_perclass_structural.tex
Reviewer Round-2 request: "compact appendix table with kept boxes and pseudo P/R
by class for R7b/R7c/R7d/R7e" — turns the structural-evidence story from plausible
to much harder to dismiss.
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import NUSC_CLASSES, NUSC_TO_COCO, box_supports

PER_CLASS_THR = {
    'car': 0.35, 'truck': 0.40, 'construction_vehicle': 0.45,
    'bus': 0.30, 'trailer': 0.30, 'barrier': 0.30,
    'motorcycle': 0.40, 'bicycle': 0.40, 'pedestrian': 0.50,
    'traffic_cone': 0.35,
}
NUSC_CLASS_IDX = {c: i for i, c in enumerate(NUSC_CLASSES)}
COCO_DETECTABLE = [c for c in NUSC_CLASSES if NUSC_TO_COCO[c]]


def compute_per_class_metrics(filtered_pseudo, infos):
    """For a filtered pseudo dict, compute per-class kept / TP / FP / FN.
    Match by BEV center distance <= 2.0 m + same class (same protocol as
    g2_filter_pseudo.py and g2_pilot_cocoF1_seed*.py).
    """
    per_class = {c: {'kept': 0, 'tp': 0, 'fp': 0, 'fn': 0} for c in NUSC_CLASSES}
    info_by_tok = {i['token']: i for i in infos}
    proc = 0
    t0 = time.time()
    for tok, pb in filtered_pseudo.items():
        info = info_by_tok.get(tok)
        if info is None:
            continue
        kept_boxes = pb['boxes_lidar']
        kept_labels = pb['labels']
        N = len(kept_boxes)
        for li in kept_labels:
            per_class[NUSC_CLASSES[int(li)]]['kept'] += 1
        gt_boxes = info.get('gt_boxes')
        gt_names = info.get('gt_names')
        if gt_boxes is None or gt_names is None or len(gt_names) == 0:
            for li in kept_labels:
                per_class[NUSC_CLASSES[int(li)]]['fp'] += 1
            continue
        gt_cls_idx = np.array([NUSC_CLASS_IDX.get(g, -1) for g in gt_names])
        if N == 0:
            for j, gn in enumerate(gt_names):
                if gn in NUSC_CLASS_IDX:
                    per_class[gn]['fn'] += 1
            continue
        pc = kept_boxes[:, :2]
        gc = gt_boxes[:, :2]
        diff = pc[:, None, :] - gc[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        same = kept_labels[:, None] == gt_cls_idx[None, :]
        dist = np.where(same, dist, 1e9)
        matched = np.zeros(len(gt_names), dtype=bool)
        for i in range(N):
            j_candidates = np.where(~matched & (dist[i] <= 2.0))[0]
            if j_candidates.size > 0:
                j = j_candidates[np.argmin(dist[i, j_candidates])]
                matched[j] = True
                per_class[NUSC_CLASSES[int(kept_labels[i])]]['tp'] += 1
            else:
                per_class[NUSC_CLASSES[int(kept_labels[i])]]['fp'] += 1
        for j, gn in enumerate(gt_names):
            if not matched[j] and gn in NUSC_CLASS_IDX:
                per_class[gn]['fn'] += 1
        proc += 1
        if proc % 5000 == 0:
            print(f'  {proc}/{len(filtered_pseudo)} tokens ({time.time()-t0:.0f}s)',
                  flush=True)
    return per_class


def build_perclass_only_pseudo(raw_pseudo):
    """Reconstruct R7b perclass-only filter output (no boost) from raw pseudo.
    R7b is class-thresholded by score only, no camera support.
    """
    out = {}
    for tok, pb in raw_pseudo.items():
        scores = pb['scores']
        labels = pb['labels']
        N = len(scores)
        if N == 0:
            out[tok] = pb
            continue
        cls_arr = np.array([NUSC_CLASSES[int(l)] for l in labels])
        thr_arr = np.array([PER_CLASS_THR[c] for c in cls_arr])
        keep = scores >= thr_arr
        out[tok] = {
            'boxes_lidar': pb['boxes_lidar'][keep],
            'scores':      pb['scores'][keep],
            'labels':      pb['labels'][keep],
        }
    return out


def main():
    print('Loading raw pseudo + source infos...', flush=True)
    raw = pickle.load(open('/root/BMVC2026/cache/pseudo_5p_s0_raw_full.pkl', 'rb'))
    src = pickle.load(open('/root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl', 'rb'))
    infos = src['infos']
    raw_pseudo = raw['pseudo']
    print(f'  {len(raw_pseudo)} tokens, {len(infos)} infos', flush=True)

    # Load R7c/R7d/R7e dumps
    print('Loading R7c/R7d/R7e filter dumps...', flush=True)
    cache_dir = Path('/root/BMVC2026/cache/pseudo_5p_s0_filtered')
    r7c = pickle.load(open(cache_dir / 'consensus.pkl', 'rb'))['pseudo']
    r7d = pickle.load(open(cache_dir / 'consensus_shuf.pkl', 'rb'))['pseudo']
    r7e = pickle.load(open(cache_dir / 'consensus_decorr.pkl', 'rb'))['pseudo']

    # Build R7b (perclass-only) inline from raw
    print('Building R7b perclass-only filter from raw...', flush=True)
    r7b = build_perclass_only_pseudo(raw_pseudo)

    filters = {'R7b_perclass_only': r7b, 'R7c_consensus': r7c,
               'R7d_row_shuffle': r7d, 'R7e_col_shuffle': r7e}
    out = {}
    for name, dump in filters.items():
        print(f'\n=== {name} ===', flush=True)
        n_kept = sum(len(p['boxes_lidar']) for p in dump.values())
        print(f'  total kept: {n_kept}', flush=True)
        t0 = time.time()
        pc = compute_per_class_metrics(dump, infos)
        # Add P/R fields
        for c, d in pc.items():
            tp = d['tp']; fp = d['fp']; fn = d['fn']
            d['P'] = 100.0 * tp / max(tp + fp, 1)
            d['R'] = 100.0 * tp / max(tp + fn, 1)
            f1 = 2 * d['P'] * d['R'] / max(d['P'] + d['R'], 1e-6)
            d['F1'] = f1
        out[name] = {'kept_total': n_kept, 'per_class': pc, 'walltime_s': round(time.time()-t0,1)}
        print(f'  walltime: {time.time()-t0:.0f}s', flush=True)
        for c in NUSC_CLASSES:
            d = pc[c]
            print(f'    {c:<22s} kept={d["kept"]:>7d}  P={d["P"]:5.1f}  R={d["R"]:5.1f}  F1={d["F1"]:5.1f}')

    out_path = Path('/root/BMVC2026/cache/perclass_kept_pr_5p_s0.json')
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'\nWrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
