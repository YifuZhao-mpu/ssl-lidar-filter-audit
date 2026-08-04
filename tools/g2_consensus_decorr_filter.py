#!/usr/bin/env python3
"""R7e — Per-camera column-wise shuffle of support vectors WITHIN class.

This is the *correct* decorrelation intervention: independently permute
each camera's column (within same class), so box i receives s_FRONT
from box perm_FRONT[i], s_FRONT_LEFT from box perm_FL[i], etc. The
6-camera joint distribution is broken; per-camera marginals and class
composition are preserved.

Unlike R7d (row-wise shuffle of whole 6-vectors, which preserves joint
dependence including inter-camera correlation), this intervention
actually destroys the cross-camera dependence structure.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, CAMERAS, NUSC_TO_COCO, box_supports,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-pseudo', default='/root/BMVC2026/cache/pseudo_5p_s0_raw_full.pkl')
    p.add_argument('--det-cache',  default='/root/BMVC2026/cache/fasterrcnn_v2_train.pkl')
    p.add_argument('--source-infos', default='/root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl')
    p.add_argument('--out', default='/root/BMVC2026/cache/pseudo_5p_s0_filtered/consensus_decorr.pkl')
    p.add_argument('--tau-c', type=float, default=0.15)
    p.add_argument('--gamma', type=float, default=0.20)
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    print('[1] Loading...', flush=True)
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    src = pickle.load(open(args.source_infos, 'rb'))
    cache = pickle.load(open(args.det_cache, 'rb'))
    pseudo = raw['pseudo']
    info_by_tok = {i['token']: i for i in src['infos']}
    det_cache = cache['dets']
    coco_cats = cache['meta']['coco_categories']

    # Step 1: compute supports
    print('[2] Computing supports...', flush=True)
    all_sup = []  # list of (tok, idx, label, sup_row, valid_row)
    for k, (tok, pb) in enumerate(pseudo.items()):
        info = info_by_tok.get(tok); dets = det_cache.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dets is None or N == 0:
            for i in range(N):
                all_sup.append((tok, i, int(pb['labels'][i]),
                                np.zeros(6), np.zeros(6, dtype=bool)))
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats,
                                   args.det_score_thr, args.iou2d_thr)
        for i in range(N):
            all_sup.append((tok, i, int(pb['labels'][i]),
                            sup[i].copy(), valid[i].copy()))
        if (k+1) % 5000 == 0:
            print(f'  {k+1}/{len(pseudo)}', flush=True)
    N_total = len(all_sup)
    print(f'  total boxes: {N_total}', flush=True)

    # Step 2: per-camera column-wise shuffle within class
    print('[3] Column-wise shuffle within class (per camera independently)...', flush=True)
    by_class = {c: [] for c in range(len(NUSC_CLASSES))}
    for idx, item in enumerate(all_sup):
        by_class[item[2]].append(idx)

    shuffled_sup = np.zeros((N_total, 6))
    shuffled_valid = np.zeros((N_total, 6), dtype=bool)
    for cls_idx, indices in by_class.items():
        if not indices: continue
        indices_arr = np.array(indices)
        sup_arr = np.stack([all_sup[i][3] for i in indices])        # (N_class, 6)
        valid_arr = np.stack([all_sup[i][4] for i in indices])      # (N_class, 6)
        # Independent permutation per camera
        for cam in range(6):
            perm = rng.permutation(len(indices))
            shuffled_sup[indices_arr, cam] = sup_arr[perm, cam]
            shuffled_valid[indices_arr, cam] = valid_arr[perm, cam]
        print(f'  {NUSC_CLASSES[cls_idx]:<22s} n={len(indices)}', flush=True)

    # Sanity: post-shuffle correlation should be near zero
    real_off = []; shuf_off = []
    for i in range(6):
        for j in range(6):
            if i == j: continue
            mask = shuffled_valid[:, i] & shuffled_valid[:, j]
            if mask.sum() < 100: continue
            x = shuffled_sup[mask, i]; y = shuffled_sup[mask, j]
            if x.std() > 1e-6 and y.std() > 1e-6:
                shuf_off.append(np.corrcoef(x, y)[0,1])
    if shuf_off:
        print(f'  post-shuffle mean off-diag r = {np.mean(shuf_off):.4f} '
              f'(should be near 0 if decorrelation succeeded)', flush=True)

    def m_visibility(N_valid):
        if N_valid <= 0: return 1
        if N_valid <= 2: return 1
        if N_valid <= 4: return 2
        return 3

    # Step 3: apply consensus rule on decorrelated supports
    print('[4] Apply consensus rule on column-shuffled supports...', flush=True)
    filtered = {}
    n_kept = 0; n_total = 0
    idx = 0
    for tok, pb in pseudo.items():
        N = len(pb['boxes_lidar'])
        if N == 0:
            filtered[tok] = pb
            continue
        labels = pb['labels']
        passthrough = np.array([
            NUSC_CLASSES[int(l)] not in NUSC_TO_COCO
            or not NUSC_TO_COCO[NUSC_CLASSES[int(l)]]
            for l in labels
        ], dtype=bool)

        sup_rows = shuffled_sup[idx:idx+N]
        valid_rows = shuffled_valid[idx:idx+N]
        idx += N

        n_valid = valid_rows.sum(axis=1)
        mean_sup = np.where(n_valid > 0,
                            sup_rows.sum(axis=1) / np.clip(n_valid, 1, None),
                            0.0)
        votes = (sup_rows >= args.gamma) & valid_rows
        vote_count = votes.sum(axis=1)
        m_req = np.array([m_visibility(int(v)) for v in n_valid])
        hard_ok = vote_count >= m_req
        soft_ok = mean_sup >= args.tau_c
        has_valid = n_valid > 0

        keep = soft_ok & hard_ok & has_valid
        keep = keep | passthrough | (~has_valid)
        n_kept += int(keep.sum())
        n_total += N
        filtered[tok] = {
            'boxes_lidar': pb['boxes_lidar'][keep],
            'scores':      pb['scores'][keep],
            'labels':      pb['labels'][keep],
        }

    print(f'[5] Total kept: {n_kept}/{n_total}', flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({'pseudo': filtered, 'meta': {
        'variant': 'consensus_decorr_columnwise',
        'shuffle_seed': args.seed,
        'shuffle_within': 'class',
        'shuffle_type': 'per_camera_independent_permutation',
        'tau_c': args.tau_c, 'gamma': args.gamma,
    }}, open(out, 'wb'))
    print(f'  → {out}')


if __name__ == '__main__':
    main()
