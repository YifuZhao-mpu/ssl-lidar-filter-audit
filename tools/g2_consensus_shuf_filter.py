#!/usr/bin/env python3
"""Support-shuffle consensus intervention (Codex Round 5 ask).

For each pseudo-box, compute 6-camera support as usual; then shuffle the
support VECTORS across boxes within the same class. This preserves per-camera
marginal distributions but breaks the across-camera correlation structure.
Apply the same consensus rule. If the family gap shrinks under this control,
view-correlation was a primary driver; if not, other factors dominate.

Outputs: cache/pseudo_5p_s0_filtered/consensus_shuf.pkl
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
    p.add_argument('--out',        default='/root/BMVC2026/cache/pseudo_5p_s0_filtered/consensus_shuf.pkl')
    p.add_argument('--tau-c', type=float, default=0.15)
    p.add_argument('--gamma', type=float, default=0.20)
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f'[1] Loading data...', flush=True)
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    src = pickle.load(open(args.source_infos, 'rb'))
    cache = pickle.load(open(args.det_cache, 'rb'))
    pseudo = raw['pseudo']
    info_by_tok = {i['token']: i for i in src['infos']}
    det_cache = cache['dets']
    coco_cats = cache['meta']['coco_categories']
    print(f'  {len(pseudo)} samples')

    # Step 1: compute supports for every box across all samples
    print('[2] Computing supports across the full dump...', flush=True)
    all_sup = []  # list of (sample_token, box_idx, label_idx, sup_row, valid_row)
    for k, (tok, pb) in enumerate(pseudo.items()):
        info = info_by_tok.get(tok); dets = det_cache.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dets is None or N == 0:
            for i in range(N):
                all_sup.append((tok, i, int(pb['labels'][i]), np.zeros(6), np.zeros(6, dtype=bool)))
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats, args.det_score_thr, args.iou2d_thr)
        for i in range(N):
            all_sup.append((tok, i, int(pb['labels'][i]), sup[i].copy(), valid[i].copy()))
        if (k+1) % 5000 == 0:
            print(f'  support {k+1}/{len(pseudo)}', flush=True)

    print(f'  total boxes: {len(all_sup)}')

    # Step 2: shuffle support VECTORS within each class to preserve marginals
    print('[3] Shuffling support vectors within each class...', flush=True)
    by_class = {c: [] for c in range(len(NUSC_CLASSES))}
    for idx, item in enumerate(all_sup):
        by_class[item[2]].append(idx)
    shuffled_sup = [None] * len(all_sup)
    shuffled_valid = [None] * len(all_sup)
    for cls_idx, indices in by_class.items():
        if not indices:
            continue
        indices_arr = np.array(indices)
        # Pull support vectors of this class
        sup_arr = np.stack([all_sup[i][3] for i in indices])
        valid_arr = np.stack([all_sup[i][4] for i in indices])
        # Permute the rows (support vectors get shuffled across boxes of same class)
        perm = rng.permutation(len(indices))
        sup_perm = sup_arr[perm]
        valid_perm = valid_arr[perm]
        for k, dst in enumerate(indices):
            shuffled_sup[dst] = sup_perm[k]
            shuffled_valid[dst] = valid_perm[k]
        print(f'  {NUSC_CLASSES[cls_idx]:<20s} n={len(indices)}', flush=True)

    # Step 3: apply consensus rule on shuffled supports + emit filtered pseudo
    print('[4] Applying consensus rule on shuffled supports...', flush=True)

    def m_visibility(N_valid):
        # Mirror Step 5's visibility-aware vote count m(N_valid)
        # m(1)=1, m(2)=1, m(3)=m(4)=2, m(5)=m(6)=3
        if N_valid <= 0: return 1
        if N_valid <= 2: return 1
        if N_valid <= 4: return 2
        return 3

    filtered = {}
    n_kept = 0; n_total = 0
    idx = 0
    for tok, pb in pseudo.items():
        N = len(pb['boxes_lidar'])
        if N == 0:
            filtered[tok] = pb
            continue
        labels = pb['labels']
        # Passthrough non-COCO classes
        passthrough = np.array([
            NUSC_CLASSES[int(l)] not in NUSC_TO_COCO
            or not NUSC_TO_COCO[NUSC_CLASSES[int(l)]]
            for l in labels
        ], dtype=bool)

        sup_rows = np.stack([shuffled_sup[idx + i] for i in range(N)])
        valid_rows = np.stack([shuffled_valid[idx + i] for i in range(N)])
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

        # Emit kept subset in same dict format as g2_filter_pseudo
        out_pb = {
            'boxes_lidar': pb['boxes_lidar'][keep],
            'scores':      pb['scores'][keep],
            'labels':      pb['labels'][keep],
        }
        filtered[tok] = out_pb

    print(f'[5] Total kept: {n_kept}/{n_total} ({100*n_kept/n_total:.1f}%)')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({'pseudo': filtered,
                 'meta': {'variant': 'consensus_shuf',
                          'shuffle_seed': args.seed,
                          'shuffle_within': 'class',
                          'tau_c': args.tau_c, 'gamma': args.gamma}},
                open(out_path, 'wb'))
    print(f'  → {out_path}')


if __name__ == '__main__':
    main()
