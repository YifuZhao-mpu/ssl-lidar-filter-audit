#!/usr/bin/env python3
"""D2 robustness: per-class breakdown of the inter-camera support correlation
on real-teacher pseudo boxes vs perturbed-GT pilot. Codex round-3 ask: confirm
the 0.42 >> 0.03 gap holds across classes, not just in the pooled mix.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, CAMERAS, NUSC_TO_COCO, box_supports,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-pseudo', required=True)
    p.add_argument('--source-infos', required=True)
    p.add_argument('--det-cache', required=True)
    p.add_argument('--out-dir', default='/root/BMVC2026/cache')
    p.add_argument('--max-samples', type=int, default=5000)
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    p.add_argument('--top-classes', type=int, default=5,
                   help='Report per-class breakdown for the N most-populated classes')
    return p.parse_args()


def main():
    args = parse_args()
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    pseudo = raw['pseudo']
    src = pickle.load(open(args.source_infos, 'rb'))
    info_by_tok = {i['token']: i for i in src['infos']}
    cache = pickle.load(open(args.det_cache, 'rb'))
    det_cache_dets = cache['dets']
    coco_cats = cache['meta']['coco_categories']

    print(f'[+] {len(pseudo)} samples in raw pseudo')
    rng = np.random.RandomState(0)
    tokens = list(pseudo.keys())
    if len(tokens) > args.max_samples:
        tokens = list(rng.choice(tokens, args.max_samples, replace=False))

    # Accumulate (sup, valid, label) per box across all sampled tokens
    rows_sup = []; rows_valid = []; rows_lbl = []
    proc = 0
    for tok in tokens:
        pb = pseudo[tok]
        info = info_by_tok.get(tok); dets = det_cache_dets.get(tok)
        if info is None or dets is None or len(pb['boxes_lidar']) == 0:
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats,
                                   args.det_score_thr, args.iou2d_thr)
        keep = valid.sum(axis=1) >= 2
        if keep.sum() == 0:
            continue
        rows_sup.append(sup[keep])
        rows_valid.append(valid[keep])
        rows_lbl.append(pb['labels'][keep])
        proc += 1
        if proc % 1000 == 0:
            print(f'  {proc}/{len(tokens)}')

    sup_real = np.concatenate(rows_sup, axis=0)        # (N, 6)
    valid_arr = np.concatenate(rows_valid, axis=0)     # (N, 6)
    labels = np.concatenate(rows_lbl, axis=0)           # (N,)
    print(f'  total boxes (≥2 valid views): {len(sup_real)}')

    # Class population counts
    counts = np.bincount(labels, minlength=len(NUSC_CLASSES))
    order = np.argsort(-counts)
    top_idx = order[:args.top_classes]
    print(f'  top {args.top_classes} classes by box count: '
          + ', '.join(f'{NUSC_CLASSES[i]}={counts[i]}' for i in top_idx))

    # Synthesize perturbed-GT pilot (independent per-view sampling from real marginal)
    flat = sup_real[valid_arr].ravel()
    sup_pilot = rng.choice(flat, size=sup_real.shape, replace=True)
    sup_real_nan = np.where(valid_arr, sup_real, np.nan)
    sup_pilot_nan = np.where(valid_arr, sup_pilot, np.nan)

    def offdiag_mean_med(sup_arr):
        K = sup_arr.shape[1]
        vals = []
        for i in range(K):
            for j in range(K):
                if i == j: continue
                mask = ~np.isnan(sup_arr[:, i]) & ~np.isnan(sup_arr[:, j])
                if mask.sum() < 50: continue
                xi = sup_arr[mask, i]; xj = sup_arr[mask, j]
                if xi.std() < 1e-8 or xj.std() < 1e-8:
                    vals.append(0.0); continue
                vals.append(float(np.corrcoef(xi, xj)[0, 1]))
        v = np.array(vals)
        if v.size == 0:
            return float('nan'), float('nan'), 0
        return float(v.mean()), float(np.median(v)), int(v.size)

    summary = {'pooled': {}, 'per_class': {}}
    rm, rmd, rn = offdiag_mean_med(sup_real_nan)
    pm, pmd, pn = offdiag_mean_med(sup_pilot_nan)
    summary['pooled'] = {
        'real_offdiag_mean': rm, 'real_offdiag_median': rmd, 'real_offdiag_n_pairs': rn,
        'pilot_offdiag_mean': pm, 'pilot_offdiag_median': pmd, 'pilot_offdiag_n_pairs': pn,
        'delta_mean': rm - pm, 'n_boxes': int(len(sup_real)),
    }
    print(f'\n[Pooled] real mean={rm:.4f} pilot mean={pm:.4f} Δ={rm-pm:.4f} n_boxes={len(sup_real)}')

    print('\n[Per class]')
    print(f"  {'class':<22s} {'n_box':>7s} {'real_r':>9s} {'pilot_r':>9s} {'Δ':>9s}")
    for ci in top_idx:
        cmask = labels == ci
        if cmask.sum() < 100:
            continue
        sr = sup_real_nan[cmask]; sp = sup_pilot_nan[cmask]
        rm_c, rmd_c, rn_c = offdiag_mean_med(sr)
        pm_c, pmd_c, pn_c = offdiag_mean_med(sp)
        cls = NUSC_CLASSES[int(ci)]
        summary['per_class'][cls] = {
            'n_boxes': int(cmask.sum()),
            'real_offdiag_mean': rm_c, 'real_offdiag_median': rmd_c, 'real_offdiag_n_pairs': rn_c,
            'pilot_offdiag_mean': pm_c, 'pilot_offdiag_median': pmd_c, 'pilot_offdiag_n_pairs': pn_c,
            'delta_mean': rm_c - pm_c,
        }
        print(f"  {cls:<22s} {int(cmask.sum()):>7d} {rm_c:>9.4f} {pm_c:>9.4f} {rm_c-pm_c:>9.4f}")

    out = Path(args.out_dir) / 'd2_correlation_perclass.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
