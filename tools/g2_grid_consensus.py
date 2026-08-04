#!/usr/bin/env python3
"""R5b-grid — grid search γ × τ for consensus variant, reusing support matrix
so we don't re-project 437k boxes 9 times.

Loads raw pseudo + FRCNN cache + source infos, computes per-view support
ONCE, then sweeps (γ, τ) × (hard-floor on/off). Reports COCO-covered and
non-COCO P/R/F1 per config + ΔP vs raw.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, CAMERAS, NUSC_TO_COCO,
    box_supports, apply_variant, compute_pr, m_thresh,
)

COCO_COVERED = {'car', 'truck', 'bus', 'trailer', 'pedestrian', 'bicycle', 'motorcycle'}
NON_COCO = {'construction_vehicle', 'barrier', 'traffic_cone'}


class GridArgs:
    """Minimal args object to pass to apply_variant()."""
    def __init__(self, tau, gamma):
        self.consensus_tau = tau
        self.consensus_gamma = gamma
        self.threshold_retune_delta = 0.10
        self.max_cam_topk = 50


def aggregate(per_class, classes):
    tp = sum(per_class.get(c, {}).get('tp', 0) for c in classes)
    fp = sum(per_class.get(c, {}).get('fp', 0) for c in classes)
    fn = sum(per_class.get(c, {}).get('fn', 0) for c in classes)
    p = tp / max(tp + fp, 1) * 100
    r = tp / max(tp + fn, 1) * 100
    f = 2 * p * r / max(p + r, 1e-6)
    return p, r, f, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-pseudo', required=True)
    ap.add_argument('--det-cache', required=True)
    ap.add_argument('--source-infos', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--taus', nargs='+', type=float,
                    default=[0.10, 0.15, 0.20])
    ap.add_argument('--gammas', nargs='+', type=float,
                    default=[0.10, 0.15, 0.20])
    ap.add_argument('--iou2d-thr', type=float, default=0.30)
    ap.add_argument('--det-score-thr', type=float, default=0.30)
    args = ap.parse_args()

    print(f'[+] Loading raw pseudo: {args.raw_pseudo}')
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    pseudo = raw['pseudo']
    print(f'    samples: {len(pseudo)}')

    print(f'[+] Loading det cache: {args.det_cache}')
    cache = pickle.load(open(args.det_cache, 'rb'))
    det_cache = cache['dets']
    coco_cats = cache['meta']['coco_categories']

    print(f'[+] Loading source infos: {args.source_infos}')
    src = pickle.load(open(args.source_infos, 'rb'))
    info_by_tok = {i['token']: i for i in src['infos']}

    # Compute support ONCE
    print('[+] Computing per-view support (one-shot)...')
    supports = {}
    proc = 0
    for tok, pb in pseudo.items():
        info = info_by_tok.get(tok)
        dets = det_cache.get(tok)
        N = len(pb['boxes_lidar'])
        if info is None or dets is None:
            supports[tok] = (np.zeros((N, 6), dtype=np.float32),
                             np.zeros((N, 6), dtype=bool))
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats,
                                   args.det_score_thr, args.iou2d_thr)
        supports[tok] = (sup, valid)
        proc += 1
        if proc % 1000 == 0:
            print(f'    {proc}/{len(pseudo)}')

    # Raw baseline — for ΔP reporting
    def raw_mask(tok):
        pb = pseudo[tok]
        return np.ones(len(pb['boxes_lidar']), dtype=bool)
    P_raw, R_raw, F_raw, tp_raw, fp_raw, fn_raw, pc_raw = compute_pr(
        pseudo, src['infos'], raw_mask)
    raw_coco = aggregate(pc_raw, COCO_COVERED)
    print(f'\n=== RAW (reference) ===')
    print(f'global: P={P_raw*100:.1f} R={R_raw*100:.1f} F1={F_raw*100:.1f}')
    print(f'coco  : P={raw_coco[0]:.1f} R={raw_coco[1]:.1f} F1={raw_coco[2]:.1f}')

    # Threshold_retune baseline (δ=0.10 default)
    def thr_mask(tok):
        pb = pseudo[tok]
        keep = np.zeros(len(pb['boxes_lidar']), dtype=bool)
        for i in range(len(pb['boxes_lidar'])):
            keep[i] = pb['scores'][i] >= 0.40
        return keep
    P_thr, R_thr, F_thr, *_rest, pc_thr = compute_pr(pseudo, src['infos'], thr_mask)
    thr_coco = aggregate(pc_thr, COCO_COVERED)
    thr_noncoco = aggregate(pc_thr, NON_COCO)
    print(f'\n=== THRESHOLD_RETUNE (δ=0.10, simple-alt baseline) ===')
    print(f'global: P={P_thr*100:.1f} R={R_thr*100:.1f} F1={F_thr*100:.1f}')
    print(f'coco  : P={thr_coco[0]:.1f} R={thr_coco[1]:.1f} F1={thr_coco[2]:.1f}')
    print(f'noncoco: P={thr_noncoco[0]:.1f} R={thr_noncoco[1]:.1f} F1={thr_noncoco[2]:.1f}')

    # Grid search
    print(f'\n=== CONSENSUS GRID ({len(args.gammas)}×{len(args.taus)}) ===')
    print(f"{'γ':<5s} {'τ':<5s} {'kept':>7s} "
          f"{'P_all':>6s} {'R_all':>6s} {'F1_all':>6s} "
          f"{'P_coco':>7s} {'R_coco':>7s} {'F1_coco':>7s} "
          f"{'ΔPcoco':>7s} "
          f"{'P_nc':>6s} {'R_nc':>6s}")

    grid_results = []
    for gamma in args.gammas:
        for tau in args.taus:
            ga = GridArgs(tau=tau, gamma=gamma)
            def cmask(tok):
                pb = pseudo[tok]
                sup, valid = supports[tok]
                return apply_variant('consensus', pb, sup, valid, ga)
            total_kept = sum(int(cmask(tok).sum()) for tok in pseudo)
            P, R, F, *_rest, pc = compute_pr(pseudo, src['infos'], cmask)
            coco = aggregate(pc, COCO_COVERED)
            nc = aggregate(pc, NON_COCO)
            dP_coco = coco[0] - raw_coco[0]
            grid_results.append({
                'gamma': gamma, 'tau': tau, 'kept': total_kept,
                'P_all': P*100, 'R_all': R*100, 'F1_all': F*100,
                'P_coco': coco[0], 'R_coco': coco[1], 'F1_coco': coco[2],
                'dP_coco': dP_coco,
                'P_noncoco': nc[0], 'R_noncoco': nc[1],
                'per_class': pc,
            })
            print(f"{gamma:<5.2f} {tau:<5.2f} {total_kept:>7d} "
                  f"{P*100:>6.1f} {R*100:>6.1f} {F*100:>6.1f} "
                  f"{coco[0]:>7.1f} {coco[1]:>7.1f} {coco[2]:>7.1f} "
                  f"{dP_coco:>+7.1f} "
                  f"{nc[0]:>6.1f} {nc[1]:>6.1f}")

    # Also note best consensus vs threshold_retune on COCO F1
    best = max(grid_results, key=lambda r: r['F1_coco'])
    print(f'\n=== VERDICT ===')
    print(f'Best consensus COCO-F1: γ={best["gamma"]:.2f} τ={best["tau"]:.2f} → F1={best["F1_coco"]:.1f} (P={best["P_coco"]:.1f} R={best["R_coco"]:.1f})')
    print(f'Threshold_retune   COCO-F1: {thr_coco[2]:.1f} (P={thr_coco[0]:.1f} R={thr_coco[1]:.1f})')
    print(f'Gap (consensus − threshold): {best["F1_coco"] - thr_coco[2]:+.1f} F1, {best["P_coco"] - thr_coco[0]:+.1f} P, {best["R_coco"] - thr_coco[1]:+.1f} R')

    out = {
        'raw_coco': {'P': raw_coco[0], 'R': raw_coco[1], 'F1': raw_coco[2]},
        'threshold_retune_coco': {'P': thr_coco[0], 'R': thr_coco[1], 'F1': thr_coco[2]},
        'grid': [
            {k: v for k, v in r.items() if k != 'per_class'}
            for r in grid_results
        ],
        'best_consensus': {k: v for k, v in best.items() if k != 'per_class'},
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x))
    print(f'\n[+] Grid results → {args.out_json}')


if __name__ == '__main__':
    main()
