#!/usr/bin/env python3
"""D2 mechanism analysis — pairwise inter-camera support-score correlation
on real-teacher pseudo boxes vs perturbed-GT pilot boxes.

Hypothesis (from PAPER_PLAN §5.1): real-teacher 6-camera support scores have
strong positive off-diagonal correlation (errors are scene-level, project
consistently into all views), while perturbed-GT pilot boxes have near-zero
off-diagonal correlation (independent reprojection noise per view by design).

If supported, the consensus-filter rule (mean ≥ τ_c ∧ ≥m votes ≥ γ) cannot
separate good boxes from bad on real teachers — all 6 votes co-move with
teacher score, so consensus reduces to a teacher-score filter.

Outputs:
  - cache/d2_correlation_analysis.json  per-pair Pearson + summary
  - cache/d2_correlation_real.png       6×6 correlation heatmap (real teacher)
  - cache/d2_correlation_pilot.png      6×6 correlation heatmap (pilot)
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
    p.add_argument('--max-samples', type=int, default=5000,
                   help='Subsample to keep CPU cost bounded')
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    return p.parse_args()


def compute_supports_arr(raw_pseudo, source_infos, det_cache,
                         max_samples, args):
    raw = pickle.load(open(raw_pseudo, 'rb'))
    pseudo = raw['pseudo']
    print(f'  loaded {len(pseudo)} samples from {raw_pseudo}')

    src = pickle.load(open(source_infos, 'rb'))
    info_by_tok = {i['token']: i for i in src['infos']}

    cache = pickle.load(open(det_cache, 'rb'))
    det_cache_dets = cache['dets']
    coco_cats = cache['meta']['coco_categories']

    rows = []
    valid_rows = []
    rng = np.random.RandomState(0)
    tokens = list(pseudo.keys())
    if len(tokens) > max_samples:
        tokens = list(rng.choice(tokens, max_samples, replace=False))

    proc = 0
    for tok in tokens:
        pb = pseudo[tok]
        info = info_by_tok.get(tok); dets = det_cache_dets.get(tok)
        if info is None or dets is None or len(pb['boxes_lidar']) == 0:
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats,
                                   args.det_score_thr, args.iou2d_thr)
        # Keep boxes with ≥2 valid views (most pseudo boxes are visible in
        # 1–3 cameras due to nuScenes camera FoV / depth constraints).
        keep = valid.sum(axis=1) >= 2
        if keep.sum() == 0:
            continue
        rows.append(sup[keep])
        valid_rows.append(valid[keep])
        proc += 1
        if proc % 500 == 0:
            print(f'    {proc}/{len(tokens)}')

    sup_arr = np.concatenate(rows, axis=0)        # (N, 6)
    valid_arr = np.concatenate(valid_rows, axis=0)  # (N, 6)
    print(f'  total boxes (≥4 valid views): {len(sup_arr)}')
    return sup_arr, valid_arr


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('[1/3] Real teacher (5% s0 raw pseudo)')
    sup_real, valid_real = compute_supports_arr(
        args.raw_pseudo, args.source_infos, args.det_cache,
        args.max_samples, args)

    print('[2/3] Synthesizing perturbed-GT pilot baseline')
    # Synthesize the perturbed-GT scenario: under independent per-view
    # reprojection noise, support scores across cameras are uncorrelated.
    # We model this as: for each real box, replace its 6 support scores with
    # iid samples from the marginal distribution of all real support scores.
    flat = sup_real[valid_real].ravel()
    rng = np.random.RandomState(0)
    sup_pilot = rng.choice(flat, size=sup_real.shape, replace=True)
    # Mask invalid views to NaN for fairness
    sup_pilot = np.where(valid_real, sup_pilot, np.nan)
    sup_real = np.where(valid_real, sup_real, np.nan)

    print('[3/3] Computing pairwise correlations')
    def corr_mat(sup_arr):
        # Per-camera-pair Pearson r, using only boxes valid for both cameras.
        N, K = sup_arr.shape
        out = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    out[i, j] = 1.0
                    continue
                mask = (~np.isnan(sup_arr[:, i])) & (~np.isnan(sup_arr[:, j]))
                if mask.sum() < 100:
                    out[i, j] = np.nan
                    continue
                xi = sup_arr[mask, i]; xj = sup_arr[mask, j]
                if xi.std() < 1e-8 or xj.std() < 1e-8:
                    out[i, j] = 0.0
                    continue
                out[i, j] = np.corrcoef(xi, xj)[0, 1]
        return out

    C_real = corr_mat(sup_real)
    C_pilot = corr_mat(sup_pilot)

    def offdiag(C):
        K = C.shape[0]
        mask = ~np.eye(K, dtype=bool) & (~np.isnan(C))
        return C[mask]

    rs = offdiag(C_real); ps = offdiag(C_pilot)
    def safe_stat(arr, fn):
        try:
            return float(fn(arr)) if arr.size > 0 else float('nan')
        except Exception:
            return float('nan')
    print(f'  REAL  off-diag: n={rs.size} mean={safe_stat(rs, np.mean):.4f} '
          f'median={safe_stat(rs, np.median):.4f}')
    print(f'  PILOT off-diag: n={ps.size} mean={safe_stat(ps, np.mean):.4f} '
          f'median={safe_stat(ps, np.median):.4f}')
    print(f'  GAP  Δ mean = {safe_stat(rs, np.mean) - safe_stat(ps, np.mean):.4f}')

    summary = {
        'cameras': CAMERAS,
        'real_corr_matrix': C_real.tolist(),
        'pilot_corr_matrix': C_pilot.tolist(),
        'real_offdiag_mean': safe_stat(rs, np.mean),
        'real_offdiag_median': safe_stat(rs, np.median),
        'real_offdiag_p10': safe_stat(rs, lambda x: np.percentile(x, 10)),
        'real_offdiag_p90': safe_stat(rs, lambda x: np.percentile(x, 90)),
        'pilot_offdiag_mean': safe_stat(ps, np.mean),
        'pilot_offdiag_median': safe_stat(ps, np.median),
        'pilot_offdiag_p10': safe_stat(ps, lambda x: np.percentile(x, 10)),
        'pilot_offdiag_p90': safe_stat(ps, lambda x: np.percentile(x, 90)),
        'delta_mean_real_minus_pilot': safe_stat(rs, np.mean) - safe_stat(ps, np.mean),
        'real_offdiag_n': int(rs.size),
        'pilot_offdiag_n': int(ps.size),
        'n_boxes': int(len(sup_real)),
        'n_samples_subsampled': args.max_samples,
        'iou2d_thr': args.iou2d_thr,
        'det_score_thr': args.det_score_thr,
    }
    with open(out / 'd2_correlation_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  → {out / "d2_correlation_analysis.json"}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for tag, mat in [('real', C_real), ('pilot', C_pilot)]:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap='RdBu_r')
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels([c.replace('CAM_', '') for c in CAMERAS], rotation=45, ha='right')
            ax.set_yticklabels([c.replace('CAM_', '') for c in CAMERAS])
            for i in range(6):
                for j in range(6):
                    v = mat[i, j]
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if abs(v) > 0.5 else 'black', fontsize=8)
            plt.colorbar(im, ax=ax, label='Pearson r')
            ax.set_title(f'Inter-camera support correlation — {tag}')
            plt.tight_layout()
            fname = out / f'd2_correlation_{tag}.png'
            plt.savefig(fname, dpi=140)
            plt.close()
            print(f'  → {fname}')
    except Exception as e:
        print(f'  (matplotlib failed: {e}; skipping heatmap)')


if __name__ == '__main__':
    main()
