#!/usr/bin/env python3
"""R5b — Apply all filter variants to raw pseudo 3D boxes using cached FRCNN
2D detections across 6 cameras.

Variants:
  raw               no filter (just class-wise score floor)
  threshold_retune  raise class-wise score floor by delta
  best_view         keep if >= 1 camera sees it (strongest support only)
  any_view          keep if any camera sees it (loose)
  max_cam           keep top-k boxes per sample by max camera score
  consensus         soft consensus score + visibility-aware floor (CORE)
  temporal_pm1      (optional) temporal ±1 consensus — placeholder for now

For each variant, compute pseudo-label precision / recall on train GT (for
diagnostic only; does not touch training).

Output:
  out_dir/<variant>.pkl    # filtered pseudo boxes in same format as input
  out_dir/metrics.json     # P/R per variant, per class, global
"""
import argparse, json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from pyquaternion import Quaternion

NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

# nuScenes class -> set of COCO class names acceptable as 2D match
NUSC_TO_COCO = {
    'car':        {'car'},
    'truck':      {'truck'},
    'bus':        {'bus', 'truck'},
    'trailer':    {'truck', 'car'},
    'construction_vehicle': {'truck'},
    'pedestrian': {'person'},
    'bicycle':    {'bicycle'},
    'motorcycle': {'motorcycle', 'bicycle'},
    'barrier':    set(),
    'traffic_cone': set(),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-pseudo', required=True,
                   help='Output of g2_dump_pseudo.py')
    p.add_argument('--det-cache', required=True,
                   help='Output of g2_cache_frcnn_nus.py')
    p.add_argument('--source-infos', required=True,
                   help='The infos pkl corresponding to raw-pseudo (for cam calibration + GT for diagnostics)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--variants', nargs='+',
                   default=['raw', 'threshold_retune', 'best_view',
                            'any_view', 'max_cam', 'consensus'])
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    p.add_argument('--consensus-tau', type=float, default=0.15,
                   help='τ_c soft consensus threshold (mean s_v)')
    p.add_argument('--consensus-gamma', type=float, default=0.20,
                   help='γ hard-floor per-view threshold')
    p.add_argument('--threshold-retune-delta', type=float, default=0.10,
                   help='Extra score floor for threshold_retune variant')
    p.add_argument('--max-cam-topk', type=int, default=50)
    return p.parse_args()


# ---------------- geometry ---------------- #

def box_corners(center, wlh, yaw):
    w, l, h = wlh
    x = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2])
    y = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2])
    z = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2])
    c = np.vstack([x, y, z])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])
    return rot @ c + np.array(center).reshape(3, 1)


def project_to_cam(corners_lidar, cam, info):
    lid_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lid_t = np.array(info['lidar2ego_translation']).reshape(3, 1)
    pts_ego = lid_r @ corners_lidar + lid_t

    leg_r = Quaternion(info['ego2global_rotation']).rotation_matrix
    leg_t = np.array(info['ego2global_translation']).reshape(3, 1)
    pts_global = leg_r @ pts_ego + leg_t

    ceg_r = Quaternion(cam['ego2global_rotation']).rotation_matrix
    ceg_t = np.array(cam['ego2global_translation']).reshape(3, 1)
    pts_ego_cam = ceg_r.T @ (pts_global - ceg_t)

    cs_r = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
    cs_t = np.array(cam['sensor2ego_translation']).reshape(3, 1)
    pts_cam = cs_r.T @ (pts_ego_cam - cs_t)

    depths = pts_cam[2, :]
    if np.all(depths <= 0.1):
        return None
    K = np.array(cam['cam_intrinsic'])
    fr = depths > 0.1
    xs = np.full(8, -1e6); ys = np.full(8, -1e6)
    pp = K @ pts_cam[:, fr]
    pp = pp / pp[2:3, :]
    xs[fr] = pp[0, :]; ys[fr] = pp[1, :]
    return xs, ys


def aabb_in_image(xs, ys, W, H, min_area=100):
    vx = xs > -1e5
    if vx.sum() < 4:
        return None
    x0, x1 = xs[vx].min(), xs[vx].max()
    y0, y1 = ys[vx].min(), ys[vx].max()
    x0, x1 = max(0, x0), min(W - 1, x1)
    y0, y1 = max(0, y0), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) < min_area:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def iou2d(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max((ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter, 1e-6)


# ---------------- support score ---------------- #

def per_view_support(aabb, cam_dets, accept_coco, coco_cats, score_thr, iou_thr):
    """Return max(conf * IoU) over valid 2D dets in this camera."""
    if aabb is None:
        return 0.0, False  # (score, valid)
    boxes = cam_dets['boxes']
    scores = cam_dets['scores']
    labels = cam_dets['labels']
    best = 0.0
    for b, s, lb in zip(boxes, scores, labels):
        if s < score_thr or lb >= len(coco_cats):
            continue
        if accept_coco and coco_cats[lb] not in accept_coco:
            continue
        i = iou2d(aabb, b)
        if i >= iou_thr:
            v = float(s * i)
            if v > best:
                best = v
    return best, True


def box_supports(pseudo_boxes, info, det_cache_sample, coco_cats,
                 score_thr, iou_thr):
    """For each pseudo box in sample, compute {cam: support_score, is_valid}."""
    N = len(pseudo_boxes['boxes_lidar'])
    sup = np.zeros((N, len(CAMERAS)), dtype=np.float32)
    valid = np.zeros((N, len(CAMERAS)), dtype=bool)
    if N == 0:
        return sup, valid
    for cam_idx, cam in enumerate(CAMERAS):
        cam_info = info['cams'][cam]
        cam_dets = det_cache_sample.get(cam, None) if det_cache_sample else None
        if cam_dets is None:
            continue
        W, H = cam_dets['img_wh']
        for i in range(N):
            b3d = pseudo_boxes['boxes_lidar'][i]
            cls_idx = int(pseudo_boxes['labels'][i])
            cls = NUSC_CLASSES[cls_idx] if cls_idx < len(NUSC_CLASSES) else NUSC_CLASSES[0]
            center = b3d[:3]; wlh = b3d[3:6]; yaw = float(b3d[6])
            corners = box_corners(center, wlh, yaw)
            proj = project_to_cam(corners, cam_info, info)
            if proj is None:
                continue
            xs, ys = proj
            aabb = aabb_in_image(xs, ys, W, H)
            if aabb is None:
                continue
            valid[i, cam_idx] = True
            accept = NUSC_TO_COCO.get(cls, set())
            if accept:
                s_v, _ = per_view_support(aabb, cam_dets, accept, coco_cats,
                                           score_thr, iou_thr)
                sup[i, cam_idx] = s_v
            else:
                # no COCO equivalent — pass-through (can't filter with FRCNN)
                sup[i, cam_idx] = 0.0  # will cause rejection in consensus;
                                        # keep for diagnostics
    return sup, valid


# ---------------- filter variants ---------------- #

def apply_variant(variant, pseudo_boxes, sup, valid, args):
    """Return mask: which boxes to keep."""
    N = len(pseudo_boxes['boxes_lidar'])
    if N == 0:
        return np.zeros((0,), dtype=bool)
    labels = pseudo_boxes['labels']

    # Classes we cannot filter with COCO (barrier, traffic_cone) — always pass
    # through (otherwise they all get killed and recall collapses)
    passthrough = np.array([
        NUSC_CLASSES[int(l)] not in NUSC_TO_COCO or not NUSC_TO_COCO[NUSC_CLASSES[int(l)]]
        for l in labels
    ], dtype=bool)

    if variant == 'raw':
        return np.ones(N, dtype=bool)

    if variant == 'threshold_retune':
        scores = pseudo_boxes['scores']
        # raise floor by delta for each class
        keep = np.zeros(N, dtype=bool)
        for i in range(N):
            cls = NUSC_CLASSES[int(labels[i])]
            floor = 0.30 + args.threshold_retune_delta
            keep[i] = scores[i] >= floor
        return keep

    # camera-aided variants: passthrough classes always kept
    max_sup = sup.max(axis=1)
    n_valid = valid.sum(axis=1)
    has_valid = n_valid > 0

    if variant == 'best_view':
        # keep if at least one camera sees it (sup > gamma)
        keep = (max_sup >= args.consensus_gamma) & has_valid
        keep = keep | passthrough | (~has_valid)  # no camera → can't filter → keep
        return keep

    if variant == 'any_view':
        # loose: any non-zero support
        keep = (max_sup > 0) & has_valid
        keep = keep | passthrough | (~has_valid)
        return keep

    if variant == 'max_cam':
        # Top-K per sample by max-camera support, restricted to
        # camera-supported candidates. Passthrough classes (barrier,
        # traffic_cone) are kept unconditionally on top — they are not
        # camera-assessable so they cannot be ranked.
        keep = np.zeros(N, dtype=bool)
        scores = pseudo_boxes['scores']
        candidates = np.where((~passthrough) & has_valid & (max_sup > 0))[0]
        if candidates.size:
            # Tie-break on teacher score so zero-IoU but high-confidence
            # camera matches don't reorder identical-support boxes randomly
            order = candidates[np.lexsort((
                -scores[candidates],
                -max_sup[candidates],
            ))]
            keep[order[:args.max_cam_topk]] = True
        keep = keep | passthrough
        return keep

    if variant == 'consensus':
        # soft mean across valid views + visibility-aware hard floor
        sum_sup = sup.sum(axis=1)
        mean_sup = np.where(n_valid > 0, sum_sup / np.clip(n_valid, 1, None), 0.0)
        hits = (sup >= args.consensus_gamma).sum(axis=1)
        thr = np.array([m_thresh(int(n)) for n in n_valid])
        soft_ok = mean_sup >= args.consensus_tau
        hard_ok = hits >= thr
        keep = soft_ok & hard_ok & has_valid
        keep = keep | passthrough | (~has_valid)
        return keep

    raise ValueError(f'Unknown variant: {variant}')


def m_thresh(n_valid):
    if n_valid <= 0: return 1
    if n_valid == 1: return 1
    if n_valid == 2: return 1
    if n_valid <= 4: return 2
    return 3


# ---------------- pseudo-GT matching for diagnostics ---------------- #

def box_iou3d_proxy(a, b):
    """Cheap BEV IoU proxy: center distance <= 2.0 m AND same class = match."""
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return (dx*dx + dy*dy) ** 0.5


def compute_pr(pseudo_per_token, infos, mask_func):
    """For each token, get kept pseudo boxes and match to GT. Return (P, R, F1)."""
    tp = 0; fp = 0; fn = 0
    info_by_tok = {i['token']: i for i in infos}
    per_class = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for tok, pb in pseudo_per_token.items():
        info = info_by_tok.get(tok, None)
        if info is None:
            continue
        keep = mask_func(tok)
        kept_boxes = pb['boxes_lidar'][keep]
        kept_labels = pb['labels'][keep]
        gt_boxes = info.get('gt_boxes', None)
        gt_names = info.get('gt_names', None)
        if gt_boxes is None or gt_names is None:
            continue
        matched_gt = set()
        for bi, lbl in enumerate(kept_labels):
            cls = NUSC_CLASSES[int(lbl)]
            c = kept_boxes[bi][:2]
            best_j = -1; best_d = 1e9
            for j, gn in enumerate(gt_names):
                if j in matched_gt or gn != cls:
                    continue
                gc = gt_boxes[j][:2]
                d = ((c[0]-gc[0])**2 + (c[1]-gc[1])**2) ** 0.5
                if d < best_d:
                    best_d = d; best_j = j
            if best_j >= 0 and best_d <= 2.0:
                matched_gt.add(best_j)
                tp += 1; per_class[cls]['tp'] += 1
            else:
                fp += 1; per_class[cls]['fp'] += 1
        for j, gn in enumerate(gt_names):
            if j not in matched_gt and gn in NUSC_CLASSES:
                fn += 1; per_class[gn]['fn'] += 1
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-6)
    return P, R, F, tp, fp, fn, dict(per_class)


# ---------------- main ---------------- #

def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[+] Loading raw pseudo: {args.raw_pseudo}')
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    pseudo = raw['pseudo']
    print(f'[+] Pseudo samples: {len(pseudo)}')

    print(f'[+] Loading detection cache: {args.det_cache}')
    cache = pickle.load(open(args.det_cache, 'rb'))
    det_cache = cache['dets']
    coco_cats = cache['meta']['coco_categories']
    print(f'[+] Detection cache tokens: {len(det_cache)}')

    print(f'[+] Loading source infos: {args.source_infos}')
    src = pickle.load(open(args.source_infos, 'rb'))
    info_by_tok = {i['token']: i for i in src['infos']}
    print(f'[+] Source info tokens: {len(info_by_tok)}')

    # Pre-compute per-sample support matrices
    print('[+] Computing per-view support for each pseudo box...')
    supports = {}   # token -> (sup matrix N×6, valid matrix N×6)
    proc = 0
    for tok, pb in pseudo.items():
        info = info_by_tok.get(tok, None)
        dets = det_cache.get(tok, None)
        if info is None or dets is None:
            supports[tok] = (np.zeros((len(pb['boxes_lidar']), 6), dtype=np.float32),
                             np.zeros((len(pb['boxes_lidar']), 6), dtype=bool))
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats,
                                   args.det_score_thr, args.iou2d_thr)
        supports[tok] = (sup, valid)
        proc += 1
        if proc % 1000 == 0:
            print(f'  processed {proc}/{len(pseudo)}')

    # Apply each variant
    metrics = {}
    for variant in args.variants:
        print(f'\n[+] Variant: {variant}')
        filtered = {}
        def mk_mask(tok):
            pb = pseudo[tok]
            sup, valid = supports[tok]
            return apply_variant(variant, pb, sup, valid, args)
        total_kept = 0
        # Diagnostic counts (per Codex review): how many kept boxes came from
        # the no-camera fallback vs. camera-supported vs. COCO-passthrough?
        n_passthrough_kept = 0
        n_no_camera_kept = 0
        n_supported_kept = 0
        for tok, pb in pseudo.items():
            m = mk_mask(tok)
            sup, valid = supports[tok]
            n_valid = valid.sum(axis=1)
            max_sup = sup.max(axis=1) if sup.size else np.array([])
            labels = pb['labels']
            passthrough = np.array([
                NUSC_CLASSES[int(l)] not in NUSC_TO_COCO
                or not NUSC_TO_COCO[NUSC_CLASSES[int(l)]]
                for l in labels
            ], dtype=bool) if len(labels) else np.zeros(0, dtype=bool)
            if m.size:
                n_passthrough_kept += int((m & passthrough).sum())
                n_no_camera_kept += int(
                    (m & (~passthrough) & (n_valid == 0)).sum())
                n_supported_kept += int(
                    (m & (~passthrough) & (n_valid > 0)).sum())
            filtered[tok] = {
                'boxes_lidar': pb['boxes_lidar'][m].copy(),
                'scores': pb['scores'][m].copy(),
                'labels': pb['labels'][m].copy(),
            }
            total_kept += int(m.sum())
        pickle.dump({'pseudo': filtered, 'meta': {'variant': variant, 'args': vars(args)}},
                    open(out / f'{variant}.pkl', 'wb'))

        print(f'    kept_breakdown: supported={n_supported_kept} '
              f'passthrough={n_passthrough_kept} no_camera_fallback={n_no_camera_kept}')

        # Diagnostic P/R
        P, R, F, tp, fp, fn, per_cls = compute_pr(pseudo, src['infos'], mk_mask)
        print(f'    kept={total_kept}  P={P*100:.1f}  R={R*100:.1f}  F1={F*100:.1f}  (tp={tp} fp={fp} fn={fn})')
        metrics[variant] = {'kept': total_kept, 'P': P, 'R': R, 'F1': F,
                            'tp': tp, 'fp': fp, 'fn': fn, 'per_class': per_cls,
                            'kept_supported': n_supported_kept,
                            'kept_passthrough': n_passthrough_kept,
                            'kept_no_camera_fallback': n_no_camera_kept}

    # Relative gain table
    print('\n=== Precision gain table (vs raw) ===')
    if 'raw' in metrics:
        P_raw = metrics['raw']['P'] * 100
        print(f"{'variant':<20s} {'P':>6s} {'R':>6s} {'F1':>6s} {'ΔP':>7s}")
        for v in args.variants:
            if v not in metrics: continue
            m = metrics[v]
            print(f'{v:<20s} {m["P"]*100:>6.1f} {m["R"]*100:>6.1f} {m["F1"]*100:>6.1f} {m["P"]*100 - P_raw:>+7.1f}')

    json.dump(metrics, open(out / 'metrics.json', 'w'),
              indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f'\n[+] Metrics → {out / "metrics.json"}')


if __name__ == '__main__':
    main()
