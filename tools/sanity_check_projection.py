#!/usr/bin/env python3
"""Sanity check projection + class mapping + valid-view masking before R8.

Loads K labeled samples (where GT is known), projects each GT box to all 6
cameras using the same code path as g2_filter_pseudo.py, then for each
projection:
  1. report valid-view count and AABB area distribution
  2. look up FRCNN cache for that sample/cam
  3. compute support score (max conf * IoU2D) against same-class COCO dets
  4. report per-class hit rate (how often FRCNN sees the projected GT box)

If projection or class mapping is broken, hit rate on labeled GT will be near 0.
If healthy, should see >50% hit rate on car/truck/bus/pedestrian for confidence
that the consensus filter is functioning as designed.
"""
import sys, pickle, json
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, CAMERAS, NUSC_TO_COCO,
    box_corners, project_to_cam, aabb_in_image,
    iou2d, per_view_support,
)

LABELED_PKL = '/root/BMVC2026/infos/nuscenes_infos_train_5p_s0_labeled.pkl'
DET_CACHE   = '/root/BMVC2026/cache/fasterrcnn_v2_train.pkl'
N_SAMPLES   = 30      # how many labeled samples to audit
DET_SCORE_THR = 0.30
IOU2D_THR     = 0.30


def main():
    print(f'[+] Loading labeled infos: {LABELED_PKL}')
    L = pickle.load(open(LABELED_PKL, 'rb'))
    print(f'    samples: {len(L["infos"])}')

    print(f'[+] Loading FRCNN cache: {DET_CACHE}')
    cache = pickle.load(open(DET_CACHE, 'rb'))
    det_by_tok = cache['dets']
    coco_cats = cache['meta']['coco_categories']
    print(f'    cached tokens: {len(det_by_tok)}')
    print(f'    coco_categories: {len(coco_cats)} entries; first 10: {coco_cats[:10]}')
    # Spot-check key COCO names exist
    KEY_COCO = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
    for k in KEY_COCO:
        if k in coco_cats:
            print(f"    ✓ '{k}' at index {coco_cats.index(k)}")
        else:
            print(f"    ✗ '{k}' NOT IN coco_cats — class mapping broken!")
            return

    # Find first N samples that have BOTH gt_boxes and a det cache entry
    samples = []
    for inf in L['infos']:
        tok = inf['token']
        if tok in det_by_tok and inf.get('gt_boxes') is not None and len(inf['gt_boxes']) > 0:
            samples.append(inf)
            if len(samples) >= N_SAMPLES:
                break
    print(f'[+] Auditing {len(samples)} samples')

    # Per-class stats: for each NUSC class, count GT boxes, valid views, hit rates
    stats = {c: {'gt_boxes': 0, 'projected_anywhere': 0,
                 'valid_view_count': [], 'aabb_area_avg': [],
                 'hit_count': 0,           # at least 1 cam supports
                 'hit_supports': []}        # max sup over cams (if >0)
             for c in NUSC_CLASSES}

    for inf in samples:
        tok = inf['token']
        gt_boxes = inf['gt_boxes']      # (N, 7) x,y,z,w,l,h,yaw
        gt_names = inf['gt_names']
        dets_for_sample = det_by_tok[tok]

        for i in range(len(gt_boxes)):
            cls = str(gt_names[i])
            if cls not in NUSC_CLASSES:
                continue
            stats[cls]['gt_boxes'] += 1
            box = gt_boxes[i]
            corners = box_corners(box[:3], box[3:6], float(box[6]))
            n_valid = 0
            max_sup = 0.0
            cls_areas = []
            for cam_idx, cam in enumerate(CAMERAS):
                cam_info = inf['cams'][cam]
                cam_dets = dets_for_sample.get(cam, None)
                if cam_dets is None:
                    continue
                W, H = cam_dets['img_wh']
                proj = project_to_cam(corners, cam_info, inf)
                if proj is None:
                    continue
                xs, ys = proj
                aabb = aabb_in_image(xs, ys, W, H)
                if aabb is None:
                    continue
                n_valid += 1
                cls_areas.append((aabb[2]-aabb[0]) * (aabb[3]-aabb[1]))
                accept = NUSC_TO_COCO.get(cls, set())
                if accept:
                    s_v, _ = per_view_support(aabb, cam_dets, accept, coco_cats,
                                               DET_SCORE_THR, IOU2D_THR)
                    if s_v > max_sup:
                        max_sup = s_v
            if n_valid > 0:
                stats[cls]['projected_anywhere'] += 1
                stats[cls]['valid_view_count'].append(n_valid)
                stats[cls]['aabb_area_avg'].append(np.mean(cls_areas))
            if max_sup > 0:
                stats[cls]['hit_count'] += 1
                stats[cls]['hit_supports'].append(max_sup)

    # Report
    print(f'\n=== Per-class GT projection + FRCNN hit-rate audit (N={len(samples)} samples) ===\n')
    print(f"{'class':<22s} {'gt':>5s} {'proj':>5s} {'proj%':>6s} "
          f"{'avg_nv':>7s} {'avg_area':>9s} {'hit':>5s} {'hit%':>6s} "
          f"{'avg_sup':>8s}")
    print('-' * 90)
    summary = {}
    for cls in NUSC_CLASSES:
        s = stats[cls]
        gt = s['gt_boxes']
        proj = s['projected_anywhere']
        proj_pct = (proj / gt * 100) if gt > 0 else 0
        hit = s['hit_count']
        hit_pct = (hit / proj * 100) if proj > 0 else 0
        avg_nv = np.mean(s['valid_view_count']) if s['valid_view_count'] else 0
        avg_area = np.mean(s['aabb_area_avg']) if s['aabb_area_avg'] else 0
        avg_sup = np.mean(s['hit_supports']) if s['hit_supports'] else 0
        accept = NUSC_TO_COCO.get(cls, set())
        flag = ' (no-COCO)' if not accept else ''
        print(f'{cls:<22s} {gt:>5d} {proj:>5d} {proj_pct:>5.1f}% '
              f'{avg_nv:>7.2f} {avg_area:>9.0f} {hit:>5d} {hit_pct:>5.1f}% '
              f'{avg_sup:>8.3f}{flag}')
        summary[cls] = dict(gt=gt, proj=proj, proj_pct=proj_pct,
                            hit=hit, hit_pct=hit_pct, avg_nv=avg_nv,
                            avg_sup=avg_sup, has_coco=bool(accept))
    print('\n=== Sanity verdicts ===')
    pass_count = 0
    fail_count = 0
    for cls in ['car', 'truck', 'bus', 'pedestrian']:
        s = summary[cls]
        if s['gt'] == 0:
            print(f'  ⚠  {cls}: 0 GT boxes in audit — too few samples')
            continue
        if s['proj_pct'] < 50:
            print(f'  ✗  {cls}: only {s["proj_pct"]:.1f}% GT projects to ≥1 cam — projection broken')
            fail_count += 1
        elif s['hit_pct'] < 30:
            print(f'  ⚠  {cls}: projects fine but FRCNN hit rate only {s["hit_pct"]:.1f}% '
                  f'— class map or thresholds suspect')
            fail_count += 1
        else:
            print(f'  ✓  {cls}: proj={s["proj_pct"]:.0f}%, FRCNN hit={s["hit_pct"]:.0f}%, '
                  f'avg sup={s["avg_sup"]:.2f}')
            pass_count += 1

    out = Path('/root/BMVC2026/.aris/traces/2026-05-02_sanity_check_projection.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'n_samples': len(samples),
               'det_score_thr': DET_SCORE_THR,
               'iou2d_thr': IOU2D_THR,
               'per_class': summary},
              open(out, 'w'), indent=2, default=lambda x: float(x) if hasattr(x,'__float__') else str(x))
    print(f'\n[+] Detailed JSON → {out}')
    print(f'\n[+] Summary: {pass_count} class(es) pass, {fail_count} class(es) flagged')


if __name__ == '__main__':
    main()
