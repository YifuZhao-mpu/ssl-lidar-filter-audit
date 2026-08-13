#!/usr/bin/env python3
"""R5c — Apply per-class+boost (or single-threshold baseline) filter to full
pseudo dump and emit a nuScenes-format infos.pkl with pseudo as gt_boxes,
ready to be ConcatDataset'd with the labeled subset for R6/R7 SSL.

Variants emitted:
  single_thr   single score floor (default 0.40) — baseline arm (R6)
  perclass_boost  per-class optimal thr + consensus-boost — claim arm (R7)

Per-class optimal thresholds discovered on partial-3k Phase B:
  car=0.35, truck=0.40, construction_vehicle=0.45, bus=0.30, trailer=0.30,
  barrier=0.30, motorcycle=0.40, bicycle=0.40, pedestrian=0.50, traffic_cone=0.35

Consensus boost: if score >= floor (0.25) AND not already-passing per-class thr,
recover box if mean per-view support >= boost_thr (0.10).
"""
import argparse, copy, json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/BMVC2026')
from tools.g2_filter_pseudo import (
    NUSC_CLASSES, CAMERAS, NUSC_TO_COCO, box_supports, m_thresh,
)


PER_CLASS_THR = {
    'car': 0.35, 'truck': 0.40, 'construction_vehicle': 0.45,
    'bus': 0.30, 'trailer': 0.30, 'barrier': 0.30,
    'motorcycle': 0.40, 'bicycle': 0.40, 'pedestrian': 0.50,
    'traffic_cone': 0.35,
}
BOOST_FLOOR = 0.25
BOOST_THR = 0.10
SINGLE_THR_DEFAULT = 0.40

# Package B hybrid (R18): rescue-eligible classes, frozen from the s0 R7b
# kept-share audit (COCO-overlap classes with >=5% of kept pseudo-labels;
# see PACKAGE_B_PROTOCOL.md). Rescue reuses BOOST_FLOOR as tau_lo and
# any-view support (max_sup > 0) — no newly tuned constants.
G_COMMON = {'car', 'pedestrian', 'truck'}

# R19 conjunctive control: the six-camera consensus constants, reused verbatim
# from g2_filter_pseudo.py so the camera term is byte-for-byte the rule audited
# as R7c. Nothing here is newly tuned.
CONSENSUS_TAU = 0.15     # tau_c, soft mean-support threshold
CONSENSUS_GAMMA = 0.20   # gamma, per-view hard floor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-pseudo', required=True)
    p.add_argument('--unlabeled-infos', required=True)
    p.add_argument('--det-cache', default=None,
                   help='Required for perclass_* / boost_only variants; unused otherwise')
    p.add_argument('--variant', required=True,
                   choices=['single_thr', 'perclass_boost',
                            'perclass_only', 'boost_only',
                            'hybrid_posonly', 'perclass_and_consensus',
                            'perclass_random_prune',
                            'passthrough'])
    p.add_argument('--single-thr', type=float, default=SINGLE_THR_DEFAULT)
    p.add_argument('--out-pkl', required=True)
    p.add_argument('--prune-seed', type=int, default=19,
                   help='RNG seed for perclass_random_prune; fixed so the '
                        'volume-matched control is reproducible')
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    return p.parse_args()


def filter_perclass_boost(pb, sup, valid, mode='all'):
    """Returns keep mask using per-class threshold + consensus-boost.

    mode='all'         box kept iff (passes per-class thr) OR (passes boost)
    mode='perclass'    box kept iff (passes per-class thr) — boost branch off
    mode='boost'       box kept iff (passes boost) AND (fails per-class thr) —
                       isolates the boost-only contribution
    """
    N = len(pb['boxes_lidar'])
    if N == 0:
        return np.zeros((0,), dtype=bool)
    scores = pb['scores']; labels = pb['labels']
    n_valid = valid.sum(axis=1)
    sum_sup = sup.sum(axis=1)
    mean_sup = np.where(n_valid > 0, sum_sup / np.clip(n_valid, 1, None), 0.0)
    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        cls = NUSC_CLASSES[int(labels[i])]
        thr = PER_CLASS_THR[cls]
        passes_pc = scores[i] >= thr
        passes_boost = (scores[i] >= BOOST_FLOOR and
                        scores[i] < thr and
                        mean_sup[i] >= BOOST_THR)
        if mode == 'all':
            keep[i] = passes_pc or passes_boost
        elif mode == 'perclass':
            keep[i] = passes_pc
        elif mode == 'boost':
            keep[i] = passes_boost
        else:
            raise ValueError(f'bad mode: {mode}')
    return keep


def filter_single_thr(pb, thr):
    return pb['scores'] >= thr


def filter_hybrid_posonly(pb, sup, valid):
    """Class-aware positive-only camera-support rescue (Package B, R18).

    Non-G_COMMON classes: keep iff score >= PER_CLASS_THR[cls] — identical
    to perclass_only. G_COMMON classes additionally rescue boxes with
    BOOST_FLOOR <= score < thr when at least one valid camera gives
    non-zero support. Support never vetoes a box that passes its
    per-class threshold, so the mask is a superset of perclass_only.
    """
    N = len(pb['boxes_lidar'])
    if N == 0:
        return np.zeros((0,), dtype=bool)
    scores = pb['scores']; labels = pb['labels']
    has_valid = valid.sum(axis=1) > 0
    max_sup = sup.max(axis=1) if sup.size else np.zeros(N, dtype=np.float32)
    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        cls = NUSC_CLASSES[int(labels[i])]
        if scores[i] >= PER_CLASS_THR[cls]:
            keep[i] = True
        elif (cls in G_COMMON and scores[i] >= BOOST_FLOOR
              and has_valid[i] and max_sup[i] > 0):
            keep[i] = True
    return keep


def filter_perclass_and_consensus(pb, sup, valid):
    """R19 — conjunctive control: per-class score threshold AND six-camera
    consensus.

        keep = passes_perclass AND (consensus OR passthrough OR no_valid_cam)

    The camera term is the six-camera consensus keep-rule of
    g2_filter_pseudo.py reproduced exactly, including its passthrough branch
    for classes with no COCO mapping and its fallback for boxes no camera can
    see. Camera support therefore acts purely as a *veto* layered on top of
    the score gate, and only where camera evidence actually exists; everywhere
    else the rule degrades to perclass_only.

    This is the control that separates "camera evidence is worthless" from
    "removing the confidence floor is harmful": the mask is by construction a
    strict subset of perclass_only, so any downstream difference is
    attributable to the camera veto alone. It is the conjunctive structure
    used by DetMatch (threshold first, agreement second), which the audited
    camera-support family does not have.
    """
    N = len(pb['boxes_lidar'])
    if N == 0:
        return np.zeros((0,), dtype=bool)
    scores = pb['scores']; labels = pb['labels']

    passes_pc = np.array(
        [scores[i] >= PER_CLASS_THR[NUSC_CLASSES[int(labels[i])]]
         for i in range(N)], dtype=bool)

    passthrough = np.array(
        [not NUSC_TO_COCO.get(NUSC_CLASSES[int(l)], set()) for l in labels],
        dtype=bool)

    n_valid = valid.sum(axis=1)
    has_valid = n_valid > 0
    sum_sup = sup.sum(axis=1)
    mean_sup = np.where(n_valid > 0, sum_sup / np.clip(n_valid, 1, None), 0.0)
    hits = (sup >= CONSENSUS_GAMMA).sum(axis=1)
    m = np.array([m_thresh(int(n)) for n in n_valid])
    consensus = (mean_sup >= CONSENSUS_TAU) & (hits >= m) & has_valid

    camera_ok = consensus | passthrough | (~has_valid)
    return passes_pc & camera_ok


def filter_perclass_random_prune(pb, sup, valid, rng):
    """Volume-matched control for R19: prune the SAME number of boxes the
    conjunctive rule prunes, but choose them at random instead of by camera.

    R19 removes 61.6% of perclass_only's boxes, so its -8.18 NDS confounds
    "camera support vetoes the wrong boxes" with "supervision volume dropped by
    61.6%". This arm holds volume fixed and randomises only *which* boxes go:

        pool_c      = boxes of class c passing the per-class score threshold
        k_c         = |pool_c| - |pool_c surviving the R19 camera veto|
        keep        = pool_c minus k_c boxes drawn uniformly from pool_c

    Matching is done per (keyframe, class), which is finer than the per-dump or
    per-scene matching the paper asks for and implies both: every keyframe
    contributes exactly as many boxes of exactly the same classes as it does
    under R19. The two arms therefore differ in the *identity* of the removed
    boxes and in nothing else -- not count, not class composition, not which
    keyframes survive.

    Both masks are computed here in the same pass rather than read back from
    R19's output, so the counts cannot drift from the arm they are matched to.
    `rng` must be a seeded Generator supplied by the caller.
    """
    N = len(pb['boxes_lidar'])
    if N == 0:
        return np.zeros((0,), dtype=bool)
    labels = pb['labels']
    pool = filter_perclass_boost(pb, sup, valid, mode='perclass')
    vetoed = filter_perclass_and_consensus(pb, sup, valid)
    # R19's mask is a strict subset of the pool by construction; assert it,
    # because if it ever stops holding the matched count below is meaningless.
    if np.any(vetoed & ~pool):
        raise AssertionError('conjunctive mask is not a subset of perclass_only')

    keep = pool.copy()
    for cls_idx in np.unique(labels[pool]):
        in_cls = pool & (labels == cls_idx)
        k = int(in_cls.sum()) - int((vetoed & (labels == cls_idx)).sum())
        if k <= 0:
            continue
        idx = np.flatnonzero(in_cls)
        drop = rng.choice(idx, size=k, replace=False)
        keep[drop] = False
    return keep


def main():
    args = parse_args()
    out = Path(args.out_pkl)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'[+] Loading raw pseudo: {args.raw_pseudo}')
    raw = pickle.load(open(args.raw_pseudo, 'rb'))
    pseudo = raw['pseudo']
    print(f'    samples: {len(pseudo)}')

    print(f'[+] Loading unlabeled infos: {args.unlabeled_infos}')
    src = pickle.load(open(args.unlabeled_infos, 'rb'))
    info_by_tok = {i['token']: i for i in src['infos']}

    needs_support = args.variant in ('perclass_boost', 'perclass_only',
                                     'boost_only', 'hybrid_posonly',
                                     'perclass_and_consensus',
                                     'perclass_random_prune')
    if needs_support:
        if not args.det_cache:
            raise SystemExit(f'--det-cache is required for variant {args.variant}')
        print(f'[+] Loading det cache: {args.det_cache}')
        cache = pickle.load(open(args.det_cache, 'rb'))
        det_cache = cache['dets']
        coco_cats = cache['meta']['coco_categories']
        print('[+] Computing per-view support (full dump)...')
        supports = {}
        proc = 0
        for tok, pb in pseudo.items():
            info = info_by_tok.get(tok); dets = det_cache.get(tok)
            N = len(pb['boxes_lidar'])
            if info is None or dets is None:
                supports[tok] = (np.zeros((N, 6), dtype=np.float32),
                                 np.zeros((N, 6), dtype=bool))
                continue
            supports[tok] = box_supports(pb, info, dets, coco_cats,
                                          args.det_score_thr, args.iou2d_thr)
            proc += 1
            if proc % 2000 == 0:
                print(f'    {proc}/{len(pseudo)}')
    else:
        supports = None

    rng = np.random.default_rng(args.prune_seed)
    if args.variant == 'perclass_random_prune':
        print(f'[+] random-prune RNG seed: {args.prune_seed} '
              f'(numpy default_rng; iteration order is dict order of the '
              f'unlabeled infos, which is deterministic for a given input)')

    new_infos = []
    n_kept_total = 0
    n_dropped_total = 0
    n_no_pseudo_samples = 0
    for tok, info in info_by_tok.items():
        pb = pseudo.get(tok)
        if pb is None or len(pb['boxes_lidar']) == 0:
            n_no_pseudo_samples += 1
            continue
        if args.variant == 'single_thr':
            keep = filter_single_thr(pb, args.single_thr)
        elif args.variant == 'passthrough':
            # raw-pseudo is assumed to be ALREADY filtered (e.g., output of
            # g2_filter_pseudo.py). Keep every box; the filter step already
            # decided what passes. Empty-sample skip happens in main loop.
            keep = np.ones(len(pb['boxes_lidar']), dtype=bool)
        elif args.variant == 'hybrid_posonly':
            sup, valid = supports[tok]
            keep = filter_hybrid_posonly(pb, sup, valid)
        elif args.variant == 'perclass_and_consensus':
            sup, valid = supports[tok]
            keep = filter_perclass_and_consensus(pb, sup, valid)
        elif args.variant == 'perclass_random_prune':
            sup, valid = supports[tok]
            keep = filter_perclass_random_prune(pb, sup, valid, rng)
        else:
            sup, valid = supports[tok]
            mode = {'perclass_boost': 'all',
                    'perclass_only': 'perclass',
                    'boost_only': 'boost'}[args.variant]
            keep = filter_perclass_boost(pb, sup, valid, mode=mode)
        n_kept = int(keep.sum())
        n_dropped = len(keep) - n_kept
        n_kept_total += n_kept
        n_dropped_total += n_dropped
        if n_kept == 0:
            continue  # no boxes left → skip sample (no useful supervision)
        boxes_lidar = pb['boxes_lidar'][keep]   # (N, 9): x,y,z,w,l,h,yaw,vx,vy
        labels = pb['labels'][keep]
        # Build new info — keep all sample-level fields, replace gt_*
        new = copy.deepcopy(info)
        new['gt_boxes'] = boxes_lidar[:, :7].astype(np.float32)
        new['gt_velocity'] = boxes_lidar[:, 7:9].astype(np.float32)
        new['gt_names'] = np.array(
            [NUSC_CLASSES[int(l)] for l in labels], dtype='<U25')
        new['valid_flag'] = np.ones(len(boxes_lidar), dtype=bool)
        new['num_lidar_pts'] = np.full(len(boxes_lidar), 100, dtype=np.int64)
        new['num_radar_pts'] = np.zeros(len(boxes_lidar), dtype=np.int64)
        # ann_infos may exist on input; replace with pseudo-equivalent so any
        # downstream code that reads it sees consistent supervision
        new['ann_infos'] = (new['gt_boxes'].copy(), new['gt_names'].copy())
        new_infos.append(new)

    out_payload = {
        'infos': new_infos,
        'metadata': src.get('metadata', {'version': 'v1.0-trainval'}),
    }
    with open(out, 'wb') as f:
        pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'\n=== {args.variant} ===')
    print(f'  samples retained:  {len(new_infos)}/{len(info_by_tok)}')
    print(f'  samples skipped (no pseudo or all-dropped): {len(info_by_tok) - len(new_infos)}')
    print(f'    of which: no pseudo at all: {n_no_pseudo_samples}')
    print(f'  pseudo boxes kept:  {n_kept_total}')
    print(f'  pseudo boxes dropped: {n_dropped_total}')
    print(f'  output → {out}  ({out.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
