#!/usr/bin/env python3
"""Inter-camera support dependence, conditioned on whether the pseudo-box is right.

The published diagnostic (tools/g2_support_correlation.py) reports the
*unconditional* mean off-diagonal Pearson r of per-camera support (0.42 on the
5% s0 dump) against an iid scaffold (0.03). Reviewers pointed out that this
mixes two things: cameras that all see the same real object agree because the
object is there (shared signal), which a consensus vote is supposed to exploit;
what a consensus vote must NOT have is correlated *errors*. This script splits
the same boxes into true and false positives and reports the dependence within
each group.

Everything upstream is identical to the published diagnostic: same raw dump,
same 5000-keyframe subsample (RandomState(0)), same >=2-valid-view filter, same
2D detection cache and thresholds, same `box_supports` routine.

TP/FP labels: greedy one-to-one same-class BEV centre matching at <= 2.0 m
against the keyframe's GT (the paper's cocoF1 matching rule), boxes taken in
descending teacher score.

Reported per group (all / TP / FP) and restricted to the eight COCO-mapped
classes (support is identically zero on barrier / traffic_cone):
  * mean off-diagonal Pearson r over camera pairs (pairwise-complete views);
  * vote lift: P(view j >= gamma | view i >= gamma) / P(view j >= gamma),
    averaged over ordered camera pairs, at the consensus floor gamma = 0.20.
    Lift 1 = votes independent; > 1 = votes co-occur.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np

from tools.g2_filter_pseudo import NUSC_CLASSES, NUSC_TO_COCO, box_supports

GAMMA = 0.20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-pseudo', required=True)
    p.add_argument('--source-infos', required=True)
    p.add_argument('--det-cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-samples', type=int, default=5000)
    p.add_argument('--iou2d-thr', type=float, default=0.30)
    p.add_argument('--det-score-thr', type=float, default=0.30)
    return p.parse_args()


def tp_mask(boxes, labels, scores, info):
    gt_boxes, gt_names = info.get('gt_boxes'), info.get('gt_names')
    out = np.zeros(len(boxes), dtype=bool)
    if gt_boxes is None or gt_names is None or len(gt_names) == 0:
        return out
    idx = {c: i for i, c in enumerate(NUSC_CLASSES)}
    gcls = np.array([idx.get(g, -1) for g in gt_names])
    d = np.sqrt(((boxes[:, None, :2] - gt_boxes[None, :, :2]) ** 2).sum(-1))
    d = np.where(labels[:, None] == gcls[None, :], d, 1e9)
    matched = np.zeros(len(gt_names), dtype=bool)
    for i in np.argsort(-scores):
        cand = np.where(~matched & (d[i] <= 2.0))[0]
        if cand.size:
            matched[cand[np.argmin(d[i, cand])]] = True
            out[i] = True
    return out


def corr_offdiag(sup, valid):
    x = np.where(valid, sup, np.nan)
    K = x.shape[1]
    rs = []
    for i in range(K):
        for j in range(i + 1, K):
            m = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            if m.sum() < 100:
                continue
            a, b = x[m, i], x[m, j]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return (float(np.mean(rs)) if rs else None), len(rs)


def vote_lift(sup, valid, gamma=GAMMA):
    K = sup.shape[1]
    lifts = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            m = valid[:, i] & valid[:, j]
            if m.sum() < 100:
                continue
            vi = sup[m, i] >= gamma
            vj = sup[m, j] >= gamma
            pj = vj.mean()
            if vi.sum() < 20 or pj <= 0:
                continue
            lifts.append(float(vj[vi].mean() / pj))
    return (float(np.mean(lifts)) if lifts else None), len(lifts)


def summarize(sup, valid, cls, sel):
    r, nr = corr_offdiag(sup[sel], valid[sel])
    lift, nl = vote_lift(sup[sel], valid[sel])
    return {'n_boxes': int(sel.sum()), 'mean_offdiag_r': r, 'n_pairs_r': nr,
            'vote_lift_at_gamma': lift, 'n_pairs_lift': nl,
            'vote_rate_at_gamma': float((sup[sel] >= GAMMA)[valid[sel]].mean()) if sel.any() else None}


def main():
    a = parse_args()
    raw = pickle.load(open(a.raw_pseudo, 'rb'))['pseudo']
    infos = {i['token']: i for i in pickle.load(open(a.source_infos, 'rb'))['infos']}
    cache = pickle.load(open(a.det_cache, 'rb'))
    dets_all, coco_cats = cache['dets'], cache['meta']['coco_categories']

    rng = np.random.RandomState(0)                       # identical subsample
    tokens = list(raw.keys())
    if len(tokens) > a.max_samples:
        tokens = list(rng.choice(tokens, a.max_samples, replace=False))

    S, V, C, T = [], [], [], []
    for n, tok in enumerate(tokens):
        pb = raw[tok]; info = infos.get(tok); dets = dets_all.get(tok)
        if info is None or dets is None or len(pb['boxes_lidar']) == 0:
            continue
        sup, valid = box_supports(pb, info, dets, coco_cats, a.det_score_thr, a.iou2d_thr)
        tp = tp_mask(pb['boxes_lidar'], pb['labels'], pb['scores'], info)
        keep = valid.sum(axis=1) >= 2
        if keep.any():
            S.append(sup[keep]); V.append(valid[keep]); C.append(pb['labels'][keep]); T.append(tp[keep])
        if (n + 1) % 1000 == 0:
            print(f'  {n+1}/{len(tokens)}', flush=True)
    sup, valid = np.concatenate(S), np.concatenate(V)
    cls, tp = np.concatenate(C), np.concatenate(T)
    coco = np.array([bool(NUSC_TO_COCO.get(NUSC_CLASSES[int(c)])) for c in cls])

    res = {'meta': {'raw_pseudo': a.raw_pseudo, 'max_samples': a.max_samples,
                    'gamma': GAMMA, 'match': 'greedy same-class BEV centre <= 2.0 m, score-desc',
                    'n_boxes_total': int(len(sup))},
           'all_classes': {g: summarize(sup, valid, cls, m) for g, m in
                           (('all', np.ones(len(sup), bool)), ('TP', tp), ('FP', ~tp))},
           'coco_mapped': {g: summarize(sup, valid, cls, m & coco) for g, m in
                           (('all', np.ones(len(sup), bool)), ('TP', tp), ('FP', ~tp))},
           'per_class': {}}
    for c in ('car', 'pedestrian', 'truck'):
        ci = NUSC_CLASSES.index(c)
        res['per_class'][c] = {g: summarize(sup, valid, cls, (cls == ci) & m) for g, m in
                               (('all', np.ones(len(sup), bool)), ('TP', tp), ('FP', ~tp))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
