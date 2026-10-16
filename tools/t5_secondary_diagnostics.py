#!/usr/bin/env python3
"""T5 secondary analyses 3-4 (PREREG_2026-10-06_TRUE5P.md): kept-box counts,
per-class pseudo P/R per arm, and the pre-flight checklist C1-C3 recomputed on
the T5 teacher dumps. Descriptive only; no decision rides on these.

Inputs per seed s:
  infos/nuscenes_pseudo_t5_5p_s{s}_{arm}.pkl   the label sets the students saw
  infos/nuscenes_infos_train_5p_s{s}_unlabeled.pkl   GT of the unlabeled split
  cache/d2_correlation_conditional_t5_5p_s{s}.json   C2 (from
      tools/g2_support_correlation_conditional.py on the T5 raw dump)

Matching, P/R and the C1/C3 definitions are the published ones
(tools/g2_perclass_kept_pr.py, tools/g2_preflight_diagnostic.py).

Usage:  PYTHONPATH=. python3 tools/t5_secondary_diagnostics.py
Output: cache/perclass_kept_pr_t5_5p_s{s}.json, cache/t5_secondary.json,
        cache/T5_SECONDARY.md
"""
import json, os, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get('ARIS_PROJECT_ROOT',
                           Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))
from tools.g2_filter_pseudo import NUSC_CLASSES                  # noqa: E402
from tools.g2_perclass_kept_pr import compute_per_class_metrics  # noqa: E402
from tools.g2_preflight_diagnostic import (check1_vocabulary,    # noqa: E402
                                           check3_selectivity)

ARMS = {0: ['single_thr', 'perclass_only', 'best_view', 'consensus',
            'perclass_and_consensus'],
        1: ['single_thr', 'perclass_only', 'best_view', 'consensus',
            'perclass_and_consensus'],
        2: ['perclass_only', 'best_view']}
SCORE_KEY, CAMERA_KEY = 'perclass_only', 'consensus'
NON_COCO = {'barrier', 'traffic_cone'}
CLS_IDX = {c: i for i, c in enumerate(NUSC_CLASSES)}


def load_arm(path, tokens):
    """Pseudo infos -> {token: {'boxes_lidar', 'labels'}} over every U token."""
    infos = pickle.load(open(path, 'rb'))['infos']
    by_tok = {i['token']: i for i in infos}
    out = {}
    for t in tokens:
        i = by_tok.get(t)
        if i is None or len(i['gt_names']) == 0:
            out[t] = {'boxes_lidar': np.zeros((0, 7), np.float32),
                      'labels': np.zeros((0,), np.int64)}
            continue
        out[t] = {'boxes_lidar': np.asarray(i['gt_boxes'])[:, :7],
                  'labels': np.array([CLS_IDX[n] for n in i['gt_names']])}
    return out, len(infos)


def add_pr(pc):
    for d in pc.values():
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        d['P'] = 100.0 * tp / max(tp + fp, 1)
        d['R'] = 100.0 * tp / max(tp + fn, 1)
        d['F1'] = 2 * d['P'] * d['R'] / max(d['P'] + d['R'], 1e-6)
    return pc


def c2_from_conditional(path):
    d = json.loads(Path(path).read_text())
    cm = d['coco_mapped']
    return {
        'pooled_r_all_classes': d['all_classes']['all']['mean_offdiag_r'],
        'coco_mapped_r': {g: cm[g]['mean_offdiag_r'] for g in ('all', 'TP', 'FP')},
        'coco_mapped_vote_lift': {g: cm[g]['vote_lift_at_gamma'] for g in ('all', 'TP', 'FP')},
        'per_class_r': {c: v['all']['mean_offdiag_r'] for c, v in d['per_class'].items()
                        if c not in NON_COCO},
        'per_class_fp_lift': {c: v['FP']['vote_lift_at_gamma'] for c, v in d['per_class'].items()
                              if c not in NON_COCO and 'FP' in v},
        'n_boxes': d['meta']['n_boxes_total'],
    }


def main():
    res = {}
    for s, arms in ARMS.items():
        src = pickle.load(open(REPO / f'infos/nuscenes_infos_train_5p_s{s}_unlabeled.pkl', 'rb'))
        gt = src['infos']
        tokens = [i['token'] for i in gt]
        kept_pr = {}
        for arm in arms:
            dump, n_inf = load_arm(REPO / f'infos/nuscenes_pseudo_t5_5p_s{s}_{arm}.pkl', tokens)
            pc = add_pr(compute_per_class_metrics(dump, gt))
            tot = {k: sum(v[k] for v in pc.values()) for k in ('kept', 'tp', 'fp', 'fn')}
            tot['P'] = 100.0 * tot['tp'] / max(tot['tp'] + tot['fp'], 1)
            tot['R'] = 100.0 * tot['tp'] / max(tot['tp'] + tot['fn'], 1)
            kept_pr[arm] = {'kept_total': tot['kept'], 'overall': tot,
                            'n_info_records': n_inf, 'per_class': pc}
            print(f's{s} {arm:24s} kept={tot["kept"]:>9,d}  P={tot["P"]:5.1f}  R={tot["R"]:5.1f}',
                  flush=True)
        (REPO / f'cache/perclass_kept_pr_t5_5p_s{s}.json').write_text(json.dumps(kept_pr, indent=2))
        r = {'kept': {a: kept_pr[a]['overall'] for a in arms},
             'C1_vocabulary': check1_vocabulary(kept_pr, SCORE_KEY)}
        if CAMERA_KEY in kept_pr:
            r['C3_selectivity'] = check3_selectivity(kept_pr, SCORE_KEY, CAMERA_KEY)
        cpath = REPO / f'cache/d2_correlation_conditional_t5_5p_s{s}.json'
        if cpath.exists():
            r['C2_independence'] = c2_from_conditional(cpath)
        res[f's{s}'] = r
    (REPO / 'cache/t5_secondary.json').write_text(json.dumps(res, indent=2))

    L = ['# T5 secondary analyses 3-4 (descriptive)', '',
         'Kept counts and pseudo P/R against U ground truth (same-class BEV centre <= 2 m).', '']
    for sk, r in res.items():
        L.append(f'## Seed {sk[1:]}')
        for a, o in r['kept'].items():
            L.append(f'- {a}: kept {o["kept"]:,}  P {o["P"]:.1f}  R {o["R"]:.1f}')
        c1 = r['C1_vocabulary']
        L.append(f'- C1: {100*c1["out_of_vocab_share"]:.1f}% of {SCORE_KEY} kept mass out of '
                 f'vocabulary ({c1["out_of_vocab_boxes"]:,} / {c1["kept_total"]:,}); '
                 f'{100*c1["approx_mapped_share"]:.1f}% approximately mapped')
        if 'C2_independence' in r:
            c2 = r['C2_independence']
            L.append(f'- C2: pooled r {c2["pooled_r_all_classes"]:.3f}; COCO-mapped r '
                     f'TP {c2["coco_mapped_r"]["TP"]:.2f} / FP {c2["coco_mapped_r"]["FP"]:.2f}; '
                     f'vote lift TP {c2["coco_mapped_vote_lift"]["TP"]:.1f} / '
                     f'FP {c2["coco_mapped_vote_lift"]["FP"]:.1f}')
        if 'C3_selectivity' in r:
            c3 = r['C3_selectivity']
            L.append(f'- C3 ({CAMERA_KEY} vs {SCORE_KEY}): {c3["n_dominated_in_vocab"]}/'
                     f'{c3["n_in_vocab"]} in-vocabulary classes dominated, '
                     f'{c3["n_trade_in_vocab"]} trade; median keep ratio '
                     f'{c3["median_keep_ratio_in_vocab"]:.2f}')
            for row in sorted(c3['rows'], key=lambda x: (x['out_of_vocab'], x['class'])):
                tag = 'OOV' if row['out_of_vocab'] else row['verdict']
                L.append(f'  - {row["class"]}: keep x{row["keep_ratio"]:.2f}, '
                         f'dP {row["dP"]:+.1f}, dR {row["dR"]:+.1f} [{tag}]')
        L.append('')
    (REPO / 'cache/T5_SECONDARY.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
