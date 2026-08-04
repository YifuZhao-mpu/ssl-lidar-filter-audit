#!/usr/bin/env python3
"""Pre-flight diagnostic for camera-support pseudo-label gating.

Motivation: the expensive way to learn that a camera-support keep rule will not
work is to train a student and read NDS 10-15 GPU-hours later. All three
components we identify as driving the family gap are visible in the pilot
caches *before* any student is trained. This script computes them.

The three checks, all from artifacts that exist after the teacher dump and the
2D detection cache -- i.e. at a cost of ~0 GPU-hours beyond what any
camera-support filter already requires:

  C1  Vocabulary coverage.   What share of the pseudo-label mass belongs to
      classes the 2D detector can fire on at all? Mass on classes outside the
      2D vocabulary cannot be gated; it either passes through unfiltered or is
      destroyed, and both change the label distribution the student sees.

  C2  View independence.     Mean off-diagonal Pearson correlation of per-camera
      support on the *real teacher* dump. Consensus rules buy error suppression
      from independence across views; if adjacent cameras agree, the votes are
      not independent and the suppression is not there to buy.

  C3  Gate selectivity.      Per class, does the camera gate remove false
      positives faster than true positives? A gate that lowers precision AND
      recall relative to the score gate is not trading anything -- it is
      strictly dominated, and is destroying labels rather than cleaning them.

Reported descriptively. These are pre-registration checks, not a calibrated
predictor -- see the caveat printed at the end and stated in the paper.

Usage:  python tools/g2_preflight_diagnostic.py [--seed 0]
Output: cache/preflight_diagnostic.json, figures/table_preflight.tex
"""
import argparse, json
from pathlib import Path

ROOT = Path('/workspace/BMVC2026')

# Frozen from the paper (sec:filters). The COCO-pretrained 2D detector has no
# label for these two nuScenes classes, so their per-camera support is
# identically zero by construction.
NON_COCO = {'barrier', 'traffic_cone'}
# construction_vehicle is mapped onto COCO 'truck' as an approximation; it is
# nominally in-vocabulary but the detector fires on it only weakly.
APPROX_MAPPED = {'construction_vehicle'}


def load(p):
    q = ROOT / p
    if not q.exists():
        raise SystemExit(f'missing required cache: {q}')
    return json.loads(q.read_text())


def check1_vocabulary(kept_pr, score_key):
    """C1 -- share of kept pseudo-label mass the 2D vocabulary can reach."""
    per_class = kept_pr[score_key]['per_class']
    total = sum(v['kept'] for v in per_class.values())
    out_of_vocab = sum(v['kept'] for c, v in per_class.items() if c in NON_COCO)
    approx = sum(v['kept'] for c, v in per_class.items() if c in APPROX_MAPPED)
    return {
        'kept_total': total,
        'out_of_vocab_boxes': out_of_vocab,
        'out_of_vocab_share': out_of_vocab / total,
        'approx_mapped_share': approx / total,
        'covered_share': 1.0 - out_of_vocab / total,
        'per_class_kept': {c: v['kept'] for c, v in per_class.items()},
    }


def check2_independence(pooled, perclass):
    """C2 -- is per-camera support independent across views on a real teacher?"""
    pc = {c: v['real_offdiag_mean'] for c, v in perclass['per_class'].items()}
    # Classes with r == 0 exactly are uninformative nulls (the detector never
    # fires), not evidence of independence -- separate them out.
    informative = {c: r for c, r in pc.items() if c not in NON_COCO}
    return {
        'real_offdiag_mean': pooled['real_offdiag_mean'],
        'real_offdiag_median': pooled['real_offdiag_median'],
        'real_offdiag_p90': pooled['real_offdiag_p90'],
        'scaffold_offdiag_mean': pooled['pilot_offdiag_mean'],
        'n_boxes': pooled['n_boxes'],
        'per_class_real_r': pc,
        'max_informative_class_r': max(informative.values()) if informative else None,
        'null_classes': sorted(c for c in pc if c in NON_COCO),
    }


def check3_selectivity(kept_pr, score_key, camera_key):
    """C3 -- does the camera gate strip noise or signal, class by class?"""
    base = kept_pr[score_key]['per_class']
    cam = kept_pr[camera_key]['per_class']
    rows, destructive = [], []
    for c in sorted(base):
        if c not in cam:
            continue
        b, k = base[c], cam[c]
        if b['kept'] == 0:
            continue
        keep_ratio = k['kept'] / b['kept']
        dP = k['P'] - b['P']
        dR = k['R'] - b['R']
        # A gate exists to buy precision. Three outcomes are worth separating:
        #   dominated -- precision AND recall both fall. The gate is not trading
        #                anything; it is destroying labels. Nothing justifies it.
        #   trade     -- precision rises, recall falls. A real trade-off, which
        #                may or may not pay off downstream.
        #   ok        -- precision rises without a recall cost.
        if dP <= 0.0 and dR < 0.0:
            verdict = 'dominated'
        elif dP > 0.0 and dR < 0.0:
            verdict = 'trade'
        else:
            verdict = 'ok'
        rows.append({'class': c, 'kept_base': b['kept'], 'kept_gate': k['kept'],
                     'keep_ratio': keep_ratio, 'P_base': b['P'], 'P_gate': k['P'],
                     'dP': dP, 'R_base': b['R'], 'R_gate': k['R'], 'dR': dR,
                     'verdict': verdict,
                     'out_of_vocab': c in NON_COCO})
        if verdict == 'dominated':
            destructive.append(c)
    in_vocab = [r for r in rows if not r['out_of_vocab']]
    return {
        'rows': rows,
        'dominated_classes': destructive,
        'n_dominated_in_vocab': sum(1 for r in in_vocab if r['verdict'] == 'dominated'),
        'n_trade_in_vocab': sum(1 for r in in_vocab if r['verdict'] == 'trade'),
        'n_in_vocab': len(in_vocab),
        'median_keep_ratio_in_vocab': sorted(r['keep_ratio'] for r in in_vocab)[len(in_vocab) // 2],
    }


def emit_table(res, out):
    """Compact LaTeX tabular: the three checks, their value, and the reading."""
    c1, c2, c3 = res['C1_vocabulary'], res['C2_independence'], res['C3_selectivity']
    L = []
    L.append(r'\begin{tabular}{clcl}')
    L.append(r'\toprule')
    L.append(r'Check & Quantity & Value & Reading \\')
    L.append(r'\midrule')
    L.append('C1 & pseudo-label mass outside the 2D vocabulary & '
             f'{c1["out_of_vocab_share"]*100:.1f}\\% & cannot be gated \\\\')
    L.append('   & \\quad of which approximately mapped & '
             f'{c1["approx_mapped_share"]*100:.1f}\\% & weak signal \\\\')
    L.append(r'\midrule')
    L.append('C2 & mean off-diag.\\ support $r$, real teacher & '
             f'{c2["real_offdiag_mean"]:.3f} & not independent \\\\')
    L.append('   & \\quad same, perturbed-GT scaffold & '
             f'{c2["scaffold_offdiag_mean"]:.3f} & holds only here \\\\')
    L.append(r'\midrule')
    L.append('C3 & in-vocabulary classes strictly dominated & '
             f'{c3["n_dominated_in_vocab"]}/{c3["n_in_vocab"]} & $P$ \\emph{{and}} $R$ both fall \\\\')
    L.append('   & \\quad median kept-box ratio (in-vocab.) & '
             f'{c3["median_keep_ratio_in_vocab"]:.2f} & '
             + r'vs.\ score gate \\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    Path(out).write_text('\n'.join(L))
    print(f'  wrote: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-filter', default='R7b_perclass_only')
    ap.add_argument('--camera-filter', default='R7c_consensus')
    args = ap.parse_args()

    kept_pr = load('cache/perclass_kept_pr_5p_s0.json')
    pooled = load('cache/d2_correlation_analysis.json')
    perclass = load('cache/d2_correlation_perclass.json')

    res = {
        'score_filter': args.score_filter,
        'camera_filter': args.camera_filter,
        'C1_vocabulary': check1_vocabulary(kept_pr, args.score_filter),
        'C2_independence': check2_independence(pooled, perclass),
        'C3_selectivity': check3_selectivity(kept_pr, args.score_filter,
                                             args.camera_filter),
    }

    out = ROOT / 'cache/preflight_diagnostic.json'
    out.write_text(json.dumps(res, indent=2))
    print(f'  wrote: {out}')
    emit_table(res, ROOT / 'figures/table_preflight.tex')

    c1, c2, c3 = res['C1_vocabulary'], res['C2_independence'], res['C3_selectivity']
    print('\n=== Pre-flight diagnostic ===')
    print(f'C1 vocabulary : {c1["out_of_vocab_share"]*100:5.1f}% of kept mass is out of vocabulary '
          f'({c1["out_of_vocab_boxes"]:,} / {c1["kept_total"]:,} boxes)')
    print(f'                {c1["approx_mapped_share"]*100:5.1f}% is only approximately mapped')
    print(f'C2 independence: real r = {c2["real_offdiag_mean"]:.3f} '
          f'(median {c2["real_offdiag_median"]:.3f}, p90 {c2["real_offdiag_p90"]:.3f}) '
          f'vs scaffold r = {c2["scaffold_offdiag_mean"]:.3f} on n={c2["n_boxes"]:,}')
    print(f'                strongest in-vocabulary class r = {c2["max_informative_class_r"]:.3f}; '
          f'null (detector-silent) classes: {", ".join(c2["null_classes"])}')
    print(f'C3 selectivity : {c3["n_dominated_in_vocab"]}/{c3["n_in_vocab"]} in-vocabulary classes strictly '
          f'dominated (P and R both fall); {c3["n_trade_in_vocab"]} trade P for R; '
          f'median kept ratio {c3["median_keep_ratio_in_vocab"]:.2f}')
    print('\n  per-class detail (in-vocabulary first):')
    for r in sorted(c3['rows'], key=lambda r: (r['out_of_vocab'], r['class'])):
        flag = 'OOV' if r['out_of_vocab'] else {'dominated': 'DOMINATED', 'trade': 'trade', 'ok': 'ok'}[r['verdict']]
        print(f'    {r["class"]:22s} keep x{r["keep_ratio"]:5.2f}  '
              f'dP {r["dP"]:+6.2f}  dR {r["dR"]:+6.2f}   [{flag:>9s}]')
    print('\nCaveat: these are descriptive pre-registration checks, not a calibrated')
    print('predictor. Validating them as a predictor would need setups where camera')
    print('gating succeeds; this study contains none.')


if __name__ == '__main__':
    main()
