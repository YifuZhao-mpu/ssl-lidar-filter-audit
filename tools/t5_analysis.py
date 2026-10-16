#!/usr/bin/env python3
"""Pre-registered analysis of the true-5% initialization replication (T5).

Reads the paired scene-level bootstrap written by
`tools/g2_bootstrap_ci.py --runs t5 --out cache/bootstrap_ci_T5.json` and applies
exactly the quantities and decision rules fixed in
review-stage/PREREG_2026-10-06_TRUE5P.md:

  primary (seeds 0, 1):
    G_s = min(NDS single_thr, NDS perclass_only) - max(NDS best_view, NDS consensus)
    plus the 4 score x camera contrasts, each with a paired interval
  decision:
    F1  all 4 contrasts > 0 with percentile 95% intervals excluding zero, both seeds
    F3  G_s <= 0 (point estimate) on at least one primary seed
    F2  otherwise, or G_s < half the published same-seed gap (s0 9.15, s1 7.23 NDS)
  secondary: student - teacher, student - supcont, conjunctive - perclass_only,
             seed-2 anchors perclass_only - best_view

Intervals: percentile (primary) and basic, in NDS points.
Usage: python3 tools/t5_analysis.py [--boot cache/bootstrap_ci_T5.json] [--out-md ...]
"""
import argparse, json
import numpy as np

PUBLISHED_GAP = {0: 9.15, 1: 7.23}
PRIMARY = (0, 1)


def load(path):
    d = json.load(open(path))
    rep = {}
    for k, v in d['replicate_nds'].items():
        bs = sorted(int(b) for b in v if int(b) >= 0)
        rep[k] = (float(v['-1']), np.array([v[str(b)] for b in bs]), bs)
    return rep


def contrast(rep, a, b):
    pa, ra, ba = rep[a]; pb, rb, bb = rep[b]
    assert ba == bb, f'replicate sets differ for {a} vs {b}'
    pt = (pa - pb) * 100
    d = (ra - rb) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'a': a, 'b': b, 'point': pt, 'pct': [lo, hi],
            'basic': [2 * pt - hi, 2 * pt - lo], 'frac_le_0': float(np.mean(d <= 0)), 'B': len(d)}


def gap(rep, s):
    S = [f'T5single_s{s}', f'T5perclass_s{s}']
    C = [f'T5best_s{s}', f'T5cons_s{s}']
    pts = {k: rep[k][0] for k in S + C}
    bs = rep[S[0]][2]
    for k in S + C:
        assert rep[k][2] == bs
    g = (np.min([rep[k][1] for k in S], 0) - np.max([rep[k][1] for k in C], 0)) * 100
    pt = (min(pts[k] for k in S) - max(pts[k] for k in C)) * 100
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {'point': pt, 'pct': [lo, hi], 'basic': [2 * pt - hi, 2 * pt - lo],
            'frac_le_0': float(np.mean(g <= 0))}


def fmt(c):
    return (f"{c['point']:+6.2f}  pct [{c['pct'][0]:+6.2f}, {c['pct'][1]:+6.2f}]"
            f"  basic [{c['basic'][0]:+6.2f}, {c['basic'][1]:+6.2f}]  P(<=0)={c['frac_le_0']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', default='cache/bootstrap_ci_T5.json')
    ap.add_argument('--out-json', default='cache/t5_analysis.json')
    ap.add_argument('--out-md', default='cache/T5_ANALYSIS.md')
    a = ap.parse_args()
    rep = load(a.boot)
    out = {'primary': {}, 'secondary': {}}
    lines = ['# T5 analysis (pre-registered rules, PREREG_2026-10-06_TRUE5P.md)', '',
             'NDS points; percentile intervals primary, basic alongside.', '']

    f1_all, f3_any, f2_flag = True, False, False
    for s in PRIMARY:
        g = gap(rep, s)
        cs = [contrast(rep, f'T5{x}_s{s}', f'T5{y}_s{s}')
              for x in ('single', 'perclass') for y in ('best', 'cons')]
        out['primary'][s] = {'gap': g, 'contrasts': cs}
        f1_all &= all(c['point'] > 0 and c['pct'][0] > 0 for c in cs)
        f3_any |= g['point'] <= 0
        f2_flag |= g['point'] < 0.5 * PUBLISHED_GAP[s]
        lines += [f'## Seed {s}', f'- family gap G = {fmt(g)}  (published same-seed gap {PUBLISHED_GAP[s]})']
        lines += [f"- {c['a']} - {c['b']}: {fmt(c)}" for c in cs] + ['']
    if f3_any:
        decision = 'F3'
    elif f1_all and not f2_flag:
        decision = 'F1'
    else:
        decision = 'F2'
    out['decision'] = decision
    lines += [f'## Decision: **{decision}**', '']

    sec = []
    for s in (0, 1, 2):
        for st in ('single', 'perclass', 'best', 'cons', 'conj'):
            k = f'T5{st}_s{s}'
            if k not in rep:
                continue
            for ref in (f'T5L_s{s}', f'T5LC_s{s}'):
                if ref in rep:
                    sec.append(contrast(rep, k, ref))
        if f'T5conj_s{s}' in rep and f'T5perclass_s{s}' in rep:
            sec.append(contrast(rep, f'T5conj_s{s}', f'T5perclass_s{s}'))
    if 'T5perclass_s2' in rep and 'T5best_s2' in rep:
        sec.append(contrast(rep, 'T5perclass_s2', 'T5best_s2'))
    out['secondary'] = sec
    lines += ['## Secondary contrasts'] + [f"- {c['a']} - {c['b']}: {fmt(c)}" for c in sec]
    lines += ['', '## Point estimates (NDS)'] + [f'- {k}: {v[0]:.4f}' for k, v in sorted(rep.items())]

    json.dump(out, open(a.out_json, 'w'), indent=2, default=float)
    open(a.out_md, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
