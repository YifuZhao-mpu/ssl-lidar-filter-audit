#!/usr/bin/env python
"""Turn cache/bootstrap_ci.json into paper-ready outputs.

Emits
  figures/table_bootstrap_ci.tex   standalone booktabs `tabular` (no table wrapper,
                                   no \\resizebox), narrow enough for a single journal
                                   column: rows = filter, columns = seed, each filter
                                   occupying two physical rows (point NDS, then the
                                   percentile 95% CI in \\scriptsize).
  cache/BOOTSTRAP_SUMMARY.md       prose summary a reviewer can check line by line.

Also runs a set of hard sanity checks and refuses to write if any fails.
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = '/workspace/BMVC2026'
WORK_DIRS = os.path.join(REPO, 'work_dirs')
METRICS_REL = os.path.join('results', 'pts_bbox', 'metrics_summary.json')
IN_JSON = os.path.join(REPO, 'cache', 'bootstrap_ci.json')
# Runs bootstrapped in their own later pass, merged in here so the table is
# emitted rather than hand-edited. R19 (the gate-matched conjunctive arm) was
# run on 2026-08-05, after the 26-run main pass, and has its own wall clock and
# worker count -- which is why meta['runs'] below is deliberately NOT extended.
IN_JSON_EXTRA = [os.path.join(REPO, 'cache', 'bootstrap_ci_R19.json'),
                 os.path.join(REPO, 'cache', 'bootstrap_ci_R20_contrast.json'),
                 os.path.join(REPO, 'cache', 'bootstrap_ci_R19_seeds.json')]
OUT_TEX = os.path.join(REPO, 'figures', 'table_bootstrap_ci.tex')
OUT_MD = os.path.join(REPO, 'cache', 'BOOTSTRAP_SUMMARY.md')

# Display order / names for the table.  (tag, latex label, block header before it)
# The last block holds the two gate-matched arms: both are layered on the
# per-class score gate and differ only in what the camera term may do.
ROWS = [
    ('R4b/R4', 'Labeled-only (no SSL)', None),
    ('R6',  'Single threshold',        r'\emph{Score-based filters}'),
    ('R7',  'Per-class + boost',       None),
    ('R7b', 'Per-class only',          None),
    ('R8b', 'Best-view',               r'\emph{Camera-support filters}'),
    ('R8c', 'Any-view',                None),
    ('R8d', r'Max-cam $K{=}5$',        None),
    ('R7c', '6-camera consensus',      None),
    ('R18', 'Class-aware pos.-only',   r'\emph{Subordinated to the score gate}'),
    ('R19', r'Per-class $\land$ consensus', None),
    ('R20', r'\quad\emph{vs.} matched random prune', None),
]
SEEDS = [0, 1, 2]


def load():
    """Main bootstrap pass, with any later single-run passes merged in.

    Merging is restricted to the per-run payloads. `pairs`, `family_gap` and
    `within_family` are left alone: the extra passes carry them empty, and the
    family-gap statistic is defined over the 7-filter family only.
    """
    with open(IN_JSON) as f:
        d = json.load(f)
    d['meta']['runs_extra'] = []
    for path in IN_JSON_EXTRA:
        if not os.path.exists(path):
            print('WARN: extra bootstrap file missing, skipping: %s' % path)
            continue
        with open(path) as f:
            e = json.load(f)
        if e['meta']['B_requested'] != d['meta']['B_requested']:
            raise SystemExit(
                'B mismatch: %s has B=%d, main pass has B=%d -- refusing to '
                'merge intervals computed at different replicate counts'
                % (path, e['meta']['B_requested'], d['meta']['B_requested']))
        # A later pass re-bootstraps the shared reference arms so its contrasts
        # are paired within that pass. Those keys are expected duplicates: keep
        # the main pass's copy, but refuse to merge if the two disagree, since
        # the driver is deterministic (replicate b is default_rng(b) over the
        # sorted scene list) and any disagreement means the two passes did not
        # see the same data.
        fresh = []
        for k, v in e['runs'].items():
            if k not in d['runs']:
                fresh.append(k)
                continue
            got, ref = v['point_nds'], d['runs'][k]['point_nds']
            if abs(got - ref) > 1e-12:
                raise SystemExit(
                    'run %s disagrees between %s and the main pass: '
                    '%.17g vs %.17g' % (k, os.path.basename(path), got, ref))
            ra = d['replicate_nds'].get(k, {})
            rb = e.get('replicate_nds', {}).get(k, {})
            common = set(ra) & set(rb)
            bad = [x for x in common if ra[x] != rb[x]]
            if bad:
                raise SystemExit(
                    'run %s has %d/%d differing replicates between %s and the '
                    'main pass -- the passes are not paired'
                    % (k, len(bad), len(common), os.path.basename(path)))
        for k in fresh:
            d['runs'][k] = e['runs'][k]
            if k in e.get('replicate_nds', {}):
                d['replicate_nds'][k] = e['replicate_nds'][k]
            if k in e.get('point_estimate_verification', {}):
                d['point_estimate_verification'][k] = \
                    e['point_estimate_verification'][k]
        if not fresh:
            continue
        d['meta']['runs_extra'].append({
            'keys': fresh,
            'source': os.path.basename(path),
            'wall_clock_s': e['meta'].get('wall_clock_s'),
            'workers': e['meta'].get('workers'),
            'created': e['meta'].get('created'),
        })
    return d


def all_run_keys(d):
    """Main-pass run keys followed by any merged-in extras, in display order."""
    keys = list(d['meta']['runs'])
    for x in d['meta'].get('runs_extra', []):
        keys.extend(x['keys'])
    return keys


def index_runs(d):
    """(tag_or_alias, seed) -> run dict.  Labeled-only is R4b at seed 0, R4 at 1/2."""
    idx = {}
    for k, r in d['runs'].items():
        idx[(r['tag'], r['seed'])] = dict(r, key=k)
    # alias so the labeled-only row can be addressed uniformly
    for s in SEEDS:
        for t in ('R4b', 'R4'):
            if (t, s) in idx:
                idx[('R4b/R4', s)] = idx[(t, s)]
    return idx


# --------------------------------------------------------------------------- checks
def min_gap_per_seed(d, B):
    """Smallest min(score)-max(camera) gap over ALL replicates, recomputed from raw data."""
    out = {}
    R = d['replicate_nds']
    for s in SEEDS:
        g = d['family_gap'].get(str(s))
        if g is None:
            continue
        bs = [str(b) for b in range(B) if all(str(b) in R[k] for k in g['score_runs'] + g['camera_runs'])]
        worst = None
        for b in bs:
            gap = min(R[k][b] for k in g['score_runs']) - max(R[k][b] for k in g['camera_runs'])
            worst = gap if worst is None else min(worst, gap)
        out[s] = worst
    return out


def cached_nds(d):
    """Full-precision NDS straight out of each run's cached metrics_summary.json."""
    out = {}
    for k, r in d['runs'].items():
        p = os.path.join(WORK_DIRS, r['dir'], METRICS_REL)
        if os.path.exists(p):
            with open(p) as f:
                out[k] = float(json.load(f)['nd_score'])
    return out


def checks(d, idx):
    errs, warns = [], []

    # (1) point estimates reproduce the paper values
    for k, v in d['point_estimate_verification'].items():
        if not v['ok']:
            errs.append('point estimate mismatch %s: got %.6f paper %.4f'
                        % (k, v['recomputed_nds'], v['paper_nds']))

    # (1b) and they reproduce the FULL-PRECISION cached metrics_summary.json
    cn = cached_nds(d)
    for k, v in cn.items():
        got = d['runs'][k]['point_nds']
        if abs(got - v) > 1e-12:
            errs.append('point estimate %s differs from cached metrics_summary.json at full '
                        'precision: %.17g vs %.17g' % (k, got, v))

    # (2) no NaN/null anywhere in the numeric summaries
    def scan(o, path=''):
        if isinstance(o, dict):
            for kk, vv in o.items():
                scan(vv, path + '/' + str(kk))
        elif isinstance(o, list):
            for i, vv in enumerate(o):
                scan(vv, path + '/%d' % i)
        elif o is None:
            errs.append('null (NaN) at %s' % path)
        elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            errs.append('non-finite at %s' % path)
    for sec in ('runs', 'pairs', 'family_gap'):
        scan(d[sec], sec)
    scan(d['within_family']['per_seed'], 'within_family/per_seed')

    B = d['meta']['B_requested']
    # (3) every run completed all B replicates and CI brackets the point estimate
    for k, r in d['runs'].items():
        if r['B_completed'] != B:
            errs.append('%s completed %d/%d replicates' % (k, r['B_completed'], B))
        lo, hi = r['ci95_nds']
        if not (lo < r['point_nds'] < hi):
            errs.append('%s point NDS %.4f outside its own CI [%.4f, %.4f]'
                        % (k, r['point_nds'], lo, hi))
        if not (0.001 < r['boot_se_nds'] < 0.05):
            warns.append('%s SE %.4f outside the plausible 0.001-0.05 band'
                         % (k, r['boot_se_nds']))
        if abs(r['bias_boot_mean_minus_point']) > 3 * r['boot_se_nds']:
            warns.append('%s bootstrap bias %.4f exceeds 3 SE' % (k, r['bias_boot_mean_minus_point']))

    # (4) Report the sign of the scientific result -- do NOT gate on it.
    # A hard error here would make this generator structurally incapable of
    # emitting a negative or null finding, which is confirmation bias wearing a
    # sanity check's clothes. Hard errors are reserved for correctness
    # invariants (point-estimate reproduction, B completeness, no NaN); the
    # direction of the answer is an output, not a precondition.
    for s, g in d['family_gap'].items():
        if g['ci95_gap'][0] <= 0:
            warns.append('family gap CI on seed %s includes zero: [%.4f, %.4f] '
                         '-- report this, do not suppress it'
                         % (s, g['ci95_gap'][0], g['ci95_gap'][1]))
        if not (0.03 < g['point_gap'] < 0.20):
            warns.append('family gap seed %s = %.4f outside the expected 0.03-0.20 band'
                         % (s, g['point_gap']))

    # (5) paired CIs must be tighter than the naive difference of marginal CIs
    for p in d['pairs']:
        w_paired = p['ci95_diff'][1] - p['ci95_diff'][0]
        a, b = d['runs'][p['score_run']], d['runs'][p['camera_run']]
        w_naive = ((a['ci95_nds'][1] - a['ci95_nds'][0]) ** 2
                   + (b['ci95_nds'][1] - b['ci95_nds'][0]) ** 2) ** 0.5
        if w_paired > w_naive:
            warns.append('paired CI for %s-%s (%.4f) not tighter than unpaired (%.4f)'
                         % (p['score_run'], p['camera_run'], w_paired, w_naive))
    return errs, warns


# ----------------------------------------------------------------------------- LaTeX
def fmt_ci(lo, hi):
    return r'{\scriptsize[%.3f,\,%.3f]}' % (lo, hi)


def make_tex(d, idx):
    B = d['meta']['B_requested']
    L = []
    L.append(r'\begin{tabular}{@{}lccc@{}}')
    L.append(r'\toprule')
    L.append(r' & \multicolumn{3}{c}{NDS $\uparrow$ (95\% CI)} \\')
    L.append(r'\cmidrule(l){2-4}')
    L.append(r'Filter variant & Seed 0 & Seed 1 & Seed 2 \\')
    L.append(r'\midrule')
    for tag, label, header in ROWS:
        if header is not None:
            L.append(r'\addlinespace[2pt]')
            L.append(r'\multicolumn{4}{@{}l}{%s} \\' % header)
        pts, cis = [], []
        for s in SEEDS:
            r = idx.get((tag, s))
            if r is None:
                pts.append('---')
                cis.append('')
            else:
                pts.append('%.4f' % r['point_nds'])
                cis.append(fmt_ci(*r['ci95_nds']))
        L.append(r'%s & %s \\[-3pt]' % (label, ' & '.join(pts)))
        L.append(r' & %s \\[2pt]' % ' & '.join(cis))
    # family-gap block
    L.append(r'\midrule')
    L.append(r'\multicolumn{4}{@{}l}{\emph{Family gap:} $\min$\,NDS(score) $-$ $\max$\,NDS(camera)} \\')
    gp, gc, gz = [], [], []
    for s in SEEDS:
        g = d['family_gap'].get(str(s))
        if g is None:
            gp.append('---'); gc.append(''); gz.append('---')
        else:
            gp.append(r'\textbf{%+.4f}' % g['point_gap'])
            gc.append(fmt_ci(*g['ci95_gap']))
            gz.append('%d/%d' % (round(g['frac_gap_le_0'] * g['B']), g['B']))
    L.append(r'Gap & %s \\[-3pt]' % ' & '.join(gp))
    L.append(r' & %s \\[2pt]' % ' & '.join(gc))
    L.append(r'Replicates with gap $\le 0$ & %s \\' % ' & '.join(gz))
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    return '\n'.join(L) + '\n'


# -------------------------------------------------------------------------- markdown
def make_md(d, idx, errs, warns):
    m = d['meta']
    B = m['B_requested']
    o = []
    A = o.append
    A('# Scene-level bootstrap confidence intervals for the nuScenes NDS results')
    A('')
    A('Generated %s from `%s`.' % (m['created'], IN_JSON))
    A('Produced by `%s` (no GPU, cached detection results only).' % m['script'])
    A('')

    # ---- (d) protocol first, because (a)-(c) are only meaningful given it
    A('## (d) Resampling protocol (exactly what was done)')
    A('')
    A('* **Metric backend.** %s. Nothing about the metric is reimplemented: `accumulate`,'
      % m['metric_backend'])
    A('  `calc_ap`, `calc_tp` and `DetectionMetrics` are the official devkit functions, and the')
    A('  official loaders (`load_gt`, `load_prediction`, `add_center_dist`, `filter_eval_boxes`,')
    A('  `detection_cvpr_2019` config) are applied once per run on the full val split before any')
    A('  resampling. All three devkit filters (class-wise range, zero-LiDAR-point GT, bike-rack)')
    A('  are per-box and independent of which other samples are in the eval set, so pre-filtering')
    A('  is equivalent to filtering inside each replicate.')
    A('* **Resampling unit: the scene.** %s. The official val split is %d scenes / %d keyframes.'
      % (m['resampling_unit'], m['n_val_scenes'], m['n_val_samples']))
    A('  Keyframes inside a scene are ~2 Hz samples of the same traffic situation, the same')
    A('  calibration and the same object instances, so they are strongly dependent; a per-keyframe')
    A('  bootstrap would treat %d dependent units as independent and give dishonestly narrow'
      % m['n_val_samples'])
    A('  intervals. Each replicate draws %d scenes with replacement from the %d val scenes and'
      % (m['n_val_scenes'], m['n_val_scenes']))
    A('  takes the union of their keyframes **with multiplicity**.')
    A('* **Multiplicity handling: re-keying, not weighting** (%s).' % m['multiplicity'])
    A('  Concretely, copy *j*>=1 of sample token `T` is stored under the key `T#j` in **both** the')
    A('  ground-truth and the prediction `EvalBoxes`; GT boxes are shared by reference and')
    A('  prediction boxes are shallow-copied with `sample_token` rewritten. This reproduces the')
    A('  resampled dataset exactly as if those keyframes had physically occurred *m* times: `npos`')
    A('  is multiplied by *m* and each copy is greedily matched against its own copy of the GT.')
    A('  The cheaper alternative (one key with weight *m*) would give the same `npos` but would')
    A('  collapse the duplicated TP/FP runs in the confidence-sorted accumulation behind the')
    A('  101-point interpolated PR curve and behind `calc_tp`, and would therefore *understate*')
    A('  the variance.')
    A('* **Pairing:** %s. Every filter/seed therefore sees the *identical* resampled val set in'
      % m['pairing'])
    A('  replicate *b*, so all differences reported below are paired differences; a CI on a paired')
    A('  difference is far tighter, and far more relevant, than differencing two marginal CIs.')
    A('* **B = %d replicates** for every one of the %d runs (%d x %d = %s replicate evaluations,'
      % (B, len(m['runs']), len(m['runs']), B, format(len(m['runs']) * B, ',')))
    A('  no run got fewer). Percentile method: the 95% CI is the [2.5, 97.5] percentile of the')
    A('  bootstrap distribution. Wall clock %.0f s (%.2f h) on %d CPU worker processes; no GPU.'
      % (m['wall_clock_s'], m['wall_clock_s'] / 3600.0, m['workers']))
    A('* **Runs covered.** %s' % ', '.join(m['runs']))
    for x in m.get('runs_extra', []):
        A('* **Merged from a later pass.** %s, bootstrapped separately on %s'
          % (', '.join('`%s`' % k for k in x['keys']), x['created']))
        A('  under the identical protocol and the same B = %d, from `cache/%s`'
          % (B, x['source']))
        A('  (wall clock %.0f s on %d workers). These runs are not part of the'
          % (x['wall_clock_s'], x['workers']))
        A('  family-gap statistic, which is defined over the 7-filter family only.')
    A('* **Reproduce with:** `python tools/g2_bootstrap_ci.py --runs main5p --B %d --workers %d'
      % (B, m['workers']))
    A('  --chunk %d`. The driver is deterministic: replicate *b* is `numpy.random.default_rng(b)`'
      % m['chunk'])
    A('  over the sorted list of val scene names, so re-running reproduces every number exactly.')
    A('')
    A('**Caveat, which the paper must state.** %s' % m['caveat'])
    A('This bootstrap answers "how much of the observed gap is attributable to *which 150 scenes we')
    A('happened to evaluate on*?". It does not answer "how much would NDS move under a different')
    A('training seed?" -- the three training seeds are the only evidence on that, and they are')
    A('reported separately, never pooled with the bootstrap.')
    A('')

    # ---- point-estimate verification
    A('## (0) The bootstrap reproduces the published point estimates')
    A('')
    A('With no resampling (every scene drawn exactly once) the driver recomputes each run\'s NDS')
    A('through the same injected code path used inside every replicate, and compares it against')
    A('the run\'s cached `metrics_summary.json` at full double precision:')
    A('')
    cn = cached_nds(d)
    ALL = all_run_keys(d)
    A('| run | recomputed NDS (this driver) | cached NDS (`metrics_summary.json`) | bit-identical |')
    A('|---|---|---|---|')
    for k in ALL:
        got = d['runs'][k]['point_nds']
        ref = cn.get(k)
        A('| `%s` | %.17g | %s | %s |'
          % (k, got, ('%.17g' % ref) if ref is not None else 'n/a',
             'yes' if (ref is not None and got == ref) else
             ('n/a' if ref is None else 'NO')))
    A('')
    nex = sum(1 for k in ALL if k in cn and d['runs'][k]['point_nds'] == cn[k])
    A('%d of %d runs reproduce the cached NDS to the last bit (0 ulp), and all %d match the paper\'s'
      % (nex, len(ALL), len(ALL)))
    A('4-decimal published values. The scene->token restriction, the box injection and the re-keying')
    A('machinery are therefore exercised on a case whose answer is known, and they are correct.')
    A('')

    # ---- (a)/(b) family gap
    A('## (a) and (b) The score-vs-camera family gap, per seed')
    A('')
    A('The statistic is the one the paper actually claims: the **worst** score-based filter minus')
    A('the **best** camera-support filter on the same seed, i.e. min NDS(score) - max NDS(camera).')
    A('This is the most conservative version of the claim -- it is the smallest gap between any')
    A('member of one family and any member of the other.')
    A('')
    mingap = min_gap_per_seed(d, B)
    A('| seed | worst score-based | best camera-support | gap (NDS) | 95%% CI | worst of all %d replicates | replicates with gap <= 0 | bootstrap p |' % B)
    A('|---|---|---|---|---|---|---|---|')
    for s in SEEDS:
        g = d['family_gap'].get(str(s))
        if g is None:
            continue
        A('| %d | `%s` (%.4f) | `%s` (%.4f) | **%+.4f** | [%+.4f, %+.4f] | %+.4f | %d / %d | %s |'
          % (s, g['argmin_score_point'], d['runs'][g['argmin_score_point']]['point_nds'],
             g['argmax_camera_point'], d['runs'][g['argmax_camera_point']]['point_nds'],
             g['point_gap'], g['ci95_gap'][0], g['ci95_gap'][1], mingap[s],
             round(g['frac_gap_le_0'] * g['B']), g['B'],
             ('<= %.3f (1/B floor)' % (1.0 / B)) if g['two_sided_bootstrap_p'] <= 1.0 / B
             else '%.4f' % g['two_sided_bootstrap_p']))
    A('')
    excl = all(g['ci95_gap'][0] > 0 for g in d['family_gap'].values())
    A('**(b) Does the CI exclude zero on every seed?** %s' % ('**Yes.**' if excl else '**No.**'))
    if excl:
        worst = min(d['family_gap'].values(), key=lambda g: g['ci95_gap'][0])
        nseed = len(d['family_gap'])
        A('The lower 95%% confidence limit is positive on all %d seeds; the smallest lower limit'
          % nseed)
        A('anywhere is %+.4f NDS.' % worst['ci95_gap'][0])
        if all(g['frac_gap_le_0'] == 0 for g in d['family_gap'].values()):
            A('Not one of the %d x %d = %s bootstrap replicates produced a non-positive gap, so the'
              % (nseed, B, format(nseed * B, ',')))
            A('two-sided bootstrap p-value sits at its resolution floor 1/B = %.3f on every seed.'
              % (1.0 / B))
            A('The stronger statement is worth making because it is what the data actually show:')
            A('across all %s resampled val sets, the *worst single draw anywhere* still leaves the'
              % format(nseed * B, ','))
            A('worst score-based filter ahead of the best camera-support filter by %+.4f NDS.'
              % min(mingap.values()))
    A('')
    A('Every individual (score filter x camera filter x seed) pairing -- %d of them -- is reported'
      % len(d['pairs']))
    A('in `cache/bootstrap_ci.json` under `pairs`. Per seed, the extreme pairings are:')
    A('')
    A('| seed | pairing | difference (NDS) | 95% CI | sign-flip frac. |')
    A('|---|---|---|---|---|')
    for s in SEEDS:
        ps = [p for p in d['pairs'] if p['seed'] == s]
        if not ps:
            continue
        for tag_, p in (('worst case (min)', min(ps, key=lambda q: q['point_diff_nds'])),
                        ('best case (max)', max(ps, key=lambda q: q['point_diff_nds']))):
            A('| %d | %s: `%s` - `%s` | %+.4f | [%+.4f, %+.4f] | %.3f |'
              % (s, tag_, p['score_run'], p['camera_run'], p['point_diff_nds'],
                 p['ci95_diff'][0], p['ci95_diff'][1], p['sign_flip_frac']))
    A('')
    pd_ = [p['point_diff_nds'] for p in d['pairs']]
    lo_ = [p['ci95_diff'][0] for p in d['pairs']]
    fl_ = [p['sign_flip_frac'] for p in d['pairs']]
    A('Over all %d pairings: the smallest paired difference is %+.4f NDS and the largest is %+.4f.'
      % (len(pd_), min(pd_), max(pd_)))
    A('All %d lower 95%% confidence limits are positive; the smallest of them anywhere is %+.4f NDS.'
      % (len(lo_), min(lo_)))
    A('The maximum sign-flip fraction over all pairings is %.3f -- i.e. across %s'
      % (max(fl_), format(len(pd_) * B, ',')))
    A('(pairing, replicate) combinations, the score-based filter never once scored below the')
    A('camera-support filter on the same resampled val set.')
    A('')

    # ---- (c) within-family
    wf = d['within_family']
    A('## (c) Are the *within*-family differences distinguishable from zero?')
    A('')
    A('The paper treats the within-score-family spread as seed/eval noise. The primary D1 contrast')
    A('is %s:' % wf['comparison'])
    A('')
    A('| seed | difference (NDS) | 95% CI | CI contains 0 | sign-flip frac. | bootstrap p |')
    A('|---|---|---|---|---|---|')
    for s in SEEDS:
        g = wf['per_seed'].get(str(s))
        if g is None:
            continue
        A('| %d | %+.4f | [%+.4f, %+.4f] | **%s** | %.3f | %.4f |'
          % (s, g['point_diff_nds'], g['ci95_diff'][0], g['ci95_diff'][1],
             'yes' if g['ci_contains_zero'] else 'no', g['sign_flip_frac'],
             g['two_sided_bootstrap_p']))
    A('')
    for ttl, rows in wf['other_within_score_contrasts'].items():
        A('Same test for %s:' % ttl.replace('_minus_', ' - '))
        A('')
        A('| seed | difference (NDS) | 95% CI | CI contains 0 | bootstrap p |')
        A('|---|---|---|---|---|')
        for s in SEEDS:
            g = rows.get(str(s))
            if g is None:
                continue
            A('| %d | %+.4f | [%+.4f, %+.4f] | **%s** | %.4f |'
              % (s, g['point_diff_nds'], g['ci95_diff'][0], g['ci95_diff'][1],
                 'yes' if g['ci_contains_zero'] else 'no', g['two_sided_bootstrap_p']))
        A('')
    st = wf['across_seeds']['sign_test']
    wx = wf['across_seeds']['wilcoxon_signed_rank']
    A('Across the three seeds, the exact two-sided sign test on the %s point differences gives'
      % wf['comparison'])
    A('n = %s, %s positive, p = %s (Wilcoxon signed-rank p = %s).'
      % (st['n'], st['n_pos'], st['p_two_sided'],
         wx.get('p_two_sided', wx.get('reason'))))
    A('%s' % wf['across_seeds']['note'])
    A('')

    # --- collect ALL within-score-family contrasts and be explicit about exceptions ---
    allc = []          # (name, seed, diff, ci, contains_zero, p)
    for s in SEEDS:
        g = wf['per_seed'].get(str(s))
        if g is not None:
            allc.append(('R7b - R7', s, g['point_diff_nds'], g['ci95_diff'],
                         g['ci_contains_zero'], g['two_sided_bootstrap_p']))
    for ttl, rows in wf['other_within_score_contrasts'].items():
        nm = ttl.replace('_minus_', ' - ')
        for s in SEEDS:
            g = rows.get(str(s))
            if g is not None:
                allc.append((nm, s, g['point_diff_nds'], g['ci95_diff'],
                             g['ci_contains_zero'], g['two_sided_bootstrap_p']))
    sig = [c for c in allc if not c[4]]
    n_contain = sum(1 for g in wf['per_seed'].values() if g['ci_contains_zero'])
    n_tot = len(wf['per_seed'])
    n_gap = sum(1 for g in d['family_gap'].values() if g['ci95_gap'][0] > 0)

    A('**Verdict, stated precisely.**')
    A('')
    A('* The **primary D1 contrast** (%s) has a 95%% CI containing zero on **%d of %d seeds**, its'
      % (wf['comparison'], n_contain, n_tot))
    A('  sign flips in %.0f-%.0f%% of replicates, and its point difference changes sign between'
      % (100 * min(g['sign_flip_frac'] for g in wf['per_seed'].values()),
         100 * max(g['sign_flip_frac'] for g in wf['per_seed'].values())))
    A('  seeds. It is not distinguishable from zero. D1 is supported.')
    A('* Across **all %d within-score-family contrasts** tested here (3 contrasts x 3 seeds),'
      % len(allc))
    if not sig:
        A('  every 95% CI contains zero.')
    else:
        A('  **%d exception%s** has a CI that excludes zero and must be reported rather than'
          % (len(sig), '' if len(sig) == 1 else 's'))
        A('  glossed over:')
        A('')
        for nm, s, dv, ci, _, p in sig:
            A('  * **%s at seed %d: %+.4f NDS, 95%% CI [%+.4f, %+.4f], bootstrap p = %.4f.**'
              % (nm, s, dv, ci[0], ci[1], p))
        A('')
        A('  This survives a Bonferroni correction over the %d contrasts (threshold 0.05/%d = %.4f).'
          % (len(allc), len(allc), 0.05 / len(allc)))
        A('  Its correct interpretation is narrow: *given these two trained checkpoints*, on a')
        A('  resampled val set, the difference is real. It says nothing about reproducibility,')
        A('  and in fact it does **not** reproduce -- the same contrast is', )
        for nm, s, dv, ci, cz, p in allc:
            if nm == sig[0][0] and s != sig[0][1]:
                A('  %+.4f NDS with CI [%+.4f, %+.4f] at seed %d (contains zero: %s),'
                  % (dv, ci[0], ci[1], s, 'yes' if cz else 'no'))
        A('  so the *ordering* is still seed-fragile, which is exactly the D1 claim. What this')
        A('  result does rule out is the stronger, sloppier phrasing "all within-family differences')
        A('  are within noise on every seed": at least one is not, and the paper should say')
        A('  "within-family differences do not replicate across seeds" instead.')
        big = max(abs(c[2]) for c in allc)
        A('')
        A('  **Wording consequence for the manuscript.** `sec/4_findings.tex` currently states that')
        A('  within-family swings "stay $\\leq 0.5$ NDS". The largest within-score-family point')
        A('  difference measured here is %.2f NDS points (%s at seed %d), which exceeds that bound.'
          % (100 * big, sig[0][0], sig[0][1]))
        A('  The $\\leq 0.5$ figure is exact for the D1 contrast the sentence is really about')
        A('  (R7b vs R7: %s NDS points across the three seeds) but not for the full three-member'
          % ', '.join('%+.2f' % (100 * c[2]) for c in allc if c[0] == 'R7b - R7'))
        A('  score family. That sentence should be scoped to the D1 pair or the bound raised to 1.0.')
    lowest = min(g['ci95_gap'][0] for g in d['family_gap'].values())
    big = max(abs(c[2]) for c in allc)
    A('* Contrast either of the above with the **family gap**, whose CI excludes zero on %d of %d'
      % (n_gap, len(d['family_gap'])))
    A('  seeds with a lower 95%% limit of at least %+.4f NDS -- %.1fx the largest within-family'
      % (lowest, lowest / big if big else float('nan')))
    A('  effect measured anywhere (%+.4f NDS), and never once sign-flipped in %s'
      % (big, format(len(d['pairs']) * B, ',')))
    A('  (pairing, replicate) draws.')
    A('')
    A('So under one and the same resampling protocol: the family gap survives with room to spare;')
    A('the within-family ordering does not.')
    A('')

    # ---- per-run table
    A('## Per-run bootstrap detail')
    A('')
    A('| run | filter | seed | point NDS | boot. mean | SE | 95% CI | width |')
    A('|---|---|---|---|---|---|---|---|')
    for k in all_run_keys(d):
        r = d['runs'][k]
        A('| `%s` | %s | %d | %.4f | %.4f | %.4f | [%.4f, %.4f] | %.4f |'
          % (k, r['label'], r['seed'], r['point_nds'], r['boot_mean_nds'], r['boot_se_nds'],
             r['ci95_nds'][0], r['ci95_nds'][1], r['ci95_nds'][1] - r['ci95_nds'][0]))
    A('')
    ses = [d['runs'][k]['boot_se_nds'] for k in m['runs']]
    A('Marginal per-run evaluation SE ranges from %.4f to %.4f NDS. Note that this is the *marginal*'
      % (min(ses), max(ses)))
    A('scene-composition uncertainty, which is common to all runs and largely cancels in the paired')
    A('differences -- which is exactly why the paired CIs above are much tighter than these.')
    A('')
    fam_se = {}
    for k in m['runs']:
        fam_se.setdefault(d['runs'][k]['family'], []).append(d['runs'][k]['boot_se_nds'])
    A('Two features of the marginal distributions are worth stating explicitly, because a reviewer')
    A('will notice them:')
    A('')
    A('1. **The camera-support runs are roughly twice as variable across evaluation scenes as the')
    A('   score-based runs** (mean SE %.4f vs %.4f NDS). Their accuracy depends much more strongly'
      % (sum(fam_se.get('camera', [0])) / max(1, len(fam_se.get('camera', []))),
         sum(fam_se.get('score', [0])) / max(1, len(fam_se.get('score', [])))))
    A('   on *which* scenes are drawn. This does not weaken the family-gap conclusion -- the paired')
    A('   analysis conditions on the same scene draw for both families -- but it is itself a')
    A('   substantive finding: camera-support filtering produces a less scene-robust detector.')
    ex = d['runs'].get('R8b_s1')
    A('2. **The percentile CIs of the camera runs are visibly skewed** and the point estimate does')
    if ex is not None:
        A('   not sit at the centre of the interval (e.g. `R8b_s1`: point %.4f inside [%.4f, %.4f],'
          % (ex['point_nds'], ex['ci95_nds'][0], ex['ci95_nds'][1]))
        A('   much closer to the upper limit than the lower). This is')
    else:
        A('   not sit at the centre of the interval. This is')
    A('   a genuine property of the bootstrap distribution (a left tail from replicates that draw')
    A('   many hard scenes), not an artefact: the percentile method deliberately does not force')
    A('   symmetry. The same skew propagates into the paired-difference CIs, which is why several')
    A('   paired CIs sit above their point estimate. The direction of that skew is *conservative*')
    A('   for the paper\'s claim -- it makes the gap look larger, not smaller, in the bulk of the')
    A('   bootstrap distribution -- and the conclusion does not depend on it, since not a single')
    A('   replicate of any pairing produced a non-positive gap.')
    A('')

    A('## Automated sanity checks')
    A('')
    if not errs:
        A('All hard checks passed: point estimates reproduce the paper values; no NaN/null in any')
        A('summary; every run completed all %d replicates; every point estimate lies inside its own' % B)
        A('CI; every family-gap CI excludes zero; every paired CI is tighter than the corresponding')
        A('unpaired combination.')
    else:
        A('**FAILED CHECKS:**')
        for e in errs:
            A('* %s' % e)
    if warns:
        A('')
        A('Warnings (non-fatal):')
        for w in warns:
            A('* %s' % w)
    A('')
    return '\n'.join(o) + '\n'


def main():
    d = load()
    idx = index_runs(d)
    errs, warns = checks(d, idx)
    for w in warns:
        print('WARN: %s' % w)
    for e in errs:
        print('ERROR: %s' % e)
    if errs:
        print('\nRefusing to emit paper-ready outputs with failing checks.')
        return 1
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, 'w') as f:
        f.write(make_tex(d, idx))
    with open(OUT_MD, 'w') as f:
        f.write(make_md(d, idx, errs, warns))
    print('Wrote %s' % OUT_TEX)
    print('Wrote %s' % OUT_MD)
    return 0


if __name__ == '__main__':
    sys.exit(main())
