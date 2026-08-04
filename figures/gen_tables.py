"""Generate Tables 1, 2, 3 as standalone LaTeX tabular snippets.
- Table 1: D2 main filter-family matrix at 5% labels.
  NOTE: the submitted paper uses the three-seed `table1_filter_family_3seed.tex`
  (Package B, 2026-07-18), not the two-seed table emitted here. This generator is
  kept for provenance of the original two-seed table only.
- Table 2: D2 mechanism — pooled + per-class correlation breakdown.
- Table 3: D1 — pilot cocoF1 vs downstream NDS across all three seeds.
"""
import sys
sys.path.insert(0, '/workspace/BMVC2026/figures')
from paper_plot_style import *

OUT = FIG_DIR

# ============== Table 1 — D2 main filter family at 5% labels ============== #

ROWS_T1 = [
    ('Labeled-only (R4b, no SSL)', 'R4b'),
    (r'\midrule\multicolumn{7}{l}{\emph{Score-based filters}} \\', None),
    ('Single threshold (R6)', 'R6'),
    ('Per-class + boost (R7)', 'R7'),
    ('Per-class only (R7b)', 'R7b'),
    (r'\midrule\multicolumn{7}{l}{\emph{Camera-support filters}} \\', None),
    ('Best-view (R8b)', 'R8b'),
    ('Any-view (R8c)', 'R8c'),
    (r'Max-cam $K{=}5$ (R8d)', 'R8d'),
    ('6-camera consensus (R7c)', 'R7c'),
    (r'\midrule\multicolumn{7}{l}{\emph{Intervention ablation (Sec.~\ref{sec:d2_mech})}} \\', None),
    ('Shuffled consensus (R7d)', 'R7d'),
]

baseline_nds = load_run('R4b')['NDS']

t1 = []
t1.append(r'\begin{tabular}{lcccccr}')
t1.append(r'\toprule')
t1.append(r'Filter variant & NDS$\uparrow$ & mAP$\uparrow$ & mATE$\downarrow$ & mASE$\downarrow$ & mAOE$\downarrow$ & $\Delta$NDS \\')
t1.append(r'\midrule')
for name, tag in ROWS_T1:
    if tag is None:
        t1.append(name)
        continue
    r = load_run(tag)
    if r is None:
        t1.append(f'{name} & — & — & — & — & — & — \\\\')
        continue
    delta = r['NDS'] - baseline_nds
    delta_str = f'{delta:+.4f}' if tag != 'R4b' else r'\,---'
    t1.append(f'{name} & {r["NDS"]:.4f} & {r["mAP"]:.4f} & '
              f'{r["mATE"]:.3f} & {r["mASE"]:.3f} & {r["mAOE"]:.3f} & {delta_str} \\\\')
t1.append(r'\midrule')
t1.append(r'\multicolumn{7}{l}{\emph{Seed-1 confirmation (full 7-filter family)}} \\')
t1.append(r'\midrule')
r = load_run('R4_s1')
t1.append(f'Labeled-only (R4-s1) & {r["NDS"]:.4f} & {r["mAP"]:.4f} & '
          f'{r["mATE"]:.3f} & {r["mASE"]:.3f} & {r["mAOE"]:.3f} & \\,---  \\\\')
base = load_run('R4_s1')['NDS']
t1.append(r'\multicolumn{7}{l}{\emph{Score-based filters (seed 1)}} \\')
for tag, name in [('R6_s1', 'Single threshold (R6-s1)'),
                  ('R7_s1', 'Per-class + boost (R7-s1)'),
                  ('R7b_s1', 'Per-class only (R7b-s1)')]:
    r = load_run(tag)
    if r is None:
        t1.append(f'{name} & — & — & — & — & — & — \\\\')
        continue
    delta = r['NDS'] - base
    t1.append(f'{name} & {r["NDS"]:.4f} & {r["mAP"]:.4f} & '
              f'{r["mATE"]:.3f} & {r["mASE"]:.3f} & {r["mAOE"]:.3f} & '
              f'{delta:+.4f} \\\\')
t1.append(r'\multicolumn{7}{l}{\emph{Camera-support filters (seed 1)}} \\')
for tag, name in [('R8b_s1', 'Best-view (R8b-s1)'),
                  ('R8c_s1', 'Any-view (R8c-s1)'),
                  ('R8d_s1', r'Max-cam $K{=}5$ (R8d-s1)'),
                  ('R7c_s1', '6-camera consensus (R7c-s1)')]:
    r = load_run(tag)
    if r is None:
        t1.append(f'{name} & — & — & — & — & — & — \\\\')
        continue
    delta = r['NDS'] - base
    t1.append(f'{name} & {r["NDS"]:.4f} & {r["mAP"]:.4f} & '
              f'{r["mATE"]:.3f} & {r["mASE"]:.3f} & {r["mAOE"]:.3f} & '
              f'{delta:+.4f} \\\\')
t1.append(r'\bottomrule')
t1.append(r'\end{tabular}')
(OUT / 'table1_filter_family.tex').write_text('\n'.join(t1))
print(f'  wrote: {OUT / "table1_filter_family.tex"}')


# ================ Table 2 — D2 correlation: pooled + per-class ============== #

corr = load_correlation()
pooled = corr['pooled']
perclass = corr['perclass']['per_class']

# Four numeric columns, not six: the p90 columns were empty (---) on every
# per-class row and pushed the tabular ~85pt past the WACV column width. The
# pooled p90 pair is reported in the caption instead.
t2 = []
t2.append(r'\begin{tabular}{lrrrr}')
t2.append(r'\toprule')
t2.append(r' & \multicolumn{2}{c}{Real teacher} & \multicolumn{2}{c}{Perturbed-GT pilot} \\')
t2.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
t2.append(r'Subset & mean & median & mean & median \\')
t2.append(r'\midrule')
t2.append(r'\textbf{Pooled} ($n{=}146{,}698$) & '
          f'\\textbf{{{pooled["real_offdiag_mean"]:.3f}}} & '
          f'{pooled["real_offdiag_median"]:.3f} & '
          f'{pooled["pilot_offdiag_mean"]:.3f} & '
          f'{pooled["pilot_offdiag_median"]:.3f} \\\\')
t2.append(r'\midrule')
t2.append(r'\multicolumn{5}{l}{\emph{Per class (top-5 by box count)}} \\')


def _fmt_n(n):
    """1234567 -> 1{,}234{,}567 for LaTeX math-mode grouping."""
    return '{,}'.join(f'{n:,}'.split(','))


for cls, v in perclass.items():
    name = cls.replace('_', r'\_')
    n_math = '($n{=}' + _fmt_n(v['n_boxes']) + '$)'
    t2.append(f'{name} {n_math} & '
              f'{v["real_offdiag_mean"]:.3f} & '
              f'{v["real_offdiag_median"]:.3f} & '
              f'{v["pilot_offdiag_mean"]:.3f} & '
              f'{v["pilot_offdiag_median"]:.3f} \\\\')
t2.append(r'\bottomrule')
t2.append(r'\end{tabular}')
(OUT / 'table2_correlation.tex').write_text('\n'.join(t2))
print(f'  wrote: {OUT / "table2_correlation.tex"}')


# ================== Table 3 — D1 pilot/downstream inversion ================= #

# Pilot cocoF1 computed on the full 5% dump under one protocol for all seeds
# (tools/g2_pilot_cocoF1.py). Seed-as-row layout: the claim D1 now makes is that
# the pilot metric picks the same filter on every seed while the downstream
# ordering flips on seed 2 — that is only legible when seeds are the rows. The
# old filter-as-row layout also overran the column by ~199pt via its footnote.
import json

PILOT = {}
for s, path in [(0, 'cache/pilot_cocoF1_seed0_full.json'),
                (1, 'cache/pilot_cocoF1_seed1_full.json'),
                (2, 'cache/pilot_cocoF1_seed2_full.json')]:
    p = Path('/workspace/BMVC2026') / path
    if p.exists():
        PILOT[s] = json.loads(p.read_text())
    else:
        print(f'  WARNING: missing pilot cocoF1 for seed {s} ({p}) — row omitted')

NDS_TAGS = {
    0: {'single_thr': 'R6', 'perclass_only': 'R7b', 'perclass_boost': 'R7'},
    1: {'single_thr': 'R6_s1', 'perclass_only': 'R7b_s1', 'perclass_boost': 'R7_s1'},
    2: {'single_thr': 'R6_ssl_5p_s2_single_thr',
        'perclass_only': 'R7b_ssl_5p_s2_perclass_only',
        'perclass_boost': 'R7_ssl_5p_s2_perclass_boost'},
}

t3 = []
t3.append(r'\begin{tabular}{lccccc}')
t3.append(r'\toprule')
t3.append(r' & \multicolumn{2}{c}{pilot cocoF1} & \multicolumn{2}{c}{downstream NDS} & Pilot picks \\')
t3.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
t3.append(r'Seed & only & +boost & only & +boost & the winner? \\')
t3.append(r'\midrule')

single_thr_note = []
for s in sorted(PILOT):
    pil, tags = PILOT[s], NDS_TAGS[s]
    f_only = pil['perclass_only']['cocoF1']
    f_boost = pil['perclass_boost']['cocoF1']
    n_only = load_run(tags['perclass_only'])['NDS']
    n_boost = load_run(tags['perclass_boost'])['NDS']
    # The pilot always prefers whichever filter has the higher cocoF1; it is
    # "correct" iff that filter also has the higher downstream NDS.
    pilot_pick_boost = f_boost > f_only
    truth_boost = n_boost > n_only
    # amssymb is loaded in the preamble; avoid pifont's \cmark/\xmark.
    correct = r'$\checkmark$' if pilot_pick_boost == truth_boost else r'$\times$'
    bf = lambda v, win: (f'\\textbf{{{v}}}' if win else f'{v}')  # noqa: E731
    t3.append(
        f'{s} & {bf(f"{f_only:.2f}", not pilot_pick_boost)} & '
        f'{bf(f"{f_boost:.2f}", pilot_pick_boost)} & '
        f'{bf(f"{n_only:.4f}", not truth_boost)} & '
        f'{bf(f"{n_boost:.4f}", truth_boost)} & {correct} \\\\')
    st = load_run(tags['single_thr'])
    single_thr_note.append(
        f'seed {s} {pil["single_thr"]["cocoF1"]:.2f}/{st["NDS"]:.4f}')
t3.append(r'\bottomrule')
t3.append(r'\end{tabular}')
print('  Table 3 single-thr reference (cocoF1/NDS): ' + '; '.join(single_thr_note))
(OUT / 'table3_d1_inversion.tex').write_text('\n'.join(t3))
print(f'  wrote: {OUT / "table3_d1_inversion.tex"}')

print('All tables done.')
