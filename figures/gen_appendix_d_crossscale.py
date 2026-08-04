"""Appendix D — cross-scale completeness table.
All measured ΔNDS cells at {1% s0, 5% s0, 5% s1, 10% s0, 10% s1}.
Explicitly labeled as descriptive context, not a contribution.
"""
import sys
sys.path.insert(0, '/root/BMVC2026/figures')
from paper_plot_style import *

# Score-based perclass-only (best within-family at every cell), paired with same-seed labeled-only
PAIRS = [
    ('1\\% s0',   'R17a',   'R17b',     '5p_s0_unlabeled'),
    ('5\\% s0',   'R4b',    'R7b',      '5p_s0_unlabeled'),
    ('5\\% s1',   'R4_s1',  'R7b_s1',   '5p_s1_unlabeled'),
    ('10\\% s0',  'R10a',   'R10b',     '10p_s0_unlabeled'),
    ('10\\% s1',  'R10a_s1','R10b_s1',  '10p_s1_unlabeled'),
]

t = []
t.append(r'\begin{tabular}{lccccc}')
t.append(r'\toprule')
t.append(r'Cell & labeled-only NDS & SSL NDS & $\Delta$NDS & sign & filter \\')
t.append(r'\midrule')
for name, base_tag, ssl_tag, _ in PAIRS:
    base = load_run(base_tag); ssl = load_run(ssl_tag)
    if base is None or ssl is None:
        t.append(f'{name} & — & — & — & — & — \\\\')
        continue
    delta = ssl['NDS'] - base['NDS']
    sign = r'\textbf{helps}' if delta > 0 else r'\textbf{hurts}'
    color = 'blue!50!black' if delta > 0 else 'red!60!black'
    t.append(f'{name} & {base["NDS"]:.4f} & {ssl["NDS"]:.4f} & '
             f'{{\\color{{{color}}}{delta:+.4f}}} & {sign} & perclass-only \\\\')
t.append(r'\bottomrule')
t.append(r'\end{tabular}')
(FIG_DIR / 'table4_crossscale_appendix.tex').write_text('\n'.join(t))
print(f'  wrote: {FIG_DIR / "table4_crossscale_appendix.tex"}')

# Print human-readable summary too
print('\n=== Cross-scale completeness ===')
print(f'  {"cell":<10s} {"base":>8s} {"SSL":>8s} {"ΔNDS":>9s}  sign')
for name, base_tag, ssl_tag, _ in PAIRS:
    base = load_run(base_tag); ssl = load_run(ssl_tag)
    if base is None or ssl is None:
        continue
    delta = ssl['NDS'] - base['NDS']
    sign = 'helps' if delta > 0 else 'hurts'
    name_clean = name.replace('\\%', '%')
    print(f'  {name_clean:<10s} {base["NDS"]:>8.4f} {ssl["NDS"]:>8.4f} '
          f'{delta:>+9.4f}  {sign}')
