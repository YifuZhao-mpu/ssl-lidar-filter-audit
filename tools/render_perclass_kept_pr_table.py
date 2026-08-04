#!/usr/bin/env python3
"""Render the per-class kept/P/R appendix table from
cache/perclass_kept_pr_5p_s0.json into LaTeX."""
import json
from pathlib import Path

NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
COCO_DETECTABLE = {'car','truck','construction_vehicle','bus','trailer',
                   'motorcycle','bicycle','pedestrian'}

CLS_DISPLAY = {
    'car': 'car', 'truck': 'truck',
    'construction_vehicle': 'CV',
    'bus': 'bus', 'trailer': 'trailer',
    'barrier': 'barrier (*)',
    'motorcycle': 'motorcyc.',
    'bicycle': 'bicycle', 'pedestrian': 'pedestrian',
    'traffic_cone': 'tr.\\_cone (*)',
}


def fmt_int(n):
    if n >= 100_000:
        return f'{n/1000:.0f}k'
    if n >= 10_000:
        return f'{n/1000:.1f}k'
    return f'{n}'


def main():
    data = json.load(open('/root/BMVC2026/cache/perclass_kept_pr_5p_s0.json'))

    cols = ['R7b_perclass_only', 'R7c_consensus',
            'R7d_row_shuffle', 'R7e_col_shuffle']
    col_labels = [r'R7b perclass-only', r'R7c consensus',
                  r'R7d row-shuffle', r'R7e col-shuffle']

    lines = []
    lines.append(r'\begin{tabular}{l|rrr|rrr|rrr|rrr}')
    lines.append(r'\toprule')
    h1 = ' & ' + ' & '.join(rf'\multicolumn{{3}}{{c|}}{{{lbl}}}' if i < len(col_labels)-1
                            else rf'\multicolumn{{3}}{{c}}{{{lbl}}}'
                            for i, lbl in enumerate(col_labels))
    lines.append(h1 + r' \\')
    lines.append(r'class & kept & P & R & kept & P & R & kept & P & R & kept & P & R \\')
    lines.append(r'\midrule')

    for c in NUSC_CLASSES:
        row = [CLS_DISPLAY[c]]
        for col in cols:
            d = data[col]['per_class'][c]
            row.append(fmt_int(d['kept']))
            row.append(f'{d["P"]:.0f}')
            row.append(f'{d["R"]:.0f}')
        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\midrule')
    # Total kept row
    row = [r'\textbf{total}']
    for col in cols:
        row.append(r'\textbf{' + fmt_int(data[col]['kept_total']) + r'}')
        row.append('---')
        row.append('---')
    lines.append(' & '.join(row) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    out = Path('/root/BMVC2026/paper/figures/table_perclass_structural.tex')
    out.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {out}')

    # Also print a digest
    print('\nDigest:')
    print(f'  COCO-detectable kept-count change R7b→R7c:')
    for c in NUSC_CLASSES:
        if c in COCO_DETECTABLE:
            r7b = data['R7b_perclass_only']['per_class'][c]['kept']
            r7c = data['R7c_consensus']['per_class'][c]['kept']
            pct = 100.0 * (r7c - r7b) / max(r7b, 1)
            print(f'    {c:<22s} {r7b:>7d} → {r7c:>7d}  ({pct:+.1f}%)')

    print(f'\n  Passthrough (non-COCO) kept-count change R7b→R7c:')
    for c in ['barrier', 'traffic_cone']:
        r7b = data['R7b_perclass_only']['per_class'][c]['kept']
        r7c = data['R7c_consensus']['per_class'][c]['kept']
        pct = 100.0 * (r7c - r7b) / max(r7b, 1)
        print(f'    {c:<22s} {r7b:>7d} → {r7c:>7d}  ({pct:+.1f}%)')


if __name__ == '__main__':
    main()
