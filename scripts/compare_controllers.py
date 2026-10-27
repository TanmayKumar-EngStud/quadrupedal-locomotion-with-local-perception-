"""Compare controllers on the hop-success tables and the fixed terrain test.

    python scripts/compare_controllers.py \
        --hops boxes v1=outputs/calibration/hops_v1.json v2=outputs/calibration/hops_v2.json \
        --hops heightfield v1=outputs/calibration/hops_v1_heightfield.json v2=outputs/calibration/hops_v2_heightfield.json \
        --fixed "v1 types" v1=outputs/evaluation/fixed_v1_v1types.json v2=outputs/evaluation/fixed_v2_v1types.json \
        --fixed piles v1=outputs/evaluation/fixed_v1_piles.json v2=outputs/evaluation/fixed_v2_piles.json \
        --output outputs/comparison

Writes one chart per comparison (hops_<ground>.png, fixed_test.png), summary.json and
summary.md. Short hops are the 0.25-1 m bin; rows are pooled over the families in the
tables (open, rough, steps, slanted) unless stated otherwise.
"""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4']     # categorical slots 1-4, fixed order
INK, INK_2, INK_3, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
SHORT = (.25, 1.)
THRESHOLD = .8


def wilson(k, n, z=1.96):
    if not n: return (0., 1.)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - h), min(1., c + h)


def labelled(items):
    out = {}
    for item in items:
        label, path = item.split('=', 1)
        out[label] = Path(path)
    return out


def short_hops(table, family=None):
    """Per row: (successes, hops) of 0.25-1 m hops, for one family or pooled."""
    rows = {}
    for r in table['records']:
        if (family is None or r['family'] == family) and SHORT[0] <= r['length_m'] < SHORT[1]:
            k, n = rows.get(r['row'], (0, 0))
            rows[r['row']] = (k + (r['outcome'] == 'arrived'), n + 1)
    return dict(sorted(rows.items()))


def hardest_rows(rows):
    """Highest row with at least 80% success, and the last row of the unbroken run from row 0."""
    passing = [row for row, (k, n) in rows.items() if n and k / n >= THRESHOLD]
    unbroken = -1
    for row, (k, n) in rows.items():
        if row != unbroken + 1 or not n or k / n < THRESHOLD: break
        unbroken = row
    return (max(passing) if passing else None), (unbroken if unbroken >= 0 else None)


def table_summary(table):
    if 'pooled' in table:   # already a summary (e.g. computed on the server by summary_of below)
        rows = lambda d: {int(k): tuple(v) for k, v in d.items()}
        return dict(table, pooled=rows(table['pooled']), by_family={f: rows(d) for f, d in table['by_family'].items()},
                    hardest_row_pooled=tuple(table['hardest_row_pooled']),
                    hardest_row_by_family={f: tuple(v) for f, v in table['hardest_row_by_family'].items()})
    records = table['records']
    first = [r for r in records if r['from_platform']]
    families = sorted({r['family'] for r in records})
    result = dict(hops=len(records), arrived=sum(r['outcome'] == 'arrived' for r in records) / len(records),
                  falls=sum(r['outcome'] == 'fall' for r in records) / len(records),
                  first_hop_success=sum(r['outcome'] == 'arrived' for r in first) / max(1, len(first)),
                  pooled=short_hops(table), by_family={f: short_hops(table, f) for f in families})
    result['hardest_row_pooled'] = hardest_rows(result['pooled'])
    result['hardest_row_by_family'] = {f: hardest_rows(rows) for f, rows in result['by_family'].items()}
    return result


def summary_of(path):
    """Compact JSON summary of one hop table, small enough to copy between machines."""
    s = table_summary(json.loads(Path(path).read_text()))
    return json.dumps({**s, 'pooled': {str(k): v for k, v in s['pooled'].items()},
                       'by_family': {f: {str(k): v for k, v in d.items()} for f, d in s['by_family'].items()}},
                      separators=(',', ':'))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=.8); ax.set_axisbelow(True)


def plot_hops(ground, summaries, path, note=None):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    ax.axhline(THRESHOLD, color=INK_3, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(19.6, THRESHOLD - .015, '80% target', color=INK_2, fontsize=8, ha='right', va='top')
    ends = []
    for i, (label, s) in enumerate(summaries.items()):
        rows = s['pooled']; x = list(rows); p = [k / n for k, n in rows.values()]
        low, high = zip(*(wilson(k, n) for k, n in rows.values()))
        ax.fill_between(x, low, high, color=SERIES[i], alpha=.10, linewidth=0)
        ax.plot(x, p, color=SERIES[i], linewidth=2, marker='o', markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ends.append([p[-1], x[-1], label])
    ends.sort()
    for j in range(1, len(ends)):             # keep end labels at least 6% apart
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + .06)
    for y, x, label in ends:
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.set_xlim(-.5, 20.2); ax.set_ylim(0, 1.02); ax.set_xticks(range(20))
    ax.set_xlabel('Pile row (0: 0.50 m piles, 3 cm gaps  →  19: 0.12 m piles, 13 cm gaps)', color=INK_2, fontsize=9)
    ax.set_ylabel('Short-hop success (0.25–1 m)', color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
    ax.set_title(f'Hop success on pile scenes, {ground} ground (4 families pooled, 95% Wilson band)',
                 color=INK, fontsize=10.5, loc='left')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=len(summaries), loc='upper left', bbox_to_anchor=(0, -.16))
    if note: fig.text(.99, .01, note, color=INK_3, fontsize=7.5, ha='right', va='bottom')
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def plot_fixed(results, path, note=None):
    types = list(next(iter(results.values()))['by_type'])
    for r in results.values(): types += [t for t in r['by_type'] if t not in types]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    style(ax)
    width = .8 / len(results)
    for i, (label, r) in enumerate(results.items()):
        xs = [j + (i - (len(results) - 1) / 2) * width for j, t in enumerate(types) if t in r['by_type']]
        ys = [r['by_type'][t]['success_rate'] for t in types if t in r['by_type']]
        groups = ' | '.join(f"{g['success_rate']:.0%}" for g in r['groups'].values())
        ax.bar(xs, ys, width - .03, color=SERIES[i], label=f'{label}: {groups}')
    ax.set_xticks(range(len(types))); ax.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=8.5, color=INK_2)
    ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.))
    ax.set_ylabel('Targets reached', color=INK_2, fontsize=9)
    trials = next(iter(results.values()))
    ax.set_title(f"Fixed terrain test (native MuJoCo): one 2 m target in 10 s, levels {'/'.join(map(str, trials['levels']))}, "
                 f"{trials['trials_per_cell']} robots each", color=INK, fontsize=10.5, loc='left')
    names = ' | '.join(next(iter(results.values()))['groups'])
    legend = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=len(results), loc='upper left', bbox_to_anchor=(0, -.2),
                       title=f'Targets reached on: {names}', title_fontsize=8.5, alignment='left')
    legend.get_title().set_color(INK_2)
    if note: fig.text(.99, .01, note, color=INK_3, fontsize=7.5, ha='right', va='bottom')
    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=SURFACE); plt.close(fig)


def row_text(row):
    return '–' if row is None else str(row)


def markdown(hops, fixed):
    lines = ['# Controller comparison', '']
    for ground, summaries in hops.items():
        lines += [f'## Hop table, {ground} ground', '',
                  '| Controller | Hops | Arrived | Ended in a fall | First hop off the platform | Hardest row ≥80% short hops (pooled) | Unbroken ≥80% from row 0 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for label, s in summaries.items():
            top, unbroken = s['hardest_row_pooled']
            lines.append(f"| {label} | {s['hops']} | {s['arrived']:.0%} | {s['falls']:.0%} | {s['first_hop_success']:.0%} | "
                         f"{row_text(top)} | {row_text(unbroken)} |")
        lines += ['', 'Short-hop success (0.25–1 m) by row, pooled over families (successes/hops):', '',
                  '| Row | ' + ' | '.join(summaries) + ' |', '| --- |' + ' --- |' * len(summaries)]
        rows = sorted({row for s in summaries.values() for row in s['pooled']})
        for row in rows:
            cells = []
            for s in summaries.values():
                k, n = s['pooled'].get(row, (0, 0))
                cells.append(f'{k / n:.0%} ({k}/{n})' if n else '–')
            lines.append(f'| {row} | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Hardest row with ≥80% short-hop success, per family (highest row / unbroken run from row 0):', '',
                  '| Family | ' + ' | '.join(summaries) + ' |', '| --- |' + ' --- |' * len(summaries)]
        for family in next(iter(summaries.values()))['by_family']:
            cells = [' / '.join(row_text(v) for v in s['hardest_row_by_family'].get(family, (None, None))) for s in summaries.values()]
            lines.append(f'| {family} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    if fixed:
        names = list(next(iter(fixed.values()))['groups'])
        lines += ['## Fixed terrain test', '', '| Controller | ' + ' | '.join(f'{n}: reached / fell' for n in names) + ' | ' +
                  ' | '.join(t.replace('_', ' ') for t in next(iter(fixed.values()))['by_type']) + ' |',
                  '| --- |' + ' --- |' * (len(names) + len(next(iter(fixed.values()))['by_type']))]
        for label, r in fixed.items():
            lines.append(f'| {label} | ' + ' | '.join(f"{g['success_rate']:.0%} / {g['fall_rate']:.0%}" for g in r['groups'].values()) + ' | ' +
                         ' | '.join(f"{v['success_rate']:.0%}" for v in r['by_type'].values()) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hops', nargs='+', action='append', default=[], metavar='GROUND LABEL=PATH',
                   help='a ground name followed by label=path hop tables (repeatable)')
    p.add_argument('--fixed', nargs='+', action='append', default=[], metavar='GROUP LABEL=PATH',
                   help='a terrain-type group name followed by label=path goal_check results (repeatable; merged per label)')
    p.add_argument('--note', help='footnote printed under every chart (e.g. what a label means)')
    p.add_argument('--output', default='outputs/comparison')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    hops = {}
    for ground, *items in a.hops:
        hops[ground] = {label: table_summary(json.loads(path.read_text())) for label, path in labelled(items).items()}
        plot_hops(ground, hops[ground], out / f'hops_{ground}.png', a.note)
    fixed = {}
    for name, *items in a.fixed:
        for label, path in labelled(items).items():
            result = json.loads(path.read_text())
            merged = fixed.setdefault(label, dict(result, by_type={}, groups={}))
            merged['by_type'].update(result['by_type'])
            merged['groups'][name] = result['overall']
    if fixed: plot_fixed(fixed, out / 'fixed_test.png', a.note)
    serialisable = {g: {l: {**s, 'pooled': {str(k): v for k, v in s['pooled'].items()},
                            'by_family': {f: {str(k): v for k, v in rows.items()} for f, rows in s['by_family'].items()}}
                        for l, s in summaries.items()} for g, summaries in hops.items()}
    (out / 'summary.json').write_text(json.dumps(dict(hops=serialisable, fixed={l: dict(groups=r['groups'], by_type=r['by_type'])
                                                                                 for l, r in fixed.items()}), indent=2) + '\n')
    (out / 'summary.md').write_text(markdown(hops, fixed))
    if a.note: (out / 'summary.md').write_text((out / 'summary.md').read_text() + f'\n{a.note}\n')
    print((out / 'summary.md').read_text())


if __name__ == '__main__':
    main()
