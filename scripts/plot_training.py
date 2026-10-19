"""Figures for a training run: python scripts/plot_training.py runs/kth-goal-v3-seed0 [runs/...-phase2]

Reads metrics.jsonl (one line per PPO update) and evaluation.jsonl (fixed
terrain test) and writes PNG figures into <last run>/figures/. Several run
directories are joined in order (a resumed run continues the update count);
each later phase start is marked with a dashed line.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
    'axes.titlesize': 11, 'axes.labelsize': 9, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
    'legend.frameon': False, 'legend.fontsize': 8, 'font.size': 9})


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def smooth(values, window=20):
    values = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        if np.isfinite(chunk).any(): out[i] = np.nanmean(chunk)
    return out


PHASES = []   # update numbers where a resumed phase starts


def line(ax, x, y, label=None, color=SERIES[0], raw=True):
    if raw: ax.plot(x, [np.nan if v is None else v for v in y], color=color, alpha=.18, linewidth=1)
    ax.plot(x, smooth(y), color=color, label=label)


def mark_phases(fig):
    for ax in fig.axes:
        for start in PHASES:
            ax.axvline(start, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), alpha=.7)


def dashboard(name, metrics, evaluations, out):
    x = [m['update'] for m in metrics]
    types = list(metrics[-1]['success_by_type'])
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    steps = metrics[-1]['steps']
    fig.suptitle(f'Go1 goal-reaching PPO: {name}', x=.06, ha='left', fontsize=16, fontweight='bold', color=INK)
    fig.text(.06, .945, f'{len(metrics)} updates  |  {steps / 1e6:.1f} M transitions  |  '
             f'{metrics[-1]["wall_seconds"] / 3600:.2f} h  |  {metrics[-1]["samples_per_second"]:,.0f} samples/s  |  '
             'thin lines: every update, thick: 20-update average', color=INK_2, fontsize=10)

    ax = axes[0, 0]; line(ax, x, [m['mean_reward'] for m in metrics])
    ax.set_title('Training reward per control step'); ax.set_xlabel('PPO update')

    ax = axes[0, 1]; line(ax, x, [m['arrivals_per_10k'] for m in metrics], color=SERIES[2])
    ax.set_title('Targets reached per 10,000 transitions'); ax.set_xlabel('PPO update')

    ax = axes[0, 2]; line(ax, x, [m['falls_per_10k'] for m in metrics], color=SERIES[7])
    ax.set_title('Falls per 10,000 transitions'); ax.set_xlabel('PPO update'); ax.set_ylim(bottom=0)

    ax = axes[1, 0]; line(ax, x, [m['target_success_rate'] for m in metrics], color=SERIES[2])
    ax.set_title('Training targets reached (reached / attempted)'); ax.set_xlabel('PPO update'); ax.set_ylim(0, 1)

    ax = axes[1, 1]
    line(ax, x, [m['mean_command_speed'] for m in metrics], 'commanded speed', SERIES[6], raw=False)
    line(ax, x, [m['mean_velocity_toward_target'] for m in metrics], 'velocity toward target', SERIES[0])
    ax.set_title('Speed toward the target (m/s)'); ax.set_xlabel('PPO update'); ax.legend(loc='lower right')

    ax = axes[1, 2]; line(ax, x, [m['mean_time_to_goal_s'] for m in metrics], color=SERIES[1])
    ax.set_title('Time to reach a training target (s)'); ax.set_xlabel('PPO update'); ax.set_ylim(bottom=0)

    ax = axes[2, 0]
    for i, t in enumerate(types):
        ax.plot(x, smooth([m['level_by_type'][t] for m in metrics], 5), color=SERIES[i], label=t.replace('_', ' '))
    ax.set_title('Curriculum: mean terrain level (0 easy - 9 hard)'); ax.set_xlabel('PPO update')
    ax.legend(ncol=2, loc='upper left')

    ax = axes[2, 1]
    if evaluations:
        ex = [e['update'] for e in evaluations]
        for i, t in enumerate(types):
            ax.plot(ex, [e['by_type'][t]['success_rate'] for e in evaluations], color=SERIES[i], label=t.replace('_', ' '),
                    marker='o', markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.plot(ex, [e['overall']['success_rate'] for e in evaluations], color=INK, linewidth=2.5, label='all terrains')
        ax.legend(ncol=2, loc='upper left')
    ax.set_title('Fixed test: target reached (2 m, 10 s)'); ax.set_xlabel('PPO update'); ax.set_ylim(-.02, 1.02)

    ax = axes[2, 2]
    if evaluations:
        ax.plot(ex, [e['overall']['mean_progress_fraction'] for e in evaluations], color=SERIES[0], marker='o',
                markersize=4, label='progress toward target', markeredgecolor=SURFACE, markeredgewidth=1)
        ax.plot(ex, [e['overall']['fall_rate'] for e in evaluations], color=SERIES[7], marker='o', markersize=4,
                label='fall rate', markeredgecolor=SURFACE, markeredgewidth=1)
        ax.legend(loc='center right')
    ax.set_title('Fixed test: progress fraction and falls'); ax.set_xlabel('PPO update'); ax.set_ylim(-.02, 1.02)
    if PHASES:
        fig.text(.06, .925, 'dashed lines: training resumed with a changed curriculum (400: pit depth grows with level; later: 80% promotion rule)',
                 color=INK_2, fontsize=9)
    mark_phases(fig)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / 'training_dashboard.png', dpi=120); plt.close(fig)


def success_heatmap(evaluation, out):
    types = list(evaluation['by_type']); levels = evaluation['levels']
    grid = np.array([[evaluation['success_by_type_level'][f'{t}/{lv}'] for lv in levels] for t in types])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues', BLUES)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.grid(False)
    for i in range(len(types)):
        for j in range(len(levels)):
            ax.text(j, i, f'{grid[i, j]:.0%}', ha='center', va='center', fontsize=9,
                    color='white' if grid[i, j] > .55 else INK)
    ax.set_xticks(range(len(levels)), [f'level {lv}' for lv in levels])
    ax.set_yticks(range(len(types)), [t.replace('_', ' ') for t in types])
    ax.set_title(f'Fixed test after update {evaluation["update"]}: targets reached', loc='left')
    fig.text(.01, .01, f'{evaluation["trials_per_cell"]} robots per cell, target {evaluation["distance_m"]} m away, '
             f'{evaluation["seconds"]:.0f} s limit, deterministic policy', color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(out / 'evaluation_heatmap.png', dpi=130); plt.close(fig)


def reward_terms(metrics, out, last=20):
    names = list(metrics[-1]['reward_terms'])
    values = np.array([[m['reward_terms'].get(n, 0.) for n in names] for m in metrics[-last:]]).mean(0)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = [SERIES[0] if values[i] >= 0 else SERIES[7] for i in order]
    ax.barh(range(len(order)), values[order], color=colors, height=.6)
    ax.set_yticks(range(len(order)), [names[i].replace('_', ' ') for i in order])
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_title(f'Reward terms, mean per step (last {last} updates)', loc='left')
    ax.set_xlabel('weighted reward per control step (blue: reward, red: penalty)')
    fig.tight_layout(); fig.savefig(out / 'reward_terms.png', dpi=130); plt.close(fig)


def ppo_diagnostics(metrics, out):
    x = [m['update'] for m in metrics]
    panels = [('action_std', 'Policy action std'), ('learning_rate', 'Learning rate (adaptive)'),
              ('kl', 'Approximate KL per minibatch'), ('explained_variance', 'Value function explained variance'),
              ('entropy', 'Policy entropy'), ('value_loss', 'Value loss')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (key, title) in zip(axes.flat, panels):
        line(ax, x, [m['ppo'][key] for m in metrics]); ax.set_title(title); ax.set_xlabel('PPO update')
    axes[0, 1].set_yscale('log'); axes[1, 2].set_yscale('log')
    mark_phases(fig)
    fig.tight_layout(); fig.savefig(out / 'ppo_diagnostics.png', dpi=110); plt.close(fig)


def report_figures(metrics, evaluations, out):
    """Three compact 1500 x 700 px figures for reports and slides."""
    x = [m['update'] for m in metrics]
    types = list(metrics[-1]['success_by_type'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    line(axes[0], x, [m['target_success_rate'] for m in metrics], color=SERIES[2])
    axes[0].set_title('Training targets reached (reached / attempted)'); axes[0].set_ylim(0, 1)
    line(axes[1], x, [m['mean_command_speed'] for m in metrics], 'commanded speed', SERIES[6], raw=False)
    line(axes[1], x, [m['mean_velocity_toward_target'] for m in metrics], 'velocity toward target', SERIES[0])
    axes[1].set_title('Speed toward the target (m/s)'); axes[1].legend(loc='lower right')
    line(axes[2], x, [m['falls_per_10k'] for m in metrics], color=SERIES[7])
    axes[2].set_title('Falls per 10,000 transitions'); axes[2].set_ylim(bottom=0)
    for ax in axes: ax.set_xlabel('PPO update')
    mark_phases(fig); fig.tight_layout(); fig.savefig(out / 'report_1_learning.png', dpi=100); plt.close(fig)

    if evaluations:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1.35, 1]))
        ex = [e['update'] for e in evaluations]
        for i, t in enumerate(types):
            axes[0].plot(ex, [e['by_type'][t]['success_rate'] for e in evaluations], color=SERIES[i], label=t.replace('_', ' '))
        axes[0].plot(ex, [e['overall']['success_rate'] for e in evaluations], color=INK, linewidth=3, label='all terrains')
        axes[0].set_title('Fixed test: targets reached (2 m away, 10 s, levels 0/3/6/9)'); axes[0].set_ylim(-.02, 1.02)
        axes[0].set_xlabel('PPO update'); axes[0].legend(ncol=2, loc='upper left')
        last = evaluations[-1]; levels = last['levels']
        grid = np.array([[last['success_by_type_level'][f'{t}/{lv}'] for lv in levels] for t in types])
        axes[1].imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blues', BLUES), vmin=0, vmax=1, aspect='auto')
        axes[1].grid(False)
        for i in range(len(types)):
            for j in range(len(levels)):
                axes[1].text(j, i, f'{grid[i, j]:.0%}', ha='center', va='center', fontsize=10, color='white' if grid[i, j] > .55 else INK)
        axes[1].set_xticks(range(len(levels)), [f'level {lv}' for lv in levels])
        axes[1].set_yticks(range(len(types)), [t.replace('_', ' ') for t in types])
        axes[1].set_title(f'Fixed test after update {last["update"]}')
        for start in PHASES: axes[0].axvline(start, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), alpha=.7)
        fig.tight_layout(); fig.savefig(out / 'report_2_fixed_test.png', dpi=100); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for i, t in enumerate(types):
        axes[0].plot(x, smooth([m['level_by_type'][t] for m in metrics], 5), color=SERIES[i], label=t.replace('_', ' '))
    axes[0].set_title('Curriculum: mean terrain level (0 easy - 9 hard)'); axes[0].set_xlabel('PPO update')
    axes[0].legend(ncol=2, loc='upper left')
    for start in PHASES: axes[0].axvline(start, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), alpha=.7)
    names = list(metrics[-1]['reward_terms'])
    values = np.array([[m['reward_terms'].get(n, 0.) for n in names] for m in metrics[-20:]]).mean(0)
    order = np.argsort(values)
    axes[1].barh(range(len(order)), values[order], color=[SERIES[0] if values[i] >= 0 else SERIES[7] for i in order], height=.6)
    axes[1].set_yticks(range(len(order)), [names[i].replace('_', ' ') for i in order]); axes[1].axvline(0, color=INK_2, linewidth=1)
    axes[1].set_title('Reward terms per control step (last 20 updates)')
    fig.tight_layout(); fig.savefig(out / 'report_3_curriculum_rewards.png', dpi=100); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('runs', nargs='+'); a = p.parse_args()
    metrics, evaluations = [], []
    for i, run in enumerate(map(Path, a.runs)):
        m, e = read_jsonl(run / 'metrics.jsonl'), read_jsonl(run / 'evaluation.jsonl')
        if i and m: PHASES.append(m[0]['update'])
        cut = m[0]['update'] if m else float('inf')     # a resumed phase replaces later updates
        metrics = [r for r in metrics if r['update'] < cut]
        offset = metrics[-1]['wall_seconds'] if metrics else 0.   # wall clock continues across phases
        metrics += [dict(r, wall_seconds=r['wall_seconds'] + offset) for r in m]
        evaluations = [r for r in evaluations if r['update'] < cut] + e
    for previous, record in zip(metrics, metrics[1:]):   # a resume inside one directory restarts the clock
        if record['wall_seconds'] < previous['wall_seconds']:
            shift = previous['wall_seconds']
            for later in metrics[metrics.index(record):]: later['wall_seconds'] += shift
    run = Path(a.runs[-1]); out = run / 'figures'; out.mkdir(exist_ok=True)
    if not metrics: raise SystemExit(f'No metrics in {a.runs}')
    dashboard(Path(a.runs[0]).name, metrics, evaluations, out); reward_terms(metrics, out); ppo_diagnostics(metrics, out)
    if evaluations: success_heatmap(evaluations[-1], out)
    report_figures(metrics, evaluations, out)
    print(out)


if __name__ == '__main__':
    main()
