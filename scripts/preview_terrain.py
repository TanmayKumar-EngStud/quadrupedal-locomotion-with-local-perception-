"""Height maps of the training terrain at several levels and irregularity scales.

    python scripts/preview_terrain.py --noise 0 0.5 1.0 --levels 0 5 9 --output outputs/terrain_preview.png

One row per terrain type, one column per (noise, level). Each panel is a 3 m x 3 m
window next to the spawn platform, coloured by height (one colour scale per row),
so the regular v1/v2 terrain (noise 0) and the irregular v3 (0.5) and v4 (1.0)
terrain can be compared side by side.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from quadruped_nav.rl.terrain_grid import ALL_TERRAIN_TYPES, TerrainGrid

INK, INK_2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def window(kind, level, noise, seed, pile_pit_ramp, size=3.):
    grid = TerrainGrid((kind,), levels=10, seed=seed, noise=noise, pile_pit_ramp=pile_pit_ramp)
    tile = next(t for t in grid.tiles() if t[0] == f'tile_{level}_0')[3]
    res = grid.tile_size / (tile.shape[0] - 1)
    start, cells = int(round((grid.tile_size / 2 + .7) / res)), int(round(size / res))
    return tile[start:start + cells, start:start + cells]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--types', nargs='+', default=list(ALL_TERRAIN_TYPES))
    p.add_argument('--levels', nargs='+', type=int, default=[0, 5, 9])
    p.add_argument('--noise', nargs='+', type=float, default=[0., .5, 1.])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pile-pit-ramp', action='store_true', help='pile pits deepen with level (as in the v3/v4 configs)')
    p.add_argument('--output', default='outputs/terrain_preview.png')
    a = p.parse_args()
    columns = [(noise, level) for noise in a.noise for level in a.levels]
    fig, axes = plt.subplots(len(a.types), len(columns), figsize=(1.9 * len(columns), 1.9 * len(a.types) + .6),
                             facecolor=SURFACE, squeeze=False)
    for row, kind in enumerate(a.types):
        maps = [window(kind, level, noise, a.seed, a.pile_pit_ramp) for noise, level in columns]
        low, high = min(m.min() for m in maps), max(m.max() for m in maps)
        for col, ((noise, level), heights) in enumerate(zip(columns, maps)):
            ax = axes[row][col]
            image = ax.imshow(heights, origin='lower', cmap='cividis', vmin=low, vmax=high, extent=(0, 3, 0, 3))
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values(): side.set_visible(False)
            if row == 0:
                ax.set_title(f'noise {noise:g}\nlevel {level}', fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(kind.replace('_', ' '), fontsize=9, color=INK)
        bar = fig.colorbar(image, ax=axes[row].tolist(), fraction=.015, pad=.01)
        bar.ax.tick_params(labelsize=7, colors=INK_2)
        bar.set_label('height (m)', fontsize=7, color=INK_2)
    fig.suptitle('Training terrain: 3 m x 3 m windows, regular (noise 0) vs irregular (v3: 0.5, v4: 1.0)',
                 fontsize=11, color=INK, x=.02, ha='left')
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches='tight'); plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
