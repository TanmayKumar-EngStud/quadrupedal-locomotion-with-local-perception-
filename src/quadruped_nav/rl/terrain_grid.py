"""Training terrain: a grid of terrain tiles.

Rows are difficulty levels (row 0 is easiest) and columns are terrain types.
Every parallel robot lives on one tile; a per-robot curriculum moves it to a
harder row after it reaches its targets and to an easier row after it fails.

The same height array is used twice:
  * as MuJoCo heightfields (one per tile, see tiles()), so the physics collides
    with it, and
  * as a HeightGrid, so observations and rewards sample terrain heights.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch

TERRAIN_TYPES = ("rough", "slope", "stairs_up", "stairs_down", "obstacles", "stepping_stones", "gaps")
# Types added for controller v2. Each draws its tiles from its own random stream,
# so adding one leaves the tiles of the seven original types unchanged.
EXTRA_TERRAIN_TYPES = ("piles",)
ALL_TERRAIN_TYPES = TERRAIN_TYPES + EXTRA_TERRAIN_TYPES
FINE_TYPES = ("piles",)  # built with fine_resolution cells (2.5 cm), so 3 cm gaps exist
HOLE_THRESHOLD = -.04   # terrain below this on stepping-stone and gap tiles is a hole


def pit_depth(d):
    """Hole depth grows with difficulty: missteps on easy tiles are survivable."""
    return .05 + .45 * d


def pile_pit_depth(d):
    """Pile tiles have the deepest pit (0.5 m) on every level, as the project's pile
    scenes have 1.2 m pits on every row (sensed as 0.5 m, see HeightGrid.from_scene).

    A depth curriculum (0.05 m pits on level 0) was tried first: pile robots reached
    93% of their targets on level 0 but fell once promoted, and stayed near level 0,
    where a misstep costs nothing. v1's stepping stones stalled the same way (level
    0.35) and v1 then failed on the deep-pit pile scenes.
    """
    return pit_depth(1.)


class HeightGrid:
    """Terrain height z(x, y) on a regular grid, sampled with bilinear interpolation."""

    def __init__(self, heights, x_min, y_min, resolution, outside, device='cpu'):
        self.values = torch.as_tensor(np.asarray(heights, dtype=np.float32), device=device)
        self.x_min, self.y_min, self.resolution, self.outside = float(x_min), float(y_min), float(resolution), float(outside)

    def sample(self, xy):
        """Heights at world points xy[..., 2]; points off the grid get the "outside" height."""
        gx = (xy[..., 0] - self.x_min) / self.resolution
        gy = (xy[..., 1] - self.y_min) / self.resolution
        ny, nx = self.values.shape
        inside = (gx >= 0) & (gx <= nx - 1) & (gy >= 0) & (gy <= ny - 1)
        gx, gy = gx.clamp(0, nx - 1.001), gy.clamp(0, ny - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - x0, gy - y0
        v = self.values
        h = (v[y0, x0] * (1 - fx) * (1 - fy) + v[y0, x0 + 1] * fx * (1 - fy)
             + v[y0 + 1, x0] * (1 - fx) * fy + v[y0 + 1, x0 + 1] * fx * fy)
        return torch.where(inside, h, torch.full_like(h, self.outside))

    @classmethod
    def from_scene(cls, scene, device='cpu', resolution=.025, fill_gaps=0.):
        """Rasterize a project scene (stones, platforms, walls) into a height grid.

        fill_gaps (metres) optionally fills narrower gaps. It is off by default:
        with the Go1's soft foot contact, MuJoCo lets a foot slip through gaps of
        2-4 cm between boxes, so even narrow gaps are real hazards to perceive.
        Pits read at most pit_depth(1) = 0.5 m deep, the deepest pit seen in
        training: a 1.2 m pit is no more survivable than a 0.5 m one.
        """
        records = scene['stones'] + scene['platforms']
        x_min = min(p['x'] - p['size'] for p in records) - 1
        y_min = min(p['y'] - p['size'] for p in records) - 1
        x_max = max(p['x'] + p['size'] for p in records) + 1
        y_max = max(p['y'] + p['size'] for p in records) + 1
        nx, ny = math.ceil((x_max - x_min) / resolution) + 1, math.ceil((y_max - y_min) / resolution) + 1
        floor = 0. if scene['parameters'].get('continuous_floor') else -min(scene['parameters']['pit_depth'], pit_depth(1.))
        heights = np.full((ny, nx), floor, dtype=np.float32)
        for p in records + scene.get('obstacles', []):
            sx, sy = p.get('size', p.get('size_x')), p.get('size', p.get('size_y'))
            r, ph = p.get('roll', 0), p.get('pitch', 0)
            # R = Ry(pitch) Rx(roll); the terrain contract has no yaw.
            R = np.array([[math.cos(ph), math.sin(ph) * math.sin(r), math.sin(ph) * math.cos(r)],
                          [0, math.cos(r), -math.sin(r)],
                          [-math.sin(ph), math.cos(ph) * math.sin(r), math.cos(ph) * math.cos(r)]])
            radius = max(sx, sy)
            ix0 = max(0, int((p['x'] - radius - x_min) / resolution)); ix1 = min(nx, int((p['x'] + radius - x_min) / resolution) + 2)
            iy0 = max(0, int((p['y'] - radius - y_min) / resolution)); iy1 = min(ny, int((p['y'] + radius - y_min) / resolution) + 2)
            x, y = np.meshgrid(x_min + np.arange(ix0, ix1) * resolution, y_min + np.arange(iy0, iy1) * resolution)
            dx, dy = x - p['x'], y - p['y']; normal = R[:, 2]
            dz = -(normal[0] * dx + normal[1] * dy) / normal[2]
            local_x = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
            local_y = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
            inside = (abs(local_x) <= sx / 2) & (abs(local_y) <= sy / 2)
            patch = heights[iy0:iy1, ix0:ix1]
            np.maximum(patch, np.where(inside, p['z_top'] + dz, floor), out=patch)
        if fill_gaps:
            heights = close_gaps(heights, int(fill_gaps / resolution) + 1)
        return cls(heights, x_min, y_min, resolution, floor, device)


def close_gaps(heights, cells):
    """Grey-scale closing with a cells x cells window: fills pits narrower than the window."""
    from numpy.lib.stride_tricks import sliding_window_view
    def filtered(a, reduce):
        lo, hi = (cells - 1) // 2, cells // 2
        a = np.pad(a, ((lo, hi), (0, 0)), mode='edge')
        a = reduce(sliding_window_view(a, cells, axis=0), axis=-1)
        a = np.pad(a, ((0, 0), (lo, hi)), mode='edge')
        return reduce(sliding_window_view(a, cells, axis=1), axis=-1)
    return filtered(filtered(heights, np.max), np.min).astype(np.float32)


# ---------------------------------------------------------------------------
# Tile generators. Each returns an (n, n) height array in metres for a square
# tile, given difficulty d in [0, 1]. Tile edges are near height 0 so that
# neighbouring tiles join without cliffs; the centre is a flat spawn platform
# (on a hill or in a valley for slopes and stairs).
# ---------------------------------------------------------------------------

def _coords(n, res):
    c = (np.arange(n) - (n - 1) / 2) * res
    x, y = np.meshgrid(c, c)            # x varies along columns, y along rows
    return x, y


def _flatten_centre(h, x, y, half):
    centre = (abs(x) <= half) & (abs(y) <= half)
    h[centre] = 0.
    return h


def rough_tile(d, rng, n, res):
    """Smooth random bumps (control points every 0.2 m, bilinear) plus fine noise."""
    amplitude = .02 + .08 * d
    every = max(1, round(.2 / res))
    coarse = rng.uniform(-amplitude, amplitude, (n // every + 2, n // every + 2))
    idx = np.arange(n) / every
    i0 = idx.astype(int); f = idx - i0
    rows = coarse[i0] * (1 - f)[:, None] + coarse[i0 + 1] * f[:, None]
    h = rows[:, i0] * (1 - f) + rows[:, i0 + 1] * f
    h += rng.uniform(-.005, .005, (n, n))
    x, y = _coords(n, res)
    return _flatten_centre(h, x, y, .5)


def slope_tile(d, rng, n, res):
    """Pyramid with slope up to ~0.45 (24 degrees); randomly a hill or a valley."""
    slope = .05 + .40 * d
    x, y = _coords(n, res)
    half = (n - 1) * res / 2
    distance_to_edge = half - np.maximum(abs(x), abs(y))
    h = slope * np.minimum(distance_to_edge, half - .75)   # edges at 0, flat top
    return h if rng.random() < .5 else -h


def stairs_tile(d, rng, n, res, up=True):
    """Pyramid stairs around a 1.5 m spawn platform (Go1: up to 15 cm steps).

    up=True: walking away from the platform climbs; up=False: it descends.
    Targets lie in every direction, so both tiles contain both directions.
    """
    step_height = .03 + .12 * d
    step_width = .35 if d < .5 else .30
    x, y = _coords(n, res)
    ring = np.ceil(np.maximum(np.maximum(abs(x), abs(y)) - .75, 0) / step_width)
    h = step_height * (ring - ring.max())   # tile edges at 0, platform in a pit
    return h if up else -h


def obstacles_tile(d, rng, n, res):
    """Random rectangular blocks raised or lowered by up to 15 cm."""
    height = .03 + .12 * d
    x, y = _coords(n, res)
    h = np.zeros((n, n))
    for _ in range(30):
        cx, cy = rng.uniform(-n * res / 2, n * res / 2, 2)
        sx, sy = rng.uniform(.3, 1.0, 2)
        mask = (abs(x - cx) < sx / 2) & (abs(y - cy) < sy / 2)
        h[mask] = rng.choice([-1., 1.]) * rng.uniform(.5, 1.) * height
    return _flatten_centre(h, x, y, .6)


def stepping_stones_tile(d, rng, n, res):
    """Square stones separated by pits; stones shrink and gaps widen with d.

    Gaps are at least one grid cell wide so the heightfield can represent them.
    """
    stone = .60 - .30 * d
    gap = max(res, .03 + .15 * d)
    pitch = stone + gap
    x, y = _coords(n, res)
    phase = rng.uniform(0, pitch, 2)
    on_stone = (np.mod(x + phase[0], pitch) < stone) & (np.mod(y + phase[1], pitch) < stone)
    # Each stone gets its own small height offset.
    ix = np.floor((x + phase[0]) / pitch).astype(int); iy = np.floor((y + phase[1]) / pitch).astype(int)
    offsets = rng.uniform(-.03 * d, .03 * d, (ix.max() - ix.min() + 1, iy.max() - iy.min() + 1))
    h = np.where(on_stone, offsets[ix - ix.min(), iy - iy.min()], -pit_depth(d))
    return _flatten_centre(h, x, y, .6)


def piles_tile(d, rng, n, res):
    """The project's pile scenes (terrain.py) with row 19 d: square piles over pits.

    Piles shrink from 0.50 to 0.12 m and gaps widen from 3 to 13 cm, as in the
    scenes. Like the rough, steps and slanted families, half the piles get a
    height offset (up to 8 cm) and half get a tilt (up to 15 degrees), both
    growing with d. A grid node is a pile node if it lies inside the pile, the
    rule HeightGrid.from_scene uses for the scenes, so at 2.5 cm cells every 3 cm
    gap holds at least one pit node.
    """
    stone, gap = .50 - .38 * d, .03 + .10 * d
    pitch = stone + gap
    x, y = _coords(n, res)
    phase = rng.uniform(0, pitch, 2)
    u, v = np.mod(x + phase[0], pitch), np.mod(y + phase[1], pitch)
    on_stone = (u <= stone) & (v <= stone)
    ix = np.floor((x + phase[0]) / pitch).astype(int); iy = np.floor((y + phase[1]) / pitch).astype(int)
    ix, iy = ix - ix.min(), iy - iy.min()
    count = (ix.max() + 1, iy.max() + 1)
    offset = rng.uniform(-.08 * d, .08 * d, count) * (rng.random(count) < .5)
    slope = np.tan(rng.uniform(-math.radians(15) * d, math.radians(15) * d, (2, *count))) * (rng.random(count) < .5)
    top = offset[ix, iy] + slope[0][ix, iy] * (u - stone / 2) + slope[1][ix, iy] * (v - stone / 2)
    h = np.where(on_stone, top, -pile_pit_depth(d))
    return _flatten_centre(h, x, y, .6)


def gaps_tile(d, rng, n, res):
    """Flat ground with random square holes (0.3 m cells)."""
    probability = .05 + .25 * d
    cell = max(1, round(.3 / res))
    holes = rng.random((n // cell + 1, n // cell + 1)) < probability
    h = np.where(holes[np.arange(n) // cell][:, np.arange(n) // cell], -pit_depth(d), 0.)
    x, y = _coords(n, res)
    return _flatten_centre(h, x, y, .6)


def make_tile(kind, d, rng, n, res):
    if kind == 'rough': return rough_tile(d, rng, n, res)
    if kind == 'slope': return slope_tile(d, rng, n, res)
    if kind == 'stairs_up': return stairs_tile(d, rng, n, res, up=True)
    if kind == 'stairs_down': return stairs_tile(d, rng, n, res, up=False)
    if kind == 'obstacles': return obstacles_tile(d, rng, n, res)
    if kind == 'stepping_stones': return stepping_stones_tile(d, rng, n, res)
    if kind == 'gaps': return gaps_tile(d, rng, n, res)
    if kind == 'piles': return piles_tile(d, rng, n, res)
    raise ValueError(f'Unknown terrain type {kind!r}')


def hole_threshold(kind, d):
    """Terrain below this height is a hole (no targets there); -inf for types without holes."""
    if kind in ('stepping_stones', 'gaps'): return HOLE_THRESHOLD
    if kind == 'piles': return -pile_pit_depth(d) / 2   # tilted or lowered piles stay above it
    return -math.inf


def refine(heights, factor):
    """The same bilinear surface sampled factor times more densely along both axes."""
    def along(a, axis):
        n = a.shape[axis]
        t = np.arange((n - 1) * factor + 1, dtype=np.float32) / factor
        i = np.minimum(t.astype(int), n - 2)
        w = np.expand_dims(t - i, 1 - axis)
        return np.take(a, i, axis) * (1 - w) + np.take(a, i + 1, axis) * w
    return heights if factor == 1 else along(along(heights, 0), 1).astype(np.float32)


@dataclass
class TerrainGrid:
    """Grid of terrain tiles; row = difficulty level, column = terrain type.

    Tiles are built with `resolution` cells, except FINE_TYPES, which use
    `fine_resolution`. With any fine tile present, the height array is stored at
    the fine resolution: coarse tiles are refined bilinearly (the same surface),
    and their heightfields keep the coarse cells (see tiles()).
    """
    types: tuple = TERRAIN_TYPES
    levels: int = 10
    tile_size: float = 8.
    resolution: float = .05
    border: float = 3.
    seed: int = 0
    fine_resolution: float = .025

    def __post_init__(self):
        self.types = tuple(self.types)
        for kind in self.types: make_tile(kind, 0., np.random.default_rng(0), 9, self.resolution)
        rng = np.random.default_rng(self.seed)
        own = {kind: np.random.default_rng([self.seed, ALL_TERRAIN_TYPES.index(kind)]) for kind in EXTRA_TERRAIN_TYPES}
        self.factor = round(self.resolution / self.fine_resolution) if set(self.types) & set(FINE_TYPES) else 1
        if abs(self.factor * self.fine_resolution - self.resolution) > 1e-9 and self.factor > 1:
            raise ValueError('resolution must be a whole multiple of fine_resolution')
        n = int(round(self.tile_size / self.resolution))
        b = int(round(self.border / self.resolution))
        f = self.factor
        heights = np.zeros((2 * b + len(self.types) * n + 1, 2 * b + self.levels * n + 1), dtype=np.float32)
        self.spawn_height = np.zeros((self.levels, len(self.types)), dtype=np.float32)
        self.hole_threshold = np.zeros((self.levels, len(self.types)), dtype=np.float32)
        fine_tiles = {}
        for level in range(self.levels):
            for column, kind in enumerate(self.types):
                d = level / max(1, self.levels - 1)
                if kind in FINE_TYPES:
                    fine = make_tile(kind, d, own.get(kind, rng), n * f, self.resolution / f)
                    fine_tiles[level, column] = fine
                    tile = fine[::f, ::f]
                else:
                    tile = make_tile(kind, d, own.get(kind, rng), n, self.resolution)
                heights[b + column * n:b + (column + 1) * n, b + level * n:b + (level + 1) * n] = tile
                self.spawn_height[level, column] = tile[n // 2, n // 2]
                self.hole_threshold[level, column] = hole_threshold(kind, d)
        if fine_tiles:
            heights = refine(heights, f)
            for (level, column), fine in fine_tiles.items():
                heights[(b + column * n) * f:(b + (column + 1) * n) * f, (b + level * n) * f:(b + (level + 1) * n) * f] = fine
        self.heights = heights
        self.height_resolution = self.resolution / f
        self.x_min = self.y_min = 0.
        # Tile (level, column) spans x in [x0, x0 + tile_size], y in [y0, y0 + tile_size].
        self.origin_x = self.border + np.arange(self.levels) * self.tile_size
        self.origin_y = self.border + np.arange(len(self.types)) * self.tile_size

    @property
    def size_x(self): return (self.heights.shape[1] - 1) * self.height_resolution

    @property
    def size_y(self): return (self.heights.shape[0] - 1) * self.height_resolution

    def height_grid(self, device):
        return HeightGrid(self.heights, self.x_min, self.y_min, self.height_resolution, outside=0., device=device)

    def tile_centres(self, device):
        """Tensor [levels, types, 3] of spawn platform centres (x, y, z)."""
        x = torch.tensor(self.origin_x + self.tile_size / 2, dtype=torch.float32)
        y = torch.tensor(self.origin_y + self.tile_size / 2, dtype=torch.float32)
        z = torch.tensor(self.spawn_height, dtype=torch.float32)
        grid = torch.stack(torch.broadcast_tensors(x[:, None], y[None, :], z), -1)
        return grid.to(device)

    def tiles(self):
        """(name, centre x, centre y, heights) for one heightfield per tile.

        One heightfield per tile keeps collision geometry within 4 m of its own
        frame. With a single 86 m heightfield, MuJoCo Warp's convex collision in
        the heightfield frame lost precision far from the centre and robots blew up.
        Neighbouring tiles share their edge samples, so the surface is continuous.
        Fine tiles keep every sample; coarse tiles take every factor-th one, which
        are exactly the samples they were built with.
        """
        n = int(round(self.tile_size / self.resolution)); b = int(round(self.border / self.resolution))
        f = self.factor
        for level in range(self.levels):
            for column, kind in enumerate(self.types):
                step = 1 if kind in FINE_TYPES else f
                heights = self.heights[(b + column * n) * f:(b + (column + 1) * n) * f + 1:step,
                                       (b + level * n) * f:(b + (level + 1) * n) * f + 1:step]
                yield (f'tile_{level}_{column}', self.origin_x[level] + self.tile_size / 2,
                       self.origin_y[column] + self.tile_size / 2, heights)

    def border_boxes(self):
        """(centre x, centre y, half x, half y) of the flat ground around the tiles (top at z=0)."""
        inner_x, inner_y = self.levels * self.tile_size, len(self.types) * self.tile_size
        b, sx, sy = self.border, self.size_x, self.size_y
        return [(b / 2, sy / 2, b / 2, sy / 2), (b + inner_x + b / 2, sy / 2, b / 2, sy / 2),
                (sx / 2, b / 2, inner_x / 2, b / 2), (sx / 2, b + inner_y + b / 2, inner_x / 2, b / 2)]
