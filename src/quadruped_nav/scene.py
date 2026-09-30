"""Deterministic scene JSON is the source of truth for simulator and future planners."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

FAMILIES = ("open", "moat", "dead_end")


def difficulty(level: int) -> tuple[float, float]:
    if not 0 <= level <= 9:
        raise ValueError("level must be in [0, 9]")
    return 0.46 - 0.04 * level, 0.04 + 0.01 * level


def generate_scene(family="moat", level=0, seed=1042, removal=0.0):
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if not 0 <= removal <= 0.2:
        raise ValueError("removal must be between 0 and 0.2")
    if family != "open" and removal:
        raise ValueError("random removal is supported only for open training fields")
    size, gap = difficulty(level)
    pitch = size + gap
    nx = math.ceil(6.5 / pitch) + 1
    half_rows = math.ceil(3.5 / pitch)
    mid = nx // 2
    bridge_row = math.ceil(2.5 / pitch)
    rng = np.random.default_rng(seed)
    stones = []
    for i in range(nx):
        for j in range(-half_rows, half_rows + 1):
            # Two missing columns; the only crossing is >2 m off the centreline.
            if family != "open" and i in (mid, mid + 1) and j != bridge_row:
                continue
            # Two side cuts enclose a pocket opening toward the start.
            if family == "dead_end" and mid - 4 <= i <= mid + 1 and abs(j) == math.ceil(1.0 / pitch):
                continue
            if family == "open" and i not in (0, nx - 1) and rng.random() < removal:
                continue
            stones.append(dict(id=f"S{i:02d}_{j + half_rows:02d}", x=round(i*pitch, 6),
                               y=round(j*pitch, 6), z_top=0.0, size=round(size, 6)))
    start_x = -(size/2 + gap + 0.5)
    goal_x = (nx-1)*pitch + size/2 + gap + 0.4
    platforms = [dict(id="P0", role="start", x=round(start_x, 6), y=0.0, z_top=0.0, size=1.0)]
    landmarks = []
    for k, (row, colour, kind) in enumerate([(-round(2/pitch), "red", "box"), (0, "blue", "cone"), (round(2/pitch), "yellow", "cone")], 1):
        y = round(row * pitch, 6)
        platforms.append(dict(id=f"P{k}", role="goal", x=round(goal_x, 6), y=y, z_top=0.0, size=0.8))
        landmarks.append(dict(id=f"L{k}", type=kind, colour=colour, x=round(goal_x+0.65, 6), y=y, z_base=0.0, platform=f"P{k}"))
    return dict(schema_version=1, seed=seed, family=family, level=level, units="metres",
                parameters=dict(stone_size=round(size,6), gap=round(gap,6), pit_depth=1.2,
                                removal_fraction=removal, lateral_shift=0.0, height_variation=0.0),
                stones=stones, platforms=platforms, landmarks=landmarks,
                start=dict(platform="P0", heading=0.0),
                metadata=dict(bridge_y=round(bridge_row*pitch,6) if family != "open" else None,
                              note="Geometric layout only; traversal limits are not controller-calibrated."))


def save_scene(scene, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene, indent=2) + "\n")


def load_scene(path):
    scene = json.loads(Path(path).read_text())
    if scene.get("schema_version") != 1:
        raise ValueError("Unsupported scene schema")
    return scene


def geometric_route(scene, target="P2"):
    """Connectivity diagnostic only. Gap threshold is nominal, not learned capability."""
    import heapq
    nodes = scene["stones"] + scene["platforms"]
    ids = [n["id"] for n in nodes]
    start, end = ids.index(scene["start"]["platform"]), ids.index(target)
    xy = np.array([[n["x"], n["y"]] for n in nodes])
    half = np.array([n["size"]/2 for n in nodes])
    heights = np.array([n["z_top"] for n in nodes])
    limit = scene["parameters"]["gap"] + 1e-5
    frontier, best, parent = [(0.0, start)], {start: 0.0}, {}
    while frontier:
        _, i = heapq.heappop(frontier)
        if i == end:
            route = [i]
            while i in parent:
                i = parent[i]
                route.append(i)
            return [nodes[k] for k in reversed(route)]
        edge_gaps = np.linalg.norm(np.maximum(np.abs(xy-xy[i])-half[:,None]-half[i], 0), axis=1)
        for j in np.flatnonzero((edge_gaps <= limit) & (np.abs(heights-heights[i]) <= 0.05)):
            cost = best[i] + float(np.linalg.norm(xy[j]-xy[i]))
            if j != i and cost < best.get(int(j), float("inf")):
                best[int(j)], parent[int(j)] = cost, i
                heapq.heappush(frontier, (cost+float(np.linalg.norm(xy[j]-xy[end])), int(j)))
    return []
