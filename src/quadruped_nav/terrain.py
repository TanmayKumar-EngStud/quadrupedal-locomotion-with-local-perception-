"""Twenty-row terrain curriculum. Scene JSON remains the common geometry source."""
from __future__ import annotations
import math
import numpy as np

TERRAINS = ("flat", "open", "holes", "moat", "dead_end", "steps", "rough", "slalom", "bridge", "slanted", "mix")

# Family "mix": a longer course of MIX_SECTION-long sections, one per family below in an order shuffled by the
# scene seed, after a regular entry of MIX_START. Each section follows its own family's rule and ranges.
MIX_FAMILIES = ("rough", "slanted", "steps", "holes", "bridge", "moat", "slalom")
MIX_START, MIX_SECTION = 1.2, 1.5

# Per-pile irregularities (--irregularity A): Gaussian deviations with mean 0 and standard deviation
# A x unit, the same for every family, clipped at IRREGULAR_CLIP standard deviations.
IRREGULAR_CLIP = 2.5
HEIGHT_UNIT, TILT_UNIT, SIZE_UNIT, SHIFT_UNIT = .02, math.radians(6), .03, .02   # m, rad, m, m
MAX_TILT, MIN_SIZE = math.radians(15), .12


def irregular_piles(scene, amount, seed):
    """Copy of scene with every stone perturbed (platforms unchanged): top height, tilt about x and y, side length, and
    centre position within its own lattice cell (so neighbouring piles never overlap). amount 0 returns the scene."""
    if not amount: return scene
    rng = np.random.default_rng(seed)
    gauss = lambda unit, n=None: np.clip(rng.normal(0., amount * unit, n), -IRREGULAR_CLIP * amount * unit, IRREGULAR_CLIP * amount * unit)
    pitch = scene['parameters']['stone_size'] + scene['parameters']['gap']
    stones = []
    for s in scene['stones']:
        size = float(np.clip(s['size'] + gauss(SIZE_UNIT), MIN_SIZE, max(MIN_SIZE, min(s['size'], pitch) + SIZE_UNIT)))
        room = max(0., (pitch - size) / 2)
        dx, dy = np.clip(gauss(SHIFT_UNIT, 2), -room, room)
        roll, tilt = np.clip(np.array([s['roll'], s['pitch']]) + gauss(TILT_UNIT, 2), -MAX_TILT, MAX_TILT)
        stones.append(dict(s, x=float(s['x'] + dx), y=float(s['y'] + dy), z_top=float(s['z_top'] + gauss(HEIGHT_UNIT)),
                           size=size, roll=float(roll), pitch=float(tilt)))
    return dict(scene, stones=stones, metadata=dict(scene['metadata'], irregularity=float(amount), irregularity_seed=int(seed)))


def _mix_pile(kind, local, y, t, size, rng, bridge_y, wall_y):
    """(keep, top height, roll, pitch) of one pile in a mix section of family kind; local = x within the section.
    The rules and ranges are those of the family itself, moved into the section."""
    z = roll = tilt = 0.
    if kind == "holes" and rng.random() <= .05 + .20*t: return False, z, roll, tilt
    if kind == "bridge" and abs(y) > .55-.3*t: return False, z, roll, tilt
    if kind == "moat" and .35 < local < 1.15 and abs(y-bridge_y) > .6: return False, z, roll, tilt
    if kind == "slalom" and abs(local-.75) < .30+size/2 and abs(y-wall_y) < 1.0+size/2: return False, z, roll, tilt
    if kind == "rough": z = float(rng.uniform(-.08*t, .08*t))
    if kind == "steps" and .25 <= local <= 1.25: z = .02+.06*t          # one terrace: a step up, then a step down
    if kind == "slanted": roll, tilt = rng.uniform(-math.radians(15)*t, math.radians(15)*t, 2)
    return True, z, float(roll), float(tilt)


def generate_terrain(family="open", row=0, seed=0, pocket_depth=2.0, irregularity=0., irregularity_seed=None):
    """One pile scene. family "mix" is a 12 m course with one section of each of MIX_FAMILIES (rough, slanted, steps,
    holes, bridge, moat, slalom) in a seed-shuffled order. irregularity > 0 then perturbs every pile of any family
    (irregular_piles); without an irregularity_seed each call draws a new layout."""
    if family not in TERRAINS or not 0 <= row <= 19:
        raise ValueError("Unknown terrain or row outside 0..19")
    if irregularity < 0: raise ValueError("irregularity must be >= 0")
    rng = np.random.default_rng(seed)
    t = row / 19
    size, gap = .50 - .38*t, .03 + .10*t
    # Fixed pitch limits geometry count and lets curricula shrink tops gradually.
    pitch = size + gap
    course = MIX_START + MIX_SECTION*len(MIX_FAMILIES) + .3 if family == "mix" else 7
    nx, ny = math.ceil(course/pitch)+1, math.ceil(3/pitch)
    bridge_y = float(rng.choice([-1, 1])) * (1.5 + 1.2*t)
    if family == "mix":
        sections = [str(f) for f in rng.permutation(MIX_FAMILIES)]
        wall_y = -.7 if bridge_y > 0 else .7
    stones = []
    for i in range(nx):
        for j in range(-ny, ny+1):
            x, y = i*pitch, j*pitch
            if family == "mix":
                k = int((x - MIX_START) // MIX_SECTION)
                kind = sections[k] if x >= MIX_START and k < len(sections) else "open"
                keep, z, roll, pitch_angle = _mix_pile(kind, x - MIX_START - k*MIX_SECTION, y, t, size, rng, bridge_y, wall_y)
                if keep:
                    stones.append(dict(id=f"T{i}_{j}", x=x, y=y, z_top=z, size=size, roll=roll, pitch=pitch_angle,
                                       friction=float(rng.uniform(.8-.3*t, 1.2))))
                continue
            keep = True
            if family == "holes" and 1 < x < 6:
                keep = rng.random() > .05 + .20*t
            if family in ("moat", "dead_end"):
                if 3 < x < 3.8 and abs(y-bridge_y) > .6:
                    keep = False
                if family == "dead_end" and 3-pocket_depth < x < 3.8 and .75 < abs(y) < 1.1:
                    keep = False
            if family == "bridge" and 2 < x < 5 and abs(y) > .55-.3*t:
                keep = False
            # Tall blocks occupy gaps in the navigable pile list: no planner can
            # nominate their tops as footholds. Height sensing still sees them.
            if family == "slalom" and any(abs(x-bx)<.30+size/2 and abs(y-by)<1.0+size/2 for bx,by in [(2,-.7),(4,.7)]):
                keep = False
            if not keep:
                continue
            z = 0.0
            if family == "rough": z = float(rng.uniform(-.08*t,.08*t))
            if family == "steps": z = (.02+.06*t)*min(int(x/1.2), int((7-x)/1.2))
            roll = pitch_angle = 0.0
            if family == "slanted":
                roll, pitch_angle = rng.uniform(-math.radians(15)*t, math.radians(15)*t, 2)
            stones.append(dict(id=f"T{i}_{j}", x=x,y=y,z_top=z,size=(pitch+.001 if family=="flat" else size),
                               roll=float(roll),pitch=float(pitch_angle),friction=float(rng.uniform(.8-.3*t,1.2))))
    start_x, goal_x = -(size/2+gap+.5), (nx-1)*pitch+size/2+gap+.5
    scene = dict(schema_version=1,units="metres",family=family,level=row,seed=seed,
                 stones=stones,platforms=[dict(id="P0",role="start",x=start_x,y=0.,z_top=0.,size=1.),
                                         dict(id="P2",role="goal",x=goal_x,y=0.,z_top=0.,size=1.)],
                 landmarks=[],start=dict(platform="P0",heading=0.),
                 parameters=dict(pit_depth=1.2,stone_size=size,gap=gap,curriculum_rows=20,pocket_depth=pocket_depth),
                 metadata=dict(bridge_y=bridge_y,calibrated=False), obstacles=[])
    if family == "slalom":
        scene["obstacles"]=[dict(id=f"wall{i}",x=x,y=y,z_top=.8,size_x=.6,size_y=2.) for i,(x,y) in enumerate([(2,-.7),(4,.7)])]
    if family == "mix":
        starts = [MIX_START + k*MIX_SECTION for k in range(len(sections))]
        scene["metadata"] = dict(scene["metadata"], mix_sections=[[f, s, s + MIX_SECTION] for f, s in zip(sections, starts)])
        scene["obstacles"] = [dict(id="wall0", x=starts[sections.index("slalom")] + .75, y=wall_y, z_top=.8, size_x=.6, size_y=2.)]
    if irregularity:
        if irregularity_seed is None: irregularity_seed = int(np.random.SeedSequence().entropy % 2**31)
        scene = irregular_piles(scene, irregularity, irregularity_seed)
    return scene


def surface_normal(roll, pitch):
    return np.array([math.sin(pitch)*math.cos(roll),-math.sin(roll),math.cos(pitch)*math.cos(roll)])


def main():
    import argparse
    from pathlib import Path
    from .scene import save_scene
    from .mjcf import build_mjcf
    parser=argparse.ArgumentParser(description='Create a twenty-row terrain scene for inspection or evaluation.')
    parser.add_argument('--family',choices=TERRAINS,default='open');parser.add_argument('--row',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--pocket-depth',type=float,default=2.)
    parser.add_argument('--output',default='outputs/terrains');args=parser.parse_args()
    scene=generate_terrain(args.family,args.row,args.seed,args.pocket_depth)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stem=f'{args.family}_row{args.row}_seed{args.seed}'
    save_scene(scene,out/f'{stem}.json');build_mjcf(scene,out/f'{stem}.xml')
    print(out/f'{stem}.json')

if __name__=='__main__':main()
