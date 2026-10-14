"""Twenty-row terrain curriculum. Scene JSON remains the common geometry source."""
from __future__ import annotations
import math
import numpy as np

TERRAINS = ("flat", "open", "holes", "moat", "dead_end", "steps", "rough", "slalom", "bridge", "slanted")


def generate_terrain(family="open", row=0, seed=0, pocket_depth=2.0):
    if family not in TERRAINS or not 0 <= row <= 19:
        raise ValueError("Unknown terrain or row outside 0..19")
    rng = np.random.default_rng(seed)
    t = row / 19
    size, gap = .50 - .38*t, .03 + .10*t
    # Fixed pitch limits geometry count and lets curricula shrink tops gradually.
    pitch = size + gap
    nx, ny = math.ceil(7/pitch)+1, math.ceil(3/pitch)
    bridge_y = float(rng.choice([-1, 1])) * (1.5 + 1.2*t)
    stones = []
    for i in range(nx):
        for j in range(-ny, ny+1):
            x, y = i*pitch, j*pitch
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
