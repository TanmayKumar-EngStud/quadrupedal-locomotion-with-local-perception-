"""Hop-success table: how often the frozen controller completes one planner subgoal (gate G2).

    python -m quadruped_nav.calibrate --checkpoint artifacts/controller_v1/controller.pt [--ground heightfield]

One trial is one episode on a generated pile scene. The robot starts on the start
platform and is sent along a chain of piles ahead of it, exactly as the physics
executor (evaluate.py) sends planner subgoals: only hops a planner may choose (the
abstract simulator's action mask: 0.25-2 m away, a supported straight approach),
the same speed (0.75 m/s), time budget (2 s + 2 x length / speed) and arrival
radius. Every hop is recorded with its outcome:

    arrived  the base came within the arrival radius before the budget ran out
    fall     the episode ended on a fall (or a physically impossible state)
    timeout  the budget ran out

A fall or a timeout ends the trial. As in the executor, the next hop starts as soon as
the robot arrives, so an unstable landing shows up as a failed next hop.

The table bins hops by family, row and length. CalibratedEnv replaces the abstract
simulator's always-successful hops with draws from these measured odds.

--ground heightfield builds each scene as the heightfield the controller senses (the
training terrain's representation) instead of boxes. --irregularity A perturbs every pile
(terrain.irregular_piles), with the trial's seed, so the table stays reproducible.
--families mix measures the mixed family (holes + rough + slanted). With boxes, the soft Go1 foot
slides through 2-4 cm cracks between piles, which a 4.6 cm foot should not do; the
two tables together separate that simulator artifact from the controller's skill.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from multiprocessing import get_context
from pathlib import Path
import random
import time
import torch
from .abstract import Limits, Terrain, observe_pose
from .benchmark import wilson
from .rl.config import EnvConfig
from .rl.env import GoalEnv, yaw_of
from .rl.policy import load_controller
from .terrain import generate_terrain

FAMILIES = ('open', 'rough', 'steps', 'slanted')
ROWS = tuple(range(20))
LENGTH_BINS = ((.25, 1.), (1., 1.5), (1.5, 2.01))
HOPS_PER_TRIAL = 6
# Families whose hops look like another family's: they only remove piles (or add walls
# the planner never steps on), so the hop geometry the controller sees is unchanged.
PROXY_FAMILY = dict(flat='open', holes='open', moat='open', dead_end='open', bridge='open', slalom='open')


def run_trial(checkpoint, family, row, trial, ground='boxes', irregularity=0.):
    """Run one chained-hop episode and return one record per hop."""
    torch.set_num_threads(1)
    policy, saved = load_controller(checkpoint)
    seed = 300000 + 1000 * row + trial
    cfg = replace(EnvConfig(**saved['config']['env']), num_envs=1, backend='native', device='cpu',
                  seed=seed, noise=0., episode_seconds=1e9)
    scene = generate_terrain(family, row, seed, irregularity=irregularity, irregularity_seed=seed)
    env, rng = GoalEnv(cfg, scene, ground=ground), random.Random(seed)
    terrain, limits = Terrain.from_scene(scene), Limits()
    piles = {p.id: p for p in terrain.piles}
    records, obs = [], env.observations(noise=False)[0]
    for hop in range(HOPS_PER_TRIAL):
        q = env.backend.tensor('qpos')[0]
        start_xyz = q[:3].tolist()
        view = observe_pose(terrain, limits, start_xyz, float(yaw_of(q[None, 3:7])[0]))
        low, high = LENGTH_BINS[(trial + hop) % len(LENGTH_BINS)]
        candidates = [p for p, allowed in zip(view.visible_piles, view.action_mask) if allowed]
        candidates = [p for p in (piles[c.id] for c in candidates)
                      if low <= math.dist((p.x, p.y), start_xyz[:2]) < high and p.x > start_xyz[0] + .1 and abs(p.y) < 2.]
        if not candidates:
            break
        pile = rng.choice(candidates)
        pile = torch.tensor([pile.x, pile.y, pile.z])
        env.set_target(pile[None, :2])
        outcome = 'timeout'
        while float(env.target_elapsed[0]) < float(env.target_budget[0]):
            with torch.no_grad():
                action = policy.act(obs)
            obs, _, _, done, info = env.step(action, auto_reset=False, chain=False)
            if bool(info['arrived'][0]):
                outcome = 'arrived'; break
            if bool(done[0]):
                outcome = 'fall'; break
        records.append(dict(family=family, row=row, trial=trial, hop=hop, from_platform=hop == 0,
                            length_m=math.dist(start_xyz[:2], pile[:2].tolist()),
                            height_change_m=float(pile[2]) - (start_xyz[2] - .3),
                            stone_size_m=scene['parameters']['stone_size'], gap_m=scene['parameters']['gap'],
                            seconds=float(env.target_elapsed[0]), outcome=outcome))
        if outcome != 'arrived':
            break
    return records


def _run(job):
    return run_trial(*job)


def summarise(records):
    """Bins of the table: one per family, row and length range."""
    bins = []
    for family in sorted({r['family'] for r in records}):
        for row in sorted({r['row'] for r in records}):
            for low, high in LENGTH_BINS:
                hops = [r for r in records if r['family'] == family and r['row'] == row and low <= r['length_m'] < high]
                if not hops:
                    continue
                arrived = sum(r['outcome'] == 'arrived' for r in hops)
                bins.append(dict(family=family, row=row, length_min=low, length_max=high, n=len(hops),
                                 successes=arrived, probability=arrived / len(hops), wilson95=wilson(arrived, len(hops)),
                                 falls=sum(r['outcome'] == 'fall' for r in hops),
                                 timeouts=sum(r['outcome'] == 'timeout' for r in hops)))
    return bins


def calibrate(checkpoint, output, trials=24, families=FAMILIES, rows=ROWS, workers=4, ground='boxes', irregularity=0.):
    jobs = [(str(checkpoint), family, row, trial, ground, irregularity) for family in families for row in rows for trial in range(trials)]
    started, records = time.monotonic(), []
    with get_context('spawn').Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_run, jobs), 1):
            records.extend(result)
            if done % 100 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} trials, {time.monotonic() - started:.0f} s', flush=True)
    records.sort(key=lambda r: (r['family'], r['row'], r['trial'], r['hop']))
    result = dict(format='hop-calibration/v2', checkpoint=str(Path(checkpoint).resolve()), ground=ground, irregularity=irregularity,
                  hops_per_trial=HOPS_PER_TRIAL, trials_per_cell=trials, records=records, bins=summarise(records),
                  wall_seconds=time.monotonic() - started,
                  note='Measured family/row/length bins; unmeasured cells fail rather than extrapolate. '
                       f'Native CPU MuJoCo with {ground} ground, deterministic actions, executor speed and budget.')
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    return path


class CalibratedEnv:
    """Wraps the abstract simulator: each hop succeeds with its measured probability.

    A family without its own measurements uses its proxy (PROXY_FAMILY). A row without
    measurements uses the next harder measured row, which can only make hops less
    likely to succeed, never more.
    """
    def __init__(self, env, table, family, row, seed=0):
        self.env, self.rng = env, random.Random(seed)
        bins = json.loads(Path(table).read_text())['bins']
        family = PROXY_FAMILY.get(family, family) if not any(b['family'] == family for b in bins) else family
        measured = sorted({b['row'] for b in bins if b['family'] == family and b['row'] >= row})
        if not measured:
            raise ValueError(f'No hop measurements for {family} at row {row} or harder; run calibrate first')
        self.bins = [b for b in bins if b['family'] == family and b['row'] == measured[0]]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, target):
        start, end = self.env.geometry.by_id[self.env.current_id], self.env.geometry.by_id.get(target) if target else None
        if end is None:
            return self.env.step(target)
        length = math.hypot(end.x - start.x, end.y - start.y)
        matching = [b for b in self.bins if b['length_min'] <= length < b['length_max']] or \
                   [b for b in self.bins if length < b['length_max']]   # hops under 0.25 m: the shortest bin
        if not matching:
            raise ValueError(f'No measured bin for a {length:.2f} m hop; collect calibration rather than extrapolating')
        before = self.env.state()
        obs, done = self.env.step(target)
        if self.env.status not in ('timeout', 'no_candidate', 'planner_no_route') and self.rng.random() > matching[0]['probability']:
            self.env.current_id, self.env.distance_m, self.env.status = before['pile_id'], before['distance_m'], 'fall'
            self.env.states[-1] = self.env.state()
            self.env.events[-1]['outcome'] = 'fall'
            return self.env.observe(), True
        return obs, done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='artifacts/controller_v1/controller.pt')
    p.add_argument('--trials', type=int, default=24, help='episodes per family and row')
    p.add_argument('--families', nargs='+', default=list(FAMILIES))
    p.add_argument('--rows', nargs='+', type=int, default=list(ROWS))
    p.add_argument('--workers', type=int, default=4, help='parallel CPU processes (keep <= 5 on the KTH pod)')
    p.add_argument('--ground', choices=['boxes', 'heightfield'], default='boxes',
                   help='scene supports as MuJoCo boxes, or as the heightfield the controller senses')
    p.add_argument('--irregularity', type=float, default=0., help='per-pile irregularities (0 = the regular scenes)')
    p.add_argument('--output', default='outputs/calibration/hops.json')
    a = p.parse_args()
    if a.trials < 1: p.error('At least one trial required')
    path = calibrate(a.checkpoint, a.output, a.trials, a.families, a.rows, a.workers, a.ground, a.irregularity)
    table = json.loads(path.read_text())
    print(f'{len(table["records"])} hops in {table["wall_seconds"]:.0f} s -> {path}')
    for b in table['bins']:
        low, high = b['wilson95']
        print(f'{b["family"]:8s} row {b["row"]:2d}  {b["length_min"]:.2f}-{b["length_max"]:.2f} m  '
              f'{b["successes"]:3d}/{b["n"]:<3d} = {b["probability"]:.2f}  [{low:.2f}, {high:.2f}]  '
              f'falls {b["falls"]}  timeouts {b["timeouts"]}')


if __name__ == '__main__':
    main()
