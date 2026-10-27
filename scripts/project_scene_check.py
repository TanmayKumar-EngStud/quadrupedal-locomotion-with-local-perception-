"""Run a trained controller with a planner on the project's scene families (CPU MuJoCo).

    python scripts/project_scene_check.py --checkpoint runs/kth-goal-v3-seed0/last.pt \
        --planner oracle --rows 0 5 --seeds 3 --workers 32

Each episode: the planner picks pile subgoals, the controller walks to them in
native MuJoCo (box stones over 1.2 m pits), 60 s limit. Writes one JSON with
every episode and a per-family/row summary.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
from pathlib import Path

FAMILIES = ['open', 'holes', 'moat', 'dead_end', 'steps', 'rough', 'slalom', 'bridge', 'slanted']


def run(job):
    import torch
    torch.set_num_threads(1)
    from quadruped_nav.evaluate import evaluate
    from quadruped_nav.terrain import generate_terrain
    checkpoint, planner, family, row, seed, seconds = job
    result = evaluate(checkpoint, generate_terrain(family, row, seed), planner, seconds, seed)
    return dict(family=family, row=row, seed=seed, planner=planner, status=result['status'],
                success=result['success'], time_s=result['time_s'], distance_m=result['distance_m'],
                decisions=len(result['decisions']))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--planner', default='oracle')
    p.add_argument('--rows', type=int, nargs='+', default=[0]); p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--families', nargs='+', default=FAMILIES); p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--output', default='outputs/evaluation/project_scenes.json')
    a = p.parse_args()
    jobs = [(a.checkpoint, a.planner, f, r, 7000 + s, a.seconds) for f in a.families for r in a.rows for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as pool:
        episodes = list(pool.map(run, jobs))
    groups = defaultdict(list)
    for e in episodes: groups[f"{e['family']}/row{e['row']}"].append(e)
    summary = {k: dict(episodes=len(v), success_rate=sum(e['success'] for e in v) / len(v),
                       statuses={s: sum(e['status'] == s for e in v) for s in {e['status'] for e in v}},
                       mean_time_s=sum(e['time_s'] for e in v) / len(v)) for k, v in groups.items()}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(checkpoint=a.checkpoint, planner=a.planner, summary=summary, episodes=episodes), indent=2) + '\n')
    for k, v in summary.items(): print(k, json.dumps(v))
    print(out)


if __name__ == '__main__':
    main()
