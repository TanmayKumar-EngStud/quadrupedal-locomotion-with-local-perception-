"""Dead-end planner comparison in physics: pocket depth x planner x seed, one frozen controller.

    .venv/bin/python scripts/dead_end_sweep.py --output docs/results/dead_end_sweep_v10.jsonl

Each line of the output is one episode (native CPU MuJoCo, box ground, 60 s limit).
"""
import argparse
import itertools
import json
from multiprocessing import Pool


def episode(job):
    checkpoint, row, planner, depth, seed = job
    import torch
    torch.set_num_threads(1)
    from quadruped_nav.evaluate import evaluate
    from quadruped_nav.terrain import generate_terrain
    scene = generate_terrain('dead_end', row, seed, pocket_depth=depth)
    result = evaluate(checkpoint, scene, planner, 60, seed, ground='boxes')
    targets = [d['target'] for d in result['decisions']]
    return dict(planner=planner, depth=depth, seed=seed, row=row, status=result['status'], time_s=round(result['time_s'], 2),
                distance_m=round(result['distance_m'], 2), subgoals=len(targets), repeats=len(targets) - len(set(targets)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='artifacts/controller_v10/controller.pt')
    p.add_argument('--row', type=int, default=3, help='pile-scene row (difficulty)')
    p.add_argument('--depths', type=float, nargs='+', default=[1., 2., 3.], help='pocket depths in metres (the course allows up to 3)')
    p.add_argument('--planners', nargs='+', default=['greedy', 'optimistic', 'oracle'])
    p.add_argument('--seeds', type=int, default=6, help='seeds per planner and depth, starting at 10000')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', default='docs/results/dead_end_sweep.jsonl')
    a = p.parse_args()
    jobs = [(a.checkpoint, a.row, pl, d, s) for pl, d, s in itertools.product(a.planners, a.depths, range(10000, 10000 + a.seeds))]
    with Pool(a.workers) as pool, open(a.output, 'w') as f:
        for result in pool.imap_unordered(episode, jobs):
            f.write(json.dumps(result) + '\n'); f.flush(); print(result, flush=True)


if __name__ == '__main__':
    main()
