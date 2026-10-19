"""Run the fixed terrain goal-reaching test on a saved checkpoint.

    python -m quadruped_nav.goal_check --checkpoint runs/goal-v3/last.pt [--backend native]

--terrain-types tests other terrain types than the checkpoint was trained on, e.g. the
seven v1 types for every controller (the same robots, tiles and targets) or piles only.

Writes the same JSON as the periodic evaluation in training (success, falls,
time to goal and progress per terrain type and difficulty level).
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from .rl.config import EnvConfig
from .rl.evaluation import TerrainEvaluator
from .rl.policy import load_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend', choices=['native', 'warp'], default='native')
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--terrain-types', nargs='+', help='terrain types to test (default: the checkpoint\'s)')
    parser.add_argument('--output', default='outputs/evaluation/terrain_goals.json')
    a = parser.parse_args()
    device = 'cpu' if a.backend == 'native' else 'cuda:0'
    policy, saved = load_controller(a.checkpoint, device)
    if device == 'cpu': torch.set_num_threads(4)
    cfg = replace(EnvConfig(**saved['config']['env']), backend=a.backend, device=device)
    if a.terrain_types:
        cfg = replace(cfg, terrain_types=a.terrain_types, terrain_weights=None)
    result = TerrainEvaluator(cfg, trials=a.trials).run(policy)
    result.update(checkpoint=a.checkpoint, checkpoint_update=saved['update'], config=asdict(cfg))
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['overall']), flush=True)


if __name__ == '__main__':
    main()
