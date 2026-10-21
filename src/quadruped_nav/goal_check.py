"""Run the fixed terrain goal-reaching test on a saved checkpoint.

    python -m quadruped_nav.goal_check --checkpoint runs/goal-v3/last.pt [--backend native]

--terrain-types tests other terrain types than the checkpoint was trained on, e.g. the
seven v1 types for every controller (the same robots, tiles and targets) or piles only.
--terrain-from takes all terrain settings from a run config, so different controllers
are tested on the same terrain; --terrain-noise and --seed then choose the noise level
and the tiles (seed 12345, the default, is never used for training or validation):

    python -m quadruped_nav.goal_check --checkpoint artifacts/controller_v2/controller.pt \
        --terrain-from configs/kth-controller-v3.json --terrain-noise 1.0

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

TERRAIN_FIELDS = ('terrain_types', 'terrain_levels', 'tile_size', 'terrain_resolution', 'fine_terrain_resolution',
                  'terrain_noise', 'terrain_noise_ramp', 'terrain_variants', 'pile_pit_ramp')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend', choices=['native', 'warp'], default='native')
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--terrain-types', nargs='+', help='terrain types to test (default: the checkpoint\'s)')
    parser.add_argument('--terrain-from', help='take the terrain settings from this run config')
    parser.add_argument('--terrain-noise', type=float, help='irregularity scale of the test tiles, the same on every level (0 = regular)')
    parser.add_argument('--seed', type=int, default=12345, help='seed of the test tiles, robots and targets')
    parser.add_argument('--output', default='outputs/evaluation/terrain_goals.json')
    a = parser.parse_args()
    device = 'cpu' if a.backend == 'native' else 'cuda:0'
    policy, saved = load_controller(a.checkpoint, device)
    if device == 'cpu': torch.set_num_threads(4)
    cfg = replace(EnvConfig(**saved['config']['env']), backend=a.backend, device=device)
    if a.terrain_from:
        env = json.loads(Path(a.terrain_from).read_text())['env']
        cfg = replace(cfg, terrain_weights=None, **{k: env[k] for k in TERRAIN_FIELDS if k in env})
    if a.terrain_types:
        cfg = replace(cfg, terrain_types=a.terrain_types, terrain_weights=None)
    if a.terrain_noise is not None:
        cfg = replace(cfg, terrain_noise=a.terrain_noise, terrain_noise_ramp=0)
    result = TerrainEvaluator(cfg, trials=a.trials, seed=a.seed).run(policy)
    result.update(checkpoint=a.checkpoint, checkpoint_update=saved['update'], config=asdict(cfg))
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['overall']), flush=True)


if __name__ == '__main__':
    main()
