"""Fixed goal-reaching test: the same robots, tiles and targets at every evaluation.

Every (terrain type, difficulty level) pair gets several robots. Each robot
starts at its tile centre and must reach one target 2 m away (direction fixed
by the seed, never inside a hole) within 10 s, holding position afterwards.
Actions are deterministic (the policy mean). Independent of training rewards.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import math
import torch
from .env import GoalEnv

EVAL_LEVELS = (0, 3, 6, 9)


class TerrainEvaluator:
    """Builds the evaluation robots once; run(policy) repeats the identical test."""

    def __init__(self, config, grid=None, levels=EVAL_LEVELS, trials=4, distance=2., seconds=10., speed=.8,
                 seed=12345, backend=None, device=None):
        self.levels = [level for level in levels if level < config.terrain_levels]
        self.types = list(config.terrain_types)
        self.distance, self.seconds, self.speed, self.seed = distance, seconds, speed, seed
        self.backend, self.device = backend or config.backend, device or config.device
        self.count = len(self.types) * len(self.levels) * trials
        self.trials = trials
        self.cfg = replace(config, num_envs=self.count, backend=self.backend, device=self.device, seed=seed,
                           curriculum=False, noise=0., target_distance=[distance, distance], max_speed=[speed, speed],
                           episode_seconds=seconds + 1.)
        self.env = GoalEnv(self.cfg, grid=grid)

    @torch.no_grad()
    def run(self, policy):
        env, count, device = self.env, self.count, self.device
        env.generator.manual_seed(self.seed)        # same starts and targets every time
        env.column = torch.arange(len(self.types), device=device).repeat_interleave(len(self.levels) * self.trials)
        env.level = torch.tensor(self.levels, device=device).repeat_interleave(self.trials).repeat(len(self.types))
        env.reset(torch.arange(count, device=device))
        env.final[:] = True
        agent = copy.deepcopy(policy).to(device).eval()
        actor, _ = env.observations(noise=False)
        reached = torch.zeros(count, dtype=torch.bool, device=device); fallen = reached.clone()
        time_to_goal = torch.full((count,), math.nan, device=device)
        min_distance = env.start_distance.clone()
        active = torch.ones(count, dtype=torch.bool, device=device)
        for _ in range(math.ceil(self.seconds / self.cfg.control_dt)):
            action = agent.act(actor) * active[:, None]
            actor, _, _, _, info = env.step(action, auto_reset=False, chain=False)
            newly = info['arrived'] & active & ~reached
            time_to_goal = torch.where(newly, info['time_to_goal'], time_to_goal)
            reached |= newly
            fallen |= (info['fall'] | info['unstable']) & active
            min_distance = torch.minimum(min_distance, torch.where(active, info['distance'], min_distance))
            active &= ~fallen
            if not active.any(): break
        final_distance = torch.where(fallen, min_distance, info['distance'])
        progress = (env.start_distance - final_distance) / env.start_distance

        def summary(mask):
            n = int(mask.sum()); ok = reached & mask
            return dict(trials=n, success_rate=float(ok.sum()) / max(n, 1), fall_rate=float((fallen & mask).sum()) / max(n, 1),
                        mean_time_to_goal_s=float(time_to_goal[ok].mean()) if ok.any() else None,
                        mean_progress_fraction=float(progress[mask].mean()), mean_final_distance_m=float(final_distance[mask].mean()))

        return dict(overall=summary(torch.ones_like(reached)),
                    by_type={kind: summary(env.column == i) for i, kind in enumerate(self.types)},
                    by_level={str(level): summary(env.level == level) for level in self.levels},
                    success_by_type_level={f'{kind}/{level}': summary((env.column == i) & (env.level == level))['success_rate']
                                           for i, kind in enumerate(self.types) for level in self.levels},
                    distance_m=self.distance, seconds=self.seconds, speed=self.speed, backend=self.backend,
                    levels=self.levels, trials_per_cell=self.trials)
