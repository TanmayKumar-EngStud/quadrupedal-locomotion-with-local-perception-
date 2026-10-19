"""Train the goal-reaching Go1 controller with PPO on the multi-terrain grid.

Run a CPU smoke test before a long GPU job:
    quad-train --config configs/smoke.json
    quad-train --config configs/kth-goal-v3.json

Every update appends one line to <output>/metrics.jsonl; every
evaluation_every updates the fixed terrain test appends to evaluation.jsonl.
scripts/plot_training.py turns both into figures.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import platform
from pathlib import Path
import subprocess
import time
import mujoco
import torch
from .rl.config import TrainConfig, OBSERVATION_SPEC
from .rl.env import GoalEnv
from .rl.evaluation import TerrainEvaluator
from .rl.policy import ActorCritic, export_bundle, load_controller
from .rl.ppo import PPO, Rollout


class Statistics:
    """Sums of per-step quantities over one rollout (kept on the device), then a metrics record."""

    def __init__(self, types, device):
        self.types, self.device = types, device
        self.sums = defaultdict(lambda: torch.zeros((), device=device))
        self.type_arrivals = torch.zeros(len(types), device=device); self.type_failures = torch.zeros(len(types), device=device)

    def add(self, reward, info):
        s = self.sums
        arrived, fell, missed, moving = info['arrived'].float(), info['fall'].float(), info['missed'].float(), info['moving'].float()
        ended = (info['terminated'] | info['truncated']).float()
        s['steps'] += reward.numel()
        s['reward'] += reward.sum()
        s['arrivals'] += arrived.sum(); s['falls'] += fell.sum(); s['missed'] += missed.sum()
        s['episodes'] += ended.sum(); s['episode_seconds'] += (info['episode_seconds'] * ended).sum()
        s['unstable'] += info['unstable'].float().sum()
        s['time_to_goal'] += (info['time_to_goal'] * arrived).sum()
        s['path_efficiency'] += (info['path_efficiency'] * arrived).sum()
        s['moving_steps'] += moving.sum()
        s['toward_velocity'] += info['toward_velocity'].sum()
        s['command_speed'] += (info['command_speed'] * moving).sum()
        s['distance'] += info['distance'].sum()
        for name, value in info['reward_terms'].items():
            s['term/' + name] += value.sum()
        if 'column' in info:
            self.type_arrivals.index_add_(0, info['column'], arrived)
            self.type_failures.index_add_(0, info['column'], torch.maximum(fell, missed))

    def record(self):
        s = {k: float(v) for k, v in self.sums.items()}
        per = lambda x, n: x / max(n, 1)
        result = dict(
            mean_reward=per(s['reward'], s['steps']),
            arrivals_per_10k=1e4 * per(s['arrivals'], s['steps']), falls_per_10k=1e4 * per(s['falls'], s['steps']),
            missed_per_10k=1e4 * per(s['missed'], s['steps']),
            target_success_rate=per(s['arrivals'], s['arrivals'] + s['missed'] + s['falls']),
            episodes_finished=s['episodes'], mean_episode_seconds=per(s['episode_seconds'], s['episodes']),
            mean_time_to_goal_s=per(s['time_to_goal'], s['arrivals']) if s['arrivals'] else None,
            mean_path_efficiency=per(s['path_efficiency'], s['arrivals']) if s['arrivals'] else None,
            mean_velocity_toward_target=per(s['toward_velocity'], s['moving_steps']),
            mean_command_speed=per(s['command_speed'], s['moving_steps']),
            mean_target_distance_m=per(s['distance'], s['steps']), unstable_per_10k=1e4 * per(s['unstable'], s['steps']),
            reward_terms={k[5:]: per(v, s['steps']) for k, v in s.items() if k.startswith('term/')})
        arrivals, attempts = self.type_arrivals.cpu(), (self.type_arrivals + self.type_failures).cpu()
        result['success_by_type'] = {t: (float(arrivals[i] / attempts[i]) if attempts[i] else None)
                                     for i, t in enumerate(self.types)}
        return result


def save_checkpoint(path, model, ppo, cfg, update, steps, revision):
    payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   optimizer=ppo.optimizer.state_dict(), learning_rate=ppo.learning_rate, config=cfg.dict(),
                   observation_spec=OBSERVATION_SPEC, update=update, steps=steps, source_revision=revision,
                   versions=dict(python=platform.python_version(), torch=str(torch.__version__), mujoco=mujoco.__version__))
    temporary = path.with_suffix('.tmp'); torch.save(payload, temporary); temporary.replace(path)


def train(cfg, resume=None):
    cfg.env.validate(); torch.manual_seed(cfg.env.seed)
    device = cfg.env.device
    if device == 'cpu': torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(cfg.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not resume:
        raise FileExistsError(f'{out} already has a checkpoint; use --resume or a new --output')
    (out / 'config.json').write_text(json.dumps(cfg.dict(), indent=2) + '\n')
    try: revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception: revision = 'unknown'

    model = ActorCritic(cfg.hidden, cfg.init_std).to(device)
    ppo = PPO(model, cfg)
    first, steps = 0, 0
    if resume:
        loaded, checkpoint = load_controller(resume, device)
        model.load_state_dict(loaded.state_dict())
        ppo.optimizer.load_state_dict(checkpoint['optimizer']); ppo.learning_rate = checkpoint['learning_rate']
        first, steps = checkpoint['update'], checkpoint['steps']

    env = GoalEnv(cfg.env)
    evaluator = TerrainEvaluator(cfg.env, grid=env.grid) if cfg.evaluation_every else None
    obs, critic_obs = env.observations()
    if not resume:
        model.actor_norm.update(obs); model.critic_norm.update(critic_obs)
    started = time.monotonic(); start_steps = steps
    for update in range(first, cfg.updates):
        stats = Statistics(cfg.env.terrain_types, device)
        buffers = defaultdict(list)
        collect_start = time.monotonic()
        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                dist = model.distribution(obs)
                action = dist.sample()
                buffers['actor_obs'].append(obs); buffers['critic_obs'].append(critic_obs)
                buffers['actions'].append(action); buffers['log_probs'].append(dist.log_prob(action).sum(-1))
                buffers['values'].append(model.value(critic_obs))
                obs, critic_obs, reward, done, info = env.step(action)
                # Episodes cut by the time limit did not fail: bootstrap their value.
                if info['truncated'].any():
                    reward = reward + cfg.gamma * model.value(info['terminal_critic']) * info['truncated']
                buffers['rewards'].append(reward); buffers['dones'].append(done)
            stats.add(reward, info)
        with torch.no_grad(): last_values = model.value(critic_obs)
        rollout = Rollout(**{k: torch.stack(v) for k, v in buffers.items()}, last_values=last_values)
        collect_seconds = time.monotonic() - collect_start
        learn_start = time.monotonic()
        ppo_stats = ppo.update(rollout)
        # Normalizer statistics stay frozen during collection and the update, so the
        # stored log-probabilities and values match what the update recomputes.
        with torch.no_grad():
            model.actor_norm.update(rollout.actor_obs); model.critic_norm.update(rollout.critic_obs)
        learn_seconds = time.monotonic() - learn_start
        steps += cfg.rollout_steps * env.n

        record = dict(update=update + 1, steps=steps, wall_seconds=time.monotonic() - started,
                      samples_per_second=(steps - start_steps) / max(time.monotonic() - started, 1e-6),
                      collect_seconds=collect_seconds, learn_seconds=learn_seconds, **stats.record(), ppo=ppo_stats)
        if env.grid is not None:
            levels = env.level.float().cpu(); columns = env.column.cpu()
            record['mean_level'] = float(levels.mean())
            record['level_by_type'] = {t: float(levels[columns == i].mean()) for i, t in enumerate(cfg.env.terrain_types)}
        if hasattr(env.backend, 'overflow'):
            record['contact_overflow'] = env.backend.overflow()
        with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
                          if k in ('update', 'steps', 'mean_reward', 'samples_per_second', 'arrivals_per_10k', 'falls_per_10k',
                                   'target_success_rate', 'mean_level', 'mean_velocity_toward_target')}), flush=True)

        if evaluator and ((update + 1) % cfg.evaluation_every == 0 or update + 1 == cfg.updates or update == first):
            evaluation = evaluator.run(model)
            evaluation.update(update=update + 1, steps=steps)
            with (out / 'evaluation.jsonl').open('a') as f: f.write(json.dumps(evaluation) + '\n')
            print(json.dumps({'evaluation_update': update + 1, **evaluation['overall']}), flush=True)
        if (update + 1) % cfg.checkpoint_every == 0 or update + 1 == cfg.updates:
            save_checkpoint(out / 'last.pt', model, ppo, cfg, update + 1, steps, revision)
            if (update + 1) % (cfg.checkpoint_every * 10) == 0 or update + 1 == cfg.updates:
                save_checkpoint(out / f'update_{update + 1:06d}.pt', model, ppo, cfg, update + 1, steps, revision)
    return out / 'last.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/smoke.json'); p.add_argument('--output'); p.add_argument('--resume')
    p.add_argument('--updates', type=int, help='override the number of updates')
    p.add_argument('--export', help='export this checkpoint without training'); p.add_argument('--bundle', default='artifacts/controller')
    a = p.parse_args()
    if a.export: print(export_bundle(a.export, a.bundle)); return
    cfg = TrainConfig.load(a.config)
    if a.output: cfg.output = a.output
    if a.updates: cfg.updates = a.updates
    checkpoint = train(cfg, a.resume)
    print(f'Checkpoint: {checkpoint}\nExport with: quad-train --export {checkpoint} --bundle {a.bundle}')


if __name__ == '__main__': main()
