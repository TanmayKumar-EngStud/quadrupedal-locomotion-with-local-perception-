"""Run on the GPU before training: CUDA, physics and CPU/GPU parity on the terrain grid."""
import json
import torch
from quadruped_nav.rl.config import EnvConfig
from quadruped_nav.rl.env import GoalEnv

if not torch.cuda.is_available(): raise SystemExit('NVIDIA CUDA not available. Check instance, drivers and torch installation.')
print(torch.cuda.get_device_name(), torch.__version__)
settings = dict(num_envs=4, noise=0., seed=7)
native = GoalEnv(EnvConfig(**settings))
gpu = GoalEnv(EnvConfig(**settings, backend='warp', device='cuda:0'), grid=native.grid)
start = native.backend.tensor('qpos').clone()
gpu.backend.reset(torch.arange(4, device='cuda:0'), start.cuda())
largest = 0.
for t in range(100):
    action = .3 * torch.sin(torch.arange(4)[:, None] * .7 + torch.arange(12)[None] * .4 + t * .12)
    for env in (native, gpu):
        env.backend.step(env.home_joints[None] + env.cfg.action_scale * action.to(env.device), env.substeps)
    largest = max(largest, float((native.backend.tensor('qpos') - gpu.backend.tensor('qpos').cpu()).abs().max()))
print(json.dumps(dict(moving_joint_max_qpos_difference=largest, seconds=2., worlds=4, gpu_overflow=gpu.backend.overflow())))
# Heightfield contacts differ slightly between CPU MuJoCo and MuJoCo Warp (Warp
# gives box-heightfield pairs one contact), so trajectories drift apart by a few
# centimetres over 1-2 s of shuffling; more solver iterations do not change it.
if not torch.isfinite(gpu.backend.tensor('qpos')).all() or largest > .15:
    raise SystemExit('CPU/GPU parity exceeds 0.15 m. Investigate before training.')
print('Parity passed. Next: quad-train --config configs/preflight.json')
