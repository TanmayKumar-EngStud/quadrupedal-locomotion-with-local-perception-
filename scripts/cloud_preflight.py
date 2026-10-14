"""Run on the cloud before training: CUDA, physics, capacity and parity checks."""
import json
import torch
from quadruped_nav.rl.env import GoalEnv
from quadruped_nav.rl.config import EnvConfig
if not torch.cuda.is_available():raise SystemExit('NVIDIA CUDA not available. Check instance, drivers and torch installation.')
print(torch.cuda.get_device_name(),torch.__version__)
native=GoalEnv(EnvConfig(num_envs=1,noise=0.))
gpu=GoalEnv(EnvConfig(num_envs=1,backend='warp',device='cuda:0',noise=0.))
initial=native.backend.tensor('qpos').clone()
gpu.backend.reset(torch.tensor([0],device='cuda:0'),initial.cuda())
for _ in range(25):
    native.backend.step(native.home[None,7:],10);gpu.backend.step(gpu.home[None,7:],10)
native.backend.check();gpu.backend.check()
error=float((native.backend.tensor('qpos')-gpu.backend.tensor('qpos').cpu()).abs().max())
print(json.dumps({'standing_qpos_max_difference':error}))
if not torch.isfinite(gpu.backend.tensor('qpos')).all() or error>.05:raise SystemExit('CPU/GPU standing parity exceeds 0.05. Investigate before training.')
print('Standing parity passed. Next: quad-train --config configs/preflight.json')
