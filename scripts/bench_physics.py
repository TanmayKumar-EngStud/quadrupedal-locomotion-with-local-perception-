"""Measure MuJoCo Warp throughput for the training terrain world.

Example: python scripts/bench_physics.py --worlds 4096 --resolution .05
Prints control steps per second (one control step = sim substeps).
"""
import argparse, json, time, types
import torch
from quadruped_nav.rl.terrain_grid import TerrainGrid
from quadruped_nav.rl.robot import build_training_model, configure_physics
from quadruped_nav.rl.backend import WarpBackend

p = argparse.ArgumentParser()
p.add_argument('--worlds', type=int, default=4096); p.add_argument('--resolution', type=float, default=.05)
p.add_argument('--sim-dt', type=float, default=.004); p.add_argument('--iterations', type=int, default=4)
p.add_argument('--ls-iterations', type=int, default=8); p.add_argument('--nconmax', type=int, default=64)
p.add_argument('--njmax', type=int, default=256); p.add_argument('--steps', type=int, default=100)
p.add_argument('--contacts', default='feet_trunk', choices=['feet_trunk', 'legs'])
p.add_argument('--levels', type=int, default=10); p.add_argument('--tile', type=float, default=8.)
p.add_argument('--types', type=int, default=7)
a = p.parse_args()
from quadruped_nav.rl.terrain_grid import TERRAIN_TYPES
grid = TerrainGrid(TERRAIN_TYPES[:a.types], a.levels, a.tile, a.resolution)
model = build_training_model(grid)
cfg = types.SimpleNamespace(sim_dt=a.sim_dt, solver_iterations=a.iterations, solver_ls_iterations=a.ls_iterations, kp=35., joint_damping=.5, contact_set=a.contacts)
configure_physics(model, cfg)
backend = WarpBackend(model, a.worlds, 'cuda:0', nconmax=a.nconmax, njmax=a.njmax)
centres = grid.tile_centres('cuda:0').reshape(-1, 3)
q = torch.tensor(model.key_qpos[0], device='cuda:0').float().repeat(a.worlds, 1)
pick = torch.arange(a.worlds, device='cuda:0') % len(centres)
q[:, :3] = centres[pick] + torch.tensor([0, 0, .30], device='cuda:0')
backend.reset(torch.arange(a.worlds, device='cuda:0'), q)
substeps = round(.02 / a.sim_dt)
home = q[:, 7:].clone()
for warm in range(3):
    backend.step(home, substeps)
torch.cuda.synchronize(); t = time.time()
for i in range(a.steps):
    action = home + .3 * torch.sin(torch.arange(12, device='cuda:0') + i * .3)
    backend.step(action, substeps)
torch.cuda.synchronize(); seconds = time.time() - t
nacon = int(backend.tensor('nacon').max()); nefc = int(backend.tensor('nefc').max())
print(json.dumps(dict(worlds=a.worlds, resolution=a.resolution, contacts=a.contacts, field_cells=int(grid.heights.size), sim_dt=a.sim_dt, iterations=a.iterations,
                      control_steps_per_s=round(a.worlds * a.steps / seconds), contacts_total=nacon,
                      contacts_per_world=round(nacon / a.worlds, 1), max_nefc=nefc,
                      mem_mb=round(torch.cuda.max_memory_allocated() / 2**20))), flush=True)
