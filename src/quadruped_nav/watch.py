"""Watch a trained controller walk to targets on one training terrain (CPU MuJoCo).

    python watch.py --terrain stairs_up --level 5
    python watch.py --checkpoint artifacts/controller_v1/controller.pt --terrain stepping_stones --level 0
    python watch.py --checkpoint artifacts/controller_v2/controller.pt --terrain piles --level 3
    python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain piles --level 3 --terrain-noise 0

--terrain-noise sets the irregularities (0 = regular terrain, 0.5 = v3's training
terrain, 1.0 = v4's); --pile-pits deep or ramp sets the pile pits. Both default to
what the controller was trained on.

The robot starts on the chosen tile and walks to a chain of random targets
(the distances it was trained on), shown as a red sphere. After a fall it restarts on the same tile.
Close the window to stop. --headless runs without a window and prints
the same events, for a quick check on any machine.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import time
import mujoco
import numpy as np
import torch
from .rl.config import EnvConfig
from .rl.env import GoalEnv
from .rl.policy import load_controller
from .rl.terrain_grid import ALL_TERRAIN_TYPES


def build(checkpoint, terrain, level, speed, seed, terrain_noise=None, pile_pits=None):
    policy, saved = load_controller(checkpoint)
    # Only the chosen terrain type is built (10 tiles instead of 70): drawing every
    # 8 m heightfield, about 51k triangles each, is what made the viewer lag.
    cfg = replace(EnvConfig(**saved['config']['env']), num_envs=1, backend='native', device='cpu', noise=0.,
                  curriculum=False, seed=seed, episode_seconds=1e9, max_speed=[speed, speed], terrain_types=[terrain],
                  terrain_weights=None, terrain_variants=1)
    if terrain_noise is not None: cfg = replace(cfg, terrain_noise=terrain_noise)
    if pile_pits is not None: cfg = replace(cfg, pile_pit_ramp=pile_pits == 'ramp')
    env = GoalEnv(cfg)
    env.column[0], env.level[0] = 0, level
    env.reset(torch.tensor([0]))
    return policy, env, saved


def hide_other_tiles(env, level):
    """Viewer only: put the other levels' heightfields in geom group 5, which the viewer hides. Physics is unchanged."""
    grid = env.grid
    centre = np.array([grid.origin_x[level], grid.origin_y[0]]) + grid.tile_size / 2
    for g in range(env.model.ngeom):
        if env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD and np.abs(env.model.geom_pos[g, :2] - centre).max() > 1:
            env.model.geom_group[g] = 5


def draw_target(scene, env):
    scene.ngeom = 0
    mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.07, 0, 0]),
                        env.target[0].numpy().astype(np.float64) + [0, 0, .07], np.eye(3).ravel(),
                        np.array([.9, .15, .1, 1.], dtype=np.float32))
    scene.ngeom = 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='artifacts/controller_v1/controller.pt')
    p.add_argument('--terrain', choices=ALL_TERRAIN_TYPES, default='rough')
    p.add_argument('--level', type=int, default=0, choices=range(10), metavar='0-9', help='difficulty, 0 easiest')
    p.add_argument('--speed', type=float, default=.8, help='commanded walking speed in m/s')
    p.add_argument('--seed', type=int, default=0, help='changes the start pose and the targets')
    p.add_argument('--seconds', type=float, default=0., help='stop after this long (0 = until the window is closed)')
    p.add_argument('--terrain-noise', type=float, help='irregularities: 0 regular, 0.5 as v3, 1.0 as v4 (default: as trained)')
    p.add_argument('--pile-pits', choices=['deep', 'ramp'], help='pile pits: 0.5 m everywhere, or deepening with level (default: as trained)')
    p.add_argument('--headless', action='store_true', help='no window; print events only')
    a = p.parse_args()
    torch.set_num_threads(2)
    policy, env, saved = build(a.checkpoint, a.terrain, a.level, a.speed, a.seed, a.terrain_noise, a.pile_pits)
    print(f'Controller update {saved["update"]}, {a.terrain} level {a.level}, target speed {a.speed} m/s, '
          f'terrain noise {env.cfg.terrain_noise:g}, pile pits {"ramp" if env.cfg.pile_pit_ramp else "deep"}', flush=True)
    dt, limit = env.cfg.control_dt, a.seconds or float('inf')
    handle = None
    if not a.headless:
        import mujoco.viewer
        hide_other_tiles(env, a.level)
        handle = mujoco.viewer.launch_passive(env.model, env.backend.data[0])
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        handle.cam.trackbodyid = env.model.body('trunk').id
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 3., 135., -25.
    obs, _ = env.observations(noise=False)
    elapsed = reached = falls = 0
    try:
        while elapsed < limit and (handle is None or handle.is_running()):
            started = time.monotonic()
            with torch.no_grad(): action = policy.act(obs)
            obs, _, _, done, info = env.step(action, auto_reset=False, chain=True)
            elapsed += dt
            if bool(info['arrived'][0]):
                reached += 1
                print(f'{elapsed:6.1f} s  target {reached} reached in {float(info["time_to_goal"][0]):.1f} s', flush=True)
            if bool(done[0]):
                falls += 1
                print(f'{elapsed:6.1f} s  fell; restarting on the same tile', flush=True)
                env.reset(torch.tensor([0])); obs, _ = env.observations(noise=False)
            if handle is not None:
                with handle.lock(): draw_target(handle.user_scn, env)
                handle.sync()
                time.sleep(max(0., dt - (time.monotonic() - started)))
    finally:
        if handle is not None: handle.close()
    print(f'Done after {elapsed:.0f} s: {reached} targets reached, {falls} falls', flush=True)


if __name__ == '__main__':
    main()
