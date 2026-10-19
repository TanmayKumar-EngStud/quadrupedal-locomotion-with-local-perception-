"""Render a trained controller reaching a chain of targets on one training tile.

    python scripts/render_rollout.py --checkpoint runs/kth-goal-v3-seed0/last.pt \
        --terrain stairs_up --level 5 --output outputs/rollouts/stairs_up_L5.gif

Uses native (CPU) MuJoCo, deterministic actions and a camera that follows the
robot. The red sphere is the current target. Headless Linux: MUJOCO_GL=egl.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import mujoco
import numpy as np
import torch
from PIL import Image
from quadruped_nav.rl.config import EnvConfig
from quadruped_nav.rl.env import GoalEnv
from quadruped_nav.rl.policy import load_controller


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--terrain', default='rough')
    p.add_argument('--level', type=int, default=0); p.add_argument('--seconds', type=float, default=12.)
    p.add_argument('--seed', type=int, default=3); p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=300); p.add_argument('--output', default='outputs/rollouts/rollout.gif')
    a = p.parse_args()
    torch.set_num_threads(2)
    policy, saved = load_controller(a.checkpoint)
    cfg = replace(EnvConfig(**saved['config']['env']), num_envs=1, backend='native', device='cpu', noise=0.,
                  curriculum=False, seed=a.seed, episode_seconds=a.seconds + 1, max_speed=[.8, .8],
                  terrain_types=[a.terrain], terrain_weights=None)
    env = GoalEnv(cfg)
    env.column[0] = 0; env.level[0] = a.level
    env.reset(torch.tensor([0]))
    renderer = mujoco.Renderer(env.model, a.height, a.width)
    camera = mujoco.MjvCamera(); camera.distance, camera.azimuth, camera.elevation = 2.8, 135., -25.
    frames, reached = [], 0
    obs, _ = env.observations(noise=False)
    for step in range(int(a.seconds / cfg.control_dt)):
        with torch.no_grad(): action = policy.act(obs)
        obs, _, _, done, info = env.step(action, auto_reset=False, chain=True)
        reached += int(info['arrived'][0])
        if bool(done[0]): print('episode ended (fall)' if bool(info['fall'][0]) else 'episode ended'); break
        if step % 3: continue          # about 17 frames per second
        data = env.backend.data[0]
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera)
        scene = renderer.scene
        if scene.ngeom < scene.maxgeom:   # target marker
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.06, 0, 0]),
                                env.target[0].numpy().astype(np.float64) + [0, 0, .06], np.eye(3).ravel(),
                                np.array([.9, .15, .1, 1.], dtype=np.float32))
            scene.ngeom += 1
        frames.append(Image.fromarray(renderer.render()))
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    print(f'{out}: {len(frames)} frames, targets reached: {reached}')


if __name__ == '__main__':
    main()
