"""Render two controllers side by side (same seed, same targets) as an MP4 clip.

    # watch.py view: a chain of random targets on one training tile
    python scripts/render_compare.py tile --terrain piles --level 2 --output outputs/comparison/clips/piles_L2.mp4
    # play.py view: the planner crossing a pile scene
    python scripts/render_compare.py scene --family open --row 0 --ground boxes --output outputs/comparison/clips/open_row0.mp4

Native (CPU) MuJoCo, deterministic actions, camera following each robot. The red
sphere is the current target. Needs ffmpeg on PATH.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from quadruped_nav.evaluate import evaluate
from quadruped_nav.rl.config import EnvConfig
from quadruped_nav.rl.env import GoalEnv
from quadruped_nav.rl.policy import load_controller
from quadruped_nav.terrain import generate_terrain

FPS = 25          # one frame every second control step (50 Hz control)


class Recorder:
    """Renders every second step of one robot, with a target marker and a caption."""

    def __init__(self, width, height, label):
        self.width, self.height, self.label = width, height, label
        self.frames, self.renderer, self.step, self.status = [], None, 0, ''
        self.camera = mujoco.MjvCamera(); self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.6, 140., -28.

    def __call__(self, env, force=False):
        self.step += 1
        if self.step % 2 and not force: return
        if self.renderer is None: self.renderer = mujoco.Renderer(env.model, self.height, self.width)
        data = env.backend.data[0]
        self.camera.lookat[:] = data.qpos[:3]
        self.renderer.update_scene(data, self.camera)
        scene = self.renderer.scene
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.05, 0, 0]),
                                env.target[0].numpy().astype(np.float64) + [0, 0, .05], np.eye(3).ravel(),
                                np.array([.9, .15, .1, 1.], dtype=np.float32))
            scene.ngeom += 1
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, self.width, 22], fill=(16, 18, 22))
        draw.text((8, 5), f'{self.label}   t = {self.step * .02:4.1f} s   {self.status}', fill=(240, 240, 236))
        self.frames.append(image)


def tile_rollout(checkpoint, recorder, terrain, level, seconds, seed, noise=None):
    policy, saved = load_controller(checkpoint)
    cfg = replace(EnvConfig(**saved['config']['env']), num_envs=1, backend='native', device='cpu', noise=0., curriculum=False,
                  seed=seed, episode_seconds=seconds + 1, max_speed=[.75, .75], terrain_types=[terrain], terrain_weights=None,
                  terrain_variants=1, pile_pit_ramp=False, terrain_noise=noise or 0.)   # same terrain for both controllers
    env = GoalEnv(cfg)
    env.column[0], env.level[0] = 0, level
    env.reset(torch.tensor([0]))
    obs, _ = env.observations(noise=False)
    reached = 0
    for _ in range(int(seconds / cfg.control_dt)):
        with torch.no_grad(): action = policy.act(obs)
        obs, _, _, done, info = env.step(action, auto_reset=False, chain=True)
        reached += int(info['arrived'][0])
        recorder.status = f'targets reached: {reached}'
        if bool(done[0]):
            recorder.status += '   FELL'; recorder(env, force=True); break
        recorder(env)
    return dict(reached=reached, fell=bool(done[0]))


def scene_rollout(checkpoint, recorder, family, row, ground, planner, seconds, seed):
    scene, last = generate_terrain(family, row, seed), []
    def on_step(env):
        recorder.status = f'{family} row {row}, {ground}'
        recorder(env); last[:] = [env]
    result = evaluate(checkpoint, scene, planner, seconds, seed, ground=ground, on_step=on_step)
    recorder.status = f'{family} row {row}, {ground}:  {result["status"].upper()}'
    recorder(last[0], force=True)
    return dict(status=result['status'], time_s=result['time_s'])


def write_mp4(frames, path, hold_seconds=1.):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        for frame in frames + [frames[-1]] * int(hold_seconds * FPS):
            ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.close()
    if ffmpeg.returncode: raise RuntimeError('ffmpeg failed')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['tile', 'scene'])
    p.add_argument('--checkpoints', nargs=2, default=['artifacts/controller_v1/controller.pt', 'artifacts/controller_v2/controller.pt'])
    p.add_argument('--labels', nargs=2, default=['v1', 'v2'])
    p.add_argument('--terrain', default='piles'); p.add_argument('--level', type=int, default=2)
    p.add_argument('--terrain-noise', type=float, help='tile mode: irregularity scale (0.5 = v3, 1.0 = v4)')
    p.add_argument('--family', default='open'); p.add_argument('--row', type=int, default=0)
    p.add_argument('--ground', choices=['boxes', 'heightfield'], default='boxes'); p.add_argument('--planner', default='oracle')
    p.add_argument('--seconds', type=float, default=12.); p.add_argument('--seed', type=int, default=3)
    p.add_argument('--width', type=int, default=480); p.add_argument('--height', type=int, default=320)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    recorders = []
    for checkpoint, label in zip(a.checkpoints, a.labels):
        recorder = Recorder(a.width, a.height, label)
        if a.mode == 'tile': outcome = tile_rollout(checkpoint, recorder, a.terrain, a.level, a.seconds, a.seed, a.terrain_noise)
        else: outcome = scene_rollout(checkpoint, recorder, a.family, a.row, a.ground, a.planner, a.seconds, a.seed)
        print(label, outcome, flush=True)
        recorders.append(recorder)
    length = max(len(r.frames) for r in recorders)
    frames = []
    for i in range(length):
        canvas = Image.new('RGB', (a.width * len(recorders) + 4 * (len(recorders) - 1), a.height), (16, 18, 22))
        for j, r in enumerate(recorders):
            canvas.paste(r.frames[min(i, len(r.frames) - 1)], (j * (a.width + 4), 0))
        frames.append(canvas)
    write_mp4(frames, Path(a.output))
    print(f'{a.output}: {len(frames)} frames ({len(frames) / FPS:.1f} s)')


if __name__ == '__main__':
    main()
