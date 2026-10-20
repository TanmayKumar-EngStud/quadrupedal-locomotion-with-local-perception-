"""Goal-reaching Go1 environment shared by native MuJoCo and batched MuJoCo Warp.

Two kinds of world:
  * training (scene=None): a heightfield grid of terrain tiles. Every robot is
    placed on a tile, gets random targets 1-3 m away, and a per-robot
    curriculum moves it to harder or easier tiles;
  * evaluation (scene=project scene JSON): the planner's stepping-stone
    layouts, built from boxes (ground='boxes') or as one heightfield rasterized
    by HeightGrid.from_scene (ground='heightfield', the training terrain's
    representation). Targets are set by the planner (set_target).

The policy is told where the target is (body frame) and the velocity a simple
guidance law would command to get there: walk toward the target, turning to
face it, and slow down near it. Rewards follow that command, pay for progress
toward the target, and pay a bonus on arrival.
"""
from __future__ import annotations
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from .backend import NativeBackend, WarpBackend
from .config import EnvConfig, OBSERVATION_SPEC, BODY_SCAN, FOOT_SCAN
from .robot import FEET, CALVES, build_training_model, configure_physics
from .terrain_grid import HeightGrid, TerrainGrid
from . import rewards as reward_terms
from ..mjcf import build_mjcf

FOOT_RADIUS = .023
STANDING_HEIGHT = .30         # nominal base height above terrain, used to scale the body scan
FALL_CLEARANCE = .12          # base closer than this to the highest terrain within 10 cm = collapsed
FALL_TILT = -.5               # projected gravity z above this (tilt > 60 degrees) = fallen
# Physically impossible speeds (m/s, rad/s, rad/s): the simulation of that robot
# failed, so its episode is ended and the values are clamped before use.
MAX_LINEAR_SPEED, MAX_BODY_RATE, MAX_JOINT_SPEED = 20., 60., 100.


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def inverse_rotate(q, v):
    """Rotate world vectors v into the body frame of quaternions q (w, x, y, z)."""
    u = -q[:, 1:]; t = 2 * torch.cross(u, v, dim=-1)
    return v + q[:, :1] * t + torch.cross(u, t, dim=-1)


def yaw_frame_points(xy, yaw, offsets):
    """World xy of offsets (k, 2) given in the yaw-aligned frame at positions xy[..., 2]."""
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    x = xy[..., None, 0] + c * offsets[:, 0] - s * offsets[:, 1]
    y = xy[..., None, 1] + s * offsets[:, 0] + c * offsets[:, 1]
    return torch.stack([x, y], -1)


def robots_per_type(n, weights):
    """Split n robots among terrain types in proportion to weights (largest remainder)."""
    share = n * np.asarray(weights, dtype=float) / sum(weights)
    counts = np.floor(share).astype(int)
    counts[np.argsort(counts - share, kind='stable')[:n - counts.sum()]] += 1
    return counts


class GoalEnv:
    def __init__(self, cfg: EnvConfig, scene=None, grid=None, ground='boxes'):
        cfg.validate()
        if ground not in ('boxes', 'heightfield'): raise ValueError('ground must be boxes or heightfield')
        self.cfg, self.n, self.device = cfg, cfg.num_envs, cfg.device
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.scene = scene
        if scene is None:
            self.grid = grid or TerrainGrid(cfg.terrain_types, cfg.terrain_levels, cfg.tile_size,
                                            cfg.terrain_resolution, seed=cfg.seed,
                                            fine_resolution=cfg.fine_terrain_resolution, noise=cfg.terrain_noise,
                                            variants=cfg.terrain_variants, pile_pit_ramp=cfg.pile_pit_ramp)
            self.model = build_training_model(self.grid)
            self.terrain = self.grid.height_grid(self.device)
        else:
            self.grid = None
            self.terrain = HeightGrid.from_scene(scene, self.device, resolution=cfg.fine_terrain_resolution)
            # The heightfield is the height map the policy senses, so physics and sensing agree exactly.
            field = None if ground == 'boxes' else (self.terrain.values.cpu().numpy(), self.terrain.x_min,
                                                    self.terrain.y_min, self.terrain.resolution)
            with tempfile.TemporaryDirectory() as temp:
                self.model = mujoco.MjModel.from_xml_path(str(build_mjcf(scene, Path(temp) / 'scene.xml', heightfield=field)))
            self.supports = scene['stones'] + scene['platforms']
            self.piles = torch.tensor([[p['x'], p['y'], p['z_top']] for p in self.supports], device=self.device)
        configure_physics(self.model, cfg)
        backend = NativeBackend if cfg.backend == 'native' else WarpBackend
        self.backend = backend(self.model, self.n, self.device, nconmax=cfg.nconmax, njmax=cfg.njmax)
        self.substeps = round(cfg.control_dt / cfg.sim_dt)

        # Robot constants.
        self.home = torch.tensor(self.model.key_qpos[0], dtype=torch.float32, device=self.device)
        self.home_joints = self.home[7:]
        self.joint_low = torch.tensor(self.model.actuator_ctrlrange[:, 0], dtype=torch.float32, device=self.device)
        self.joint_high = torch.tensor(self.model.actuator_ctrlrange[:, 1], dtype=torch.float32, device=self.device)
        middle, half = (self.joint_low + self.joint_high) / 2, (self.joint_high - self.joint_low) / 2
        self.soft_low, self.soft_high = middle - .95 * half, middle + .95 * half
        self.foot_sites = [self.model.site(name).id for name in FEET]
        self.knee_bodies = [self.model.body(name).id for name in CALVES]
        self.body_scan = torch.tensor(BODY_SCAN, dtype=torch.float32, device=self.device)
        self.foot_scan = torch.tensor(FOOT_SCAN, dtype=torch.float32, device=self.device)
        self.clearance_scan = torch.tensor([[0, 0], [.1, 0], [-.1, 0], [0, .08], [0, -.08]], dtype=torch.float32, device=self.device)
        self.gap_scan = torch.tensor([[0, 0], [.04, 0], [-.04, 0], [0, .04], [0, -.04]], dtype=torch.float32, device=self.device)
        self.speed_limits = torch.tensor([MAX_LINEAR_SPEED] * 3 + [MAX_BODY_RATE] * 3 + [MAX_JOINT_SPEED] * 12, device=self.device)

        # Per-robot task state.
        zeros = lambda *shape, dtype=torch.float32: torch.zeros((self.n, *shape), device=self.device, dtype=dtype)
        self.target = zeros(3)
        self.start_distance = zeros()
        self.max_speed = torch.full((self.n,), sum(cfg.max_speed) / 2, device=self.device)
        self.target_budget = zeros()
        self.target_elapsed = zeros()
        self.elapsed = zeros()
        self.reached = zeros(dtype=torch.bool)
        self.final = zeros(dtype=torch.bool)
        self.hold_left = zeros()
        self.path_length = zeros()
        self.previous_action = zeros(12)
        self.previous_feet = zeros(4, 3)
        self.previous_joint_vel = zeros(12)
        self.air_time = zeros(4)
        self.last_contact = zeros(4, dtype=torch.bool)
        self.episode_arrivals = zeros()
        self.episode_misses = zeros()
        # Curriculum: terrain column fixed per robot, difficulty level adapts.
        if self.grid is not None:
            self.centres = self.grid.tile_centres(self.device)
            if cfg.terrain_weights is None:
                self.column = torch.arange(self.n, device=self.device) % len(self.grid.types)
            else:
                counts = torch.as_tensor(robots_per_type(self.n, cfg.terrain_weights), device=self.device)
                self.column = torch.repeat_interleave(torch.arange(len(self.grid.types), device=self.device), counts)
            self.level = torch.randint(0, cfg.max_start_level + 1, (self.n,), device=self.device, generator=self.generator)
            self.variant = torch.zeros(self.n, dtype=torch.long, device=self.device)   # tile copy, drawn at reset
            self.hole_threshold = torch.tensor(self.grid.hole_threshold, device=self.device)   # [level, tile column]
            self.tile_x = torch.tensor(self.grid.origin_x, dtype=torch.float32, device=self.device)
            self.tile_y = torch.tensor(self.grid.origin_y, dtype=torch.float32, device=self.device)
        self.reset(torch.arange(self.n, device=self.device))

    # ------------------------------------------------------------------ utils
    def rand(self, *shape): return torch.rand(shape, device=self.device, generator=self.generator)

    def _tile(self, ids):
        """Grid column of each robot's tile: its terrain type and its tile variant."""
        return self.column[ids] * self.grid.variants + self.variant[ids]

    def uniform(self, bounds, *shape): return bounds[0] + (bounds[1] - bounds[0]) * self.rand(*shape)

    def _kinematics(self):
        """Robot state in the frames the observations and rewards need."""
        q, v = self.backend.tensor('qpos'), self.backend.tensor('qvel')
        finite = torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1)
        q, v = torch.nan_to_num(q), torch.nan_to_num(v)
        valid = finite & (v.abs() <= self.speed_limits).all(-1)
        v = torch.maximum(torch.minimum(v, self.speed_limits), -self.speed_limits)
        quat = q[:, 3:7]
        down = torch.tensor([0., 0., -1.], device=self.device).expand(self.n, 3)
        feet = torch.nan_to_num(self.backend.tensor('site_xpos')[:, self.foot_sites]).clone()
        knees = torch.nan_to_num(self.backend.tensor('xpos')[:, self.knee_bodies])
        return SimpleNamespace(
            valid=valid, pos=q[:, :3].clone(), quat=quat.clone(), yaw=yaw_of(quat),
            lin_vel_world=v[:, :3].clone(), lin_vel_body=inverse_rotate(quat, v[:, :3]), ang_vel_body=v[:, 3:6].clone(),
            gravity_body=inverse_rotate(quat, down), joint_pos=q[:, 7:].clone(), joint_vel=v[:, 6:].clone(),
            feet=feet, foot_height=feet[:, :, 2] - self.terrain.sample(feet[:, :, :2]) - FOOT_RADIUS,
            knee_height=knees[:, :, 2] - self.terrain.sample(knees[:, :, :2]))

    def _target_geometry(self, k):
        """Distance, world direction and body-frame offset of each robot's target."""
        delta = self.target[:, :2] - k.pos[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1e-3)[:, None]
        c, s = torch.cos(k.yaw), torch.sin(k.yaw)
        local = torch.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], -1)
        return distance, direction, local

    def _command(self, distance, local):
        """Guidance law: velocity (vx, vy, wz) in the yaw frame that leads to the target.

        The linear part points at the target with the same magnitude whatever the
        robot's heading. Scaling it down when the target is behind let robots face
        away from their targets and earn the tracking reward by standing still.
        The yaw rate turns the robot to face the target.
        """
        cfg = self.cfg
        heading_error = torch.atan2(local[:, 1], local[:, 0])
        speed = self.max_speed * (distance / cfg.slow_down_radius).clamp(0, 1)
        v = speed[:, None] * local / distance.clamp_min(1e-3)[:, None]
        wz = (cfg.yaw_gain * heading_error).clamp(-1.5, 1.5) * (distance > cfg.slow_down_radius)
        return torch.cat([v, wz[:, None]], -1) * (~self.reached)[:, None]

    # ------------------------------------------------------------ observations
    def observations(self, noise=True):
        return self._observe(self._kinematics(), noise)

    def _observe(self, k, noise=True):
        """Actor observation (optionally noisy) and critic observation (clean + privileged)."""
        clean = self._actor_observation(k, 0.)
        actor = self._actor_observation(k, self.cfg.noise) if noise and self.cfg.noise else clean
        privileged = torch.cat([
            (k.pos[:, 2] - self.terrain.sample(k.pos[:, :2]))[:, None], k.foot_height,
            (self.target_budget - self.target_elapsed)[:, None]], -1)
        return actor, torch.cat([clean, privileged], -1).clamp(-100, 100)

    def _actor_observation(self, k, noise_scale):
        jitter = lambda x, amount: x + (2 * self.rand(*x.shape) - 1) * amount * noise_scale if noise_scale else x
        distance, _, local = self._target_geometry(k)
        body_points = yaw_frame_points(k.pos[:, :2], k.yaw, self.body_scan)                  # (n, 70, 2)
        foot_points = yaw_frame_points(k.feet[:, :, :2], k.yaw[:, None], self.foot_scan)     # (n, 4, 9, 2)
        body_scan = (k.pos[:, 2:3] - STANDING_HEIGHT - jitter(self.terrain.sample(body_points), .02)).clamp(-1, 1)
        foot_scan = (k.feet[:, :, 2:3] - jitter(self.terrain.sample(foot_points), .02)).clamp(-1, 1).flatten(1)
        actor = torch.cat([
            jitter(k.lin_vel_body, .1) * 2., jitter(k.ang_vel_body, .2) * .25, jitter(k.gravity_body, .05),
            self._command(distance, local), local * (3. / distance.clamp_min(3.))[:, None], distance.clamp(max=3.)[:, None],
            jitter(k.joint_pos - self.home_joints, .01), jitter(k.joint_vel, 1.5) * .05, self.previous_action,
            (k.foot_height < .02).float(), body_scan, foot_scan], -1)
        assert actor.shape[-1] == OBSERVATION_SPEC['actor_dim']
        return actor.clamp(-100, 100)

    # ------------------------------------------------------------ resets/targets
    def reset(self, ids):
        if not len(ids): return
        cfg, count = self.cfg, len(ids)
        q = self.home.repeat(count, 1)
        if self.grid is not None:
            if self.grid.variants > 1:
                self.variant[ids] = torch.randint(0, self.grid.variants, (count,), device=self.device, generator=self.generator)
            spawn = self.centres[self.level[ids], self._tile(ids)]
            q[:, :2] = spawn[:, :2] + (self.rand(count, 2) - .5) * 1.
            q[:, 2] = self.terrain.sample(q[:, :2]) + STANDING_HEIGHT + .02
            yaw = (2 * self.rand(count) - 1) * math.pi
        else:
            q[:, :2] += (self.rand(count, 2) - .5) * .10
            yaw = (self.rand(count) - .5) * .8
        q[:, 3], q[:, 4:6], q[:, 6] = torch.cos(yaw / 2), 0, torch.sin(yaw / 2)
        self.backend.reset(ids, q)
        for buffer in (self.elapsed, self.previous_action, self.previous_joint_vel, self.air_time,
                       self.episode_arrivals, self.episode_misses):
            buffer[ids] = 0
        self.last_contact[ids] = True
        self.previous_feet[ids] = self.backend.tensor('site_xpos')[ids][:, self.foot_sites]
        self.final[ids] = False
        self._new_targets(ids)

    def _new_targets(self, ids):
        """Random reachable targets: 1-3 m away, on the robot's tile, not in a hole."""
        if not len(ids): return
        cfg, count = self.cfg, len(ids)
        pos = self.backend.tensor('qpos')[ids, :2]
        if self.grid is None:
            # Project scene: a random pile 1-2 m away, else stay in place.
            distance = torch.linalg.vector_norm(self.piles[None, :, :2] - pos[:, None], dim=-1)
            eligible = (distance >= 1.) & (distance <= 2.)
            choice = self.rand(*eligible.shape).masked_fill(~eligible, -1).argmax(-1)
            xy = torch.where(eligible.any(-1, keepdim=True), self.piles[choice, :2], pos)
        else:
            candidates = 8
            angle = (2 * self.rand(count, candidates) - 1) * math.pi
            radius = self.uniform(cfg.target_distance, count, candidates)
            xy = pos[:, None] + radius[..., None] * torch.stack([angle.cos(), angle.sin()], -1)
            low = torch.stack([self.tile_x[self.level[ids]], self.tile_y[self._tile(ids)]], -1)   # tile corner
            xy = torch.maximum(torch.minimum(xy, (low + cfg.tile_size - .5)[:, None]), (low + .5)[:, None])
            far_enough = torch.linalg.vector_norm(xy - pos[:, None], dim=-1) >= .75 * cfg.target_distance[0]
            in_hole = self.terrain.sample(xy) < self.hole_threshold[self.level[ids], self._tile(ids)][:, None]
            valid = far_enough & ~in_hole
            first = torch.where(valid.any(-1), valid.float().argmax(-1), torch.full((count,), candidates - 1, device=self.device))
            xy = xy[torch.arange(count, device=self.device), first]
        self._assign(ids, xy, final=False)
        self.max_speed[ids] = self.uniform(cfg.max_speed, count)
        self.target_budget[ids] = 2. + 2. * self.start_distance[ids] / self.max_speed[ids]

    def _assign(self, ids, xy, final):
        pos = self.backend.tensor('qpos')[ids, :2]
        self.target[ids, :2] = xy
        self.target[ids, 2] = self.terrain.sample(xy)
        self.start_distance[ids] = torch.linalg.vector_norm(xy - pos, dim=-1)
        self.final[ids] = final
        self.reached[ids] = False
        self.target_elapsed[ids] = 0
        self.hold_left[ids] = 0
        self.path_length[ids] = 0

    def set_target(self, xy, heading=None, final=False, ids=None, speed=None):
        """Planner interface: give the robots in ids a new target (heading is ignored).

        A final target is held forever once reached; otherwise the robot stops
        at the target and waits for the next set_target call.
        """
        ids = torch.arange(self.n, device=self.device) if ids is None else torch.as_tensor(ids, device=self.device)
        xy = torch.as_tensor(xy, device=self.device, dtype=torch.float32).reshape(-1, 2).expand(len(ids), 2)
        self._assign(ids, xy, final)
        self.max_speed[ids] = sum(self.cfg.max_speed) / 2 if speed is None else speed
        self.target_budget[ids] = 2. + 2. * self.start_distance[ids] / self.max_speed[ids]

    def _update_levels(self, ids, fell):
        """Curriculum: promote after reaching >= 80% of targets without a fall, demote after a fall or no progress.

        (Requiring a perfect episode kept stepping-stone robots on level 0 even at
        91% target success.)
        """
        arrivals, misses = self.episode_arrivals[ids], self.episode_misses[ids]
        promote = (arrivals >= 2) & (arrivals >= 4 * misses) & ~fell
        demote = fell | ((arrivals == 0) & (misses >= 1))
        level = self.level[ids] + promote.long() - demote.long()
        level = level.clamp_min(0)
        # Robots that solved the hardest row restart on a random row (legged_gym).
        top = level >= self.cfg.terrain_levels
        level[top] = torch.randint(0, self.cfg.terrain_levels, (int(top.sum()),), device=self.device, generator=self.generator)
        self.level[ids] = level

    # -------------------------------------------------------------------- step
    def step(self, action, auto_reset=True, chain=True):
        cfg, dt = self.cfg, self.cfg.control_dt
        action = action.clamp(-cfg.action_clip, cfg.action_clip)
        joint_targets = (self.home_joints + cfg.action_scale * action).clamp(self.joint_low, self.joint_high)
        before = self.backend.tensor('qpos')[:, :2].clone()
        self.backend.step(joint_targets, self.substeps)
        k = self._kinematics()
        self.elapsed += dt; self.target_elapsed += dt
        self.path_length += torch.linalg.vector_norm(k.pos[:, :2] - before, dim=-1)
        distance, direction, local = self._target_geometry(k)

        # Feet: contact, swing timing (legged_gym touchdown bookkeeping), slip.
        contact = k.foot_height < .02
        touchdown = (self.air_time > 0) & (contact | self.last_contact)
        self.air_time += dt
        air_time_on_touchdown = self.air_time.clone()
        self.air_time *= ~(contact | self.last_contact)
        self.last_contact = contact
        foot_velocity = (k.feet - self.previous_feet) / dt

        # Outcomes.
        nearby = self.terrain.sample(yaw_frame_points(k.pos[:, :2], k.yaw, self.clearance_scan)).amax(-1)
        fall = ((k.gravity_body[:, 2] > FALL_TILT) | (k.pos[:, 2] - nearby < FALL_CLEARANCE)) & k.valid
        arrived = (distance < cfg.arrival_radius) & ~fall & k.valid
        newly_arrived = arrived & ~self.reached
        time_to_goal = self.target_elapsed.clone()          # meaningful where newly_arrived
        efficiency = (self.start_distance / self.path_length.clamp_min(1e-3)).clamp(max=1.)
        self.reached |= arrived
        hold = self.uniform(cfg.hold_seconds, self.n)
        self.hold_left = torch.where(newly_arrived, torch.where(self.final, torch.full_like(hold, math.inf), hold), self.hold_left - dt)
        missed = (self.target_elapsed >= self.target_budget) & ~self.reached & ~fall
        terminated = fall | ~k.valid
        truncated = (self.elapsed >= cfg.episode_seconds) & ~terminated
        done = terminated | truncated
        self.episode_arrivals += newly_arrived.float(); self.episode_misses += missed.float()

        # Rewards (see rewards.py for the definitions and the reasoning).
        moving = ~self.reached
        s = SimpleNamespace(
            lin_vel_body=k.lin_vel_body, lin_vel_world=k.lin_vel_world, ang_vel_body=k.ang_vel_body,
            gravity_body=k.gravity_body, command=self._command(distance, local), target_direction=direction,
            max_speed=self.max_speed, moving=moving, touchdown=touchdown, air_time_on_touchdown=air_time_on_touchdown,
            foot_velocity=foot_velocity, contact=contact, knee_height=k.knee_height, joint_pos=k.joint_pos,
            home_joints=self.home_joints, torques=torch.nan_to_num(self.backend.tensor('qfrc_actuator')[:, 6:]),
            joint_acc=(k.joint_vel - self.previous_joint_vel) / dt, action=action, previous_action=self.previous_action,
            soft_low=self.soft_low, soft_high=self.soft_high, newly_arrived=newly_arrived, fall=fall,
            speed_xy=torch.linalg.vector_norm(k.lin_vel_world[:, :2], dim=-1),
            # How far each foot's sole sits below the highest terrain within 4 cm: positive when it slips into a gap.
            foot_depth=self.terrain.sample(k.feet[:, :, None, :2] + self.gap_scan).amax(-1) - (k.feet[:, :, 2] - FOOT_RADIUS))
        reward, terms = reward_terms.compute(s, cfg.rewards, cfg.reward_params, dt, cfg.only_positive_rewards)
        self.previous_action = action.clone(); self.previous_feet = k.feet; self.previous_joint_vel = k.joint_vel

        # Observation of the state reached by this step (needed to bootstrap
        # the value of episodes cut by the time limit), before any reset.
        terminal_actor, terminal_critic = self._observe(k, noise=False)
        info = dict(terminated=terminated, truncated=truncated, terminal_critic=terminal_critic,
                    arrived=newly_arrived, within_goal=arrived, fall=fall, missed=missed, distance=distance,
                    unstable=~k.valid, toward_velocity=(k.lin_vel_world[:, :2] * direction).sum(-1) * moving,
                    moving=moving, time_to_goal=time_to_goal, path_efficiency=efficiency, reward_terms=terms,
                    episode_seconds=self.elapsed.clone(), command_speed=torch.linalg.vector_norm(s.command[:, :2], dim=-1))
        if self.grid is not None:
            info.update(column=self.column, level=self.level.clone())
        if chain:
            switch = ((self.reached & (self.hold_left <= 0)) | missed) & ~done
            self._new_targets(torch.where(switch)[0])
        if auto_reset:
            ids = torch.where(done)[0]
            if self.grid is not None and self.cfg.curriculum and len(ids):
                self._update_levels(ids, fall[ids])
            self.reset(ids)
        actor, critic = self.observations()
        return actor, critic, reward, done, info
