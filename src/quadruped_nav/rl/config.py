"""Configuration of the goal-reaching task, the reward weights and PPO.

All values are plain dataclass fields so a run is fully described by the
config.json written next to its checkpoints.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from .terrain_grid import TERRAIN_TYPES, ALL_TERRAIN_TYPES


def default_rewards():
    """Reward weights. Dense terms are multiplied by the control timestep;
    'arrival' and 'fall' are one-off event rewards (not scaled by dt)."""
    return dict(
        alive=1.,                 # per second survived
        # Task: move toward the target at the commanded velocity.
        tracking_lin_vel=1.5,     # exp(-|v_xy - v_cmd|^2 / 0.25), body frame
        tracking_ang_vel=.75,     # exp(-(w_z - w_cmd)^2 / 0.25)
        goal_progress=.5,         # velocity component toward the target (m/s)
        # Body stability.
        lin_vel_z=-2.,            # vertical bouncing
        ang_vel_xy=-.05,          # roll/pitch rates
        orientation=-.5,          # tilt of the body
        # Gait shaping.
        feet_air_time=.5,         # reward swings longer than 0.25 s, paid on touchdown
        feet_slip=-.1,            # horizontal foot speed while in contact
        knee_collision=-1.,       # knees closer than 4 cm to the terrain
        stand_still=-.5,          # joint deviation from home while holding at a goal
        # Smooth, efficient control.
        torques=-1e-4, dof_acc=-2.5e-7, action_rate=-.01, dof_pos_limits=-10.,
        # Events.
        arrival=10.,              # once per reached target
        fall=-10.,                # once, on termination
        # Failure-targeted terms, off unless a run config sets a weight (v3/v4 candidates):
        arrival_speed=0.,         # once per arrival: (horizontal speed - 0.5 m/s)^2 above 0.5 m/s; -10 cancels the bonus at 1.5 m/s
        feet_in_gap=0.)           # per second: metres each foot sits more than 3 cm below the support within 4 cm (capped at 10 cm)


@dataclass
class EnvConfig:
    # Simulation backend.
    num_envs: int = 8
    backend: str = "native"          # 'native' (CPU MuJoCo) or 'warp' (GPU MuJoCo Warp)
    device: str = "cpu"
    seed: int = 0
    # Training terrain: a grid of tiles (rows = difficulty, columns = types).
    terrain_types: list = field(default_factory=lambda: list(TERRAIN_TYPES))
    terrain_levels: int = 10
    tile_size: float = 8.
    terrain_resolution: float = .05
    fine_terrain_resolution: float = .025   # cell size of the pile tiles (terrain_grid.FINE_TYPES)
    terrain_weights: list = None     # share of robots per terrain type (None: equal shares)
    terrain_noise: float = 0.        # per-element irregularities: 0.5 for v3, 1.0 for v4 (see terrain_grid)
    terrain_variants: int = 1        # independent tiles per (level, type); robots draw one at every reset
    pile_pit_ramp: bool = False      # pile pits deepen with level (for runs that start from scratch)
    curriculum: bool = True          # per-robot level promotion / demotion
    max_start_level: int = 2         # robots start on rows 0..max_start_level
    # Physics and control.
    sim_dt: float = .004
    control_dt: float = .02
    solver_iterations: int = 20      # fewer iterations made random-action rollouts blow up
    solver_ls_iterations: int = 20
    contact_set: str = 'feet_trunk'  # robot geoms colliding with terrain: 'feet_trunk' or 'legs'
    nconmax: int = 64                # contacts per world (pooled across worlds)
    njmax: int = 256                 # constraint rows per world
    kp: float = 35.
    joint_damping: float = .5
    action_scale: float = .25        # joint target = home + action_scale * action
    action_clip: float = 4.
    # Goal-reaching task.
    episode_seconds: float = 20.
    target_distance: list = field(default_factory=lambda: [1., 3.])
    max_speed: list = field(default_factory=lambda: [.5, 1.])   # m/s, sampled per target
    arrival_radius: float = .25
    hold_seconds: list = field(default_factory=lambda: [0., 2.])  # standing at a goal before the next one
    slow_down_radius: float = .5     # commanded speed ramps down inside this distance
    yaw_gain: float = 1.5
    noise: float = 1.                # observation noise multiplier (0 disables)
    only_positive_rewards: bool = False  # optionally clip the dense reward at 0 (legged_gym); events added after
    rewards: dict = field(default_factory=default_rewards)

    def validate(self):
        if self.backend not in ("native", "warp"): raise ValueError("backend must be native or warp")
        if self.contact_set not in ('feet_trunk', 'legs'): raise ValueError('contact_set must be feet_trunk or legs')
        if self.backend == 'native' and self.device != 'cpu': raise ValueError('Native physics requires device=cpu')
        if self.backend == 'warp' and not self.device.startswith('cuda'): raise ValueError('Warp physics requires device=cuda')
        if self.num_envs < 1 or self.sim_dt <= 0 or self.control_dt <= 0 or self.action_scale <= 0:
            raise ValueError("Invalid environment size/timing/action scale")
        if abs(round(self.control_dt / self.sim_dt) * self.sim_dt - self.control_dt) > 1e-8:
            raise ValueError("Control timestep must be a multiple of physics timestep")
        if unknown := set(self.terrain_types) - set(ALL_TERRAIN_TYPES): raise ValueError(f'Unknown terrain types {unknown}')
        if self.terrain_weights is not None and (len(self.terrain_weights) != len(self.terrain_types)
                                                 or min(self.terrain_weights) < 0 or sum(self.terrain_weights) <= 0):
            raise ValueError('terrain_weights needs one non-negative weight per terrain type')
        if self.terrain_noise < 0 or self.terrain_variants < 1: raise ValueError('Invalid terrain noise or variants')
        if not 0 <= self.max_start_level < self.terrain_levels: raise ValueError('Invalid start level')
        if not 0 < self.target_distance[0] <= self.target_distance[1]: raise ValueError('Invalid target distance range')
        if self.target_distance[0] <= self.arrival_radius: raise ValueError('Targets must start outside the arrival radius')
        if not all(math.isfinite(v) for v in self.rewards.values()): raise ValueError('Rewards must be finite')
        if unknown := set(self.rewards) - set(default_rewards()): raise ValueError(f'Unknown reward terms {unknown}')


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    updates: int = 3000
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1e-3      # adapted to keep the KL near desired_kl
    desired_kl: float = .01
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .005
    value_coef: float = 1.
    max_grad_norm: float = 1.
    init_std: float = 1.
    hidden: list = field(default_factory=lambda: [512, 256, 128])
    checkpoint_every: int = 50
    evaluation_every: int = 50       # fixed-target evaluation on every terrain type
    validation_seed: int = None      # run that evaluation on unseen tiles from this seed (None: training tiles)
    max_hours: float = None          # stop after this much training time, whatever `updates` says
    output: str = "runs/goal-v3"

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        data["env"] = EnvConfig(**data.get("env", {}))
        result = cls(**data); result.env.validate()
        if min(result.updates, result.rollout_steps, result.epochs, result.minibatches, result.checkpoint_every) < 1:
            raise ValueError("Training counts must be positive")
        return result

    def dict(self): return asdict(self)


# Terrain samples used by the policy (Lab 1 style): body-anchored points plus
# points anchored at every foot, all in the yaw-aligned body frame (x forward,
# y left). At most 200 samples in total.
BODY_SCAN = [(x / 100, y / 100) for x in range(-45, 91, 15) for y in range(-45, 46, 15)]
FOOT_SCAN = [(0., 0.), (.08, 0.), (-.08, 0.), (0., .08), (0., -.08), (.16, 0.), (.24, 0.), (.16, .08), (.16, -.08)]


def observation_spec():
    """Names and sizes of the actor and critic observation blocks (frozen contract)."""
    actor = [["base_linear_velocity_body_x2", 3], ["base_angular_velocity_body_x0.25", 3],
             ["projected_gravity", 3], ["velocity_command_vx_vy_wz", 3],
             ["target_xy_body_clipped_3m", 2], ["target_distance_clipped_3m", 1],
             ["joint_position_minus_home", 12], ["joint_velocity_x0.05", 12], ["previous_action", 12],
             ["foot_contact", 4],
             ["body_scan_base_height_minus_0.3_minus_terrain", len(BODY_SCAN)],
             ["foot_scan_foot_height_minus_terrain", 4 * len(FOOT_SCAN)]]
    critic_extra = [["base_height_above_terrain", 1], ["foot_height_above_terrain", 4], ["target_time_left_s", 1]]
    actor_dim = sum(n for _, n in actor)
    return {"version": 3, "actor_fields": actor, "critic_extra_fields": critic_extra,
            "actor_dim": actor_dim, "critic_dim": actor_dim + sum(n for _, n in critic_extra),
            "body_scan": BODY_SCAN, "foot_scan": FOOT_SCAN, "foot_order": ["FR", "FL", "RR", "RL"],
            "joint_order": [f"{leg}_{joint}" for leg in ["FR", "FL", "RR", "RL"] for joint in ["hip", "thigh", "calf"]]}


OBSERVATION_SPEC = observation_spec()
