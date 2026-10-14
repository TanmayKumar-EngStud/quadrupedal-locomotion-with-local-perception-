from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class EnvConfig:
    family: str = "flat"
    row: int = 0
    seed: int = 0
    num_envs: int = 8
    backend: str = "native"
    device: str = "cpu"
    sim_dt: float = .002
    control_dt: float = .02
    action_scale: float = .5
    kp: float = 35.
    kd: float = .5
    episode_seconds: float = 20.
    subgoal_seconds: float = 4.
    arrival_radius: float = .25
    noise: float = .02
    final_probability: float = .2
    curriculum: str = "smooth"
    scan_spacing: float = .05
    nconmax: int = 128
    njmax: int = 512
    rewards: dict = field(default_factory=lambda: dict(
        position=2.,heading=.5,progress=.3,tracking_lin_vel=1.,tracking_ang_vel=.3,
        orientation=-2.,lin_vel_z=-.5,ang_vel_xy=-.05,pose=.1,
        feet_clearance=.1,feet_slip=-.1,feet_air_time=.05,
        action_rate=-.01,torques=-.0002,energy=-.001,dof_acc=-1e-7,
        dof_pos_limits=-1.,stall=-.2,stand_still=-1.,arrival=2.,fall=-5.))

    def validate(self):
        if self.backend not in ("native","warp") or self.curriculum not in ("smooth","uniform","none"):
            raise ValueError("Invalid backend/curriculum")
        if self.backend=='native' and self.device!='cpu':raise ValueError('Native physics requires device=cpu')
        if self.backend=='warp' and not self.device.startswith('cuda'):raise ValueError('Warp physics requires device=cuda')
        if self.num_envs<1 or self.sim_dt<=0 or self.control_dt<=0 or self.action_scale<=0:
            raise ValueError("Invalid environment size/timing/action scale")
        if abs(round(self.control_dt/self.sim_dt)*self.sim_dt-self.control_dt)>1e-8:
            raise ValueError("Control timestep must be a multiple of physics timestep")
        if self.scan_spacing != .05:
            raise ValueError("Observation contract v1 fixes height scan spacing at 5 cm")
        if self.episode_seconds<=0 or self.subgoal_seconds<=0 or not 0<=self.final_probability<=1:
            raise ValueError("Invalid task duration or final-goal probability")


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    updates: int = 2000
    rollout_steps: int = 32
    epochs: int = 5
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .005
    value_coef: float = .5
    max_grad_norm: float = 1.
    symmetry_coef: float = .1
    target_kl: float = .02
    hidden: list = field(default_factory=lambda:[256,128,64])
    checkpoint_every: int = 50
    terrain_every: int = 100
    terrain_families: list = field(default_factory=list)
    output: str = "runs/controller"

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        data["env"]=EnvConfig(**data.get("env",{}))
        result=cls(**data);result.env.validate()
        if min(result.updates,result.rollout_steps,result.epochs,result.minibatch_size,result.checkpoint_every,result.terrain_every)<1:
            raise ValueError("Training counts must be positive")
        return result

    def dict(self): return asdict(self)


OBSERVATION_SPEC = {
    "version":1,"actor_dim":873,"critic_dim":876,
    "actor_fields":[["body_angular_velocity",3],["projected_gravity",3],
                    ["joint_position_minus_home",12],["joint_velocity_scaled_0.05",12],
                    ["previous_tanh_action",12],["relative_target_xy",2],
                    ["heading_error_sin_cos",2],["time_left_fraction",1],["final_goal",1],
                    ["base_height_minus_surface_minus_0.30",825]],
    "scan":{"x":[-.8,.8,33],"y":[-.6,.6,25],"order":"x-major, y-minor","frame":"yaw-only","clip_m":[-1.,1.]},
    "joint_order":[f"{leg}_{joint}" for leg in ["FR","FL","RR","RL"] for joint in ["hip","thigh","calf"]],
    "critic_extra":"body linear velocity (simulation privileged)",
}
