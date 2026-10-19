"""Reward terms for goal-reaching locomotion, in physical units before weighting.

Every function takes the step's quantities and returns one value per robot.
The environment multiplies dense terms by their weight and the control
timestep; 'arrival' and 'fall' are event rewards paid once.

Why the policy cannot settle for "standing still is good enough":
  * the commanded velocity points at the target, so standing still earns only
    exp(-|v_cmd|^2/0.25) of the tracking reward (0.08 at 0.8 m/s), while moving
    correctly earns 1.0;
  * goal_progress pays for velocity toward the target and is never switched off;
  * reaching the target pays a large one-off bonus, and a new target follows.

Rewards are not clipped at zero: with exploration noise the gait penalties
exceed the tracking reward at the start of training, and a zero clip would then
hide every learning signal except falls. Instead a small 'alive' reward keeps
surviving positive and the -10 fall penalty makes ending the episode costly.
"""
import torch

TRACKING_SIGMA = .25
AIR_TIME_TARGET = .25        # seconds; Go1 swings are short
KNEE_CLEARANCE = .04


def alive(s): return torch.ones_like(s.fall, dtype=torch.float32)
def tracking_lin_vel(s): return torch.exp(-(s.lin_vel_body[:, :2] - s.command[:, :2]).square().sum(-1) / TRACKING_SIGMA)
def tracking_ang_vel(s): return torch.exp(-(s.ang_vel_body[:, 2] - s.command[:, 2]).square() / TRACKING_SIGMA)
def goal_progress(s): return torch.minimum((s.lin_vel_world[:, :2] * s.target_direction).sum(-1), s.max_speed) * s.moving
def lin_vel_z(s): return s.lin_vel_body[:, 2].square()
def ang_vel_xy(s): return s.ang_vel_body[:, :2].square().sum(-1)
def orientation(s): return s.gravity_body[:, :2].square().sum(-1)
def feet_air_time(s): return ((s.air_time_on_touchdown - AIR_TIME_TARGET) * s.touchdown).sum(-1) * s.moving
def feet_slip(s): return (s.foot_velocity[:, :, :2].square().sum(-1) * s.contact).sum(-1)
def knee_collision(s): return (s.knee_height < KNEE_CLEARANCE).float().sum(-1)
def stand_still(s): return (s.joint_pos - s.home_joints).abs().sum(-1) * ~s.moving
def torques(s): return s.torques.square().sum(-1)
def dof_acc(s): return s.joint_acc.square().sum(-1)
def action_rate(s): return (s.action - s.previous_action).square().sum(-1)
def dof_pos_limits(s): return ((s.soft_low - s.joint_pos).clamp_min(0) + (s.joint_pos - s.soft_high).clamp_min(0)).sum(-1)
def arrival(s): return s.newly_arrived.float()
def fall(s): return s.fall.float()


DENSE = ('alive', 'tracking_lin_vel', 'tracking_ang_vel', 'goal_progress', 'lin_vel_z', 'ang_vel_xy', 'orientation',
         'feet_air_time', 'feet_slip', 'knee_collision', 'stand_still', 'torques', 'dof_acc', 'action_rate',
         'dof_pos_limits')
EVENTS = ('arrival', 'fall')


def compute(s, weights, dt, only_positive=True):
    """Total reward per robot and the weighted value of every term (for logging)."""
    terms = {}
    for name in DENSE:
        if weights.get(name, 0.):
            terms[name] = torch.nan_to_num(globals()[name](s)) * weights[name] * dt
    dense = sum(terms.values()) if terms else torch.zeros_like(s.fall, dtype=torch.float32)
    if only_positive:
        dense = dense.clamp_min(0.)
    for name in EVENTS:
        if weights.get(name, 0.):
            terms[name] = globals()[name](s) * weights[name]
    return dense + sum(terms[name] for name in EVENTS if name in terms), terms
