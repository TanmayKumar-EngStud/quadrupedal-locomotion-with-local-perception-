"""Reward function of the goal-reaching controller, laid out like Lab 1 Level C.

HOW TO TUNE IT
  * reward_config() below holds every reward weight ("scales") and every shaping
    parameter ("params"), grouped by purpose. Positive weights reward a behaviour,
    negative weights penalise it, and 0 disables the term.
  * To change them for one training run, add "rewards" (weights) and/or
    "reward_params" (shaping parameters) to the "env" block of the run config,
    e.g. configs/kth-controller-v3.json. List only the values you change; every
    other value keeps its default from reward_config().
  * Each weight belongs to the _reward_<name>() function with the same name,
    further down. Those functions return physical quantities (m/s, rad, m, ...)
    before weighting.

HOW THE TOTAL IS FORMED (every control step, 50 times per second)
  reward = sum of  weight x _reward_<name>() x dt   over the per-second terms
         + sum of  weight x _reward_<name>()        over the event terms
  Event terms (arrival, fall, arrival_speed) are paid once, when the event
  happens, so they are not multiplied by the control timestep dt.

WHY THE ROBOT CANNOT SETTLE FOR "STANDING STILL IS GOOD ENOUGH"
  * the commanded velocity points at the target, so standing still earns only
    exp(-|v_cmd|^2 / tracking_sigma) of the tracking reward (0.08 at 0.8 m/s),
    while moving correctly earns 1.0;
  * goal_progress pays for velocity toward the target and is never switched off;
  * reaching the target pays a large one-off bonus, and a new target follows.
  Rewards are not clipped at zero: with exploration noise the gait penalties
  exceed the tracking reward at the start of training, and a zero clip would then
  hide every learning signal except falls. The small 'alive' reward keeps
  surviving positive, and the -10 fall penalty makes ending the episode costly.
"""
import torch


def reward_config():
    """Default reward weights and shaping parameters (edit here, or override them in a run config)."""
    return dict(
        ### ----- REWARD WEIGHTS ----- ###
        scales=dict(
            # 1. Reach the target.
            alive=1.0,               # Per second survived: surviving is worth more than ending the episode.
            tracking_lin_vel=1.5,    # Match the commanded velocity toward the target (body frame).
            tracking_ang_vel=0.75,   # Match the commanded turning rate (turn to face the target).
            goal_progress=0.5,       # Velocity toward the target in m/s, capped at the commanded speed.
            heading_alignment=0.0,   # Cosine between heading and direction of travel (+1 forwards, -1 backwards). Off here; 0.5 in variant 10.
            arrival=10.0,            # EVENT: once for every target reached.

            # 2. Keep the body controlled.
            lin_vel_z=-2.0,          # Penalise vertical bouncing.
            ang_vel_xy=-0.05,        # Penalise roll and pitch rotation.
            orientation=-0.5,        # Penalise body tilt.

            # 3. Shape foot placement and gait.
            feet_air_time=0.5,       # Reward swings longer than air_time_target, paid on touchdown.
            feet_slip=-0.1,          # Penalise feet sliding while in contact.
            knee_collision=-1.0,     # Penalise knees closer than knee_clearance to the terrain.
            stand_still=-0.5,        # Penalise joint deviation from the standing pose while holding at a target.
            feet_clearance=0.0,      # Reward lifting each foot during its swing, paid on touchdown (Lab 1 C). Off here; 0.5 in variant 5 round 3.

            # 4. Encourage smooth, efficient control.
            torques=-1e-4,           # Penalise large motor torques.
            dof_acc=-2.5e-7,         # Penalise joint accelerations.
            action_rate=-0.01,       # Penalise abrupt changes between consecutive actions.

            # 5. Discourage unsafe or failed behaviour.
            dof_pos_limits=-10.0,    # Penalise joints beyond 95% of their range.
            feet_in_gap=0.0,         # Penalise feet sitting in gaps (see gap_deadband). Off here; -20 in v3/v4.
            feet_edge=0.0,           # Penalise feet standing within 5 cm of a drop deeper than edge_drop. Off here; -1 in variants 3 and 6.
            fall=-10.0,              # EVENT: once, when the robot falls (the episode ends).
            arrival_speed=0.0,       # EVENT: penalise arriving faster than arrival_speed_limit. Off here; -10 in v3/v4.
        ),
        ### ----- SHAPING PARAMETERS ----- ###
        params=dict(
            tracking_sigma=0.25,         # Smaller means stricter velocity tracking.
            air_time_target=0.25,        # Seconds; Go1 swings are short. Longer swings earn feet_air_time.
            knee_clearance=0.04,         # Metres; knees closer than this to the terrain count as a collision.
            arrival_speed_limit=0.5,     # m/s; the guidance command is about 0.4 m/s at the arrival circle.
            gap_deadband=0.03,           # Metres a foot may sit below the support within 4 cm before it counts.
                                         # (v2 on pile scenes: at most 2.6 cm in successful episodes,
                                         #  10-37 cm in the last second before 10 of 12 falls.)
            gap_cap=0.10,                # Metres; a foot deeper in a gap than this counts the same.
            edge_drop=0.05,              # Metres; a drop this deep within 5 cm of a standing foot makes it an edge.
            clearance_target=0.10,       # Metres; feet_clearance saturates when a swing lifts the foot this high (Lab 1 C).
            clearance_min_air_time=0.08, # Seconds; shorter swings (contact flicker) earn no feet_clearance (Lab 1 C).
            heading_min_speed=0.2,       # m/s; slower movement earns no heading_alignment (its direction is noise).
        ),
    )


EVENTS = ('arrival', 'fall', 'arrival_speed')   # paid once; every other term is paid per second


# ------------------------------------------------------------------------------
# Reward terms, one function per weight. s holds the step's quantities (built in
# GoalEnv.step) and p the shaping parameters. Each returns one value per robot.
# ------------------------------------------------------------------------------

# 1. Reach the target.
def _reward_alive(s, p):
    """1 for every robot still running."""
    return torch.ones_like(s.fall, dtype=torch.float32)


def _reward_tracking_lin_vel(s, p):
    """exp(-|v_xy - v_cmd|^2 / tracking_sigma): 1 when the body moves exactly as commanded."""
    error = (s.lin_vel_body[:, :2] - s.command[:, :2]).square().sum(-1)
    return torch.exp(-error / p['tracking_sigma'])


def _reward_tracking_ang_vel(s, p):
    """exp(-(w_z - w_cmd)^2 / tracking_sigma) for the turning rate."""
    error = (s.ang_vel_body[:, 2] - s.command[:, 2]).square()
    return torch.exp(-error / p['tracking_sigma'])


def _reward_goal_progress(s, p):
    """Velocity toward the target (m/s), at most the commanded speed; 0 while holding at a target."""
    toward = (s.lin_vel_world[:, :2] * s.target_direction).sum(-1)
    return torch.minimum(toward, s.max_speed) * s.moving


def _reward_heading_alignment(s, p):
    """Cosine between the body's heading and its horizontal velocity: +1 walking forwards, -1 backwards, 0 sideways.

    Paid while walking faster than heading_min_speed to a target more than slow_down_radius away, where the command
    also turns the robot toward the target; close to a target, stepping sideways or back stays free.
    """
    return s.heading_cos * (s.speed_xy > p['heading_min_speed']) * s.moving * s.target_far


def _reward_arrival(s, p):
    """1 on the step a robot first comes within the arrival radius of its target."""
    return s.newly_arrived.float()


# 2. Keep the body controlled.
def _reward_lin_vel_z(s, p):
    """Vertical body speed squared."""
    return s.lin_vel_body[:, 2].square()


def _reward_ang_vel_xy(s, p):
    """Roll and pitch rates squared."""
    return s.ang_vel_body[:, :2].square().sum(-1)


def _reward_orientation(s, p):
    """Tilt: the horizontal part of gravity in the body frame, squared (0 when level)."""
    return s.gravity_body[:, :2].square().sum(-1)


# 3. Shape foot placement and gait.
def _reward_feet_air_time(s, p):
    """On touchdown, the swing time beyond air_time_target (negative for shorter swings)."""
    return ((s.air_time_on_touchdown - p['air_time_target']) * s.touchdown).sum(-1) * s.moving


def _reward_feet_clearance(s, p):
    """On touchdown after a swing of at least clearance_min_air_time, each foot's highest point above the terrain
    during that swing as a share of clearance_target (at most 1), while moving (Lab 1 C's feet_clearance)."""
    completed = s.touchdown & (s.air_time_on_touchdown >= p['clearance_min_air_time'])
    return ((s.swing_peak / p['clearance_target']).clamp(0, 1) * completed).sum(-1) * s.moving


def _reward_feet_slip(s, p):
    """Horizontal foot speed squared, for feet in contact."""
    return (s.foot_velocity[:, :, :2].square().sum(-1) * s.contact).sum(-1)


def _reward_knee_collision(s, p):
    """Number of knees closer than knee_clearance to the terrain."""
    return (s.knee_height < p['knee_clearance']).float().sum(-1)


def _reward_stand_still(s, p):
    """Joint deviation from the standing pose, only while holding at a target."""
    return (s.joint_pos - s.home_joints).abs().sum(-1) * ~s.moving


# 4. Encourage smooth, efficient control.
def _reward_torques(s, p):
    """Motor torques squared."""
    return s.torques.square().sum(-1)


def _reward_dof_acc(s, p):
    """Joint accelerations squared."""
    return s.joint_acc.square().sum(-1)


def _reward_action_rate(s, p):
    """Change between consecutive actions, squared."""
    return (s.action - s.previous_action).square().sum(-1)


# 5. Discourage unsafe or failed behaviour.
def _reward_dof_pos_limits(s, p):
    """How far joints are beyond their soft limits (95% of the range)."""
    return ((s.soft_low - s.joint_pos).clamp_min(0) + (s.joint_pos - s.soft_high).clamp_min(0)).sum(-1)


def _reward_feet_in_gap(s, p):
    """Metres each foot sits more than gap_deadband below the highest terrain within 4 cm (capped at gap_cap)."""
    return (s.foot_depth - p['gap_deadband']).clamp(0, p['gap_cap'] - p['gap_deadband']).sum(-1)


def _reward_feet_edge(s, p):
    """Number of feet standing at an edge: the terrain within 5 cm drops more than edge_drop below the sole
    (the feet_edge term of Cheng et al., Extreme Parkour, 2023). Catches feet before they slip off a pile or stone."""
    return (s.edge_drop > p['edge_drop']).float().sum(-1)


def _reward_fall(s, p):
    """1 on the step a robot falls."""
    return s.fall.float()


def _reward_arrival_speed(s, p):
    """On arrival, (horizontal speed - arrival_speed_limit)^2 above the limit."""
    return (s.speed_xy - p['arrival_speed_limit']).clamp_min(0).square() * s.newly_arrived


# ------------------------------------------------------------------------------
# Total reward.
# ------------------------------------------------------------------------------
TERMS = {name[len('_reward_'):]: fn for name, fn in globals().items() if name.startswith('_reward_')}


def compute(s, weights, params, dt, only_positive=False):
    """Total reward per robot and the weighted value of every term (for logging).

    Terms are added in the order of reward_config(); a term with weight 0 is skipped.
    """
    terms = {}
    for name, weight in weights.items():
        if weight and name not in EVENTS:
            terms[name] = torch.nan_to_num(TERMS[name](s, params)) * weight * dt
    dense = sum(terms.values()) if terms else torch.zeros_like(s.fall, dtype=torch.float32)
    if only_positive:
        dense = dense.clamp_min(0.)
    for name in EVENTS:
        if weights.get(name, 0.):
            terms[name] = TERMS[name](s, params) * weights[name]
    return dense + sum(terms[name] for name in EVENTS if name in terms), terms
