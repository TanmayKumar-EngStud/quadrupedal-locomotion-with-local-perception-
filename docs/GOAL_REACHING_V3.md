# Goal-reaching controller v3 (October 14, 2026)

The v1/v2 controller learned to stand and shuffle but never reached its targets
("standing still is good enough"). v3 keeps the project's stack (PyTorch PPO +
MuJoCo Warp on the KTH GPU, native MuJoCo on the CPU for the planner) and
changes the task, rewards, terrain and training recipe so that walking to the
target is the only way to score well.

## Why v1/v2 stalled

| Cause | Effect |
| --- | --- |
| About 6,000 samples/s (256 robots, 1 ms physics step, 100 solver iterations, full-body self-collision, elliptic cones with impratio 100) | 4.4 M transitions after 538 updates. Legged-robot PPO normally needs 100-300 M before a gait emerges, so the policy was still at the "stand still" stage |
| Progress reward switched off after update 150; position reward only in the last second | Very little reward for moving at all; standing still scored about as well |
| Negative dense reward, not clipped | Falling (ending the episode) was a way to stop paying penalties |
| Actor did not see its own velocity, so it could not tell whether it was moving | A harder learning problem than needed |
| All 256 robots shared one terrain per batch | Little diversity; curriculum updates were global |

## What changed

**Physics (src/quadruped_nav/rl/robot.py).** The same settings are applied to the
training world and the planner's box scenes:
- 4 ms physics step (5 substeps per 50 Hz control step).
- Pyramidal friction cones.
- 20 Newton iterations. With 4, random-action rollouts blew up on both CPU and
  GPU; the extra iterations cost about 7% of throughput.
- PD gains kp=35 and damping 0.5 (MuJoCo Playground's Go1 values, as in Lab 1).
- Robot geoms collide with the terrain only (no self-collision), and only the
  feet and trunk take part (`contact_set: legs` also adds shins and thighs).
- Kernel overflow printing is disabled, because Warp's printf was the largest
  single slowdown.

Result: about 30,000-60,000 physics steps/s with 4,096 robots on the H100 MIG
slice, against about 6,000 before.

**One heightfield per tile.** A single 86 m x 62 m heightfield made MuJoCo Warp
unstable: robots more than about 30 m from the heightfield's centre blew up
under random actions, with joint speeds over 1,000 rad/s and then NaN. Warp
computes heightfield contacts in the heightfield frame, and float32 rounding at
that distance exceeds its collision tolerance. Building one 8 m heightfield per
tile (70 heightfields) plus four flat border boxes removed the problem: 0
unstable worlds out of 1,792. The only exceptions are robots that fall into
stepping-stone or gap pits, and those episodes are ended as falls.

**Terrain (src/quadruped_nav/rl/terrain_grid.py).** A 5 cm height grid of 8 m tiles. Each of the 10 rows is a difficulty level; each of the 7
columns is a terrain type: rough, slope (up to 24 degrees), stairs up, stairs
down (3-15 cm steps), obstacles, stepping stones (0.6 m stones with 5 cm gaps up
to 0.3 m stones with 18 cm gaps) and gaps (random 0.3 m holes).
Tile edges meet at height 0, so there are no cliffs between tiles.

Pit depth on the stepping-stone and gap tiles grows with the level, from 5 cm
on level 0 to 50 cm on level 9. With 50 cm pits from level 0 onwards (phase 1,
updates 1-400), every misstep ended the episode and stepping-stone success
stayed near 0%. The run was resumed at update 400 with the depth curriculum
(phase 2): stepping-stone training success rose to about 80% within 15 updates.

**Per-robot curriculum (env.py).** Each robot keeps its terrain type and
starts on level 0-2. After an episode it moves up a level if it reached at
least 2 targets and at least 80% of the targets it attempted, without falling.
It moves down a level if it fell or reached nothing. Robots that clear level 9
restart on a random level, as in legged_gym.

The first rule required a perfect episode. With about five targets per episode,
stepping-stone robots stayed on level 0 even though they reached 91% of their
targets, so the run was resumed at update 750 with the 80% rule (phase 3).

**Task and guidance law (env.py).** Targets are 1-3 m away in any direction,
on the robot's tile, never in a hole, with a time budget of 2 s + 2 x
(distance / speed). After arriving, the robot holds position for 0-2 s and then
gets the next target, so one 20 s episode contains several targets. The policy
sees the target position in its body frame, and also the velocity command a
simple guidance law would give: walk toward the target at 0.5-1.0 m/s, turn to
face it, and slow down inside 0.5 m.

The commanded speed does not depend on the robot's heading. An earlier version
slowed the command when the target was behind (turn first). Robots then learned
to face away from their targets, which shrank the command, and standing still
earned about 90% of the tracking reward (training attempt 5).

**Rewards (src/quadruped_nav/rl/rewards.py).**
- Task: velocity tracking toward the target (1.5), yaw-rate tracking (0.75),
  velocity toward the target (0.5, never switched off), +10 per target reached,
  -10 per fall and +1 per second alive.
- Regularizers (legged_gym style): vertical bounce, roll/pitch rate, tilt, swing
  time, foot slip, knee clearance, stand-still at the goal, torque, joint
  acceleration, action rate and joint limits.
- Standing still with a target 1 m away now earns at most 8% of the tracking
  reward.
- Rewards are not clipped at zero. A legged_gym-style zero clip was tried first
  (training attempt 4): at the initial exploration noise, the gait penalties
  (about -0.045 per step) outweighed tracking (+0.018), so almost every step
  scored exactly 0. The robots learned not to fall but never to walk.

**Observations (config.py, OBSERVATION_SPEC v3; 161 actor values).**
- Proprioception: base linear velocity and angular velocity in the body frame,
  gravity direction, joint positions and velocities, previous action and a
  contact flag per foot.
- Task inputs: the velocity command, the target position in the body frame
  (clipped to 3 m) and the target distance.
- Terrain samples, following Lab 1's terrain_sample_constructor: 70 points on a
  1.35 m x 0.9 m body grid, plus 9 points anchored at each foot (the foot
  itself, a cross at plus or minus 8 cm, and points 16 cm and 24 cm ahead).
  That is 106 samples, under Lab 1's 200-sample budget.

The critic also receives privileged values: base and foot heights above the
terrain, and the time left on the target.

**PPO (src/quadruped_nav/rl/ppo.py).** rsl_rl/legged_gym recipe:
- A Gaussian policy (initial std 1.0, actions scaled by 0.25 rad).
- Actor and critic networks of 512-256-128 units.
- 24-step rollouts, 5 epochs over 4 minibatches.
- Clipped surrogate and clipped value loss.
- GAE with gamma 0.99 and lambda 0.95.
- An adaptive learning rate that targets a KL of 0.01.
- Value bootstrapping on time-outs.
- Observation normalization whose statistics are frozen during collection and
  the update and refreshed after each update. Updating them mid-rollout
  (training attempt 3) inflated the KL and collapsed the adaptive learning rate.

## Measurements saved during training

`runs/<run>/metrics.jsonl` gets one record per update:
- Rewards: mean reward and every weighted reward term.
- Task outcomes: targets reached, falls and missed targets per 10,000
  transitions; training success rate, overall and per terrain type; time to
  reach a target; path efficiency (straight-line distance / distance walked).
- Motion: velocity toward the target versus commanded speed; episode length.
- Curriculum: mean terrain level, overall and per type.
- PPO: losses, KL, learning rate, entropy, action std, explained variance,
  clip fraction.
- System: samples/s, contact and constraint usage, non-finite states.

`runs/<run>/evaluation.jsonl` gets one record every 25 updates. The same 112
robots (7 types x levels 0/3/6/9 x 4 trials) each try to reach one target 2 m
away within 10 s, using deterministic actions. Each record holds success rate,
fall rate, time to goal and progress, per type, per level, and for every
type/level cell.

`python scripts/plot_training.py runs/<run>` writes four figures: a training
dashboard, a success heatmap, a reward-term breakdown and PPO diagnostics.

## Commands

```sh
uv pip install --python .venv/bin/python matplotlib     # plots only
.venv/bin/python scripts/bench_physics.py --worlds 4096  # GPU throughput
.venv/bin/python scripts/cloud_preflight.py              # CPU/GPU agreement
.venv/bin/python -m quadruped_nav.train --config configs/preflight.json
.venv/bin/python -m quadruped_nav.train --config configs/kth-goal-v3.json
.venv/bin/python -m quadruped_nav.goal_check --checkpoint runs/kth-goal-v3-seed0/last.pt --backend native
.venv/bin/python -m quadruped_nav.evaluate --checkpoint runs/kth-goal-v3-seed0/last.pt --family open --row 0
# Controller v2 (pile fine-tune) and the v1/v2 comparison
.venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v2.json --resume runs/kth-goal-v3-seed0-phase3/update_003000.pt
.venv/bin/python -m quadruped_nav.calibrate --workers 5 --checkpoint artifacts/controller_v2/controller.pt --ground heightfield --output outputs/calibration/hops_v2_heightfield.json
.venv/bin/python -m quadruped_nav.goal_check --checkpoint artifacts/controller_v2/controller.pt --terrain-types piles
.venv/bin/python scripts/compare_controllers.py --help
.venv/bin/python scripts/render_compare.py scene --family open --row 0 --output outputs/comparison/clips/open_row0.mp4
```

## Controller v2: pile fine-tune (October 15, 2026)

Controller v1 (`artifacts/controller_v1`, update 3000) reaches 86% of the fixed-test
targets, but on the project's pile scenes only 24% of first hops off the start
platform arrive and 65% of hops end in falls (`outputs/calibration/hops_v1.json`).
Its curriculum reached stepping-stone level 0.35 of 9: about 1 robot in 7 trained on
stones, mostly on level 0, where pits were 5 cm deep and a misstep cost nothing.

**Pile tiles (terrain_grid.py, type `piles`).** An 8th tile type copies the pile
scenes (terrain.py) with row = 19 d: piles shrink from 0.50 to 0.12 m and gaps
widen from 3 to 13 cm; half the piles get a height offset (up to 8 cm) and half a
tilt (up to 15 degrees), as in the rough, steps and slanted families. Pile tiles use
2.5 cm heightfield cells, so every 3 cm gap holds at least one pit node; the rest of
the grid keeps 5 cm cells (the height map is stored at 2.5 cm, with the 5 cm tiles
refined bilinearly, which is the same surface). Each added type draws from its own
random stream, so the seven v1 tile types are byte-identical with or without piles.

Pits on pile tiles are 0.5 m deep on every level. A first attempt kept the depth
curriculum (5 cm pits on level 0, 0.5 m from level 5): within 40 updates the pile
robots reached 93% of their targets on level 0 but fell as soon as they were
promoted, and sat at mean level 0.2, the same trap as v1's stepping stones. That
run is kept as `runs/kth-controller-v2-attempt1-shallowpits` on the server.

**Mix.** `terrain_weights` puts 40% of the robots on piles and 10% on stepping
stones (half on stone tiles); the other six types keep 8.3% each. Robots start on
random levels 0-9 (`max_start_level: 9`), so the old types continue near the levels
v1 had reached instead of climbing from level 0.

**Heightfield pile scenes (`--ground heightfield`).** `GoalEnv(scene, ground=...)`,
`calibrate.py` and `evaluate.py` can build a pile scene as one MuJoCo heightfield,
rasterized by `HeightGrid.from_scene` at 2.5 cm: exactly the height map the policy
senses, and the training terrain's representation. Box scenes stay the default.

**Run.** `configs/kth-controller-v2.json`, resumed from v1's update 3000 for 1000
updates (about 50 min on the H100 MIG slice; the 2.5 cm tiles did not change the
2.4 s per update). `configs/kth-controller-v1-control.json` is the fair control: the
same 1000 updates with the original seven types.

**Results (v2 = update 4000, `artifacts/controller_v2`).** All measured in native
CPU MuJoCo with deterministic actions; charts and full tables in `outputs/comparison/`
(`summary.md`). Hop tables: `python -m quadruped_nav.calibrate --workers 5`, 24
episodes per family and row, families open/rough/steps/slanted, short hops = 0.25-1 m.

| | v1 | control | v2 |
| --- | --- | --- | --- |
| Box scenes: hops arrived / ended in a fall | 26% / 65% | 32% / 65% | 63% / 26% |
| Box scenes: first hop off the start platform | 24% | 29% | 48% |
| Box scenes: short hops, row 0 (pooled) | 74% (60/81) | 81% (71/88) | 98% (173/176) |
| Box scenes: rows with >=80% short hops, unbroken from row 0 | none | 0 | 0-5 (row 5: 81%, 52/64) |
| Heightfield scenes: hops arrived / ended in a fall | 18% / 62% | 21% / 65% | 51% / 26% |
| Heightfield scenes: short hops, row 0 | 72% (42/58) | 63% (38/60) | 85% (126/148) |
| Heightfield scenes: rows with >=80% short hops, unbroken from row 0 | none | none | 0-3 (row 3: 82%, 71/87) |
| Fixed test, the seven v1 types | 82% | 83% | 88% |
| Fixed test, piles (levels 0/3/6/9) | 6% | 19% | 38% |

control = v1 plus the same 1000 updates on the original seven types
(`configs/kth-controller-v1-control.json`). Its tables were computed on the KTH
server (summaries in `outputs/calibration/summary_hops_v1_control*.json`); re-running
v2's box table there gave 63% arrived, 26% falls, 48% first hops and the same
rows (0-5 unbroken), so server and laptop results agree within noise.

- The gain comes from the pile tiles, not from extra training: the control improves
  on v1 only slightly.
- v1 is no better on heightfield scenes than on box scenes, so the crack artifact is
  not what limited it. Heightfield scenes are harder for v2 than box scenes (row 0:
  85% against 98%): a gap narrower than two 2.5 cm cells opens as a 5-7.5 cm notch.
- No forgetting on the seven v1 types (stepping stones 44% -> 56%, gaps 69% -> 88%,
  all others equal or better).
- Above row 5 (piles under 0.40 m, gaps over 5.6 cm) success falls off; on box
  scenes rows 12-19 stay at or below 50% for short hops and near 0% for hops over 1 m. All pile robots
  stayed on level 0 during the fine-tune (level-0 training success rose from 23% to
  59%), so the harder levels were barely trained.
- The policy's action std rose from 0.81 to 0.98 over the fine-tune (entropy bonus
  0.005), which works against precise foot placement; a lower entropy coefficient
  for fine-tuning is the obvious next change.

## Known limitations

- In MuJoCo, the Menagerie Go1 foot contact is soft (solimp width 2.3 cm). A
  foot pressed into a 2-4 cm gap between two boxes slides through it, which
  scripts/project_scene_check.py showed on the project scenes: a rear foot
  dropped into the 3 cm gap between the start platform and the first stone.
  Narrow gaps are therefore real hazards. The height map shows them, and
  HeightGrid.from_scene(fill_gaps=...) is off by default.

- Any robot whose state becomes physically impossible (above 20 m/s, 60 rad/s
  body rate or 100 rad/s joint speed, or non-finite) has its episode ended
  without the fall penalty. It is logged as unstable_per_10k; in the current run
  this is about 0.1-0.7 per 10,000 transitions.

- CPU MuJoCo and MuJoCo Warp handle heightfield contacts slightly differently:
  scripted shuffling drifts apart by about 7 cm over 2 s, whatever the solver
  iteration count. The planner evaluation runs on CPU with box terrain, so
  policies are re-tested there (goal_check --backend native, and evaluate.py).
- The heightfield bevels step edges over one 5 cm cell (2.5 cm on pile tiles and in
  heightfield pile scenes). A gap narrower than two cells therefore opens as a
  5-7.5 cm notch at the top: wider than the real 3 cm gap, so heightfield scenes are
  not easier than box scenes for the narrowest gaps.
- Only feet and trunk collide by default. Shins are kept off the terrain by the
  knee-clearance penalty, not by physics.
- Perfect height sensing (no perception noise model beyond 2 cm jitter), as in Lab 1.
