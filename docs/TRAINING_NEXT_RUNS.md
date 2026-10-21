# Training the next controllers (v3 and v4, RQ1)

Updated October 20, 2026. RQ1 compares two controllers trained **from scratch for
8 hours each**, identical except for the size of the terrain irregularities (v3
moderate, v4 high), against v2. Background on the training recipe and every pitfall
found so far: [GOAL_REACHING_V3.md](GOAL_REACHING_V3.md). Server setup in general:
[RUNBOOK.md](RUNBOOK.md) sections 2-4.

## Update, October 21: fine-tune from v2 instead

v3 trained from scratch (`configs/kth-controller-v3.json`) finished its 8 hours but is
worse than v2 on every test (pile-scene hops 31% vs 86%, fixed test 66% vs 88%):
validation stopped improving after about 2 hours while the action noise drifted from
1.0 to 1.86, and pile robots stayed on the shallow-pit levels. The RQ1 runs are
therefore fine-tunes from v2's weights, as v2 was fine-tuned from v1:

| Config | Noise | Purpose |
| --- | --- | --- |
| `configs/kth-controller-v3-from-v2.json` | 0.5 | moderate-noise controller |
| `configs/kth-controller-v4-from-v2.json` | 1.0 | high-noise controller |
| `configs/kth-controller-v2-continued.json` | 0 | control: the same extra training without noise |

All three: v2's terrain mix, deep pile pits, start levels 0-9, entropy 0.001 (stops
the noise drift), `arrival_speed` and `feet_in_gap` on, targets 0.4-3 m, 3 hours each.
Launch all three in parallel (each about 3-4 GB of GPU memory), each with
`--resume runs/kth-controller-v2/update_004000.pt`:

```sh
for RUN in v3-from-v2 v4-from-v2 v2-continued; do
  setsid nohup .venv/bin/python -m quadruped_nav.train --config configs/kth-controller-$RUN.json \
      --resume runs/kth-controller-v2/update_004000.pt > runs/kth-controller-$RUN.log 2>&1 < /dev/null &
  sleep 5
done
```

The sections below describe the settings and the from-scratch v3 run.

**Later on October 21:** the team chose eight 2-hour variants on top of v2 instead (four
per GPU). They are described, with launch and evaluation steps, in
[TRAINING_VARIANTS.md](TRAINING_VARIANTS.md).

## What is implemented

| Setting (in `env` of the run config) | v3 | v4 | Meaning |
| --- | --- | --- | --- |
| `terrain_noise` | 0.5 | 1.0 | size of the per-element irregularities (0 = the regular v1/v2 terrain) |
| `terrain_variants` | 2 | 2 | independent tiles per level and type; each robot draws one at every reset |
| `pile_pit_ramp` | true | true | pile pits deepen from 5 cm (level 0) to 0.5 m (level 5), for runs from scratch |
| `terrain_weights` | 8/8/14/10/8/15/12/25 | same | share of robots per type (rough, slope, stairs up, stairs down, obstacles, stepping stones, gaps, piles) |
| `max_start_level` | 2 | 2 | from scratch: robots start on levels 0-2 |
| `target_distance` | 0.4-3 m | same | v1/v2 used 1-3 m, so the controller never practised short hops (planner hops are 0.25-2 m); targets pulled back inside the tile can be as close as 0.3 m, still outside the 0.25 m arrival circle |
| `rewards.arrival_speed` | -10 | same | see below |
| `rewards.feet_in_gap` | -20 | same | see below |

| Setting (top level of the run config) | v3 and v4 | Meaning |
| --- | --- | --- |
| `max_hours` | 8 | training stops after 8 hours, whatever the GPU speed (`updates` is only an upper limit) |
| `validation_seed` | 100 | the fixed test every 25 updates runs on unseen tiles from seed 100 (validation set) |

Configs: `configs/kth-controller-v3.json` and `configs/kth-controller-v4.json`
differ only in `terrain_noise` and `output`.

**Irregularities** (`src/quadruped_nav/rl/terrain_grid.py`). Every terrain element
gets its own truncated Gaussian deviations (clipped at 2.5 sigma, then to physical
limits). sigma = `terrain_noise` x the unit below; "one level's change" is how much
the nominal value moves from one level to the next.

| Terrain | Parameters per element | Unit (sigma at noise 1.0) | Limits |
| --- | --- | --- | --- |
| stepping stones, piles | width, length | one level's size change (3.3 cm / 4.2 cm) | 0.10 m to the lattice spacing |
| | height | min(2 cm, pit depth / 10) | stays far above the pit |
| | centre x, y in its lattice cell | half the level's gap | stays inside its cell (gaps may close) |
| | tilt about x, y | 6 degrees | at most 15 degrees; corners move at most pit / 8 |
| stairs up / down | riser height, tread depth (every step) | 1.3 cm (one level's change), 5 cm | riser 2-20 cm, tread at least 20 cm |
| obstacles | block width, length; height; tilt about x, y | 10 cm; 1.3 cm; 6 degrees | size 0.2-1.2 m, tilt at most 15 degrees |
| gaps | hole width, length, centre x, y | 5 cm | hole size 0.15-0.45 m |
| slope | slope along x and along y (per tile) | one level's change (2.5 degrees) | rise/run 0.02-0.55 |
| rough | unchanged (already random everywhere) | | |

See the effect: `python scripts/preview_terrain.py --pile-pit-ramp` writes
`outputs/terrain_preview.png` (3 m windows of every type at levels 0, 5, 9 and noise
0, 0.5, 1.0).

**Failure-targeted reward terms** (off by default, enabled in the v3 and v4 configs).
All reward weights and shaping parameters, with one comment each, are in
`reward_config()` at the top of `src/quadruped_nav/rl/rewards.py` (laid out like
Lab 1 C). A run config lists only what it changes, under `env.rewards` (weights) and
`env.reward_params` (e.g. `{"gap_deadband": 0.04}`); everything else keeps its default.

| Term | What it penalises | Threshold | Suggested weight |
| --- | --- | --- | --- |
| `arrival_speed` | crossing the arrival circle fast: (speed - 0.5 m/s)^2, paid once per arrival | 0.5 m/s | -10 (cancels the +10 bonus at 1.5 m/s) |
| `feet_in_gap` | each foot more than 3 cm below the highest terrain within 4 cm, in metres per second | 3 cm, capped at 10 cm | -20 |

Evidence: in planner runs on pile scenes, v2 crossed arrival circles at 1.0-1.9 m/s
(the command there is about 0.4 m/s), and in 10 of 12 falls a foot sat 10-37 cm deep
in a gap during the last second, against at most 2.6 cm in successful episodes.
v2 walking under the v3 rewards would lose about 3.0 per robot-second to
arrival_speed (against 4.3 gained from arrivals) and 0.2 to feet_in_gap; a robot that
follows its slowing command arrives at 0.5 m/s or less and pays nothing. In the probe
run, check both terms in `reward_terms`: they should be small next to `arrival`.

**Not included yet** (team plan, RQ1): re-spaced difficulty levels (Amedeo), the edge
and stumble penalties, the lower entropy bonus and the relaxed demotion rule
(Damien). If any of them is added, v3 and v4 must both be trained with it, so that
the noise stays the only difference between them.

## Steps to train v3 on the KTH GPU

Run these in a JupyterLab terminal on the GPU server, in the repository folder.

### 1. Get the code

The server checkout holds files that were uploaded by hand during development,
some of them untracked copies of files the branch now contains (for example
`configs/kth-controller-v2.json`). Git refuses to pull over untracked files, and a
plain `git stash` does not move them, so stash with `--include-untracked`. Ignored
folders (`runs/`, `artifacts/`, `.venv/`) are not touched.

```sh
cd "$HOME/RL&EAI/Project/Language-Guided-Quadrupedal-Navigation"   # or wherever the repository is
git stash push --include-untracked -m "server copies before v3"
git fetch origin
git checkout tanmay
git pull origin tanmay
git log --oneline -3                                               # the newest commit must be on top
ls configs/kth-controller-v3.json                                  # must exist
```

On a fresh machine instead: `git clone -b tanmay https://github.com/TanmayKumar-EngStud/Language-Guided-Quadrupedal-Navigation.git`.

### 2. Environment and GPU check

```sh
bash scripts/cloud_setup.sh          # installs uv if needed, uv sync --extra cloud, CPU/GPU parity check, 3-update GPU run
.venv/bin/python -m pytest -q        # all tests must pass
```

### 3. Probe run (about 5 minutes)

```sh
.venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v3.json \
    --output runs/v3-probe --updates 40
```

Check the last lines of the output (and `runs/v3-probe/metrics.jsonl`):

- no `Traceback` or `NaN`;
- `samples_per_second` (about 30,000-40,000 on the 10 GB slice; a larger GPU should be faster);
- in `metrics.jsonl`: `unstable_per_10k` below about 2, `contact_overflow.worlds_at_constraint_capacity` 0.

Then delete the probe: `rm -r runs/v3-probe`.

### 4. Launch the 8-hour run

```sh
setsid nohup .venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v3.json \
    > runs/kth-controller-v3.log 2>&1 < /dev/null &
```

`setsid nohup ... < /dev/null &` keeps the run alive when the browser or terminal
closes. The run stops by itself after 8 hours (`max_hours`) and saves
`runs/kth-controller-v3/last.pt` and `update_XXXXXX.pt`. To train v4 at the same time
on a GPU with enough memory (each run uses about 3-4 GB), start the same command with
`kth-controller-v4`. On the 24 GB pod, keep CPU jobs beside training to 5 workers or
fewer.

### 5. Monitor

```sh
tail -n 3 runs/kth-controller-v3.log                       # one line per update
.venv/bin/python scripts/plot_training.py runs/kth-controller-v3   # dashboard figures (needs matplotlib)
```

| Field in `metrics.jsonl` | Healthy | Problem seen before |
| --- | --- | --- |
| `level_by_type` | rises for every type over the hours | stuck at 0 (v2 piles, v1 stepping stones) |
| `target_success_rate` | rises within the first few hundred updates | near 0: robots stand still |
| `mean_velocity_toward_target` | towards 0.5-1 m/s | near 0: "standing still pays" |
| `unstable_per_10k` | below about 2 | robots blowing up |
| `ppo.learning_rate` | mostly 1e-4 to 1e-3, short dips are normal | stuck at 1e-5 |
| `ppo.action_std` | slowly falling | rising: too much exploration |

`runs/kth-controller-v3/evaluation.jsonl` holds the fixed test on the validation
tiles every 25 updates. If the run crashes, resume it:
`--resume runs/kth-controller-v3/last.pt` with the same config. The 8-hour clock
restarts on resume, so lower `max_hours` by the time already trained.

### 6. Export and download

```sh
.venv/bin/python -m quadruped_nav.train --export runs/kth-controller-v3/last.pt --bundle artifacts/controller_v3
```

Download `artifacts/controller_v3.zip` (JupyterLab file browser: right click,
Download), then on the laptop:

```sh
unzip artifacts/controller_v3.zip -d artifacts/controller_v3     # manifest.json holds SHA-256 hashes to check
```

### 7. Evaluate (laptop, CPU)

Use the validation seed (100) for any choice between checkpoints, and the test seed
(12345, the default) only for the final numbers. The same terrain for every
controller comes from `--terrain-from`:

```sh
C=artifacts/controller_v3/controller.pt
# test matrix: repeat for v2, v3, v4 and noise 0, 0.5, 1.0, 1.5
uv run --extra train python -m quadruped_nav.goal_check --checkpoint $C \
    --terrain-from configs/kth-controller-v3.json --terrain-noise 0.5 --output outputs/evaluation/fixed_v3_noise0.5.json
# hop tables on the pile scenes
uv run --extra train python -m quadruped_nav.calibrate --workers 5 --checkpoint $C --output outputs/calibration/hops_v3.json
uv run --extra train python -m quadruped_nav.calibrate --workers 5 --checkpoint $C --ground heightfield \
    --output outputs/calibration/hops_v3_heightfield.json
# side-by-side clip against v2 on irregular piles
uv run --extra train python scripts/render_compare.py tile --checkpoints artifacts/controller_v2/controller.pt $C \
    --labels v2 v3 --terrain piles --level 2 --terrain-noise 0.5 --output outputs/comparison/clips/piles_v2_v3.mp4
```

Add v3 and v4 to `outputs/comparison/make_charts.sh` and run it for the charts.
Freeze the controller with the best hop success on the pile rows used for RQ2.
