# Training the next controllers (v3 and v4, RQ1)

Written October 19, 2026. This is the step-by-step guide for the RQ1 runs in the
proposal: two controllers trained **from scratch for 8 hours each**, identical except
for the amount of Gaussian noise in the training-terrain parameters (v3 moderate,
v4 high), and compared against v2. Background on the training recipe and on every
pitfall found so far: [GOAL_REACHING_V3.md](GOAL_REACHING_V3.md). Server setup:
[RUNBOOK.md](RUNBOOK.md) sections 2-4.

Steps 1-2 are code changes that do not exist yet (owners from the proposal). Steps
3-8 use commands that work today.

## 1. Terrain changes (Amedeo)

All terrain lives in `src/quadruped_nav/rl/terrain_grid.py`. Each tile generator
(`rough_tile`, `stairs_tile`, `stepping_stones_tile`, `gaps_tile`, `piles_tile`, ...)
takes a difficulty `d` in [0, 1]; level `L` uses `d = L / 9`.

- **Re-spaced levels.** Change how `d` maps to step height, stone size, gap width
  and pit depth so each level is a step robots can actually climb. Evidence: in v2
  every pile robot stayed on level 0, and v1's stepping stones reached level 0.35.
- **Gaussian parameter noise.** Add a noise setting to `EnvConfig`
  (`src/quadruped_nav/rl/config.py`), e.g. `terrain_noise` in units of one level's
  change: 0.5 for v3, 1.0 for v4. Perturb each tile's parameters (riser height,
  tread depth, slope, stone size, gap width, pit depth, roughness, block height), and
  individual steps and stones within a tile.
- **More tile variants.** The grid builds one tile per (level, type) once, at start-up.
  With noise, one tile per cell gives only one noise sample per cell, so build
  several variants per (level, type), e.g. extra copies of each column. Memory is not
  the limit: 80 tiles at 2.5-5 cm cells use about 40 MB.
- **Held-out test terrain.** `goal_check` already builds its terrain from seed
  12345, while training uses `seed: 0`, so its tiles are never trained on. Build the
  noisy test set the same way, with a seed no training run uses.
- **Warning for from-scratch runs.** v2 has 0.5 m pits on every pile level, which
  worked because v2 started from a walking v1. From scratch, deep pits on the easiest
  level stalled learning before (phase 1 of v1: stepping-stone success near 0%).
  Give piles a pit-depth ramp at the lowest levels, or check in the probe run (step 4)
  that pile robots leave level 0.

## 2. Reward and PPO changes (Damien)

- **New reward terms** (edge penalty: stance feet near a drop; stumble penalty: toes
  catching stair risers; optional arrival-speed penalty). For each term: add a
  function to `src/quadruped_nav/rl/rewards.py` and its name to `DENSE`, add its
  weight to `default_rewards()` in `config.py` (unknown names are rejected), and pass
  any new quantity it needs through the state built in `GoalEnv.step`
  (`src/quadruped_nav/rl/env.py`).
- **Lower entropy bonus.** `entropy` in the run config (currently 0.005); try 0.001.
  During the v2 fine-tune the action std rose from 0.81 to 0.98.
- Keep a change only if a 30-minute probe (step 4) shows it reduces the failure it
  targets without slowing the robot (`mean_velocity_toward_target`).

## 3. Local checks (any laptop, CPU)

```sh
uv sync --extra train --frozen
uv run --extra train pytest -q                                   # all tests must pass
uv run --extra train quad-train --config configs/smoke.json --output runs/my-smoke
uv run --extra train python scripts/render_rollout.py --checkpoint artifacts/controller_v2/controller.pt \
    --terrain piles --level 3 --output outputs/rollouts/check.gif   # look at the new terrain
```

Add a test for every new terrain or reward feature (see `tests/test_training.py`).

## 4. GPU preflight and 30-minute probes (KTH JupyterHub)

Bring the server checkout to the branch first. It holds files that were uploaded by
hand during development, so run `git stash && git pull` there (the stash keeps those
copies), or make a fresh clone and run `scripts/cloud_setup.sh` in it. Then:

```sh
bash scripts/cloud_setup.sh                               # CUDA check, CPU/GPU parity, short GPU run
.venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v3.json \
    --output runs/v3-probe --updates 600                  # about 30 min
```

Check in `runs/v3-probe/metrics.jsonl` (or `python scripts/plot_training.py runs/v3-probe`):

| Field | Healthy | Problem seen before |
| --- | --- | --- |
| `unstable_per_10k` | below about 2 (v1, v2: 0.1-1.5) | robots far from a single large heightfield blew up (fixed: one heightfield per tile) |
| `level_by_type` | rises for every type | stuck at 0: no promotion (v2 piles, v1 stepping stones) |
| `ppo.learning_rate`, `ppo.kl` | learning rate mostly 1e-4 to 1e-3 (short dips are normal) | stuck at 1e-5: normalizer updated mid-rollout |
| `ppo.action_std` | slowly falling | rising: too much entropy bonus |
| `mean_velocity_toward_target` | towards 0.5-1 m/s | near 0: "standing still pays" (see GOAL_REACHING_V3.md) |
| `samples_per_second` | about 30,000-40,000 on the 10 GB slice | |

Also time an update: `collect_seconds + learn_seconds` (about 2.4 s on the 10 GB
slice; about 3 s with the fixed test every 25 updates).

## 5. Configs for the two runs

Copy `configs/kth-controller-v2.json` to `configs/kth-controller-v3.json` and
`configs/kth-controller-v4.json` and change:

- `output`: `runs/kth-controller-v3` / `runs/kth-controller-v4`;
- the noise setting from step 1 (0.5 / 1.0); everything else identical, same `seed`;
- `entropy` and reward weights chosen in step 2;
- `max_start_level`: 2 for from-scratch runs (9 was for fine-tuning a walking policy);
- `updates`: fill the 8-hour budget. 8 h = 28,800 s; at about 3 s per update that is
  about 9,500 updates. Use the update time measured in the probe on the partition you
  actually get. Two runs sharing one GPU each run at about half speed.

## 6. Launch (from scratch: no `--resume`)

```sh
setsid nohup .venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v3.json \
    > runs/kth-controller-v3.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python -m quadruped_nav.train --config configs/kth-controller-v4.json \
    > runs/kth-controller-v4.log 2>&1 < /dev/null &
```

`setsid nohup ... < /dev/null &` keeps the jobs alive when the browser terminal
closes. The output folder must not already contain `last.pt`. Checkpoints:
`last.pt` every 25 updates, `update_XXXXXX.pt` every 250. The pod has 24 GB RAM: keep
CPU jobs running beside training (e.g. calibration) to 5 workers or fewer, or the pod
restarts and kills everything.

## 7. Monitor during the 8 hours

- `tail -f runs/kth-controller-v3.log` prints one line per update.
- `runs/<run>/evaluation.jsonl` gets the fixed terrain test every 25 updates.
- `python scripts/plot_training.py runs/kth-controller-v3` writes the dashboard figures.
- If a run crashes, resume it with `--resume runs/<run>/last.pt` and the same config.

## 8. Export, download, evaluate, freeze (19-21 Oct)

```sh
# on the server
.venv/bin/python -m quadruped_nav.train --export runs/kth-controller-v3/last.pt --bundle artifacts/controller_v3
# download artifacts/controller_v3.zip (JupyterLab file browser: right click, Download), then locally:
unzip artifacts/controller_v3.zip -d artifacts/controller_v3   # manifest.json holds SHA-256 hashes to check
C=artifacts/controller_v3/controller.pt
uv run --extra train python -m quadruped_nav.goal_check --checkpoint $C \
    --terrain-types rough slope stairs_up stairs_down obstacles stepping_stones gaps --output outputs/evaluation/fixed_v3_v1types.json
uv run --extra train python -m quadruped_nav.goal_check --checkpoint $C --terrain-types piles --output outputs/evaluation/fixed_v3_piles.json
uv run --extra train python -m quadruped_nav.calibrate --workers 5 --checkpoint $C --output outputs/calibration/hops_v3.json
uv run --extra train python -m quadruped_nav.calibrate --workers 5 --checkpoint $C --ground heightfield --output outputs/calibration/hops_v3_heightfield.json
```

Then add v3 and v4 to `outputs/comparison/make_charts.sh` and run it, and render
side-by-side clips against v2 with `scripts/render_compare.py`. Freeze the controller
with the best hop success on the pile rows used for RQ2, and re-run the calibrated RQ2
experiments with its hop table.
