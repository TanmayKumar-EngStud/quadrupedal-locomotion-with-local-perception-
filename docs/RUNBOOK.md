# Training and local execution

Implementation date: October 14, 2026. Read `IMPLEMENTATION_STATUS.md` before interpreting results. The code can train and evaluate; a smoke checkpoint is NOT a walking policy.

## 1. Local checks

From the repository root, with Python 3.12 and uv:

```sh
uv sync --extra train --frozen
uv run --extra train pytest -q
uv run --extra train quad-train --config configs/smoke.json --output runs/my-smoke
uv run --extra train quad-evaluate --checkpoint runs/my-smoke/last.pt --family open --seconds 2
```

Use a fresh run directory each time, or resume explicitly. `uv run python simulate.py --open` still runs the abstract replay without requiring a trained controller. Its JSON now also contains the optimistic planner episode.

## 2. GPU setup and preflight (KTH JupyterHub H100 MIG slice)

```sh
bash scripts/cloud_setup.sh                                  # CUDA check, CPU/GPU parity, 3-update GPU run
uv pip install --python .venv/bin/python matplotlib          # for scripts/plot_training.py
.venv/bin/python scripts/bench_physics.py --worlds 4096      # physics throughput
```

The parity check moves four robots with scripted joint motions on the training
terrain in both CPU MuJoCo and MuJoCo Warp. Heightfield contacts differ
slightly between the two (Warp gives box-heightfield pairs one contact), so a
drift of a few centimetres over 2 s is expected. The check fails above 0.15 m.
Never interpret a preflight checkpoint as trained.

## 3. Training (goal-reaching controller v3)

```sh
nohup .venv/bin/python -m quadruped_nav.train --config configs/kth-goal-v3.json > runs/kth-goal-v3-seed0.log 2>&1 &
.venv/bin/python scripts/plot_training.py runs/kth-goal-v3-seed0      # any time during training
```

The run uses 4,096 robots on a grid of 70 terrain tiles (7 types x 10
difficulty levels), with a per-robot curriculum and random targets 1-3 m away.
It runs 3,000 PPO updates of 24 steps each, about 295 M transitions. On the
H100 1g.10gb slice it trains at about 30,000-40,000 transitions/s, so allow
roughly 2.5 h. See GOAL_REACHING_V3.md for the design.

Outputs in the run directory:
- `metrics.jsonl`: one record per update with every training metric.
- `evaluation.jsonl`: the fixed terrain test, every 25 updates.
- `last.pt`, updated every 25 updates, and `update_000250.pt` and later
  snapshots every 250 updates.
- `figures/`, written by plot_training.py.

`--resume runs/<run>/last.pt` continues a stopped run with the same config.

## 4. Export and download

After training, evaluate held-out episodes and select a checkpoint based on navigation/locomotion metrics, not reward alone. Export a bundle:

```sh
uv run --extra cloud quad-train --export runs/kth-goal-v3-seed0/last.pt --bundle artifacts/controller
.venv/bin/python -m quadruped_nav.goal_check --checkpoint runs/kth-goal-v3-seed0/last.pt --backend native   # CPU re-test
```

The ZIP contains `controller.pt` (actor, critic, normalizers, optimizer and config), `actor.ts` (deterministic TorchScript actor with normalization), and `manifest.json` (contract, versions, source revision and SHA-256 hashes). Keep all three together. It does not contain robot meshes; use this repository locally.

From your LOCAL terminal (replace user/host/path with the cloud provider's details):

```sh
scp user@GPU_HOST:/workspace/Language-Guided-Quadrupedal-Navigation/artifacts/controller.zip ./artifacts/
unzip artifacts/controller.zip -d artifacts/controller
```

An SSH key may be required (`scp -i /path/to/key ...`). Alternatively use the provider's file download UI. Do not share keys or tokens in the repository.

## 5. Local inference

The trained controller v1 (update 3000) is unpacked in `artifacts/controller_v1/` (controller.pt, actor.ts, manifest.json with SHA-256 hashes).

```sh
.venv/bin/python watch.py --terrain stairs_up --level 5            # one training terrain, live viewer
.venv/bin/python watch.py --terrain stepping_stones --level 0 --headless --seconds 30
uv sync --extra train --frozen
uv run --extra train python play.py --checkpoint artifacts/controller/controller.pt --planner optimistic --family open --row 0
uv run --extra train quad-evaluate --checkpoint artifacts/controller/controller.pt --planner optimistic --family dead_end --row 0 --output outputs/evaluation/dead-end.json
```

`play.py` handles macOS's `mjpython` viewer requirement. Headless evaluation needs no renderer. Local inference uses CPU MuJoCo and the SAME feature construction, action scale and PD gains stored in the checkpoint. CPU/GPU solver differences still require paired validation. This is simulator deployment, not deployment to a physical Go1.

## 6. Learned planner

```sh
uv run --extra train quad-learn-planner --variant L0 --rounds 20 --episodes 8 --seed 0 --output runs/planner-L0-seed0
uv run --extra train quad-learn-planner --variant L3 --rounds 20 --episodes 8 --rl-rounds 20 --seed 0 --output runs/planner-L3-rl-seed0
```

Repeat L0/L1/L2/L3 with seeds 0,1,2, with/without optional RL. L1 uses a GRU, L2 per-pile visit counts, L3 both. No full-map feature reaches the network. DAgger aggregates labels at states visited by the mixed student/expert rollout. Optional REINFORCE uses full-map walkable-distance progress for training rewards only. Visit counts are not penalised: backtracking can be necessary.

## 7. Benchmarks and calibration

```sh
uv run --extra train quad-benchmark --create --count 20 --split test --row 5 --manifest benchmarks/test-row5.json
uv run --extra train quad-benchmark --manifest benchmarks/test-row5.json --output outputs/benchmarks/abstract
uv run --extra train quad-benchmark --manifest benchmarks/test-row5.json --checkpoint artifacts/controller/controller.pt --output outputs/benchmarks/physics
uv run --extra train quad-benchmark --manifest benchmarks/test-row5.json --planners learned --planner-checkpoint runs/planner-L3-seed0/planner.pt --output outputs/benchmarks/learned
uv run --extra train python -m quadruped_nav.calibrate --checkpoint artifacts/controller_v1/controller.pt --workers 5 --output outputs/calibration/hops_v1.json
```

Use the SAME saved scene list and frozen controller for planner comparisons. `--count` is per terrain family. Preserve impossible/unreachable cases as failures and report their count; do not quietly remove them after seeing results. Generated train/validation/test seed ranges do not overlap. Wilson intervals summarize success proportions. Pocket entry/exit uses the generator's rectangular region and 5 cm path samples; escape also requires goal success. Use scripts/summarize_runs.py with separate summary files to report across-training-seed mean/std.

Calibration (the hop-success table, gate G2) sends the controller along chains of up to 6 planner-eligible hops (the abstract simulator's action mask) on open, rough, steps and slanted scenes, all 20 rows, 24 episodes each, with the executor's speed, time budget and arrival radius. It saves every hop and family/row/length bins with Wilson intervals; about 3 minutes with 5 workers on a laptop. `quad-benchmark --calibration outputs/calibration/hops_v1.json ...` samples those probabilities in abstract episodes. Families that only remove piles (holes, moat, dead end, bridge, slalom, flat) use the open-field bins, and an unmeasured row uses the next harder measured row; a missing bin fails explicitly. This coarse table is not the proposal's full multidimensional gap/height/slant fit.

Controller v1 result (15 October 2026): at most 77% hop success on row 0, below 30% for 1-2 m hops from row 3, 65% of hops end in a fall. It is not yet usable for physics planner comparisons on the pile scenes; see the revised proposal.

## Acceptance before the report/demo

1. GPU preflight passes, warnings/capacity errors are absent, throughput and memory are recorded.
2. On flat ground, the policy reaches short targets without repeated falls; confirm a trajectory/video, not just reward.
3. Repeat on sparse piles before increasing rows or interpreting planner comparisons.
4. A downloaded checkpoint produces the same CPU actions as the exported actor for saved observations; compare native/Warp episode statistics.
5. Run held-out paired layouts with oracle, greedy, optimistic and learned planners. Record falls, timeouts, loop behaviour and actual paths.
6. Report failures and scope changes. Do not claim robot competence, a successful curriculum or the research hypotheses from smoke tests alone.

## KTH H100 10 GB MIG allocation

The observed KTH allocation is one 10 GB MIG slice, not the whole H100. Use `configs/kth-flat.json` for 256 parallel worlds, 2,000 flat-ground updates (16,384,000 transitions), and checkpoints every 10 updates. The terrain configuration uses the same batch size and 4,000 updates (32,768,000 transitions), so it has a smaller sample budget than the default 1,024-world experiment. Do not mix these budgets in a curriculum comparison.

After GPU preflight and a short PPO test pass, start the flat run in a persistent terminal or with `nohup`. Save output to `runs/kth-flat.log`; checkpoints go to `runs/kth-flat-seed0`. Closing a browser tab does not necessarily stop a detached process, but stopping or recycling the KTH Jupyter server does. Keep backups of checkpoints outside that server.

```sh
uv run --extra cloud quad-train --config configs/kth-flat.json
# Only after inspecting a successful flat policy:
uv run --extra cloud quad-train --config configs/kth-terrain.json --warm-start runs/kth-flat-seed0/last.pt
```

GoalEnv clears collision margins on both CPU and GPU because Warp MULTICCD rejects the Go1 asset's original nonzero margins. Multi-contact collision remains enabled. Warp currently warns that capsule-cylinder pairs receive at most one contact; this must be considered in native/GPU validation.

### First KTH training launch — October 14, 2026

The first flat-ground job is detached from the browser terminal and uses `configs/kth-flat.json`. Its PID is recorded in `runs/kth-flat.pid`, its log in `runs/kth-flat.log`, and its latest checkpoint in `runs/kth-flat-seed0/last.pt`. A checkpoint was verified at update 10. Initial throughput was about 6,000 samples/s and observed MIG memory usage 962 MiB; these are not steady-state benchmark results. The completed 256-world smoke run had no solver-limit or capacity errors, and its checkpoint ran in native CPU physics.

After successful training, the launch wrapper exports `artifacts/kth-flat-controller.zip`. When the wrapper finishes, `runs/kth-flat.exit` records its exit code (`0` means both training and export succeeded). The bundle is not automatically downloaded to the laptop. A saved/exported checkpoint still needs movement evaluation before obstacle training. The manual PDF predates these GPU checks; this runbook and IMPLEMENTATION_STATUS.md contain the current validation status.

To inspect progress from the project terminal:

```sh
tail -5 runs/kth-flat.log
ls -lh runs/kth-flat-seed0/last.pt
# When present, inspect completion status:
cat runs/kth-flat.exit
```

Do not start a duplicate run in the same output folder. Keep the KTH server running; a server shutdown/restart interrupts the job. If interrupted, resume with the same flat config and `--resume runs/kth-flat-seed0/last.pt` after confirming the previous process has stopped.
