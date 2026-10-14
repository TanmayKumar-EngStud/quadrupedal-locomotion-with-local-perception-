# Implementation status — October 14, 2026

The executable project pipeline is implemented. KTH H100 10 GB MIG preflight, a 256-world PPO smoke run, checkpoint saving and GPU-checkpoint CPU execution have passed. The flat-ground training job has started; completed training and research validation are pending. No validated walking policy is included. See the KTH section of RUNBOOK.md for run locations and sample budgets.

## Delivered

- Native MuJoCo / batched MuJoCo Warp adapters, goal-reaching environment, 50 Hz position-target control and 12-joint PD configuration.
- Frozen actor/critic observation contract; 1.6 x 1.2 m terrain scan at 5 cm; 2.5 cm precomputed elevation raster.
- PPO clipped actor loss, value regression, GAE, time-limit bootstrapping, entropy, gradient clipping, KL stopping, running normalization and reflection consistency loss.
- Twenty-row terrain generation: flat, open, holes, moat, dead end, steps, rough heights, slalom walls, narrow bridge and tilted tops.
- Configurable Lab-1-inspired reward groups with per-term logging and goal-reaching adaptations.
- Atomic checkpoints, restart/warm-start, deterministic actor export and a downloadable bundle with manifest.
- Greedy, oracle A*, optimistic local-memory search, learned pile scorer L0-L3, DAgger and optional policy-gradient fine-tuning.
- Continuous-pose physics executor, local viewer, saved-layout paired benchmarks, Wilson success intervals and coarse measured-hop calibration.
- CPU tests, a GitHub Actions CPU workflow, GPU preflight and an experiment-config generator.

## Explicit differences from the proposal

| Proposal | Implemented choice / implication |
| --- | --- |
| mjlab task framework | Direct MuJoCo Warp + PyTorch. Same simulator family, different task/training integration; CUDA smoke execution was validated on the KTH H100 MIG allocation; long-run locomotion validation is pending. |
| Ray-cast scan | Perfect precomputed elevation raster, following the actual Lab 1 approach. Sampled heights are checked against MuJoCo rays on tilted tops. No dynamic-object sensing or real perception. |
| Per-world smooth curriculum | One global row/family per batch, refreshed at rollout boundaries. Promotion/demotion uses aggregate outcomes; uniform uses one sampled row. Document this in RQ1. |
| Symmetry augmentation | Actor reflection consistency regularizer, not augmented PPO likelihood samples. |
| Optimistic A* | Reverse Dijkstra cost-to-go on an optimistic occupancy grid (A* with zero heuristic), then eligible pile selection. Resolution-dependent; no completeness claim. |
| Grid visit memory with decay | Per-pile episode visit counts, without decay. This isolates the value of a simple explicit memory. |
| Fully calibrated transition model | Empirical family/row/length bins; absent bins fail rather than extrapolate. Needs adequate controller trials before use. |
| Detailed dead-end escape and seed statistics | Trajectories, success, repeats/loops, geometric pocket entry/escape and Wilson intervals are available. scripts/summarize_runs.py reports mean/std across independent training runs; interpretation still requires completed experiments. |

The original proposal/planner documents are preserved unchanged. This status file describes what the code actually does. A benchmark that passes in the abstract surrogate is not evidence of stable physics execution.

## Important limitations to address in experiments

- The Go1 task uses zero collision margins for Warp compatibility, with 100 solver iterations and 50 line-search iterations on both backends. Warp warns that capsule-cylinder pairs receive at most one contact.
- All parallel GPU worlds share terrain geometry within a batch; random poses, commands and policy noise provide diversity, with scene reseeding across batches. This may reduce sample efficiency.
- Random short targets are not guaranteed physically reachable on difficult layouts. Train on easy terrain first, inspect failures and tune weights/limits before increasing difficulty.
- Foot-contact/clearance reward signals use elevation proximity, not exact contact impulses. The simulator itself still resolves physical contacts.
- The abstract straight-line surrogate is a point-body model with approximate tilted-footprint support. Slalom walls remove navigation support under them, but no full-body clearance planning is performed.
- Smoke training intentionally uses only a handful of updates. Its output is unsuitable for a walking demonstration.
- Resume restarts environment state and is not bitwise reproducible across CPU/GPU or interrupted CUDA streams.
