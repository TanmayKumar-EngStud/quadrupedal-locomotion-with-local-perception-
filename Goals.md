# Project goals

**Day 1:** October 5, 2026. **Project deadline:** October 16, 2026.

Track substantive project checkpoints rather than administrative activity. Add a separate dated section and table for each working day. Keep earlier days as history; add a continuation row when unfinished work carries forward. Status cells contain only `[ ]` for incomplete work or `[X]` for completed work. Record actual time with its measurement type; use `Not recorded` when unknown and `—` when not started. Individual agent elapsed time is not the total team's working time.

## October 5, 2026 — Day 1

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Build the initial abstract simulator and demonstrate an oracle/greedy episode with a trajectory and log | Tanmay | [X] | 14 min agent elapsed (implementation, tests and viewer inspection; publishing excluded) |
| Run the mjlab/Go1 smoke test on the H100 allocation; measure throughput and GPU memory; check pile representation | Amedeo | [ ] | — |
| Establish controller training configuration with Amedeo, including symmetry and chained-target conventions | Damien | [ ] | — |

The MuJoCo scene/viewer predates this workday. The initial abstract simulator now runs oracle/greedy episodes and exports an interactive trajectory replay and JSON logs; see [implementation and validation](docs/ABSTRACT_SIMULATOR.md). It uses provisional geometric movement, not a trained controller. The remaining Day-1 goals stay unchecked.

## October 14, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Implement controller/planner training, expanded terrains, checkpoint export and local physics execution; validate the CPU pipeline | Tanmay | [X] | 33 min agent elapsed (implementation, smoke runs, tests and documentation; publishing excluded) |
| Deliver the project manual with architecture, concepts, important functions and cloud-to-local instructions | Tanmay | [X] | Included in pipeline work above |
| Verify H100 GPU physics, PPO updates and GPU-checkpoint CPU loading | Tanmay | [X] | Not recorded |
| Train and validate the locomotion controller on the KTH GPU | Tanmay, Amedeo, Damien | [ ] | In progress; job elapsed time is separate from contributor time |
| Freeze a validated controller, calibrate transitions and complete held-out physics experiments | Tanmay, Amedeo, Damien | [ ] | — |

CPU verification: 56 tests passed locally and on the KTH Linux server. H100 10 GB MIG validation passed: standing CPU/GPU maximum qpos difference 3.42e-6; 256-world PPO smoke run completed 16,384 transitions and saved a checkpoint; that checkpoint loaded and ran in native CPU physics. The final batch smoke log had no solver-limit, capacity or traceback errors. Warp warns that capsule-cylinder collisions provide at most one contact.

Cloud fixes use zero collision margins, an explicitly synchronized Warp-owned CUDA stream, and solver budgets of 100 iterations / 50 line-search iterations. Reproducible KTH configs are committed. The flat training run is active at `runs/kth-flat-seed0`, configured for 2,000 updates / 16,384,000 transitions; update 10 checkpoint creation was verified. Early throughput was about 6,000 samples/s and allocated GPU usage about 962 MiB. These are startup observations, not a final performance benchmark or evidence of successful walking. Export is queued after successful completion; terrain training has not started. See [runbook](docs/RUNBOOK.md) and [implementation status](docs/IMPLEMENTATION_STATUS.md).

## October 15, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Train and validate the locomotion controller on the KTH GPU (continued from October 14): controller v1, update 3000 | Tanmay | [X] | Not recorded |
| Measure the v1 hop-success table on the pile scenes | Tanmay | [X] | Not recorded |
| Fine-tune controller v2 on pile terrain and a control run; compare v1, v2 and control | Tanmay | [X] | Not recorded |
| Check both lines of feet in the planners' hop rule | Tanmay | [X] | Not recorded |
| Freeze a validated controller, calibrate transitions and complete held-out physics experiments | Tanmay, Amedeo, Damien | [ ] | — |

Controller v1 (3,000 PPO updates, 295M transitions) reaches 86% of the fixed-test targets on the KTH GPU and 82% in CPU MuJoCo. On the pile scenes 26% of its hops arrive. Controller v2 (v1 + 1,000 updates with a pile terrain type) reaches 63% of hops and 88% / 38% on the fixed test (seven original types / piles); the control run (same 1,000 updates, no piles) reaches 32%. With the corrected hop rule, v2 reaches 86% of hops on pile rows 0-9 (80% under the old rule). Bundles are on the KTH server and in `artifacts/` locally (not in Git). See [GOAL_REACHING_V3.md](docs/GOAL_REACHING_V3.md) and [the comparison](outputs/comparison/summary.md). 64 tests pass.

## October 19, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Write the step-by-step guide for the v3/v4 training runs (RQ1) | Tanmay | [X] | Not recorded |
| Re-spaced terrain levels and Gaussian terrain noise (RQ1) | Amedeo | [ ] | — |
| Edge and stumble penalties, lower entropy bonus (RQ1) | Damien | [ ] | — |
| Train v3 and v4 for 8 hours each | Tanmay, Amedeo, Damien | [ ] | — |

The guide is [TRAINING_NEXT_RUNS.md](docs/TRAINING_NEXT_RUNS.md).

## October 20, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Gaussian terrain irregularities per terrain element, tile variants, 8-hour time budget and validation tiles; v3/v4 configs | Tanmay | [X] | Not recorded |
| Re-spaced terrain levels (RQ1, continued from October 19) | Amedeo | [ ] | — |
| Train v3 for 8 hours on the KTH GPU | Tanmay | [ ] | — |
| Train v4 for 8 hours on the KTH GPU | Tanmay, Amedeo, Damien | [ ] | — |

`terrain_noise` adds per-element Gaussian deviations (width, length, height, centre position, tilt; risers and treads for stairs) to every training terrain except rough ground; 0.5 for v3, 1.0 for v4. With noise 0 the terrain is unchanged from v1/v2. Steps to run: [TRAINING_NEXT_RUNS.md](docs/TRAINING_NEXT_RUNS.md). The v3/v4 configs also enable two failure-targeted penalties (`arrival_speed` -10 above 0.5 m/s at arrival, `feet_in_gap` -20 for feet more than 3 cm below the neighbouring support) and train on targets 0.4-3 m away. Validated with unit tests (69 pass), a terrain preview and CPU smoke runs; not yet run on the GPU.

## October 21, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Train v3 for 8 hours on the KTH GPU (from scratch; continued from October 20) | Tanmay | [X] | 8 h GPU run (elapsed) |
| Evaluate v3 against v2 (hop tables, fixed tests at noise 0, 0.5, 1.0) | Tanmay | [X] | Not recorded |
| Fine-tune configs from v2 at noise 0.5, 1.0 and a control | Tanmay | [X] | Not recorded |
| Eight 2-hour training variants on top of v2: 49-sample foot scan, noise ramp, relaxed curriculum, edge penalty; configs, viewer markers, guide | Tanmay | [X] | Not recorded |
| Train variants 5-8 for 2 hours on Amedeo's KTH GPU | Amedeo | [ ] | — |
| Train variants 1-4 for 2 hours on Tanmay's KTH GPU and evaluate them against v1 and v2 | Tanmay | [X] | 2 h GPU runs (elapsed) |
| Continue variant 1 as variants 5 and 8 for 6 hours (8192 robots per run, 4 tile variants, larger validation set) | Tanmay | [ ] | — |

v3 from scratch is worse than v2 on every test (pile-scene hops 31% vs 86%), so the next runs start from v2. The eight variants, their configs, and the launch and evaluation steps are in [TRAINING_VARIANTS.md](docs/TRAINING_VARIANTS.md); watch.py and play.py now draw the terrain samples each controller senses. Validated with unit tests (74 pass) and a 2-update CPU run of every variant config; not yet run on the GPU.

Variants 1-4 ran for 2 hours each on Tanmay's GPU; variant 3 (noise 0.5 + edge penalty) is the best so far (94% of pile-scene hops on box ground against v2's 86%). Variants 5 and 8 now continue from variant 1's 49-point network for 6 hours with 8192 robots per run, four tile variants and a 640-robot validation set (round 2 in [TRAINING_VARIANTS.md](docs/TRAINING_VARIANTS.md)).

## October 22, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Train variants 5 and 8 for 6 hours from variant 1 (continued from October 21) | Tanmay | [X] | 6 h GPU runs (elapsed) |
| Fix the planner loop on pile scenes: 0.20 m pile arrival, hops judged from the pile centre, no bounce-back | Tanmay | [X] | Not recorded |
| Evaluate variants 5 and 8 on the larger test set against v2, variant 1 and variant 3 | Tanmay | [X] | Not recorded |
| Foot-clearance reward (Lab 1 C) and the round 3 config: variant 5 + edge penalty + foot clearance | Tanmay | [X] | Not recorded |
| Train variant 9 (variant 5 + edge penalty + foot clearance) for 6 hours | Tanmay | [X] | 6 h GPU run (elapsed, resumed once after a server stop) |
| Evaluate variant 9 on the larger test set against v2 and variants 1, 3, 5, 8 | Tanmay | [X] | Not recorded |

Planner loop: a robot standing on the seam between two piles counted as arriving at both (0.25 m arrival circle against 0.23 m half spacing on row 5), so the optimistic planner bounced between them. With the fix, variant 3 reaches the goal on 6 of 10 test scenes instead of 1 (open rows 3/5/7, holes, rough, slanted, moat, dead end); the remaining failures are falls. Hop tables are measured as before (`observe_pose(snap=False)`).

## October 26, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Share the trained controllers in the repository (v1, v2, v3, variant 3, variant 9) | Tanmay | [X] | Not recorded |
| Heading-alignment reward (face the direction of travel) and the variant 10 config | Tanmay | [X] | Not recorded |
| Train variant 10 (variant 9 + heading alignment 0.5) for 6 hours: the server stopped at 5 h; the robot still walks backwards 57% of the time | Tanmay | [X] | 5 h |
| Train variant 10 again with heading alignment 0.7 for 8 hours, with a keep-alive so the hub does not stop the server | Tanmay | [ ] | Not recorded |
| Pile-scene family `mix` and per-pile irregularities (`--irregularity`) for play.py and the hop tables | Tanmay | [X] | Not recorded |
| Rebuild `mix` as seven family sections (rough, slanted, steps, holes, bridge, moat, slalom), order shuffled by seed | Tanmay | [X] | Not recorded |

The controller bundles in `artifacts/` are now tracked (zips and the smoke-test bundle stay ignored); `artifacts/README.md` maps each folder to its name in [TRAINING_VARIANTS.md](docs/TRAINING_VARIANTS.md). `controller_v5` holds variant 3.

## October 27, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Train variant 10 with heading alignment 0.7 for 8 hours (continued from October 26) | Tanmay | [X] | 8 h GPU run (elapsed) |
| Evaluate variant 10 (heading 0.7) on the larger test set and the heading check against variant 9 | Tanmay | [X] | Not recorded |
| Add variant 10 to the round 3 charts, TRAINING_VARIANTS.md and `artifacts/controller_v10` | Tanmay | [X] | Not recorded |
| Dead-end planner comparison in physics (RQ2): variant 10, row 3, pockets 1, 2 and 3 m, greedy / optimistic / oracle, 6 seeds each | Tanmay | [X] | Not recorded |

Variant 10 (heading 0.7, 8 h) is the best controller so far: 98% of box-ground pile hops (variant 9: 96%), 84% on heightfield ground (81%), and 88.7% on the fixed test (87.5%). The heading reward still had no effect: it walks backwards 57% of the time, like variant 9.

Dead ends with variant 10: greedy reaches the goal in 0 of 18 episodes, optimistic in 12 of 18 and the oracle in 16 of 18. Optimistic paths are 1.28, 1.45 and 1.57 times the oracle's for 1, 2 and 3 m pockets. Greedy loops at the end of the pocket until it falls or times out; the oracle's two failures were falls in the first 4 s.

## October 28, 2026

| Goals | Contributor | Status | Time taken |
| --- | --- | --- | --- |
| Rewrite README.md: results first (GIFs, charts, controller table, terrain design), then a brief reproduce section | Tanmay | [X] | Not recorded |
| README GIFs (variant 10 escaping an irregular dead end; variant 10 on irregular pile training tiles) and figures in `docs/media` | Tanmay | [X] | Not recorded |
| `scripts/dead_end_sweep.py` and the variant 10 dead-end results in `docs/results` | Tanmay | [X] | Not recorded |

The GIFs use selected seeds; the README gives the measured rates. `scripts/dead_end_sweep.py` reproduces a recorded episode exactly (oracle, 1 m pocket, seed 10000: 9.2 s, 9.64 m).
