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
