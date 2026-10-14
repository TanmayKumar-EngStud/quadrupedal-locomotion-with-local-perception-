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
| Run H100/A100 preflight, measure throughput/memory and train/validate the locomotion controller | Tanmay, Amedeo, Damien | [ ] | — |
| Freeze a validated controller, calibrate transitions and complete held-out physics experiments | Tanmay, Amedeo, Damien | [ ] | — |

CPU verification: 56 tests passed; short PPO, export/reload, learned-planner and physics execution runs completed. CUDA execution and trained locomotion quality remain unverified locally. See [implementation status](docs/IMPLEMENTATION_STATUS.md) and the [manual](output/pdf/Manual.pdf).
