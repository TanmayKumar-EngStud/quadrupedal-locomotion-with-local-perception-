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
