# Abstract simulator — Day 1

Historical Day-1 design note. The October 14 implementation also includes the optimistic and learned planners, controller training and physics execution; see [current status](IMPLEMENTATION_STATUS.md) and [runbook](RUNBOOK.md). The deterministic baseline below remains available. The adapter now accepts roll/pitch with an approximate conservative footprint; arbitrary yaw rotations are still rejected.

Run `uv run python simulate.py --open` from the repository root. This runs paired oracle/greedy episodes on open, moat and dead-end layouts, then opens the generated offline replay. No MuJoCo import, GPU, browser network connection or trained policy is needed for the abstract run. The command is also available as `uv run python -m quadruped_nav.experiment`.

## Components

| Component | Responsibility |
| --- | --- |
| `Terrain.from_scene()` | Adapt the existing v1 JSON, including start and goal platforms, to horizontal square piles; replace coordinate-encoded source IDs with opaque stable IDs |
| `Geometry` | Generate visible piles and admissible straight-line transitions; full terrain remains internal to the environment/oracle |
| `AbstractEnv` | Reset episodes, produce immutable local observations, validate selections, execute abstract transitions, terminate and log |
| `GreedyPlanner` | Select the eligible visible pile nearest the goal; receives no full terrain or visit history |
| `OraclePlanner` | A* using the full terrain, with the same transition eligibility, Euclidean distance costs and heuristic |
| `run_episode()` / `run_comparison()` | Run both planners from the same initial state under identical model limits and record metrics |
| Replay viewer | Show local coverage, greedy motion and optional privileged reference overlays; it replays actual Python-generated logs |

`reset()` returns an `Observation`. `planner.select_subgoal(observation)` returns a pile ID or `None`. `step(target_id)` returns `(next_observation, done)`; outcomes and state are available on the environment. A hidden, unknown, current or masked target raises `ValueError` without advancing the episode. Reset is required after termination. The privileged oracle alone is constructed with a `Geometry` instance.

## Initial model assumptions

- Metres, seconds and radians; fixed world/odometry with +Z up. Local observations rotate with robot yaw. The abstract base height is 0.30 m above its support. This is a coordinate convention, not a body dynamics model.
- Perfect odometry and pile association. Visibility uses a pile centre within a 2 m horizontal radius, with no occlusion. Only these piles determine the local action mask. Full-map connectivity and generator metadata are not passed to greedy.
- Horizontal axis-aligned square tops, with optional height differences. Slanted/rotated records are not supported by the initial adapter. The earlier scene generator still provides ten levels; this is not the proposal's future 20-row controller curriculum.
- Each selected target is at most 2 m away. Targets below 1 m are allowed, including escape/recovery. There is no heading restriction and turning is instantaneous. These defaults are provisional, not a silently approved teammate interface.
- The point-body moves along a straight centreline. Exact line/rectangle intersections identify support intervals; every unsupported interval must be at most 0.14 m and every support transition's height change at most 0.05 m. Winding connectivity elsewhere cannot make an unsupported direct segment valid. Finite feet, landing stability and contact forces are not modelled. Since 15 October 2026 body width is approximated: the same gap rule is also applied to two parallel lines 0.15 m either side of the centreline (`Limits.body_half_width_m`; 0 restores the centreline-only rule), because diagonal hops that clipped a platform corner put one pair of legs over the pit in physics.
- Speed is a fixed 0.6 m/s; the target must fit within a 4 s subgoal budget. Episode duration is capped at 60 s. A completed subgoal lands at the pile centre. Success means reaching the designated goal pile, not physical stability or a measured fall-free arrival.
- Subgoals are atomic. If remaining episode time cannot fit the next transition, the log records a timeout at the last confirmed support with no completed distance added. It does not claim a partially simulated physical trajectory.
- No stochastic failures, controller training or policy inference. Movement limits are configuration independent of the terrain difficulty label; future measured controller data must replace/calibrate this surrogate.

The oracle graph connects admissible subgoal targets directly, rather than hiding a winding path behind a single endpoint command. An edge can cross several intermediate support piles but must pass the same straight-approach rule as greedy. This provides a consistent reference within this abstract model; it is not an execution guarantee in MuJoCo.

## Logs and metrics

Each comparison exports the source family/level/seed, selected goal, adapted terrain, limits, oracle reference path and both episodes. Each episode stores:

- Initial state and the state after every attempted decision: pile, position, yaw, elapsed simulated seconds, completed distance and status.
- Each decision's full **local** observation, aligned action mask, selected target, timestamps and outcome.
- Success/status, completed path distance, simulated time, decision count, repeated-target fraction, number of piles selected at least three times, and timeout-with-loop status.
- Oracle reference distance and completed-path/oracle-distance ratio. The ratio is `null` on failure or when no usable oracle reference exists, so failed short paths are not reported as efficient.

`None` with no eligible actions terminates as `no_candidate`; `None` despite eligible actions terminates as `planner_no_route`. Repeated selections include selected timeout attempts. Counts refer to chosen subgoals, not every intermediate support crossed. The viewer shows interpolated motion for inspection; local planner decisions occur only at the logged decision boundaries.

Outputs live in ignored `outputs/abstract/`. The HTML contains the complete replay data and no external dependencies. The viewer's full-world/oracle overlays are privileged inspection tools; hiding them does not change the already-recorded planner decisions. The visible-pile count during interpolated motion is a display-only ideal sensor estimate; the decision panel's eligible-target count comes from the recorded observation.

## Current validation and example results

`uv run pytest -q` exercises local visibility and hidden-map isolation, reset/replay determinism, invalid selections, direct-gap rejection despite winding support, height steps, coordinate rotation, terminal states, time budgets, all three paired layouts and safe JSON/HTML export. Existing MuJoCo scene tests remain in the suite.

Default level 0, seed 1042, goal P2:

| Layout | Greedy | Oracle |
| --- | --- | --- |
| Open | Success; 7.94 m; 13.23 s | Success; 7.94 m; 13.23 s |
| Moat | Timeout; 35.77 m completed; 60 s | Success; 9.47 m; 15.78 s |
| Dead end | Timeout; 35.77 m completed; 60 s | Success; 9.47 m; 15.78 s |

These default moat and dead-end examples happen to produce identical metric totals. They are different layouts, not independent evidence of generalisation. The generator's canonical layouts with zero random removal do not vary across seeds. Do not treat these demonstrations as the final benchmark or statistical results.

## Next integration work

The optimistic replanner, its persistent observed/unknown map, randomised benchmark manifests, pocket-entry/escape metrics, the DAgger interface and the physics executor remain future work. The shared interface contract is still under team review. Amedeo's frozen-controller measurements will calibrate the abstract movement model before planner comparisons are interpreted as robot performance.
