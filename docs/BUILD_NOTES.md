# Environment milestone — September 30, 2026

This build follows the scene and terrain section of the new Technical Design and Build Plan. It does not adopt the earlier suggestion to reduce the research project to curriculum learning alone.

Implemented: seeded JSON layouts, ten proposal difficulty levels, open/moat/dead-end terrain, P0 start, P1–P3 target platforms, red/blue/yellow landmarks, pinned Go1 asset, native MuJoCo viewer, standing position control, rendered views and regression tests.

## Verified locally

- MuJoCo 3.3.7 on macOS, Python 3.12.
- 13 automated checks passed, including all three layout families at levels 0, 5 and 9.
- Geometric routes exist from start to every goal in these canonical layouts.
- Moat/dead-end routes to the central target detour at least 2 m sideways.
- No overlapping support surfaces fill the gaps.
- All landmarks have collision disabled.
- Go1 has 12 actuators, stays on P0 for five simulated seconds, and remains numerically stable.
- Moving the robot over the moat causes a fall to the pit floor.
- Overview, robot close-up and top views rendered in MuJoCo.

## Design choices to calibrate later

The world spans approximately 6.5 m in x and ±3.5 m in y. Core stones have zero height variation and lateral shift. Start size is 1 m; goal size is 0.8 m. Pit depth is 1.2 m. These dimensions are initial choices; nominal graph connectivity is not proof of dynamically feasible locomotion.

Canonical scenes have no random removals, so their geometry does not vary with the seed. Open-field random removal supports up to 20%, is deterministic by seed, and can disconnect the graph. Rejecting disconnected training samples and introducing valid layout diversity should be addressed with the future training generator.

## Next integration boundary

1. Import scene geometry and Go1 into the pinned mjlab training stack on the actual NVIDIA allocation.
2. Add terrain scans, target-relative observations, reset/termination handling, rewards and the goal-reaching policy.
3. Calibrate gap/height limits using real policy outcomes before treating geometric routes as executable.
4. Build the language ID resolver, calibrated planner/subgoal executor and episode log against the shared JSON contract.

`Simulation.status()` exposes a pit-contact diagnostic, not the proposal's complete episode termination classifier. There is no trained policy, language model, automatic navigation, symmetry augmentation, curriculum trainer or execution-aware edge classifier in this milestone.
