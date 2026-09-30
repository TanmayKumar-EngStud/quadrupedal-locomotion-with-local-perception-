# Language-Guided Quadrupedal Navigation

A MuJoCo environment foundation for a simulated **Unitree Go1** navigating stepping-stone fields toward platforms identified by language.

![Go1 in the moat environment](outputs/moat_presentation.png)

## Current build

- Three terrain families: **open**, **moat**, and **dead-end**.
- Ten difficulty levels, interpolating the proposal's stone size/gap anchors: level 0 = 46/4 cm, level 5 = 26/9 cm, level 9 = 10/13 cm.
- A 1.2 m deep pit floor, physical stone collisions, one start platform and three goal platforms.
- Three visual-only landmarks: red box, blue cone and yellow cone.
- One JSON scene per layout, used to build portable MuJoCo XML.
- Official Menagerie Go1 meshes, collision geometry and 12 position actuators.
- An interactive viewer, reproducible renders and physics/connectivity checks.

**The robot currently holds a standing pose using joint position servos. It does not yet walk, follow language instructions or execute routes.** This build establishes the scene and robot assets for the subsequent controller, grounding and planner work. No trained policy or training result is claimed.

## Run

Install [uv](https://docs.astral.sh/uv/), then from this folder:

```sh
uv sync
uv run python -m quadruped_nav check
uv run python -m quadruped_nav render
```

Open the interactive simulation on **macOS**:

```sh
uv run python view.py
```

The same launcher works on Linux/Windows. On macOS it starts MuJoCo’s required `mjpython` process and resolves the Python shared-library directory for uv environments. The standalone environment runs on CPU; no NVIDIA GPU is needed for scene inspection. A desktop display/OpenGL context is required for the viewer. On headless NVIDIA Linux, set `MUJOCO_GL=egl` for rendering, or use an available OSMesa backend on CPU.

In the viewer:

| Control | Action |
| --- | --- |
| 1 | Overview |
| 2 | Go1 close-up |
| 3 | Top view |
| Space | Pause/resume |
| R | Reset standing pose |
| Left mouse drag | Rotate camera |
| Right mouse drag | Pan camera |
| Scroll | Zoom |
| Tab / Shift+Tab | Toggle MuJoCo panels |

On macOS you can also double-click **Launch Environment.command**.

## Generate another scene

```sh
uv run python view.py --family dead_end --level 5 --seed 1042
uv run python -m quadruped_nav build --family open --level 3 --seed 42 --removal 0.1
uv run python view.py --scene scenes/moat_level0_seed1042.json
uv run pytest
```

Generated JSON and XML are saved in `scenes/`. Renders are saved in `outputs/` (a new render of the same family replaces its preview). Seed controls random stone removal; canonical scenes with no random removal are intentionally the same geometry across seeds. Open layouts with random removal are **not guaranteed connected**; the `check` command reports absent routes. Connected-layout rejection sampling and train/validation/test manifests remain future work.

## Scene contract

The schema has `seed`, `family`, `level`, `stones`, `platforms`, `landmarks`, `start` and explicit SI-unit parameters. A stone has an ID, centre `(x,y)`, `z_top`, and square `size`. Platform P0 is the start; P1–P3 are targets. Landmarks are simulator metadata and non-colliding visuals; object detection is outside the proposal's scope.

- Core terrain tops are at z=0; pit floor is at z=-1.2 m.
- Stones extend down to the pit floor, leaving real empty gaps between them.
- The moat removes two full stone columns except one bridge at y>=2.5 m, outside the proposed local scan around the direct centreline approach.
- The dead-end field adds side cuts around the direct approach, forming a pocket and forcing a detour.
- Start platform is 1 m wide to safely initialise Go1. Targets are 0.8 m wide. These are initial environment choices, pending controller calibration.
- `geometric_route()` is an A* **connectivity diagnostic** using nominal gap thresholds. It is not an execution-feasibility guarantee, calibrated planner, foothold plan or controller.

## Repository layout

```text
docs/                       Current project proposal and implementation notes
assets/unitree_go1/         Pinned robot model, meshes and original license
src/quadruped_nav/scene.py  Deterministic JSON scene generator
src/quadruped_nav/mjcf.py   JSON → MJCF builder
src/quadruped_nav/simulation.py  Stand controller, render and camera utilities
scenes/                     Generated sample layouts and portable XML
outputs/                    Actual MuJoCo renders
tests/                     Geometry, graph connectivity and physics checks
```

## Relationship to the proposal

The authoritative proposal is [Language-Guided Legged Navigation: Technical Design and Build Plan](docs/Language-Guided%20Legged%20Navigation%20Technical%20Design%20and%20Build%20Plan.pdf), dated September 30, 2026. The older proposal is superseded.

This first milestone uses native MuJoCo to inspect scene geometry on macOS. **mjlab integration, the height scan, PPO training, the goal executor, language grounding and execution-aware edge costs are not implemented.** The JSON layouts and robot asset are reusable inputs for that integration. Proposed terrain difficulty is a hypothesis to calibrate, not demonstrated robot capability.

## Asset provenance

Go1 is vendored from [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1), commit `4d038b3feae26ec82b46a4d586379114012a8ac7`. Its original BSD-3-Clause license is retained in `assets/unitree_go1/LICENSE`. `UPSTREAM.json` records provenance. Generated scene XML incorporates the model; retain the asset license when redistributing it.
