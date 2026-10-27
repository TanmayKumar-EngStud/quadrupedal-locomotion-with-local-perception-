# Viewer commands: watch.py and play.py

Run everything from the repository root on the laptop. On macOS both scripts start
MuJoCo's `mjpython` themselves, so a window opens; close it to stop. Add
`--headless --seconds 10` to `watch.py` for a text-only check.

| Script | Shows | Terrain |
| --- | --- | --- |
| `watch.py` | the controller alone, walking to random targets on one training tile | the 8 training terrain types, levels 0-9, regular or irregular |
| `play.py` | a planner choosing piles and the controller walking to them | the pile scenes: 10 layout families, rows 0-9 |

Controllers: `artifacts/controller_v1/controller.pt`, `artifacts/controller_v2/controller.pt`,
`artifacts/controller_v3/controller.pt` (see "Get controller v3" below), and the 2-hour
variants exported to `artifacts/variant_<RUN>/controller.pt` ([TRAINING_VARIANTS.md](TRAINING_VARIANTS.md)).

**Terrain samples.** Both viewers draw the terrain samples the controller senses as
small spheres at the sensed height:
- white: anchored at the base;
- one colour per foot: FR red, FL green, RR blue, RL yellow.

v1, v2 and v3 show their 10 x 7 body grid (reaching 0.9 m ahead) and 9 samples per
foot. A `feet49` controller (variants 1, 5, 8) shows a 7 x 7 square at every foot and
4 samples under the trunk. The terminal names the layout at the start. `--no-scan`
hides the spheres. On the pile scenes with box ground the sensed pits are 0.5 m deep,
so samples over a pit float inside the 1.2 m pit: that is what the controller senses.

## Get controller v3

The bundle is on the KTH server at `artifacts/controller_v3.zip` (6.5 MB). In
JupyterLab, right-click it in the file browser and choose Download, then:

```sh
mkdir -p artifacts/controller_v3 && unzip -o ~/Downloads/controller_v3.zip -d artifacts/controller_v3
.venv/bin/python -c "import hashlib, json; d='artifacts/controller_v3'; m=json.load(open(d+'/manifest.json')); print({n: hashlib.sha256(open(d+'/'+n,'rb').read()).hexdigest() == h for n, h in m['sha256'].items()}, 'update', m['update'])"
```

Both hashes must print `True`, with update 12724.

## watch.py

| Option | Meaning |
| --- | --- |
| `--checkpoint PATH` | controller to run (default v1) |
| `--terrain TYPE` | `rough`, `slope`, `stairs_up`, `stairs_down`, `obstacles`, `stepping_stones`, `gaps`, `piles` |
| `--level 0-9` | difficulty, 0 easiest |
| `--terrain-noise X` | irregularities on the chosen level: `0` regular (v1/v2 terrain), `0.5` v3's training terrain, `1.0` v4's; default: what the controller trained on (with a noise ramp, the noise of the chosen level) |
| `--pile-pits deep\|ramp` | pile pits 0.5 m on every level (like the pile scenes), or deepening with level (v3's training); default: as trained |
| `--speed M/S` | commanded walking speed (default 0.8) |
| `--seed N` | other start pose and targets |
| `--no-scan` | hide the terrain samples |

### v3 on regular terrain (noise 0)

```sh
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain rough --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain slope --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stairs_up --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stairs_down --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain obstacles --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stepping_stones --level 2 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain gaps --level 3 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain piles --level 2 --terrain-noise 0 --pile-pits deep
```

### v3 on irregular terrain (noise 0.5, as trained)

```sh
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain rough --level 5 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain slope --level 5 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stairs_up --level 5 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stairs_down --level 5 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain obstacles --level 5 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stepping_stones --level 2 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain gaps --level 3 --terrain-noise 0.5
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain piles --level 2 --terrain-noise 0.5
```

For the stronger irregularities v4 would train on, use `--terrain-noise 1.0`. On
piles, `--pile-pits deep` shows the 0.5 m pits of the pile scenes; v3 trained mostly
on the shallow pits of `ramp`. Rough ground looks the same at every noise level (it is
random everywhere already).

### Compare v1, v2 and v3 on the same tile

Replace the checkpoint and keep everything else:

```sh
python watch.py --checkpoint artifacts/controller_v1/controller.pt --terrain stairs_up --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v2/controller.pt --terrain stairs_up --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain stairs_up --level 5 --terrain-noise 0
python watch.py --checkpoint artifacts/controller_v2/controller.pt --terrain piles --level 2 --terrain-noise 0.5 --pile-pits deep
python watch.py --checkpoint artifacts/controller_v3/controller.pt --terrain piles --level 2 --terrain-noise 0.5 --pile-pits deep
```

(`piles` does not exist for v1's own config, but every controller can be watched on
it: the tile is built from the options above.)

## play.py

| Option | Meaning |
| --- | --- |
| `--checkpoint PATH` | controller (required) |
| `--planner` | `optimistic` (default), `greedy`, `oracle` |
| `--family` | `flat`, `open`, `holes`, `moat`, `dead_end`, `steps`, `rough`, `slalom`, `bridge`, `slanted`, `mix` (one section of each of seven families on one course) |
| `--row 0-9` | difficulty. Rows 10-19 end at once with `no_candidate`: their piles are narrower than the robot's 30 cm foot spacing, so the hop rule allows no hop |
| `--ground boxes\|heightfield` | piles as MuJoCo boxes (default), or as the heightfield the controller senses |
| `--seed N` | other layout of the same family and row |
| `--irregularity A` | per-pile irregularities on any family: height, tilt, size and position, drawn anew on every run (0 = regular) |
| `--irregularity-seed N` | repeat one irregular layout; every irregular run prints its seed |
| `--seconds S` | time limit (default 60) |
| `--arrival-radius M` | a subgoal pile counts as reached within this distance of its centre (default 0.20 m) |
| `--no-scan` | hide the terrain samples |

In the window, the red sphere inside a pale circle is the current subgoal and its
0.20 m arrival circle; blue spheres are earlier subgoals. The terminal prints every
planner decision (time, pile position, hop length, reason).

Since October 22 the planner loop:
- counts a pile as reached only within 0.20 m of its centre;
- judges each hop from the centre of the pile the robot stands on;
- does not send the robot straight back to the pile it came from (unless no other
  pile is allowed).

Before, a robot on the seam between two piles "reached" both, and the planner
bounced between them (variant 3: 1 of 10 test scenes reached the goal before the
fix, 6 of 10 after). Calibration's hop tables are measured as before.

### Mixed and irregular pile scenes

`mix` is a 12.9 m course: a 1.2 m regular entry, then seven 1.5 m sections, one per
family, each built with its own family's rule at the chosen row:

| Section | What it holds (row 9 values) |
| --- | --- |
| `rough` | random pile heights, up to ±4 cm |
| `slanted` | tilted pile tops, up to about 7° |
| `steps` | one terrace: a step up, about 1 m of raised piles, a step down (2 cm at row 0, 5 cm at row 9) |
| `holes` | missing piles (5% at row 0, about 15% at row 9) |
| `bridge` | only a strip of piles remains (\|y\| ≤ 0.55 m at row 0, 0.41 m at row 9) |
| `moat` | a 0.8 m band of pits with one crossing at the side (y ≈ ±1.5 to ±2.1 m) |
| `slalom` | one 0.6 × 2 m wall on the side opposite the moat's crossing |

`--seed` shuffles the section order (seed 10000: steps, moat, rough, slanted, holes,
slalom, bridge). The terminal and the scene's `metadata.mix_sections` list the order with
each section's start and end. Rough, slanted, steps, holes and bridge grow with the row;
the moat, wall and course length stay the same.

![mix rows 3 and 8](../outputs/scene_preview/mix_sections.png)

Variant 9 crosses mix row 0 (16 s). In eight runs on rows 1-7 it fell every time: six
times at the slalom wall, twice at the steps terrace. It also falls on the standalone
`slalom` (rows 3 and 7) and `steps` (row 5) families, so mix inherits those failures.

`--irregularity A` perturbs every pile of any family, `mix` included, with Gaussian
deviations of mean 0. The standard deviation is the same for every family:

| Per pile | Standard deviation | Limit |
| --- | --- | --- |
| top height | A × 2 cm | ±2.5 standard deviations |
| tilt about x and y | A × 6° | 15° |
| side length | A × 3 cm | at least 12 cm |
| centre shift | A × 2 cm | stays inside its own cell, so piles never overlap |

Each run draws a new layout and prints its seed. Pass `--irregularity-seed` to repeat a
layout. The hop tables take `--irregularity` too (`calibrate`, seeded per trial, so
they stay reproducible).

```sh
python play.py --checkpoint artifacts/controller_v9/controller.pt --family mix --row 0
python play.py --checkpoint artifacts/controller_v9/controller.pt --family mix --row 3
python play.py --checkpoint artifacts/controller_v9/controller.pt --family mix --row 7 --seed 10002
python play.py --checkpoint artifacts/controller_v9/controller.pt --family open --row 5 --irregularity 1.0
python play.py --checkpoint artifacts/controller_v9/controller.pt --family mix --row 5 --irregularity 1.0
python play.py --checkpoint artifacts/controller_v9/controller.pt --family open --row 5 --irregularity 1.0 --irregularity-seed 4
python play.py --checkpoint artifacts/controller_v9/controller.pt --family dead_end --row 3 --irregularity 0.5 --planner oracle
.venv/bin/python -m quadruped_nav.calibrate --workers 5 --checkpoint artifacts/controller_v9/controller.pt \
    --families mix --rows 0 1 2 3 4 5 6 7 8 9 --output outputs/calibration/hops_v9_mix.json
.venv/bin/python -m quadruped_nav.calibrate --workers 5 --checkpoint artifacts/controller_v9/controller.pt \
    --irregularity 1.0 --rows 0 1 2 3 4 5 6 7 8 9 --output outputs/calibration/hops_v9_irregular1.json
```

### v3 on every layout family

```sh
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family flat --row 0
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family open --row 0
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family holes --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family moat --row 0
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family dead_end --row 0
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family steps --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family rough --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family slalom --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family bridge --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family slanted --row 3
```

The irregular pile scenes are the `rough` (random pile heights), `slanted` (tilted
piles) and `steps` (terraces) families; the Gaussian training irregularities exist
only on the training tiles (watch.py).

### Planners, rows and ground

```sh
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner greedy --family dead_end --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner oracle --family dead_end --row 3
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family open --row 6
python play.py --checkpoint artifacts/controller_v3/controller.pt --planner optimistic --family open --row 0 --ground heightfield
python play.py --checkpoint artifacts/controller_v2/controller.pt --planner optimistic --family open --row 0 --ground heightfield
```

Each run also writes the full episode (trajectory and every decision) to
`outputs/evaluation/episode.json`.
