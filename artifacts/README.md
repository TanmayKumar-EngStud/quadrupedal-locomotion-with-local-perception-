# Trained controllers

Each folder is an exported controller bundle:
- `controller.pt` is the full checkpoint, used by `watch.py`, `play.py`, `goal_check` and `calibrate`, and resumable with
  `--resume`.
- `actor.ts` is the TorchScript actor.
- `manifest.json` holds the training config and the SHA-256 hashes of both files.

The names in [docs/TRAINING_VARIANTS.md](../docs/TRAINING_VARIANTS.md) (family tree, results) are given in the second
column.

| Folder | Controller in the docs | Training run | Update | Terrain scan | Box-ground pile hops |
| --- | --- | --- | --- | --- | --- |
| `controller_v1` | v1 | `runs/kth-goal-v3-seed0-phase3` | 3000 | old (106 samples) | 42% |
| `controller_v2` | v2 | `runs/kth-controller-v2` | 4000 | old | 86% |
| `controller_v3` | v3 | `runs/kth-controller-v3` | 12724 | old | 31% |
| `controller_v5` | **variant 3** (v2 + noise 0.5, edge, gap and arrival-speed penalties) | `runs/kth-variant-3-noise-edge` | 4916 | old | 94% |
| `controller_v9` | variant 9 (variant 5 + edge penalty + foot clearance), the best so far | `runs/kth-variant-5-edge-clearance-6h` | 10176 | 49 points per foot (`feet49`) | 96% |

Check a bundle after downloading:

```sh
.venv/bin/python -c "import hashlib, json, sys; d=sys.argv[1]; m=json.load(open(d+'/manifest.json')); print({n: hashlib.sha256(open(d+'/'+n,'rb').read()).hexdigest() == h for n, h in m['sha256'].items()}, 'update', m['update'])" artifacts/controller_v9
```

Watch one:

```sh
python watch.py --checkpoint artifacts/controller_v9/controller.pt --terrain piles --level 5
python play.py --checkpoint artifacts/controller_v9/controller.pt --family open --row 5
```
