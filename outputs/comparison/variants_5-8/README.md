# Variants 5-8, round 1 (Amedeo's KTH GPU)

The four round 1 runs of [TRAINING_VARIANTS.md](../../../docs/TRAINING_VARIANTS.md) on Amedeo's GPU (not the
round 2 runs that continue variant 1 under the same numbers), evaluated with
step 8 of that guide. Trained October 21, 2026 at commit `ef07a3e` on a KTH H100 MIG 2g.20gb slice, four runs at
once, 2 hours each (16:47-18:52). Evaluated twice, both in native CPU MuJoCo with deterministic actions: on the KTH
server right after training, and on Amedeo's laptop (Apple M1) together with v2, so that v2 is measured the same way.
The two machines agree within 1 percentage point on the hop tables and within 4 points on the fixed test overall
(up to 12 points on its 16-trial pile cells).

| # | Starts from | What changes | Final update | Last validation |
| --- | --- | --- | --- | --- |
| 5 | scratch | 49-sample scan + noise ramp | 947 | 62% |
| 6 | v2 | noise 0.5 + halved momentum penalty + edge penalty + relaxed curriculum | 4959 | 83% |
| 7 | v2 | relaxed curriculum + noise ramp | 4951 | 80% |
| 8 | scratch | 49-sample scan + relaxed curriculum | 959 | 47% |

## Against v2 (laptop)

| | v2 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| Pile-scene hops arrived, boxes | 86% | 0% | 81% | 81% | 0% |
| Pile-scene hops ended in a fall, boxes | 11% | 68% | 6% | 5% | 34% |
| First hop off the platform, boxes | 88% | 0% | 65% | 63% | 0% |
| Pile-scene hops arrived, heightfield | 72% | 0% | 51% | 50% | 0% |
| Fixed test noise 0: 7 original types / piles | 83% / 25% | 51% / 6% | 87% / 25% | 77% / 19% | 33% / 12% |
| Fixed test noise 0.5: all types / piles | 76% / 19% | 45% / 12% | 73% / 25% | 70% / 19% | 26% / 6% |
| Fixed test noise 1.0: all types / piles | 73% / 6% | 44% / 0% | 73% / 12% | 69% / 0% | 31% / 0% |
| Fixed test noise 0: fell | 25% | 18% | 3% | 9% | 6% |
| Pile level at the end of training | 0 (strict rule) | 0.0 | 5.6 | 5.1 | 2.0 |

- **6 and 7 do not beat v2 yet.** They fall far less (6% instead of 11% of box hops, 3-9% instead of 25% in the
  fixed test), but they leave the platform less often and time out more, so they are more cautious rather than more
  capable. The relaxed curriculum did keep their pile robots on levels 5-6 instead of 0. Their action noise fell
  (from about 1.0 to 0.58 and 0.65), unlike v3's drift.
- **5 and 8 cannot cross pile scenes after about 950 updates**, as the guide expected. Judge them by their
  validation curves against variant 1 (and reference 0s) at the same update, not against v2.
- The fixed test has 16 trials per terrain type, so a pile cell moves in steps of 6%; treat differences under about
  15 points there as chance. The hop tables (900-2900 hops each) are much tighter.

## Against variants 1-4 (the pairs of the guide)

Pile-scene hops arrived, rows 0-9; variants 1-4 from the round 1 results in TRAINING_VARIANTS.md (Tanmay's
server), variants 5-8 from the KTH server here:

| Pair | Differs only in | Boxes | Heightfield |
| --- | --- | --- | --- |
| 1 vs 8 | relaxed curriculum | 0% vs 0% | 0% vs 0% |
| 1 vs 5 | noise ramp | 0% vs 0% | 0% vs 0% |
| 2 vs 7 | relaxed curriculum | 86% vs 82% | 76% vs 51% |
| 3 vs 6 | relaxed curriculum | 94% vs 81% | 75% vs 51% |
| 4 vs 7 | noise ramp | 84% vs 82% | 62% vs 51% |
| 4 vs 6 | noise 0.5, edge and momentum penalties | 84% vs 81% | 62% vs 51% |

Adding the relaxed curriculum to a fine-tune (2 vs 7, 3 vs 6) costs 4-13 points on boxes and about 25 on the
heightfield. The two from-scratch pairs cannot be told apart on hops yet; use the validation curves below at the
last update both runs reached.

## Validation at the numbered checkpoints (seed 100, every 25 updates)

| Run | | | | | |
| --- | --- | --- | --- | --- | --- |
| 5 | 250: 37% | 500: 54% | 750: 60% | 900: 66% | 947: 62% |
| 8 | 250: 20% | 500: 24% | 750: 40% | 900: 44% | 959: 47% |
| 6 | 4250: 80% | 4500: 79% | 4750: 82% | 4950: 88% | 4959: 83% |
| 7 | 4250: 81% | 4500: 86% | 4750: 77% | 4950: 81% | 4951: 80% |

The full curves are in `data/validation_variant_*.jsonl` (with `by_type`), to compare each pair of the guide at
the last update both runs reached: 1 vs 5 and 1 vs 8, 2 vs 7, 3 vs 6, 4 vs 6 and 4 vs 7.

## Files

- `finetune/`: v2 vs 6 and 7; `scratch/`: v2 vs 5 and 8. Each has the hop charts (boxes, heightfield), the
  fixed-test chart and `summary.md`, made by `scripts/compare_controllers.py` from the laptop results.
- `data/`: compact hop-table summaries (`summary_of` in `compare_controllers.py`, which reads them back like full
  tables) and the fixed-test results at noise 0, 0.5 and 1.0, each from the laptop (`_laptop`) and the KTH server
  (`_kth`; v2 only from the laptop), plus the validation curves. To add variants 1-4, pass these files to
  `compare_controllers.py` next to your own tables.
- `make_charts.sh`: rebuilds everything here from `outputs/{evaluation,calibration}/` and `runs/`.

The controller bundles (`artifacts/variant_<RUN>/`, about 6.5 MB each) are not in Git, like v1-v3. Amedeo has them
on his laptop and his KTH server.
