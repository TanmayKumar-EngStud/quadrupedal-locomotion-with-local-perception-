# Controller comparison

## Hop table, boxes ground

| Controller | Hops | Arrived | Ended in a fall | First hop off the platform | Hardest row ≥80% short hops (pooled) | Unbroken ≥80% from row 0 |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 2862 | 86% | 11% | 88% | 9 | 4 |
| var6 | 2460 | 81% | 6% | 65% | 8 | 8 |
| var7 | 2510 | 81% | 5% | 63% | 9 | 6 |

Short-hop success (0.25–1 m) by row, pooled over families (successes/hops):

| Row | v2 | var6 | var7 |
| --- | --- | --- | --- |
| 0 | 98% (173/177) | 95% (129/136) | 99% (160/162) |
| 1 | 94% (165/175) | 95% (125/132) | 96% (128/134) |
| 2 | 93% (145/156) | 93% (95/102) | 93% (91/98) |
| 3 | 89% (111/125) | 94% (106/113) | 96% (97/101) |
| 4 | 87% (84/97) | 98% (79/81) | 89% (65/73) |
| 5 | 71% (39/55) | 83% (50/60) | 80% (52/65) |
| 6 | 82% (40/49) | 94% (44/47) | 94% (48/51) |
| 7 | 82% (37/45) | 100% (33/33) | 78% (28/36) |
| 8 | 91% (32/35) | 81% (26/32) | 61% (20/33) |
| 9 | 100% (32/32) | 75% (24/32) | 88% (28/32) |

Hardest row with ≥80% short-hop success, per family (highest row / unbroken run from row 0):

| Family | v2 | var6 | var7 |
| --- | --- | --- | --- |
| open | 9 / 2 | 7 / 7 | 9 / 6 |
| rough | 9 / 4 | 8 / 8 | 9 / 6 |
| slanted | 9 / 5 | 8 / 4 | 9 / 1 |
| steps | 9 / 4 | 7 / 4 | 9 / 4 |

## Hop table, heightfield ground

| Controller | Hops | Arrived | Ended in a fall | First hop off the platform | Hardest row ≥80% short hops (pooled) | Unbroken ≥80% from row 0 |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 2282 | 72% | 15% | 75% | 9 | 3 |
| var6 | 1565 | 51% | 7% | 53% | 7 | – |
| var7 | 1545 | 50% | 9% | 52% | 9 | – |

Short-hop success (0.25–1 m) by row, pooled over families (successes/hops):

| Row | v2 | var6 | var7 |
| --- | --- | --- | --- |
| 0 | 87% (124/142) | 75% (49/65) | 79% (58/73) |
| 1 | 88% (114/130) | 84% (54/64) | 81% (51/63) |
| 2 | 81% (99/122) | 85% (46/54) | 78% (42/54) |
| 3 | 83% (74/89) | 85% (45/53) | 89% (41/46) |
| 4 | 78% (50/64) | 82% (33/40) | 85% (34/40) |
| 5 | 79% (33/42) | 92% (35/38) | 85% (33/39) |
| 6 | 94% (34/36) | 79% (26/33) | 34% (11/32) |
| 7 | 69% (27/39) | 97% (31/32) | 70% (23/33) |
| 8 | 79% (27/34) | 78% (25/32) | 62% (20/32) |
| 9 | 88% (28/32) | 75% (24/32) | 100% (32/32) |

Hardest row with ≥80% short-hop success, per family (highest row / unbroken run from row 0):

| Family | v2 | var6 | var7 |
| --- | --- | --- | --- |
| open | 6 / 3 | 7 / – | 9 / – |
| rough | 9 / 1 | 7 / – | 9 / – |
| slanted | 9 / 4 | 8 / – | 9 / – |
| steps | 9 / 2 | 7 / 5 | 9 / 0 |

## Fixed terrain test

| Controller | all 8 types: reached / fell | rough | slope | stairs up | stairs down | obstacles | stepping stones | gaps | piles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | 76% / 25% | 100% | 100% | 69% | 75% | 94% | 56% | 88% | 25% |
| var6 | 79% / 3% | 94% | 100% | 81% | 94% | 94% | 50% | 94% | 25% |
| var7 | 70% / 9% | 81% | 100% | 69% | 81% | 81% | 44% | 81% | 19% |

Laptop (Apple M1), native CPU MuJoCo, deterministic actions. v2 = update 4000; variants trained 2 h each on a KTH H100 MIG slice.
