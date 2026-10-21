# Controller comparison

## Hop table, boxes ground

| Controller | Hops | Arrived | Ended in a fall | First hop off the platform | Hardest row ≥80% short hops (pooled) | Unbroken ≥80% from row 0 |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 2862 | 86% | 11% | 88% | 9 | 4 |
| var5 | 903 | 0% | 68% | 0% | – | – |
| var8 | 903 | 0% | 34% | 0% | – | – |

Short-hop success (0.25–1 m) by row, pooled over families (successes/hops):

| Row | v2 | var5 | var8 |
| --- | --- | --- | --- |
| 0 | 98% (173/177) | 0% (0/32) | 0% (0/32) |
| 1 | 94% (165/175) | 0% (0/32) | 0% (0/32) |
| 2 | 93% (145/156) | 0% (0/32) | 0% (0/32) |
| 3 | 89% (111/125) | 0% (0/32) | 0% (0/32) |
| 4 | 87% (84/97) | 0% (0/32) | 0% (0/32) |
| 5 | 71% (39/55) | 0% (0/32) | 0% (0/32) |
| 6 | 82% (40/49) | 0% (0/32) | 0% (0/32) |
| 7 | 82% (37/45) | 0% (0/32) | 0% (0/32) |
| 8 | 91% (32/35) | 0% (0/32) | 0% (0/32) |
| 9 | 100% (32/32) | 0% (0/32) | 0% (0/32) |

Hardest row with ≥80% short-hop success, per family (highest row / unbroken run from row 0):

| Family | v2 | var5 | var8 |
| --- | --- | --- | --- |
| open | 9 / 2 | – / – | – / – |
| rough | 9 / 4 | – / – | – / – |
| slanted | 9 / 5 | – / – | – / – |
| steps | 9 / 4 | – / – | – / – |

## Hop table, heightfield ground

| Controller | Hops | Arrived | Ended in a fall | First hop off the platform | Hardest row ≥80% short hops (pooled) | Unbroken ≥80% from row 0 |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 2282 | 72% | 15% | 75% | 9 | 3 |
| var5 | 903 | 0% | 60% | 0% | – | – |
| var8 | 903 | 0% | 30% | 0% | – | – |

Short-hop success (0.25–1 m) by row, pooled over families (successes/hops):

| Row | v2 | var5 | var8 |
| --- | --- | --- | --- |
| 0 | 87% (124/142) | 0% (0/32) | 0% (0/32) |
| 1 | 88% (114/130) | 0% (0/32) | 0% (0/32) |
| 2 | 81% (99/122) | 0% (0/32) | 0% (0/32) |
| 3 | 83% (74/89) | 0% (0/32) | 0% (0/32) |
| 4 | 78% (50/64) | 0% (0/32) | 0% (0/32) |
| 5 | 79% (33/42) | 0% (0/32) | 0% (0/32) |
| 6 | 94% (34/36) | 0% (0/32) | 0% (0/32) |
| 7 | 69% (27/39) | 0% (0/32) | 0% (0/32) |
| 8 | 79% (27/34) | 0% (0/32) | 0% (0/32) |
| 9 | 88% (28/32) | 0% (0/32) | 0% (0/32) |

Hardest row with ≥80% short-hop success, per family (highest row / unbroken run from row 0):

| Family | v2 | var5 | var8 |
| --- | --- | --- | --- |
| open | 6 / 3 | – / – | – / – |
| rough | 9 / 1 | – / – | – / – |
| slanted | 9 / 4 | – / – | – / – |
| steps | 9 / 2 | – / – | – / – |

## Fixed terrain test

| Controller | all 8 types: reached / fell | rough | slope | stairs up | stairs down | obstacles | stepping stones | gaps | piles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | 76% / 25% | 100% | 100% | 69% | 75% | 94% | 56% | 88% | 25% |
| var5 | 45% / 18% | 81% | 75% | 50% | 56% | 50% | 25% | 19% | 6% |
| var8 | 30% / 6% | 38% | 69% | 12% | 44% | 38% | 0% | 31% | 12% |

Laptop (Apple M1), native CPU MuJoCo, deterministic actions. v2 = update 4000; variants trained 2 h each on a KTH H100 MIG slice.
