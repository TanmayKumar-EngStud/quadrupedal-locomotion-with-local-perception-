#!/bin/sh
# Rebuild outputs/comparison from the hop tables and fixed-test results (run from the repo root).
.venv/bin/python scripts/compare_controllers.py \
  --hops boxes v1=outputs/calibration/hops_v1.json v2=outputs/calibration/hops_v2.json control=outputs/calibration/summary_hops_v1_control.json \
  --hops heightfield v1=outputs/calibration/hops_v1_heightfield.json v2=outputs/calibration/hops_v2_heightfield.json control=outputs/calibration/summary_hops_v1_control_heightfield.json \
  --fixed "7 v1 types" v1=outputs/evaluation/fixed_v1_v1types.json v2=outputs/evaluation/fixed_v2_v1types.json control=outputs/evaluation/fixed_v1_control_v1types.json \
  --fixed piles v1=outputs/evaluation/fixed_v1_piles.json v2=outputs/evaluation/fixed_v2_piles.json control=outputs/evaluation/fixed_v1_control_piles.json \
  --note "control = v1 + the same 1000 updates on the original seven terrain types (no piles). All: native CPU MuJoCo, deterministic actions." \
  --output outputs/comparison
