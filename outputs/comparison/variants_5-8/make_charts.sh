#!/bin/sh
# Rebuild outputs/comparison/variants_5-8 from the step 8 results of docs/TRAINING_VARIANTS.md (run from the repo root).
# Charts use the laptop evaluation (outputs/{evaluation,calibration}/laptop/), where v2 was measured the same way.
# data/ also keeps the KTH server results and the validation curves, for comparing with variants 1-4.
set -e
OUT=outputs/comparison/variants_5-8
PY="uv run --frozen --extra train --with matplotlib python"
E=outputs/evaluation/laptop; K=outputs/calibration/laptop
NOTE="Laptop (Apple M1), native CPU MuJoCo, deterministic actions. v2 = update 4000; variants trained 2 h each on a KTH H100 MIG slice."
V5=5-feet49-noise-ramp; V6=6-noise-edge-relaxed; V7=7-noise-ramp-relaxed; V8=8-feet49-relaxed
$PY scripts/compare_controllers.py \
  --hops boxes v2=$K/hops_variant_v2.json var6=$K/hops_variant_$V6.json var7=$K/hops_variant_$V7.json \
  --hops heightfield v2=$K/hops_variant_v2_heightfield.json var6=$K/hops_variant_${V6}_heightfield.json var7=$K/hops_variant_${V7}_heightfield.json \
  --fixed "all 8 types" v2=$E/fixed_variant_v2_noise0.json var6=$E/fixed_variant_${V6}_noise0.json var7=$E/fixed_variant_${V7}_noise0.json \
  --note "$NOTE" --output $OUT/finetune > /dev/null
$PY scripts/compare_controllers.py \
  --hops boxes v2=$K/hops_variant_v2.json var5=$K/hops_variant_$V5.json var8=$K/hops_variant_$V8.json \
  --hops heightfield v2=$K/hops_variant_v2_heightfield.json var5=$K/hops_variant_${V5}_heightfield.json var8=$K/hops_variant_${V8}_heightfield.json \
  --fixed "all 8 types" v2=$E/fixed_variant_v2_noise0.json var5=$E/fixed_variant_${V5}_noise0.json var8=$E/fixed_variant_${V8}_noise0.json \
  --note "$NOTE" --output $OUT/scratch > /dev/null
mkdir -p $OUT/data
SUMMARY="import sys; sys.path.insert(0, 'scripts'); from compare_controllers import summary_of; print(summary_of(sys.argv[1]))"
for RUN in $V5 $V6 $V7 $V8 v2; do
  for G in "" _heightfield; do
    $PY -c "$SUMMARY" $K/hops_variant_$RUN$G.json > $OUT/data/hops_variant_$RUN${G}_laptop.json
    [ $RUN = v2 ] || $PY -c "$SUMMARY" outputs/calibration/hops_variant_$RUN$G.json > $OUT/data/hops_variant_$RUN${G}_kth.json
  done
  for NOISE in 0 0.5 1.0; do
    cp $E/fixed_variant_${RUN}_noise$NOISE.json $OUT/data/fixed_variant_${RUN}_noise${NOISE}_laptop.json
    [ $RUN = v2 ] || cp outputs/evaluation/fixed_variant_${RUN}_noise$NOISE.json $OUT/data/fixed_variant_${RUN}_noise${NOISE}_kth.json
  done
  [ $RUN = v2 ] || cp runs/kth-variant-$RUN/evaluation.jsonl $OUT/data/validation_variant_$RUN.jsonl
done
