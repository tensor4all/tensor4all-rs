#!/bin/bash
# One seed: patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, k = 3, rtol 1e-3 (L2 with audit),
# with per-call traces and the per-pass history of every kernel run below the cap.
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v timeout 2h taskset -c $core ./bin g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock \
  --samples 30000 --seed $seed --cap 64 --rtol 1e-3 --engine-tol-factor 3 --trace-calls true --trace-history true \
  > s$seed-history.jsonl 2> s$seed-history.time
