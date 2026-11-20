#!/bin/bash
# One seed: unpatched TreeACI on tt_nblock, T = 0.01, R = 12, absolute tol 1e-3, 1e-4, 1e-5 (max_sweeps 50; #784 applies).
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v timeout 6h taskset -c $core ./bin g0 unpatched-aci --r 12 --temperature 0.01 --topology tt_nblock \
  --samples 30000 --seed $seed --variants "tol=1e-3;tol=1e-4;tol=1e-5" \
  > s$seed-unpatched.jsonl 2> s$seed-unpatched.time
