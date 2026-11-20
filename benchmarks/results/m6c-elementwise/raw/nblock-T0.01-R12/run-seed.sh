#!/bin/bash
# One seed: patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, k = 3 (the library default since 06ebafaa).
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v timeout 4h taskset -c $core ./bin g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock \
  --samples 30000 --seed $seed --cap 64 \
  --variants "rtol=1e-2,engine-tol-factor=3;rtol=1e-3,engine-tol-factor=3;rtol=1e-4,engine-tol-factor=3" \
  > s$seed-patched.jsonl 2> s$seed-patched.time
