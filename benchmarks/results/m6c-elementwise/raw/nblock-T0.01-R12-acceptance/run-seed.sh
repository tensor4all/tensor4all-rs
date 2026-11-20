#!/bin/bash
# One seed: patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, with a per-call trace.
# First L2 acceptance (k = 3) with the audit off, then sampled-max acceptance (k = 1, no measurement).
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common="g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock --samples 30000 --seed $seed --cap 64 --trace-calls true"
/usr/bin/time -v timeout 3h taskset -c $core ./bin $common \
  --variants "rtol=1e-2,engine-tol-factor=3,audit=false;rtol=1e-3,engine-tol-factor=3,audit=false;rtol=1e-4,engine-tol-factor=3,audit=false" \
  > s$seed-l2-noaudit.jsonl 2> s$seed-l2-noaudit.time
/usr/bin/time -v timeout 3h taskset -c $core ./bin $common \
  --variants "rtol=1e-3,error-norm=sampled-max,engine-tol-factor=1;rtol=1e-4,error-norm=sampled-max,engine-tol-factor=1;rtol=1e-5,error-norm=sampled-max,engine-tol-factor=1" \
  > s$seed-sampledmax.jsonl 2> s$seed-sampledmax.time
