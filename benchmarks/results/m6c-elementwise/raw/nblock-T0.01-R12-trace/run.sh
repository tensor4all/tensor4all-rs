#!/bin/bash
# Seed 42, patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, k = 3, rtol 1e-3, with a per-call engine trace.
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v timeout 2h taskset -c 4 ./bin g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock \
  --samples 30000 --seed 42 --cap 64 --rtol 1e-3 --engine-tol-factor 3 --trace-calls true \
  > s42-patched-trace.jsonl 2> s42-patched-trace.time
