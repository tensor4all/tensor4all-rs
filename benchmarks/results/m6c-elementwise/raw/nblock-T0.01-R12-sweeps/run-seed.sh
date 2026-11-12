#!/bin/bash
# One seed: unpatched TreeACI on tt_nblock, T = 0.01, R = 12, absolute tol 1e-3, 1e-4, 1e-5,
# each stopped after 10, 20 and 30 sweeps (prefixes of the 50-sweep runs of nblock-T0.01-R12).
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
v=""
for tol in 1e-3 1e-4 1e-5; do for s in 10 20 30; do v="$v;tol=$tol,max-sweeps=$s"; done; done
/usr/bin/time -v timeout 4h taskset -c $core ./bin g0 unpatched-aci --r 12 --temperature 0.01 --topology tt_nblock \
  --samples 30000 --seed $seed --variants "${v#;}" \
  > s$seed-unpatched.jsonl 2> s$seed-unpatched.time
