#!/bin/bash
# One seed: patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, sampled-max acceptance with the
# searched root reference (#1274, no max-reference given), at rtol 1e-5, 1e-6, 1e-7, first with k = 1,
# then with k = 3 (one process per k): the nblock-T0.01-R12-given protocol without max-reference=exact.
seed=$1; core=$2; k=$3
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common="g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock --samples 30000 --seed $seed --cap 64 --trace-calls true"
v=""
for rtol in 1e-5 1e-6 1e-7; do v="$v;rtol=$rtol,error-norm=sampled-max,engine-tol-factor=$k"; done
/usr/bin/time -v timeout 3h taskset -c $core ./bin $common --variants "${v#;}" \
  > s$seed-searched-k$k.jsonl 2> s$seed-searched-k$k.time
