#!/bin/bash
# One seed: patched TreeACI on tt_nblock, T = 0.01, R = 12, cap 64, sampled-max acceptance at rtol 1e-3, 1e-4, 1e-5,
# first with k = 1 (repeats nblock-T0.01-R12-acceptance, now recording the pinned max_reference), then with k = 3
# (k-matched with the L2 k = 3 runs there). One process per k, so each has its own peak RSS.
seed=$1; core=$2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common="g0 patched-aci --r 12 --temperature 0.01 --topology tt_nblock --samples 30000 --seed $seed --cap 64 --trace-calls true"
for k in 1 3; do
  /usr/bin/time -v timeout 3h taskset -c $core ./bin $common \
    --variants "rtol=1e-3,error-norm=sampled-max,engine-tol-factor=$k;rtol=1e-4,error-norm=sampled-max,engine-tol-factor=$k;rtol=1e-5,error-norm=sampled-max,engine-tol-factor=$k" \
    > s$seed-sampledmax-k$k.jsonl 2> s$seed-sampledmax-k$k.time
done
