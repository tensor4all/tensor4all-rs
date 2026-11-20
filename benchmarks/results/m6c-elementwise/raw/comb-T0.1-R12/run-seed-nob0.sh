#!/bin/bash
# One seed without B0: patched TreeACI (k=1 and k=3) and unpatched TreeACI.
seed=$1; c1=$2; c2=$3
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common="--r 12 --temperature 0.1 --topology cttn --samples 30000 --seed $seed"
timeout 4h /usr/bin/time -v taskset -c $c1 ./bin g0 patched-aci $common --cap 64 \
  --variants "rtol=1e-2;rtol=1e-3;rtol=1e-4;rtol=1e-2,engine-tol-factor=3;rtol=1e-3,engine-tol-factor=3;rtol=1e-4,engine-tol-factor=3" \
  > s$seed-patched.jsonl 2> s$seed-patched.time &
timeout 4h /usr/bin/time -v taskset -c $c2 ./bin g0 unpatched-aci $common \
  --variants "tol=1e-3;tol=3e-4;tol=1e-4;tol=3e-5;tol=1e-5" \
  > s$seed-unpatched.jsonl 2> s$seed-unpatched.time &
wait
