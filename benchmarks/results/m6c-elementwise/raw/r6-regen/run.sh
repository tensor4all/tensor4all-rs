#!/bin/bash
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
v(){ k=$1; echo "rtol=1e-2,engine-tol-factor=$k;rtol=1e-3,engine-tol-factor=$k;rtol=3e-4,engine-tol-factor=$k;rtol=1e-4,engine-tol-factor=$k"; }
mixed="--r 6 --temperature 0.1 --cap 16 --in-cap 16 --in-rtol 1e-4 --in2-cap 8 --in2-rtol 1e-3 --exhaustive true"
g0r6="--r 6 --temperature 0.1 --cap 16 --exhaustive true"
core=$1; shift
case $1 in
  mixed) /usr/bin/time -v taskset -c $core ./bin g0sq-mixed patched-aci $mixed --variants "$(v $2)" > mixed-k$2.jsonl 2> mixed-k$2.time ;;
  g0r6) /usr/bin/time -v taskset -c $core ./bin g0 patched-aci $g0r6 --variants "$(v $2)" > g0-r6-k$2.jsonl 2> g0-r6-k$2.time ;;
  g0r7) /usr/bin/time -v taskset -c $core ./bin g0 patched-aci --r 7 --temperature 0.05 --cap 32 --exhaustive true --variants "rtol=1e-2;rtol=1e-3;rtol=1e-4" > g0-r7.jsonl 2> g0-r7.time ;;
  g0sq) /usr/bin/time -v taskset -c $core ./bin g0sq patched-aci --r 6 --temperature 0.1 --cap 16 --in-cap 16 --in-rtol 1e-4 --exhaustive true --variants "rtol=1e-2;rtol=1e-3;rtol=1e-4" > g0sq-r6.jsonl 2> g0sq-r6.time ;;
  audit) /usr/bin/time -v taskset -c $core ./bin g0sq-mixed patched-aci ${mixed/--exhaustive true/} --rtol $2 --aci-margin $3 --samples 2000 --audit-kernel true > audit-rtol$2-m$3.jsonl 2> audit-rtol$2-m$3.time ;;
esac
