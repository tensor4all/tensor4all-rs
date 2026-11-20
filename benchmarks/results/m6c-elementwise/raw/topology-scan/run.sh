#!/bin/bash
# Usage: run.sh <core> <topology> <temperature>
core=$1; topo=$2; t=$3
timeout 2h /usr/bin/time -v taskset -c $core ./bin g0 unpatched-aci --r 10 --temperature $t --topology $topo --scale-tolerance true --tol 1e-4 --samples 5000 > $topo-T$t.jsonl 2> $topo-T$t.time
