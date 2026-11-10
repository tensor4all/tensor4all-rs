#!/bin/bash
# Run the #1146 driver (final arm swapped per copy) with the #1146 arguments.
# An attempt whose idle gate fails times nothing; the next attempt reuses the builds.
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd ~/t4a-1224-bench
for v in before memo; do
  for a in 1 2 3; do
    out=$HOME/bench-1224-$v-b$a
    echo "=== $v attempt $a -> $out ($(date -Is), load $(cut -d' ' -f1-3 /proc/loadavg))"
    python3 driver-$v.py --repo ~/projects/tensor4all/tensor4all-rs --output $out \
      --cpu 4 --idle-seconds 60 --idle-max-load 2.0 --target-dir ~/t4a-1224-bench/td-$v > $out.log 2>&1
    rc=$?
    echo "rc=$rc"; tail -3 $out.log
    [ -f $out/summary.json ] && break
    grep -q "idle check failed" $out.log || break
  done
done
echo ALLDONE
