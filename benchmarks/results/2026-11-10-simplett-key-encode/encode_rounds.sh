#!/bin/bash
# Encode-only microbench, 3 rotated rounds, pinned to CPU 4, idle-gated.
set -eu
OUT=$HOME/bench-1224-encode
mkdir -p $OUT
export RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
declare -A BIN=(
 [before]=/tmp/t4a-1224-td-before/release/deps/index_key_encode-32ac1248d21cd8ac
 [after]=/tmp/t4a-1224-td-after/release/deps/index_key_encode-32ac1248d21cd8ac
 [unbox]=/tmp/t4a-1224-td-unbox/release/deps/index_key_encode-32ac1248d21cd8ac)
for v in before after unbox; do cp ${BIN[$v]} $OUT/bin-$v; done
sha256sum $OUT/bin-* > $OUT/binaries.sha256
{ uname -a; lscpu | grep -E "Model name|MHz"; rustc -vV; } > $OUT/host.txt 2>&1 || true
l0=$(cut -d' ' -f1 /proc/loadavg); sleep 60; l1=$(cut -d' ' -f1 /proc/loadavg)
echo "idle before=$l0 after=$l1 max=2.0" | tee $OUT/idle.txt
awk -v a=$l0 -v b=$l1 'BEGIN{exit !(a<=2.0 && b<=2.0)}' || { echo "IDLE FAIL; nothing timed" | tee -a $OUT/idle.txt; exit 3; }
r=0
for order in "before after unbox" "unbox before after" "after unbox before"; do
  r=$((r+1))
  for v in $order; do
    echo "round $r $v load_start=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $OUT/loads.txt
    (cd $OUT && CRITERION_HOME=$OUT/criterion-$v-r$r taskset -c 4 $OUT/bin-$v --bench --warm-up-time 0.5 --measurement-time 2 '/(encode|vec_key_hash)/' > $OUT/raw-$v-r$r.txt 2>&1)
    echo "round $r $v load_end=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $OUT/loads.txt
  done
done
echo DONE
