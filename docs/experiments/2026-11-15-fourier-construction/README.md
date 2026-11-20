# Fourier construction measurements

The [raw output](raw-output.txt) records the November 15 runs on an Apple M3
Pro with Rust 1.98.1 and Accelerate. `baseline.log` and
`grouped-baseline.log` are from `1647bc66`. `candidate.log` and
`grouped-candidate.log` are from the Fourier implementation in `dcda800c`.
The [construction design](../../design/treetn-fourier-construction.md)
summarizes the measurements.

Run these commands from each revision's checkout on macOS. Point `probe` at
the same [probe source](probe.rs) for both builds.

```bash
set -euo pipefail
probe=/absolute/path/to/probe.rs
build=$(mktemp -d)
cargo build --release --locked -p tensor4all-quanticstransform \
    --features partitioned --example patched_fourier_large \
    --target-dir "$build" --message-format=json |
    jq -r 'select(.reason == "compiler-artifact" and
        .target.name == "tensor4all_quanticstransform") |
        .filenames[] | select(endswith(".rlib"))' > "$build/rlib"
rustc --edition 2021 -C opt-level=3 -L "dependency=$build/release/deps" \
    --extern "tensor4all_quanticstransform=$(cat "$build/rlib")" \
    "$probe" -o "$build/probe"
export RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for bits in 10 20; do
    for case in '1 grouped' '2 grouped' '2 interleaved'; do
        read -r axes layout <<< "$case"
        for run in 1 2 3 4 5; do
            /usr/bin/time -l "$build/probe" "$bits" "$axes" "$layout"
        done
    done
done
for bits in 10 20; do
    for axes in 1 2; do
        "$build/release/examples/patched_fourier_large" "$bits" "$axes" grouped frontend-only
    done
done
```

Each probe process reports the median in milliseconds from 15 constructions
after one warmup. For each layout and revision, take the median, minimum, and
maximum of the five reported times. Take the median of the five maximum
resident set sizes and divide by `2^20` to get MiB. Compare the full `bonds`
vectors. The grouped example reports the sampled error against the analytic
spectrum and the scheduler reconstruction bound as separate fields.
