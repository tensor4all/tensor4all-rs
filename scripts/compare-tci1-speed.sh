#!/usr/bin/env bash
set -euo pipefail

JULIA_REPO="${1:-../TensorCrossInterpolation.jl}"
REPEATS="${T4A_TCI1_BENCH_REPEATS:-5}"
THRESHOLD="${T4A_TCI1_SPEED_RATIO_THRESHOLD:-2.0}"

if ! command -v julia >/dev/null 2>&1; then
  echo "error: julia executable not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$JULIA_REPO" ]]; then
  echo "error: Julia repo not found: $JULIA_REPO" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

RUST_OUT="$TMPDIR/rust.out"
JULIA_OUT="$TMPDIR/julia.out"
JULIA_SCRIPT="$TMPDIR/tci1_speed.jl"

cargo run --release -p tensor4all-tensorci --example tci1_speed -- --repeats "$REPEATS" | tee "$RUST_OUT"

cat > "$JULIA_SCRIPT" <<'JULIA'
using Statistics
using TensorCrossInterpolation

function lorentz(v)
    return 1.0 / (sum(abs2, v) + 1.0)
end

function run_case(; repeats::Int)
    name = "lorentz5d_d10_tol1e-8_maxiter8"
    localdims = fill(10, 5)
    firstpivot = ones(Int, 5)
    kwargs = (
        tolerance = 1e-8,
        maxiter = 8,
        sweepstrategy = :backandforth,
        normalizeerror = true,
    )

    TensorCrossInterpolation.crossinterpolate1(Float64, lorentz, localdims, firstpivot; kwargs...)

    durations = Float64[]
    rank_value = 0
    last_error = 0.0
    for _ in 1:repeats
        GC.gc()
        start = time_ns()
        tci, ranks, errors = TensorCrossInterpolation.crossinterpolate1(
            Float64, lorentz, localdims, firstpivot; kwargs...
        )
        push!(durations, (time_ns() - start) / 1e9)
        rank_value = TensorCrossInterpolation.rank(tci)
        last_error = isempty(errors) ? 0.0 : errors[end]
    end

    sort!(durations)
    median_seconds = durations[cld(length(durations), 2)]
    best_seconds = first(durations)
    println(
        "impl=julia case=$(name) repeats=$(repeats) median_seconds=$(median_seconds) " *
        "best_seconds=$(best_seconds) rank=$(rank_value) last_error=$(last_error)"
    )
end

repeats = parse(Int, get(ENV, "T4A_TCI1_BENCH_REPEATS", "5"))
run_case(; repeats)
JULIA

T4A_TCI1_BENCH_REPEATS="$REPEATS" julia --project="$JULIA_REPO" "$JULIA_SCRIPT" | tee "$JULIA_OUT"

python3 - "$RUST_OUT" "$JULIA_OUT" "$THRESHOLD" <<'PY'
import re
import sys

rust_path, julia_path, threshold_text = sys.argv[1:4]
threshold = float(threshold_text)
pattern = re.compile(r"impl=(?P<impl>\w+) case=(?P<case>\S+) .*median_seconds=(?P<median>[0-9.eE+-]+)")

def load(path):
    data = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            match = pattern.search(line)
            if match:
                data[match.group("case")] = float(match.group("median"))
    return data

rust = load(rust_path)
julia = load(julia_path)
missing = sorted(set(rust) ^ set(julia))
if missing:
    print(f"error: benchmark case mismatch: {missing}", file=sys.stderr)
    sys.exit(1)

failed = False
for case in sorted(rust):
    ratio = rust[case] / julia[case]
    print(
        f"case={case} rust_median_seconds={rust[case]:.9f} "
        f"julia_median_seconds={julia[case]:.9f} ratio={ratio:.3f} threshold={threshold:.3f}"
    )
    if ratio > threshold:
        failed = True

if failed:
    print("error: Rust TCI1 median runtime exceeded the Julia threshold", file=sys.stderr)
    sys.exit(1)
PY
