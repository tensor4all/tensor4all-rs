#!/bin/bash
# Run Julia tests for Tensor4all.jl
#
# This script builds the Rust library and runs Julia tests.
# Designed to be used both locally and in CI.
#
# Usage:
#   ./scripts/run_julia_tests.sh                  # Build with all features
#   ./scripts/run_julia_tests.sh --no-hdf5        # Build without HDF5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
CARGO_FEATURES=""
SKIP_HDF5=false
for arg in "$@"; do
    case "$arg" in
        --no-hdf5)
            CARGO_FEATURES="--no-default-features"
            SKIP_HDF5=true
            ;;
    esac
done

echo "=== Building tensor4all-capi (release) ==="
cargo build --release -p tensor4all-capi $CARGO_FEATURES

echo "=== Copying library to Julia deps ==="
JULIA_PKG_DIR="$REPO_ROOT/julia/Tensor4all.jl"
DEPS_DIR="$JULIA_PKG_DIR/deps"
mkdir -p "$DEPS_DIR"

# Detect library extension based on OS
case "$(uname -s)" in
    Linux*)  LIB_EXT="so" ;;
    Darwin*) LIB_EXT="dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIB_EXT="dll" ;;
    *)       echo "Unknown OS"; exit 1 ;;
esac

LIB_NAME="libtensor4all_capi.$LIB_EXT"
SRC_LIB="$REPO_ROOT/target/release/$LIB_NAME"
DST_LIB="$DEPS_DIR/$LIB_NAME"

if [[ ! -f "$SRC_LIB" ]]; then
    echo "ERROR: Library not found at $SRC_LIB"
    exit 1
fi

cp "$SRC_LIB" "$DST_LIB"
echo "Copied $LIB_NAME to $DEPS_DIR"

echo "=== Running Julia tests ==="
cd "$JULIA_PKG_DIR"

if [ "$SKIP_HDF5" = true ]; then
    echo "HDF5 disabled: skipping HDF5 tests"
    export T4A_SKIP_HDF5_TESTS=1
fi

julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.test()
'

echo "=== Running Julia doc examples ==="
cd "$REPO_ROOT"
for f in docs/examples/julia/*.jl; do
    if [ "$SKIP_HDF5" = true ] && [[ "$f" == *"/hdf5.jl" ]]; then
        echo "=== Skipping $f (HDF5 disabled) ==="
        continue
    fi
    echo "=== Running $f ==="
    julia --project=julia/Tensor4all.jl "$f"
done

echo "=== All tests passed ==="
