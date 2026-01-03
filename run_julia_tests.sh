#!/bin/bash
# Run Julia tests for Tensor4all.jl
#
# This script builds the Rust library and runs Julia tests.
# Designed to be used both locally and in CI.
#
# Usage:
#   ./run_julia_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building tensor4all-capi (release) ==="
cargo build --release -p tensor4all-capi

echo "=== Copying library to Julia deps ==="
JULIA_PKG_DIR="$SCRIPT_DIR/Tensor4all.jl"
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
SRC_LIB="$SCRIPT_DIR/target/release/$LIB_NAME"
DST_LIB="$DEPS_DIR/$LIB_NAME"

if [[ ! -f "$SRC_LIB" ]]; then
    echo "ERROR: Library not found at $SRC_LIB"
    exit 1
fi

cp "$SRC_LIB" "$DST_LIB"
echo "Copied $LIB_NAME to $DEPS_DIR"

echo "=== Running Julia tests ==="
cd "$JULIA_PKG_DIR"
julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.test()
'

echo "=== All tests passed ==="
