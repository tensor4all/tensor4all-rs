#!/bin/bash
# Run Julia tests for Tensor4all.jl
#
# This script clones Tensor4all.jl, builds using local tensor4all-rs,
# and runs Julia tests.
#
# Usage:
#   ./scripts/run_julia_tests.sh                  # Build with all features
#   ./scripts/run_julia_tests.sh --no-hdf5        # Build without HDF5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
SKIP_HDF5=false
for arg in "$@"; do
    case "$arg" in
        --no-hdf5)
            SKIP_HDF5=true
            ;;
    esac
done

# Clone or update Tensor4all.jl
JULIA_PKG_DIR="$REPO_ROOT/.julia-tests/Tensor4all.jl"
if [[ -d "$JULIA_PKG_DIR" ]]; then
    echo "=== Updating Tensor4all.jl ==="
    cd "$JULIA_PKG_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "=== Cloning Tensor4all.jl ==="
    mkdir -p "$REPO_ROOT/.julia-tests"
    git clone https://github.com/tensor4all/Tensor4all.jl.git "$JULIA_PKG_DIR"
fi

cd "$REPO_ROOT"

echo "=== Running Julia tests (using local tensor4all-rs) ==="
export TENSOR4ALL_RS_PATH="$REPO_ROOT"

cd "$JULIA_PKG_DIR"

if [ "$SKIP_HDF5" = true ]; then
    echo "HDF5 disabled: skipping HDF5 tests"
    export T4A_SKIP_HDF5_TESTS=1
fi

julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.build()
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
    TENSOR4ALL_RS_PATH="$REPO_ROOT" julia --project="$JULIA_PKG_DIR" "$f"
done

echo "=== All tests passed ==="
