#!/bin/bash
# Run Python tests for tensor4all (Python package)
#
# Usage:
#   ./scripts/run_python_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Building tensor4all-capi ==="
cargo build --release -p tensor4all-capi

echo "=== Setting up Python environment ==="
cd "$REPO_ROOT/python/tensor4all"

# Install dependencies and package
if command -v uv &> /dev/null; then
    uv run python scripts/build_capi.py
    echo "Using uv..."
    uv sync --all-extras
    uv pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    uv run pytest -v
else
    python3 scripts/build_capi.py
    echo "Using pip..."
    python3 -m pip install --upgrade pip
    pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    pytest -v
fi

echo "=== All Python tests passed ==="
