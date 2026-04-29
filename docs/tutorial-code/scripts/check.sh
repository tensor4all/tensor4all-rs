#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$root_dir/target/check/data"

mkdir -p "$data_dir"

cd "$root_dir"

cargo fmt --check
cargo check --offline
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_function
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_interval
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_multivariate
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_integral
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_integral_sweep
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_r_sweep
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_elementwise_product
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_affine
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_fourier
TENSOR4ALL_DATA_DIR="$data_dir" cargo run --offline --bin qtt_partial_fourier2d
