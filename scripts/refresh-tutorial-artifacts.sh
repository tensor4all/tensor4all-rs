#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/docs/tutorial-code"

cargo run --release --bin qtt_function
cargo run --release --bin qtt_interval
cargo run --release --bin qtt_integral
cargo run --release --bin qtt_integral_sweep
cargo run --release --bin qtt_r_sweep
cargo run --release --bin qtt_multivariate
cargo run --release --bin qtt_elementwise_product
cargo run --release --bin qtt_affine
cargo run --release --bin qtt_fourier
cargo run --release --bin qtt_partial_fourier2d

julia --project=docs/plotting docs/plotting/qtt_function_plot.jl
julia --project=docs/plotting docs/plotting/qtt_interval_plot.jl
julia --project=docs/plotting docs/plotting/qtt_integral_sweep_plot.jl
julia --project=docs/plotting docs/plotting/qtt_r_sweep_plot.jl
julia --project=docs/plotting docs/plotting/qtt_multivariate_plot.jl
julia --project=docs/plotting docs/plotting/qtt_elementwise_product_plot.jl
julia --project=docs/plotting docs/plotting/qtt_affine_plot.jl
julia --project=docs/plotting docs/plotting/qtt_fourier_plot.jl
julia --project=docs/plotting docs/plotting/qtt_partial_fourier2d_plot.jl

cd "$repo_root"

cp docs/tutorial-code/docs/plots/qtt_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_function_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_function_vs_qtt.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_interval_bond_dims.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_integral_sweep.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_samples.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_r_sweep_runtime.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_values.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_error.png docs/book/src/tutorials/quantics-basics/
cp docs/tutorial-code/docs/plots/qtt_multivariate_bond_dims.png docs/book/src/tutorials/quantics-basics/

cp docs/tutorial-code/docs/plots/qtt_elementwise_product_factors.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_elementwise_product_product.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_elementwise_product_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_error.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_affine_operator_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_transform.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_fourier_operator_bond_dims.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_values.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_error.png docs/book/src/tutorials/computations-with-qtt/
cp docs/tutorial-code/docs/plots/qtt_partial_fourier2d_bond_dims.png docs/book/src/tutorials/computations-with-qtt/

./scripts/test-mdbook.sh
cargo test --release -p tensor4all-tutorial-code

git status --short docs/tutorial-code/docs/data docs/tutorial-code/docs/plots docs/book/src/tutorials
