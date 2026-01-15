# Benchmarks

This directory contains benchmark code for comparing Rust and Julia implementations.

## Structure

- `rust/`: Rust benchmark code using `tensor4all-rs`
- `julia/`: Julia benchmark code using `ITensors.jl` and `ITensorMPS.jl`

## Running Benchmarks

### Rust

```bash
cd crates/tensor4all-itensorlike
cargo run --release --example benchmark_contract
```

### Julia

```bash
cd external/ITensorMPS.jl
julia benchmark/benchmark_contract.jl
```

## Benchmark Details

Both benchmarks perform MPO-MPO contraction using zip-up algorithm:
- Length: 10 sites
- Physical dimension: 2 (input and output per site)
- Bond dimension: 50
- Max rank: 50
- Includes orthogonalization time in measurements
