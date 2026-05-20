# Benchmarks

This directory contains benchmark code for comparing Rust and Julia implementations.

## Structure

- `rust/`: Rust benchmark code using `tensor4all-rs`
- `julia/`: Julia benchmark code using `ITensors.jl` and `ITensorMPS.jl`
- `results/`: saved benchmark commands and representative local outputs

## Running Benchmarks

### Rust

```bash
cd crates/tensor4all-itensorlike
cargo run --release --example benchmark_contract
```

Projected local-operator apply:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_projected_apply --release -- 38 32 32 3 0
```

Prepared local linsolve:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- 38 32 32 1 10 30 0
```

Non-AD local tensor operations:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-core --example benchmark_tensor_ops --release -- 20000 6 2 2 6
```

TensorTrain-level operations against ITensorMPS:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-itensorlike --example benchmark_tt_ops --release -- --L 32 --zipup-L 10 --chis 4,8,16,32,64
```

Inspect Julia-dumped local linsolve inputs:

```bash
cargo run -p tensor4all-hdf5 --example inspect_mps_inputs --release -- benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5
```

### Julia

```bash
cd external/ITensorMPS.jl
julia benchmark/benchmark_contract.jl
```

Projected local-operator apply:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl 38 32 32 3 0
```

Prepared local linsolve:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl 38 32 32 1 1 10
```

Non-AD local tensor operations:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tensor_ops.jl 20000 6 2 2 6
```

TensorTrain-level operations against tensor4all:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tt_ops.jl --L 32 --zipup-L 10 --chis 4,8,16,32,64
```

Dump local linsolve inputs as ITensorMPS-compatible HDF5:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5 38 32 32
```

## Benchmark Details

Both benchmarks perform MPO-MPO contraction using zip-up algorithm:
- Length: 10 sites
- Physical dimension: 2 (input and output per site)
- Bond dimension: 50
- Max rank: 50
- Includes orthogonalization time in measurements

The projected local-operator apply benchmarks isolate the local matvec hot path
used by two-site TreeTN/ITensor-style local solves. The Rust source of truth is
`benchmarks/rust/benchmark_projected_apply.rs`, included by the cargo example
target under `tensor4all-treetn`; the Julia counterpart is
`benchmarks/julia/benchmark_projected_apply.jl`.

The prepared local linsolve benchmarks construct the operator, right-hand side,
and initial state once, then time the local solve body. They also report local
GMRES/apply/RHS/factorization buckets. Use Julia `maxiter=1, krylovdim=10` as a
rough match to Rust's `krylov_maxiter=10` total-iteration cap; KrylovKit's
`maxiter=10, krylovdim=30` performs far more local operator applications.

The non-AD local tensor operation benchmarks isolate `inner`, `norm`, affine
addition, and explicit `conj`-then-contract on a small dense tensor. The default
shape `[6, 2, 2, 6]` mirrors the small two-site local tensors observed in the
QuanticsNEGF long-time local Krylov test, where dispatch/allocation overhead can
dominate floating-point work.

The TensorTrain-level operation benchmarks compare tensor4all's
`TensorTrain::inner`, strict direct-sum MPS addition, and prepared MPO×MPO zipup
contraction against ITensorMPS.jl. They use deterministic Complex64 fixtures and
print CSV-style rows with sample counts, min/median/mean/max milliseconds,
result max bond dimension, and a checksum. The Rust source of truth is
`benchmarks/rust/benchmark_tt_ops.rs`, included by
`tensor4all-itensorlike/examples/benchmark_tt_ops.rs`; the Julia counterpart is
`benchmarks/julia/benchmark_tt_ops.jl`.

`dump_local_linsolve_inputs.jl` writes the prepared local operator as
`operator_as_mps`, plus `rhs` and `init`, in one HDF5 file. The operator is a
Julia `MPO` stored through the `MPS` schema by saving its site tensors as
`MPS([H[i] for i in 1:length(H)])`; Rust reads all three groups with
`tensor4all_hdf5::load_mps`.
