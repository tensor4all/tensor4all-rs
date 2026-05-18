# HDF5 input dump for local linsolve parity

## Commands

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5 8 4 4
cargo run -p tensor4all-hdf5 --example inspect_mps_inputs --release -- benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5

BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5 38 32 32
cargo run -p tensor4all-hdf5 --example inspect_mps_inputs --release -- benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5
```

## Findings

- Julia can write the prepared local operator and states in one
  ITensorMPS-compatible HDF5 file.
- The operator is a Julia `MPO`, stored as `operator_as_mps` via
  `MPS([H[i] for i in 1:length(H)])`.
- Rust reads `operator_as_mps`, `rhs`, and `init` with
  `tensor4all_hdf5::load_mps`.
- For `N=38, state_bond_dim=32, operator_bond_dim=32`, Rust sees:
  - `operator_as_mps.length = 38`
  - `operator_as_mps.bond_dims = [32, ..., 32]`
  - `rhs.length = init.length = 38`
  - `rhs.bond_dims = init.bond_dims = [32, ..., 32]`
- Raw site tensors loaded with `load_itensor("operator_as_mps/MPS[i]")`
  preserve the Julia HDF5 index order. Loading the whole object with
  `load_mps` normalizes site tensors into `TensorTrain` chain order, so
  endpoint link indices may move relative to the raw HDF5 order. The index
  identities, prime levels, dimensions, and tags are preserved.

## Generated files

The generated `.h5` files are local benchmark artifacts and are ignored by git:

- `benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5`
- `benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5`
