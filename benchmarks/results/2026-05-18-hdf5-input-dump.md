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
  preserve the Julia HDF5 index order. After `de51179` identified the
  unwanted behavior and the follow-up TensorTrain fix, loading the whole object
  with `load_mps` also preserves the site tensor index order. The index
  identities, prime levels, dimensions, tags, and axis order are preserved.

## TensorTrain index-order normalization relaxation

The old `TensorTrain` constructor path permuted site tensor indices into a
chain-friendly convention:

```text
[left_link, site_indices..., right_link]
```

This can be useful for APIs that require a simple chain layout, likely including
conversion to or from `tensor4all-simplett::TensorTrain`. It is not needed at
the ITensor-like `TensorTrain` boundary itself. In particular, HDF5
interoperability and Julia/Rust parity debugging require `load_mps` to preserve
the raw ITensors.jl site tensor index order.

Implemented policy:

- Keep raw `ITensor` HDF5 load/store order-preserving.
- Keep `TensorTrain::new`, `TensorTrain::with_ortho`, `TensorTrain::from_treetn`,
  and `TensorTrain::set_tensor_checked` order-preserving.
- Keep chain-order normalization explicit and local to operations that actually
  require it. The current `norm_squared_fast_path` still applies it only to a
  clone before packing sites.
- Do not reorder fit-contraction inputs to satisfy an algorithm precondition.
  Fit contraction is covered by a regression with non-chain-ordered site tensor
  axes and must rely on index identity/topology rather than axis position.
- If a future SimpleTT conversion or dense evaluation path needs canonical chain
  axis order, move the `permuteinds` step into that conversion path or expose an
  explicitly named helper such as `into_chain_ordered`.
- Regression tests now cover constructor, `with_ortho`, `from_treetn`, setter,
  and HDF5 `load_mps` order preservation.

## Generated files

The generated `.h5` files are local benchmark artifacts and are ignored by git:

- `benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5`
- `benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5`
