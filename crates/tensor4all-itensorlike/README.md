# tensor4all-itensorlike

ITensors.jl-inspired high-level API for Tensor Train (MPS) operations with orthogonality tracking. Provides an ergonomic interface similar to the Julia ITensorMPS.jl library.

## Features

- **TensorTrain**: Tensor train with orthogonality center tracking
- **Multiple canonical forms**: Unitary (QR), LU, or CI canonicalization
- **Truncation**: Bond dimension truncation with configurable tolerance
- **Contraction**: Contract two tensor trains (zipup, fit, or naive method)
- **Inner products**: Efficient norm and inner product computation

## Usage

This low-level constructor takes `TensorDynLen` values from `tensor4all-core`.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions};

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
let t1 = TensorDynLen::from_dense(
    vec![b01.clone(), s1.clone(), b12.clone()],
    vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
)?;
let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], vec![1.0, 0.0, 0.0, 1.0])?;

let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
tt.orthogonalize(1)?;
tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(2))?;

assert!(tt.isortho());
assert!(tt.norm().is_finite());
# Ok(())
# }
```

## Differences from ITensorMPS.jl

- 0-indexed sites (Julia uses 1-indexed)
- Multiple canonical forms: Unitary (QR), LU, CI
- Uses `conj` instead of `dag` (no QN direction flipping)
- **Sweep counting**: Uses `nhalfsweeps` (number of half-sweeps, must be a multiple of 2)
  - A half-sweep visits edges in one direction only (forward or backward)
  - `nhalfsweeps=2` means 1 full sweep (forward + backward)
  - This matches ITensorMPS.jl's `nsweeps` semantics when `reverse_step=false`

## License

MIT License
