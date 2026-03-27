# tensor4all-itensorlike

ITensors.jl-inspired high-level API for Tensor Train (MPS) operations with orthogonality tracking. Provides an ergonomic interface similar to the Julia ITensorMPS.jl library.

## Features

- **TensorTrain**: Tensor train with orthogonality center tracking
- **Multiple canonical forms**: Unitary (QR), LU, or CI canonicalization
- **Truncation**: Bond dimension truncation with configurable tolerance
- **Contraction**: Contract two tensor trains (zipup, fit, or naive method)
- **Inner products**: Efficient norm and inner product computation

## Important Conventions

- **Column-major data layout**: `TensorDynLen::from_dense` expects data in column-major
  (Fortran) order. For a 2x3 matrix, provide `[a00, a10, a01, a11, a02, a12]`.
- **Inner product conjugation**: `inner()` computes `<self|other>` with complex conjugation
  on `self` (the left argument).
- **0-indexed sites**: Sites are 0-indexed (unlike Julia's 1-indexed convention).

## Usage

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen, TensorLike};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions, CanonicalForm};

// Create indices: site indices + bond indices
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

// Build tensors (column-major data layout)
let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
let t1 = TensorDynLen::from_dense(
    vec![b01.clone(), s1.clone(), b12.clone()],
    vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
)?;
let t2 = TensorDynLen::from_dense(vec![b12.clone(), s2.clone()], vec![1.0, 0.0, 0.0, 1.0])?;

// Create tensor train and orthogonalize
let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
tt.orthogonalize(1)?;

// Truncate with SVD
tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(2))?;

assert!(tt.isortho());

// Norm and inner product
let norm = tt.norm();
assert!(norm.is_finite());

// Inner product: <tt|tt> = norm^2
let inner = tt.inner(&tt);
assert!((inner.real() - norm * norm).abs() < 1e-10);
# Ok(())
# }
```

### Complex64 Example

```rust
# fn main() -> anyhow::Result<()> {
use num_complex::Complex64;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::TensorTrain;

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let b = DynIndex::new_bond(2)?;

let t0 = TensorDynLen::from_dense(
    vec![s0, b.clone()],
    vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),
         Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)],
)?;
let t1 = TensorDynLen::from_dense(
    vec![b, s1],
    vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
         Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
)?;

let tt = TensorTrain::new(vec![t0, t1])?;
let norm = tt.norm();
assert!(norm > 0.0);
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
- **Convenience**: `with_nsweeps(n)` is equivalent to `with_nhalfsweeps(2 * n)`

## License

MIT License
