# tensor4all-itensorlike

ITensors.jl-inspired high-level API for Tensor Train (MPS) operations with orthogonality tracking. Provides an ergonomic interface similar to the Julia ITensorMPS.jl library.

## Features

- **TensorTrain**: Tensor train with orthogonality center tracking
- **Multiple canonical forms**: Unitary (QR), LU, or CI canonicalization
- **Truncation**: Bond dimension truncation with configurable tolerance
- **Contraction**: Contract two tensor trains (zipup, fit, or naive method)
- **Inner products**: Efficient norm and inner product computation

## Usage

```rust
use tensor4all_itensorlike::{TensorTrain, TruncateOptions, CanonicalForm};

// Create tensor train from tensors
let tt = TensorTrain::new(tensors)?;

// Orthogonalize to site 2 using QR
tt.orthogonalize(2)?;

// Truncate with SVD
tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(20))?;

// Compute norm
let norm = tt.norm();
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
