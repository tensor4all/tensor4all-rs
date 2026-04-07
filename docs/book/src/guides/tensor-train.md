# Tensor Train

A **Tensor Train** (TT), also known as a Matrix Product State (MPS), represents a
high-dimensional tensor as a chain of low-rank cores.  tensor4all-rs provides two
complementary implementations:

| Crate | Best for |
|---|---|
| `tensor4all-simplett` | Lightweight numerical work with raw arrays |
| `tensor4all-itensorlike` | ITensors.jl-like Index semantics, orthogonality tracking, canonical forms |

---

## SimpleTT

The `tensor4all-simplett` crate offers a minimal, efficient TT implementation.
It works with plain `f64` and `Complex64` scalars and does not require you to
manage named indices.

### Creating a tensor train

```rust,ignore
use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, TensorTrain};

// Constant TT: all entries equal to 1.0, physical dimensions [2, 3, 4]
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Random TT with bond dimension 4
let tt_rand = TensorTrain::<f64>::random(&[2, 3, 4], 4);
```

### Evaluating and summing

```rust,ignore
// Evaluate the tensor at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;

// Sum over all multi-indices (equivalent to contracting with all-ones vectors)
let total = tt.sum();
```

For the constant TT above `tt.sum()` returns `2 × 3 × 4 = 24`.

### Compressing

`CompressionOptions` controls the accuracy–cost trade-off:

```rust,ignore
let options = CompressionOptions {
    tolerance: 1e-10,
    max_bond_dim: 20,
    ..Default::default()
};
let compressed = tt.compressed(&options)?;
```

The compression reduces bond dimensions while keeping the approximation error
below `tolerance` (relative `L∞` norm), up to `max_bond_dim`.

---

## ITensorLike TensorTrain

The `tensor4all-itensorlike` crate provides a higher-level API modelled after
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).  Each tensor carries
named `DynIndex` objects so that contractions are unambiguous regardless of axis
ordering.

> **Key conventions**
> - Sites are **0-indexed** (Julia uses 1-indexed).
> - `TensorDynLen::from_dense` expects data in **column-major** (Fortran) order.
> - `inner()` computes `<self|other>` with complex conjugation on `self`.

### Creating and orthogonalizing

```rust,ignore
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain, TruncateOptions};

// Site and bond indices
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

// Build site tensors (column-major data)
let t0 = TensorDynLen::from_dense(
    vec![s0.clone(), b01.clone()],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let t1 = TensorDynLen::from_dense(
    vec![b01.clone(), s1.clone(), b12.clone()],
    vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
)?;
let t2 = TensorDynLen::from_dense(
    vec![b12.clone(), s2.clone()],
    vec![1.0, 0.0, 0.0, 1.0],
)?;

// Assemble and orthogonalize with center at site 1
let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
tt.orthogonalize(1)?;

assert!(tt.isortho());
```

### Truncating

After orthogonalization you can truncate bond dimensions by SVD:

```rust,ignore
tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(2))?;
```

`rtol` is the relative truncation tolerance (equivalent to `√cutoff` in
ITensorMPS.jl notation).

### Norm and inner product

```rust,ignore
let norm = tt.norm();
assert!(norm.is_finite());

// <tt|tt> = ‖tt‖²
let inner = tt.inner(&tt);
assert!((inner.real() - norm * norm).abs() < 1e-10);
```

### Complex scalars

The same API works with `Complex64`:

```rust,ignore
use num_complex::Complex64;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::TensorTrain;

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let b  = DynIndex::new_bond(2)?;

let t0 = TensorDynLen::from_dense(
    vec![s0, b.clone()],
    vec![
        Complex64::new(1.0,  0.0), Complex64::new(0.0,  1.0),
        Complex64::new(0.0, -1.0), Complex64::new(1.0,  0.0),
    ],
)?;
let t1 = TensorDynLen::from_dense(
    vec![b, s1],
    vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
    ],
)?;

let tt = TensorTrain::new(vec![t0, t1])?;
assert!(tt.norm() > 0.0);
```

---

## Differences from ITensorMPS.jl

| Feature | tensor4all-itensorlike | ITensorMPS.jl |
|---|---|---|
| Site indexing | 0-indexed | 1-indexed |
| Canonical forms | Unitary (QR), LU, CI | Unitary (SVD) |
| Conjugation | `conj` | `dag` (with QN direction flip) |
| Sweep counting | `nhalfsweeps` | `nsweeps` |
