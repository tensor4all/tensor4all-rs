# tensor4all-itensorlike

ITensors.jl-inspired TensorTrain API with orthogonality tracking and multiple canonical forms.

## Key Types

- `TensorTrain` — tensor train with orthogonality center tracking
- `orthogonalize()` — move orthogonality center to a given site
- `truncate()` — bond dimension truncation (SVD, LU, or CI)
- `inner()` — inner product `<self|other>` with complex conjugation on `self`
- `norm()` — efficient norm via the orthogonality center

## Example

```rust,ignore
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions};

// Build a 3-site tensor train
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
let t1 = TensorDynLen::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 8])?;
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0, 0.0, 0.0, 1.0])?;

let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
tt.orthogonalize(1)?;
tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(2))?;

assert!(tt.isortho());
let norm = tt.norm();
assert!(norm.is_finite());

// Inner product: <tt|tt> = norm^2
let inner = tt.inner(&tt);
assert!((inner.real() - norm * norm).abs() < 1e-10);
```

## Documentation

- [User Guide: Tensor Train](https://tensor4all.github.io/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_itensorlike/)
