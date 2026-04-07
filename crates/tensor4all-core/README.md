# tensor4all-core

Core tensor library: Index system, dynamic-rank Tensor, contraction, SVD/QR/LU factorization.

## Key Types

- `Index` — flexible index with tags and prime levels
- `TensorDynLen` — dynamic-rank tensor with flexible index types
- `Storage` — dense or diagonal storage for `f64` and `Complex64`
- `contract()` / `contract_multi()` — pairwise and multi-tensor contraction
- `svd()` / `qr()` — factorizations with truncation support

## Example

```rust,ignore
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{factorize, FactorizeOptions, TensorDynLen};
use rand::rng;

// Create indices
let i = Index::<DynId>::new_dyn_with_tag(3, "i")?;
let j = Index::<DynId>::new_dyn_with_tag(4, "j")?;

// Create a random tensor
let mut rng = rng();
let tensor = TensorDynLen::random_f64(&mut rng, vec![i.clone(), j.clone()]);
assert_eq!(tensor.ndim(), 2);

// SVD factorization with truncation
let result = factorize(
    &tensor,
    &[i],
    &FactorizeOptions::svd().with_rtol(1e-10),
)?;
```

## Documentation

- [User Guide: Tensor Basics](https://tensor4all.github.io/tensor4all-rs/guides/tensor-basics.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_core/)
