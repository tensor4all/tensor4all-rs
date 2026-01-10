# tensor4all-core

Foundation library providing core data structures and algorithms for tensor operations. It implements tensor types, index management, and fundamental linear algebra operations.

## Features

- **Index**: Flexible tensor index with identity and tags (symmetry/quantum numbers are intentionally not in the default implementation)
- **Tensor**: Dynamic-rank tensor (`TensorDynLen`) with flexible index types
- **Storage**: Dense and diagonal storage backends for `f64` and `Complex64`
- **Factorization**: SVD, QR, LU, and CI decompositions with truncation support
- **Contraction**: Optimal multi-tensor contraction via `contract_multi()`

## Usage

```rust
use anyhow::Result;
use rand::thread_rng;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{factorize, FactorizeOptions, TensorDynLen};

// Create indices
let i = Index::<DynId>::new_dyn_with_tag(3, "i")?;
let j = Index::<DynId>::new_dyn_with_tag(4, "j")?;

// Create a random tensor
let mut rng = thread_rng();
let tensor = TensorDynLen::random_f64(&mut rng, vec![i.clone(), j.clone()]);

// SVD factorization with truncation
let result = factorize(
    &tensor,
    &[i],
    &FactorizeOptions::svd().with_rtol(1e-10)
)?;

# Ok::<(), anyhow::Error>(())
```

## License

MIT License
