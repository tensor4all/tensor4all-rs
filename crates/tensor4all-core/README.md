# tensor4all-core

Foundation library providing core data structures and algorithms for tensor operations. It implements tensor types, index management, and fundamental linear algebra operations.

## Features

- **Index**: Flexible tensor index with identity, symmetry info, and tags
- **Tensor**: Dynamic-rank tensor (`TensorDynLen`) with flexible index types
- **Storage**: Dense and diagonal storage backends for `f64` and `Complex64`
- **Factorization**: SVD, QR, LU, and CI decompositions with truncation support
- **Contraction**: Optimal multi-tensor contraction via `contract_multi()`

## Usage

```rust
use tensor4all_core::{Index, TensorDynLen, factorize, FactorizeOptions};

// Create indices
let i = Index::new_dyn(3).with_tags("i");
let j = Index::new_dyn(4).with_tags("j");

// Create a random tensor
let tensor = TensorDynLen::random(&mut rng, &[i.clone(), j.clone()]);

// SVD factorization with truncation
let result = factorize(
    &tensor,
    &[i],
    &FactorizeOptions::svd().with_rtol(1e-10)
)?;
```

## License

MIT License
