# tensor4all-core

Foundation library providing core data structures and algorithms for tensor operations. It implements tensor types, index management, and fundamental linear algebra operations.

## Features

- **Index**: Flexible tensor index with identity and tags (symmetry/quantum numbers are intentionally not in the default implementation)
- **Tensor**: Dynamic-rank tensor (`TensorDynLen`) with flexible index types
- **Storage**: Dense and diagonal storage backends for `f64` and `Complex64`
- **Factorization**: SVD, QR, LU, and CI decompositions with truncation support
- **Contraction**: Optimal multi-tensor contraction via `contract_multi()` with selective pair control

## Usage

```rust
use anyhow::Result;
use rand::rng;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{factorize, FactorizeOptions, TensorDynLen};

// Create indices
let i = Index::<DynId>::new_dyn_with_tag(3, "i")?;
let j = Index::<DynId>::new_dyn_with_tag(4, "j")?;

// Create a random tensor
let mut rng = rng();
let tensor = TensorDynLen::random_f64(&mut rng, vec![i.clone(), j.clone()]);

// SVD factorization with truncation
let result = factorize(
    &tensor,
    &[i],
    &FactorizeOptions::svd().with_rtol(1e-10)
)?;

# Ok::<(), anyhow::Error>(())
```

## Tensor Contraction

The library provides flexible multi-tensor contraction with control over which tensor pairs are allowed to contract:

```rust
use anyhow::Result;
use rand::rng;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{contract_multi, contract_connected, AllowedPairs, TensorDynLen};

// Create tensors A(i,j), B(j,k), C(k,l)
let i = Index::<DynId>::new_dyn_with_tag(2, "i")?;
let j = Index::<DynId>::new_dyn_with_tag(3, "j")?;
let k = Index::<DynId>::new_dyn_with_tag(4, "k")?;
let l = Index::<DynId>::new_dyn_with_tag(5, "l")?;

let mut rng = rng();
let a = TensorDynLen::random_f64(&mut rng, vec![i.clone(), j.clone()]);
let b = TensorDynLen::random_f64(&mut rng, vec![j.clone(), k.clone()]);
let c = TensorDynLen::random_f64(&mut rng, vec![k.clone(), l.clone()]);

// Contract all tensor pairs (default behavior)
let result = contract_multi(&[a.clone(), b.clone(), c.clone()], AllowedPairs::All)?;

// Contract only specified pairs (useful for tree tensor networks)
// Only A-B and B-C are connected, so j and k are contracted
let pairs = [(0, 1), (1, 2)];  // tensor indices
let result = contract_multi(&[a.clone(), b.clone(), c.clone()], AllowedPairs::Specified(&pairs))?;

// contract_connected requires the tensor graph to be connected (errors on disconnected)
let result = contract_connected(&[a, b, c], AllowedPairs::All)?;

# Ok::<(), anyhow::Error>(())
```

### AllowedPairs

- `AllowedPairs::All`: Contract all tensor pairs with matching indices (default behavior)
- `AllowedPairs::Specified(&[(usize, usize)])`: Only contract indices between specified tensor pairs

### contract_multi vs contract_connected

- `contract_multi`: Handles disconnected tensor graphs by combining components via outer product
- `contract_connected`: Requires the tensor graph to be connected; returns an error otherwise

## License

MIT License
