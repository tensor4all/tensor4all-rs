# Tensor Basics

This guide covers the `tensor4all-core` crate, which provides the foundation for
all tensor operations: indices, tensors, contraction, and factorization.

Add it to your `Cargo.toml`:

```toml
[dependencies]
tensor4all-core = "0.1"
```

## Index

Every tensor axis is identified by an `Index`. Indices carry a unique identity
(so two indices with the same dimension are still distinct), an optional tag, and
an optional prime level.

```rust
use tensor4all_core::index::{Index, DynId};
use tensor4all_core::IndexLike; // needed for .dim() and .plev()

// Simplest form: just give a dimension.
let i = Index::new_dyn(3);   // dimension 3, auto-generated ID, no tags
let j = Index::new_dyn(4);

// A tag names the index (useful for debugging and tag-based operations).
let site = Index::new_dyn_with_tag(2, "Site").unwrap();
assert_eq!(site.dim(), 2);

// Two indices created independently are always distinct, even with the same dim.
let a = Index::new_dyn(3);
let b = Index::new_dyn(3);
assert_ne!(a, b);

// Prime levels distinguish related indices (e.g. ket vs bra in quantum physics).
let bra = site.prime();       // plev 0 -> 1
let ket = site.noprime();     // always plev 0
assert_ne!(bra, ket);

// Inspect properties.
assert_eq!(site.dim(), 2);
assert_eq!(site.plev(), 0);
assert_eq!(bra.plev(), 1);
```

## Tensor (TensorDynLen)

`TensorDynLen` is a dynamic-rank tensor parameterized by a list of `Index`
values and backed by compact storage that may be dense, diagonal, or explicitly
structured. Each index uniquely identifies an axis; there is no fixed axis
ordering in the abstract sense — operations match axes by index identity.

### Creating tensors

```rust
use tensor4all_core::{TensorDynLen, Index};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);

// From explicit column-major data (2×3 tensor, 6 elements).
let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
assert_eq!(t.dims(), vec![2, 3]);

// All-zeros tensor.
let zeros = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

// Random tensor (standard normal).
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
let mut rng = ChaCha8Rng::seed_from_u64(42);
let rand_t: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]);
assert_eq!(rand_t.dims(), vec![2, 3]);
```

### Extracting data

```rust
use tensor4all_core::{TensorDynLen, Index};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let data = vec![10.0_f64, 20.0];
let t = TensorDynLen::from_dense(vec![i], data).unwrap();

// Extract all elements in column-major order.
let out: Vec<f64> = t.to_vec().unwrap();
assert_eq!(out, vec![10.0, 20.0]);

// Sum all elements.
let s = t.sum();
assert_eq!(s.real(), 30.0);
```

## Contraction

Contraction sums over all shared (common) indices between two or more tensors.
Think of it as a generalization of matrix multiplication.

### Pairwise contraction

```rust
use tensor4all_core::{TensorDynLen, Index};
use tensor4all_core::index::DynId;

// A[i,j] and B[j,k] — contracting over j gives C[i,k].
let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);

let a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();
let b = TensorDynLen::zeros::<f64>(vec![j.clone(), k.clone()]).unwrap();

let c = a.contract(&b);      // or equivalently: &a * &b
assert_eq!(c.dims(), vec![2, 4]);  // j is summed away
```

### Multi-tensor contraction

`contract_multi` contracts a list of tensors, handling disconnected components
via outer products. `contract_connected` is the same but returns an error if the
contraction graph is disconnected.

```rust
use tensor4all_core::{TensorDynLen, Index, contract_multi, contract_connected, AllowedPairs};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);
let l = Index::new_dyn(5);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(0)
};
let a: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]);
let b: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![j.clone(), k.clone()]);
let c: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![k.clone(), l.clone()]);

// Contract A(i,j) * B(j,k) * C(k,l) -> result(i,l)
let result = contract_multi(&[&a, &b, &c], AllowedPairs::All).unwrap();
assert_eq!(result.dims().iter().product::<usize>(), 2 * 5); // i * l

// Restrict which tensor pairs may contract (useful for tree tensor networks).
// Here only (A,B) and (B,C) are connected, so j and k are contracted.
let pairs = [(0usize, 1usize), (1, 2)];
let result2 = contract_multi(&[&a, &b, &c], AllowedPairs::Specified(&pairs)).unwrap();
assert_eq!(result2.dims().iter().product::<usize>(), 2 * 5);
```

`AllowedPairs::All` contracts all tensor pairs with matching indices.
`AllowedPairs::Specified` takes a slice of `(usize, usize)` tensor-index pairs
and only contracts between those pairs — useful when the connectivity is known
(e.g. tree tensor networks).

## Factorization

The unified `factorize()` function dispatches to SVD, QR, LU, or CI based on
`FactorizeOptions`. The result splits the input tensor into a `left` and `right`
factor connected by a new bond index.

### SVD with truncation

```rust
use tensor4all_core::{TensorDynLen, Index, factorize, FactorizeOptions, SvdTruncationPolicy};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(4);
let j = Index::new_dyn(6);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(1)
};
let t: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]);

// SVD: split along i | j, discarding singular values below the chosen policy threshold.
let opts = FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1e-10));
let result = factorize(&t, &[i.clone()], &opts).unwrap();

// result.left  has indices [i, bond]
// result.right has indices [bond, j]
// result.left * result.right ≈ t  (within tolerance)
let bond_dim = result.rank;
println!("bond dimension after SVD: {bond_dim}");

// Limit bond dimension explicitly.
let opts_capped = FactorizeOptions::svd().with_max_rank(2);
let result_capped = factorize(&t, &[i], &opts_capped).unwrap();
assert!(result_capped.rank <= 2);
```

### QR decomposition

```rust
use tensor4all_core::{TensorDynLen, Index, factorize, FactorizeOptions};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(4);
let j = Index::new_dyn(6);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(2)
};
let t: TensorDynLen = TensorDynLen::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]);

// QR: left factor is orthogonal (Q), right factor is upper-triangular (R).
let opts = FactorizeOptions::qr();
let result = factorize(&t, &[i], &opts).unwrap();
// result.left  = Q  (orthogonal columns)
// result.right = R  (upper triangular)
```

Both `FactorizeOptions::svd()` and `FactorizeOptions::qr()` return builder
structs. Use `.with_svd_policy(policy)` for SVD and `.with_qr_rtol(tol)` for
QR-specific rank control. `with_max_rank(n)` remains available as an
algorithm-independent hard cap. For SVD, `result.singular_values` holds the
retained singular values.
