# tensor4all-treetn

Tree tensor networks with arbitrary graph topology. Supports canonicalization, truncation, contraction, and linear solving.

## Key Types

- `TreeTN` — tree tensor network with named nodes and arbitrary tree structure
- `from_tensors()` — build a TreeTN from tensors with shared bond indices (auto-connected)
- `canonicalize()` — orthogonalize towards any center node
- `truncate()` — bond dimension truncation with configurable tolerance
- `to_dense()` — contract all bonds into a single dense tensor

## Example

```rust,ignore
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::{TreeTN, CanonicalizationOptions, TruncationOptions};

// 3-site MPS chain: t0 -- t1 -- t2
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_dyn(4);
let b12 = DynIndex::new_dyn(4);

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0; 8])?;
let t1 = TensorDynLen::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 16])?;
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0; 8])?;

let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2])?;
assert_eq!(ttn.node_count(), 3);
assert_eq!(ttn.edge_count(), 2);

// Canonicalize and truncate
let ttn = ttn.canonicalize([0], CanonicalizationOptions::default())?;
let ttn = ttn.truncate([0], TruncationOptions::default().with_max_rank(2))?;

// Compute norm
let norm = ttn.norm()?;
assert!(norm > 0.0);
```

## Documentation

- [User Guide: Tree Tensor Networks](https://tensor4all.github.io/tensor4all-rs/guides/tree-tn.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_treetn/)
