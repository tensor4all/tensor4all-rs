# tensor4all-treetn

Tree tensor network (TreeTN) implementation for general tensor networks beyond linear chains. Supports arbitrary tree topologies with local update algorithms, linear solving, and generalized contractions.

## Features

- **TreeTN**: Generic tree tensor network with named nodes
- **Arbitrary topology**: Build networks with any tree structure
- **Canonicalization**: Orthogonalize towards any center node
- **Truncation**: Bond dimension truncation with configurable tolerance
- **Contraction**: Full contraction and variational (fit) contraction
- **Linear solving**: Solve linear systems in tree tensor network form
- **Local updates**: DMRG-like sweep algorithms

## Usage

### Creating a TreeTN with `from_tensors`

The easiest way to build a TreeTN: provide tensors with shared bond indices
and they are auto-connected.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::TreeTN;

// 3-site MPS chain: t0 -- t1 -- t2
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_dyn(3);
let b12 = DynIndex::new_dyn(3);

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0; 6])?;
let t1 = TensorDynLen::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 12])?;
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0; 6])?;

let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(
    vec![t0, t1, t2],
    vec![0, 1, 2],
)?;
assert_eq!(ttn.node_count(), 3);
assert_eq!(ttn.edge_count(), 2);
# Ok(())
# }
```

### Canonicalization and Truncation

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::{DynIndex, TensorDynLen};
# use tensor4all_treetn::TreeTN;
use tensor4all_treetn::{CanonicalizationOptions, TruncationOptions};
# let s0 = DynIndex::new_dyn(2);
# let s1 = DynIndex::new_dyn(2);
# let b = DynIndex::new_dyn(4);
# let t0 = TensorDynLen::from_dense(vec![s0, b.clone()], vec![1.0; 8])?;
# let t1 = TensorDynLen::from_dense(vec![b, s1], vec![1.0; 8])?;
# let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;

// Canonicalize towards node 0
let ttn = ttn.canonicalize([0], CanonicalizationOptions::default())?;

// Truncate bond dimensions
let ttn = ttn.truncate([0], TruncationOptions::default().with_max_rank(2))?;
# Ok(())
# }
```

### Norm and Dense Conversion

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::{DynIndex, TensorDynLen};
# use tensor4all_treetn::TreeTN;
# let s0 = DynIndex::new_dyn(2);
# let b = DynIndex::new_dyn(2);
# let s1 = DynIndex::new_dyn(2);
# let t0 = TensorDynLen::from_dense(vec![s0, b.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
# let t1 = TensorDynLen::from_dense(vec![b, s1], vec![1.0, 0.0, 0.0, 1.0])?;
# let mut ttn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;

// Compute norm (canonicalizes internally)
let norm = ttn.norm()?;
assert!(norm > 0.0);

// Convert to a single dense tensor (contracts all bonds)
let dense = ttn.to_dense()?;
# Ok(())
# }
```

### Addition

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::{DynIndex, TensorDynLen};
# use tensor4all_treetn::TreeTN;
# let s = DynIndex::new_dyn(2);
# let b_a = DynIndex::new_dyn(2);
# let b_b = DynIndex::new_dyn(2);
# let ta0 = TensorDynLen::from_dense(vec![s.clone(), b_a.clone()], vec![1.0; 4])?;
# let ta1 = TensorDynLen::from_dense(vec![b_a, s.clone()], vec![1.0; 4])?;
# let tb0 = TensorDynLen::from_dense(vec![s.clone(), b_b.clone()], vec![2.0; 4])?;
# let tb1 = TensorDynLen::from_dense(vec![b_b, s.clone()], vec![2.0; 4])?;
# let tn_a = TreeTN::<TensorDynLen, usize>::from_tensors(vec![ta0, ta1], vec![0, 1])?;
# let tn_b = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tb0, tb1], vec![0, 1])?;

// Add two TreeTNs with same topology (uses direct sum on bonds)
let sum = tn_a.add(&tn_b)?;
// Bond dimension of sum = bond_dim(a) + bond_dim(b)
# Ok(())
# }
```

## Sweep Counting

This crate uses **`nfullsweeps`** (number of full sweeps) for iterative algorithms:

- **Full sweep**: Visits each edge twice (forward and backward) using an Euler tour
- **Half sweep**: Visits edges in one direction only (forward or backward)
- **Relationship**: `nfullsweeps = nhalfsweeps / 2` (where `nhalfsweeps` is used in `tensor4all-itensorlike`)

For example:
- `nfullsweeps=1` means each edge is updated twice (once in each direction)
- `nfullsweeps=2` means each edge is updated 4 times total

## License

MIT License
