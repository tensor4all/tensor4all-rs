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

```rust
use anyhow::Result;
use rand::rng;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::TensorDynLen;
use tensor4all_treetn::{CanonicalizationOptions, TreeTN, TruncationOptions};

// Build a simple 2-node TreeTN: A -- B (shared bond index)
let bond = Index::<DynId>::new_dyn(10);
let a_site = Index::<DynId>::new_dyn(2);
let b_site = Index::<DynId>::new_dyn(2);

let mut rng = rng();
let a = TensorDynLen::random_f64(&mut rng, vec![a_site.clone(), bond.clone()]);
let b = TensorDynLen::random_f64(&mut rng, vec![bond.clone(), b_site.clone()]);

let mut ttn = TreeTN::<TensorDynLen, String>::new();
let node_a = ttn.add_tensor("A".to_string(), a)?;
let node_b = ttn.add_tensor("B".to_string(), b)?;
ttn.connect(node_a, &bond, node_b, &bond)?;

// Canonicalize towards center node B
let ttn = ttn.canonicalize(["B".to_string()], CanonicalizationOptions::default())?;

// Truncate bond dimensions
let ttn = ttn.truncate(["B".to_string()], TruncationOptions::default().with_max_rank(5))?;

# Ok::<(), anyhow::Error>(())
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
