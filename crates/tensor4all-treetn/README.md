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
use tensor4all_treetn::{
    SiteIndexNetwork, LinkSpace, random_treetn_f64,
    CanonicalizationOptions, TruncationOptions,
};
use tensor4all_core::Index;

// Build a tree topology: A -- B -- C
let mut network = SiteIndexNetwork::new();
network.add_node("A".into(), vec![Index::new_dyn(2)])?;
network.add_node("B".into(), vec![Index::new_dyn(2)])?;
network.add_node("C".into(), vec![Index::new_dyn(2)])?;
network.add_edge(&"A".into(), &"B".into())?;
network.add_edge(&"B".into(), &"C".into())?;

// Create random tree tensor network
let ttn = random_treetn_f64(&mut rng, &network, LinkSpace::uniform(10));

// Canonicalize towards center node B
let ttn = ttn.canonicalize(["B".into()], CanonicalizationOptions::default())?;

// Truncate bond dimensions
let ttn = ttn.truncate(["B".into()], TruncationOptions::default().with_max_rank(5))?;
```

## License

MIT License
