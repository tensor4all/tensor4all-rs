# Tree Tensor Networks

The `tensor4all-treetn` crate provides a generic tree tensor network (`TreeTN`) that supports
arbitrary tree topologies — not just linear chains. This guide covers construction, canonicalization,
common operations, and the sweep-counting convention used by iterative algorithms.

## Creating a TreeTN

Use `TreeTN::from_tensors` to build a network from individual tensors. The topology is inferred
automatically: two tensors are connected by an edge when they share a **bond index** (a `DynIndex`
that appears in both tensors). Physical (site) indices appear in exactly one tensor.

The example below builds a 3-site MPS chain `t0 -- t1 -- t2`:

```rust,ignore
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::TreeTN;

// Site indices (appear in one tensor each)
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);

// Bond indices (shared between adjacent tensors)
let b01 = DynIndex::new_dyn(3);
let b12 = DynIndex::new_dyn(3);

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0; 6])?;
let t1 = TensorDynLen::from_dense(vec![b01, s1, b12.clone()], vec![1.0; 12])?;
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0; 6])?;

let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(
    vec![t0, t1, t2],
    vec![0, 1, 2],   // vertex labels
)?;
assert_eq!(ttn.node_count(), 3);
assert_eq!(ttn.edge_count(), 2);
```

Each vertex is labelled by the user-supplied key (here `usize`). Any type that is `Eq + Hash` works.
The tensor at vertex `v` is retrieved with `ttn[v]`.

## Canonicalization

Canonicalization orthogonalizes the network toward a chosen **root vertex**, turning all tensors
except the root into isometries. This puts the full norm information into the root tensor and is a
prerequisite for efficient norm computation and truncation.

```rust,ignore
use tensor4all_treetn::{CanonicalizationOptions, TruncationOptions};

// Canonicalize toward vertex 0
let ttn = ttn.canonicalize([0], CanonicalizationOptions::default())?;

// Truncate bond dimensions after canonicalization
let ttn = ttn.truncate([0], TruncationOptions::default().with_max_rank(2))?;
```

`TruncationOptions` supports both a maximum rank (`with_max_rank`) and a relative tolerance
(`with_rtol`). Truncation discards small singular values on each bond, reducing memory and
contraction cost at the expense of a controlled approximation error.

## Operations

### Norm Computation

`norm()` returns the Frobenius norm of the tensor represented by the network. It canonicalizes
internally so the result is always exact (up to floating-point precision).

```rust,ignore
let norm = ttn.norm()?;
assert!(norm > 0.0);
```

### Dense Conversion

`to_dense()` contracts all bond indices and returns a single `TensorDynLen` whose indices are the
physical (site) indices of the network. For large networks this can be expensive — use it mainly
for testing or small examples.

```rust,ignore
let dense = ttn.to_dense()?;
```

### Addition

Two `TreeTN`s with the **same topology and matching site indices** can be added with `add`. The
result has the same tree structure, but each bond dimension is the sum of the two input bond
dimensions (direct-sum construction).

```rust,ignore
// sum represents tn_a + tn_b; bond dims are additive
let sum = tn_a.add(&tn_b)?;
```

After addition the bond dimensions grow, so it is common to follow up with `truncate` to keep them
manageable.

### Contraction

A `TreeTN` can be fully contracted to a scalar or dense tensor via `to_dense()`. For
network-to-network contractions (e.g., computing inner products or applying operators), use the
higher-level contraction APIs provided by `tensor4all-treetn`. Refer to the crate documentation for
variational (fit) contraction, which avoids the exponential cost of naive full contraction.

## Sweep Counting

Iterative algorithms (DMRG-like local updates, fitting) sweep through the network edges multiple
times. `tensor4all-treetn` uses the **`nfullsweeps`** convention:

| Term | Meaning |
|------|---------|
| **Half sweep** | Visit each edge once in a single direction (used in `tensor4all-itensorlike`) |
| **Full sweep** | Visit each edge twice — forward **and** backward (Euler tour) |

The relationship is `nfullsweeps = nhalfsweeps / 2`.

Concretely:

- `nfullsweeps = 1` — each edge is updated twice (once per direction).
- `nfullsweeps = 2` — each edge is updated four times total.

When interoperating with code that uses `nhalfsweeps`, divide by two before passing the value to
`tensor4all-treetn` APIs.
