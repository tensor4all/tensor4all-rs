# Partitioned TreeTNs

`tensor4all-partitionedtreetn` stores TreeTN subdomains as eagerly masked
patches. It is the TreeTN-native successor to the deprecated
`tensor4all-partitionedtt` crate and supports named chains, branched trees, and
multiple site indices on one node.

This crate provides partition algebra and TreeTN-general adaptive patching. It
does not provide adaptive interpolation or TCI.

## Construct an eager patch

Projectors use zero-based coordinates and full index identity. Construction
retains every site axis but masks values outside the selected coordinates:

```rust
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
# use tensor4all_treetn::TreeTN;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let site = DynIndex::new_dyn(2);
let tensor = IdxTensor::from_dense(
    vec![site.clone()],
    vec![3.0_f64, 1.0e12],
)?;
let tree = TreeTN::from_tensors(vec![tensor], vec!["root".to_string()])?;
let patch = SubDomainTreeTN::new(
    tree,
    Projector::from_pairs([(site.clone(), 0)])?,
)?;

let node = patch.data().node_index(&"root".to_string()).ok_or("missing root")?;
assert_eq!(patch.data().tensor(node).ok_or("missing tensor")?.to_vec::<f64>()?,
           vec![3.0, 0.0]);
assert!((patch.norm_squared()? - 9.0).abs() < 1.0e-12);
# Ok(())
# }
```

Norms, inner products, contraction, truncation, and summation use this stored
masked value directly. No projector is re-applied and no full network is
densified.

## Adaptive patching

Every truncating or contracting operation takes an explicit existing node name
as its center. `add_with_patching` first assigns absolute local discarded-weight
cutoffs proportional to logical patch volume
(`cutoff * ||F||^2 * volume_p / total_volume`), applies each whole threshold at
the patch's local SVD truncations, then splits patches that remain above the
bond cap. The `cutoff` is best effort for the final whole-network error;
`max_bond_dim` is a hard cap. Inputs that share an equal projector key are
summed before patching:

```rust
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{
#     add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
# };
# use tensor4all_treetn::TreeTN;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let site0 = DynIndex::new_dyn(2);
let bond = DynIndex::new_dyn(2);
let site1 = DynIndex::new_dyn(2);
let left = IdxTensor::from_dense(
    vec![site0.clone(), bond.clone()],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let right = IdxTensor::from_dense(
    vec![bond, site1],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let patch = SubDomainTreeTN::from_treetn(
    TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?,
)?;
let result = add_with_patching(
    vec![patch],
    &0,
    &PatchingOptions {
        cutoff: 0.0,
        max_bond_dim: Some(1),
        patch_order: vec![site0],
        split_strategy: PatchSplitStrategy::Sequential,
    },
)?;

assert_eq!(result.len(), 2);
assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
# Ok(())
# }
```

`PatchSplitStrategy::Sequential` follows `patch_order`. The default
`ExactParameterGain` forms and budget-truncates every candidate's children,
then compares checked sums of logical local tensor element counts. Structured
storage payload length and AD state are not used as the metric.

## Reconstruction with a fixed global L2 tolerance

Use `reconstruction::reconstruct` when approximation must be measured against
one immutable target, rather than the local discarded-weight `cutoff` used
above. `ReconstructionTarget::from_partition` validates and snapshots disjoint
patches and pins their combined L2 norm. `ReconstructionTolerance { rtol, atol }`
sets the fixed allowance `max(atol, rtol * reference_norm)`.

The rank goal is soft: a split must improve rank, and a pairwise merge must
reduce the sum of operand ranks. Unprofitable sums remain as superposition
terms. The output's regions are disjoint, but terms within a region can overlap.
Use `regions()` to consume them. `into_partition()` rejects a region containing
multiple terms; it never implicitly sums them.

This example is included directly from the checked executable source:

```rust
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{reconstruction::*, PartitionedTreeTN, PatchSplitStrategy, SubDomainTreeTN, TreeTN};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
{{#include ../../../../crates/tensor4all-partitionedtreetn/examples/reconstruct.rs:reconstruction}}
# Ok(())
# }
```

`ReconstructionTarget::from_tensor_products` accepts pairs of patches on the
same named topology and independent site spaces. Their output node owns both
factors' external indices. Each product norm is the product of its factor norms;
orthogonal product-patch norm squares are then added. Products remain factorized
until reconstruction begins. This is a tensor product, not an elementwise
product or an induced operator norm.

Each accepted compression is checked against its uncompressed local input by
an explicit difference-network norm. Residuals add within a region and combine
in quadrature across disjoint regions. The report is a numerical a posteriori
bound; it excludes floating-point roundoff. `rtol = atol = 0` disables
approximate compression. Reaching `max_regions` retains higher rank without
relaxing the error allowance. A partial `patch_order` list constrains the
search independently of any future QFT-selected index subset.

Reconstruction reuses `PatchSplitStrategy`. `Sequential` tries only the first
unprojected nontrivial index in `patch_order` and stops that region on no gain
or insufficient region capacity, without trying later indices. The default
`ExactParameterGain` compares all permitted candidates by logical parameter
count. For contiguous QTT intervals, supply the bits MSB first and select
`Sequential`, even when the TT stores those bits in reverse order. An empty
order uses all external indices in deterministic identity order, not numeric
bit significance.

### Applying a QFT to a subset of sites

`ReconstructionTarget::from_subset_operator(&preimage, &center, &operator,
&selection)` prepares the images of an existing linear operator
acting on an ordered subset of the target's site indices. `selection` holds one
full site index per operator node, in the operator's own node order; for a
quantics Fourier transform its node 0 is the most significant input bit. Build
that operator with `tensor4all_quanticstransform::quantics_fourier_operator` and
the crate stays free of a simplett-stack runtime dependency.

The selection may skip sites and spectators keep their identity, dimension, and
node assignment. A spectator may share its node with a selected index, but two
selected indices on one node are rejected. The operator is applied exactly and
the prepared target's global norm is measured from the explicit sum of the
images, so a non-unitary operator gets its correct norm instead of the preimage
norm. The operator's construction error (for example
`FourierOptions::tolerance`) is accounted separately from the reconstruction
bound; approximate application is not exposed yet.

The transform stores frequency bit `t` at selected position `t` without an
output bit-reversal permutation. For contiguous output patches, supply
`patch_order = [r1, ..., rR]` with `PatchSplitStrategy::Sequential`.

## Dtype and topology

A partition is homogeneous: all patches must use the same `IdxTensor` scalar
dtype and the same named topology and site-index assignment. Both `f64` and
`Complex64` are supported. Topology is not restricted to a chain; a TreeTN
with a central named node and three named leaves is a valid partition input.
See the [Tree Tensor Networks guide](tree-tn.md) for constructing branched
networks and selecting contraction/truncation options.

## Migration

Use this crate for new named TreeTN partition work. The old
`tensor4all-partitionedtt` crate remains buildable during migration and receives
correctness and security fixes only; no removal date has been set.
