# tensor4all-partitionedtreetn

TreeTN-native partitioned tensor networks with eagerly masked subdomain
patches. The crate supports arbitrary named tree topologies, multiple site
indices per node, homogeneous `f64`/`Complex64` partitions, strict algebra, and
volume-proportional adaptive patching.

Adaptive patching is bond-cap-driven and independent of adaptive interpolation:
this crate does **not** provide TCI, sampled-zero inference, or a dependency on
`tensor4all-treetci`.

## Truncation convention

The scalar adaptive truncation parameter is `PatchingOptions::cutoff`, a local
discarded-weight cutoff following the ITensorMPS convention. One absolute local
threshold `cutoff * ||F||^2 * volume_p / total_volume` is derived per operation
and applied whole at every local SVD. It is **best effort** for the final
whole-network error; `max_bond_dim` is a hard cap and takes precedence. No API
in the existing adaptive-patching API claims a global relative-error bound.

## Reconstruction with a global L2 tolerance

The separate `reconstruction` module accepts an immutable
`ReconstructionTarget`, a `ReconstructionTolerance { rtol, atol }`, and
`ReconstructionOptions`. It pins the target's L2 norm and accepts local
approximations only after checking their difference-network norms against
the fixed allowance `max(atol, rtol * reference_scale)`.

`ReconstructionTarget::from_partition` snapshots disjoint eager patches.
`from_tensor_products` retains independent factor pairs on the same named tree
topology, validates disjoint product supports, and computes each product norm
by multiplying factor norms before any product is formed.

`target_bond_dim` is a soft goal. Profitable sums are merged; over-goal terms
split only when rank improves. Otherwise the output keeps high-rank terms or
superposition lists. `ReconstructedTreeTN::regions()` exposes the disjoint
regions and their terms; `into_partition()` succeeds only when every region
has one term, and never silently sums a list. Its report contains an accumulated
measured-residual error bound; floating-point roundoff is not rigorously bounded.

Reconstruction reuses `patch_order` and `PatchSplitStrategy`: `Sequential`
tries only the next unprojected nontrivial index and stops on no gain or a
region-capacity limit; the default `ExactParameterGain` compares all permitted
candidates. Use an explicit MSB-first order with `Sequential` for contiguous
QTT intervals, independently of the indices' placement on the tree.

`ReconstructionTarget::from_subset_operator` prepares the images of an existing
linear operator (for example a quantics Fourier transform built elsewhere) on an
ordered subset of full site indices. Spectators keep their identity, dimension,
and node assignment. Selected indices that share one tree node are supported by
fusing the operator MPO nodes that carry them into one multi-site node, which
requires those operator nodes to form one connected group. The
transformed output norm is never measured and the images are never summed into
one network. `SubsetOperatorOptions::unitary` instead pins the amplification
factor: `false` (default) uses the selected-space operator's Frobenius norm as an
upper bound, `true` is a caller guarantee that the operator preserves the L2
norm (factor one). The preimage reference scale is multiplied by that factor and
successive applications propagate it. The operator's own construction error
stays separate from the reported reconstruction bound.

Run the asserted examples with
`cargo run --release -p tensor4all-partitionedtreetn --example reconstruct` and
`cargo run --release -p tensor4all-partitionedtreetn --example merge_refine`.
The [guide](https://tensor4all.org/tensor4all-rs/guides/partitioned-treetn.html#reconstruction-with-a-fixed-global-l2-tolerance)
includes the same executable sources.

`reconstruction::schedule_merge_refine` runs the complementary
input-merge/output-refine QFT trajectory instead of the greedy engine: level `t`
merges one input bit and fixes one output prefix bit, restricting each
contribution to its output region before adding it, so no sum over the whole
output domain is assembled. The preimage must be the dyadic input leaves of
the selected binary indices: by default all `2^d` of them, or a sparse subset under
`CoverageContract::ZeroForMissingLeaves`, where an omitted leaf contributes exactly
zero. `MergeRefineOptions` selects the output depth, the
work limit, an optional soft rank goal, and the retained term budget;
`MergeRefineReport` records the structural counters plus the pinned allowance and
the measured bound. With `target_bond_dim` unset the trajectory is uniform and
exact (`error_bound` zero). With a goal it is adaptive: a region is refined only
when that lowers its maximum retained rank, a merged pair is combined only when
combining pays off, and merged items are truncated only within an equal share of
the allowance, so the measured bound never exceeds it. Exceeding `max_terms`
returns a resource-limit error. `MergeRefineOptions::apply_options` opts into a
truncating application whose measured deviation from the exact application is
charged to the same bound, and an application error above the allowance is
rejected. Nonuniform input trees and automatic zero-padding remain separate
follow-up work; padding is never applied silently.

## Quick start

```rust
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

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
let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
let patch = SubDomainTreeTN::from_treetn(tree)?;
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
# Ok::<(), Box<dyn std::error::Error>>(())
```

All flat tensor buffers use column-major order: the first listed index varies
fastest. Coordinates are zero-based. An operation that truncates or contracts
requires an explicit existing TreeTN node name as its center.

## Documentation

- [Tensor4all-rs user guide](https://tensor4all.org/tensor4all-rs/)
- [Partitioned TreeTN guide](https://tensor4all.org/tensor4all-rs/guides/partitioned-treetn.html)
- [API reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_partitionedtreetn/)
- [Migration design](../../docs/design/partitioned-treetn.md)
- [Provenance and citation policy](../../docs/PROVENANCE_AND_CITATION_POLICY.md)

The deprecated `tensor4all-partitionedtt` crate remains buildable during the
migration window. It is limited to correctness and security fixes; no removal
date has been set.
