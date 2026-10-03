# Partitioned TreeTNs

`tensor4all-partitionedtreetn` stores TreeTN subdomains as eagerly masked
patches. It is the TreeTN-native successor to the deprecated
`tensor4all-partitionedtt` crate and supports named chains, branched trees, and
multiple site indices on one node.

This crate provides partition algebra, TreeTN-general adaptive patching, and
adaptive patched interpolation of a function through any tree interpolation
engine. It does not implement an engine itself; the TreeTCI engine lives in
`tensor4all-treetci`.

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

## Adaptive patched interpolation

`adaptive_interpolation::patched_interpolate` interpolates a function on an
arbitrary named tree directly into a partition. It runs an engine implementing
`tensor4all_treetn::interpolation::TreeInterpolator`, such as
`tensor4all_treetci::TreeTciInterpolator`, on the whole domain. A patch that
cannot be accepted is split: the next unfixed site of `patch_order` is fixed
and every child is interpolated in turn. For quantics grids, list the most
significant bits first.

The evaluator receives a column-major `[n_sites, n_points]` batch of
zero-based full-domain points in the derived site order: nodes in ascending
name order, each node's sites in the given order. Each patch caches its
samples and hands them to its children, so no point is evaluated twice, and a
patch with at most one unfixed site is evaluated exactly without the engine.

The accuracy requirement is an `ErrorNorm` with an
`ErrorTolerance { rtol, atol }`. The default, `ErrorNorm::L2`, bounds the L2
error over the whole domain by `delta = max(atol, rtol * S)`, where `S` is an
L2 norm of the function, usually given as `L2Reference::Given(S)`. The driver
measures every accepted and zero patch itself.

```rust
# use std::collections::BTreeMap;
# use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::adaptive_interpolation::{
#     patched_interpolate, GlobalL2Error, PatchedInterpolationOptions,
# };
# use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
# use tensor4all_treetci::TreeTciInterpolator;
# use tensor4all_treetn::NodeNameNetwork;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
// A junction "c" of degree three without a site, and three leaves.
let (x, y, z) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3), DynIndex::new_dyn(3));
let mut topology = NodeNameNetwork::new();
for node in ["c", "x", "y", "z"] {
    topology.add_node(node.to_string())?;
}
for leaf in ["x", "y", "z"] {
    topology.add_edge(&"c".to_string(), &leaf.to_string())?;
}
let node_sites = BTreeMap::from([
    ("c".to_string(), vec![]),
    ("x".to_string(), vec![x.clone()]),
    ("y".to_string(), vec![y.clone()]),
    ("z".to_string(), vec![z.clone()]),
]);

// Site order [x, y, z]. f vanishes for x = 0 and has rank three otherwise.
let f = |p: &[usize]| (p[0] * (1 + p[1] + p[2]).pow(2)) as f64;
let values: Vec<f64> = (0..18).map(|k| f(&[k % 2, (k / 2) % 3, k / 6])).collect();
let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
let options = PatchedInterpolationOptions::new(3)
    .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
    .with_tolerance(ErrorTolerance { rtol: 1e-10, atol: 0.0 })
    .with_patch_order(vec![x.clone(), y.clone()]);
let result = patched_interpolate(
    &TreeTciInterpolator::default(),
    topology,
    node_sites,
    ColMajorArray::new(vec![1, 2, 2], vec![3, 1])?,
    |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
        Ok(batch.data().chunks(3).map(f).collect())
    },
    &options,
)?;

// The x = 0 half is a measured zero patch; the rest is split at y.
assert_eq!(result.report.zero_patches.len(), 1);
assert_eq!(result.partition.len(), 3);
// Every patch was measured exhaustively: the global error is certified.
let error = result.report.norm.l2_error().unwrap();
assert!(matches!(error.global, GlobalL2Error::Certified { .. }));

let reference = IdxTensor::from_dense(vec![x, y, z], values)?;
let dense = result.partition.to_treetn()?.contract_to_tensor()?;
assert!(dense.sub(&reference)?.norm()? <= result.report.norm.delta().unwrap() + 1e-12 * norm);
# Ok(())
# }
```

Under `ErrorNorm::L2` the allowance is split by patch volume: every accepted
or zero patch must have a root-mean-square residual of at most
`tau = delta / sqrt(|X|)`, which is also the engine's absolute tolerance. A
patch with at most `max(max_exhaustive_points, samples)` points (see
`VerificationOptions`) is measured at every point; a larger patch on fresh
uniform samples, followed by an independent audit sample. A failed
measurement reruns the engine with the worst measured points added to its
pivots, then splits the patch. What a run can claim is the report's
`GlobalL2Error`:

- `Certified`: every contribution is exact or exhaustive, and the absolute
  error is at most `delta` up to a small relative margin and a rounding term
  from a calibrated, not proven, model (`rounding_limited` says when that term is not below `tau`);
  `relative_error_bound` bounds `E / ||f||` when `||f~||` exceeds `E`.
- `Audited`: some contribution was sampled, and every sampled one was
  audited. The audited mean square is an unbiased estimate (its square root
  is not) with a standard error, never a bound and not a guarantee: a
  residual concentrated on unsampled points is missed by the acceptance
  sample and the audit alike, and the standard error, computed from the same
  points, does not reveal it. A localized feature that enters a patch only
  through a corner or an edge is such a case; the audited error can then be
  orders of magnitude too small. More `samples` make a miss less likely but
  do not exclude it.
- `AcceptanceOnly`: audits were disabled. The combined acceptance statistics
  are neither a bound nor an estimate.

The reference norm defaults to `L2Reference::Required`, which fails before any
evaluation unless `rtol = 0` (use `atol` alone) or the root has at most one
site (the reference is then exact). `L2Reference::MonteCarlo` estimates it
from uniform root samples; the estimate is heavy-tailed for localized
functions and can make the allowance looser than requested. With a tiny or
zero tolerance the driver may split down to exact patches, so set
`max_patches`.

`ErrorNorm::sampled_max()` keeps the M2 criterion: the engine's sampled error
estimate against `max(atol, rtol * max_reference)`, where `max_reference` is a
function value (`sampled_max_with_reference(max_abs)`, or the largest root
sample). It runs no measurement, is neither a certified bound nor a measured
error, and makes no L2 claim. Under every norm, a patch whose candidate
samples are all exactly zero is first screened (and under L2 measured); zero
patches are reported in `report.zero_patches` and left out of the partition,
which treats an absent patch as zero.

Execution is sequential. For a fixed `seed`, a deterministic evaluator, and a
deterministic engine, the report and every stored node tensor (values and
positional axis order) are identical across runs on fresh threads within one
process as long as the measured network values are reproducible. This is
tested for `f64` on trees with exactly one site per node. On trees with a
site-free node or a node with several sites, or with `f32`/`c32` data, the
cached evaluator can round differently between threads and processes, an
open issue ([#795](https://github.com/tensor4all/tensor4all-rs/issues/795)).
The intended scope is bitwise identical results on the same machine and
build across threads, thread counts, and processes, with no cross-machine
promise. It is not reached yet: those trees need #795 fixed, and
reproducibility across processes is not claimed until a two-process test
passes.
What is derived from the stored `TreeTN`s may still differ across runs, for a
single patch as for the whole partition, on any topology
([issue #791](https://github.com/tensor4all/tensor4all-rs/issues/791)):
materializing (`to_dense`, `contract_to_tensor`, `to_treetn`) in axis order and
at rounding level, and the iteration order of `external_indices`, `site_space`,
and `neighbors`.

## Reconstruction with a fixed global L2 tolerance

Use `reconstruction::reconstruct` when approximation must be measured against
one immutable target, rather than the local discarded-weight `cutoff` used
above. `ReconstructionTarget::from_partition` validates and snapshots disjoint
patches and pins their combined L2 norm. `ReconstructionTolerance { rtol, atol }`
sets the fixed allowance `max(atol, rtol * reference_scale)`.

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
&selection, &options)` prepares the images of an existing linear operator acting
on an ordered subset of the target's site indices. `selection` holds one full
site index per operator node, in the operator's own node order; for a quantics
Fourier transform its node 0 is the most significant input bit. Build that
operator with `tensor4all_quanticstransform::quantics_fourier_operator` and the
crate stays free of a simplett-stack runtime dependency.

The selection may skip sites and spectators keep their identity, dimension, and
node assignment. A spectator may share its node with a selected index. Several
selected indices may share one node as well: the operator MPO nodes carrying
them are fused into one multi-site node, which is exact and keeps the preimage
node name. Only a group whose operator nodes are threaded through another
owner's node cannot be fused locally, and that selection is rejected with repair
guidance.

The transformed output norm is never measured. `SubsetOperatorOptions::unitary`
selects the amplification factor: `false` (default) uses the selected-space
operator's Frobenius norm, an upper bound on its induced amplification; `true` is
a caller guarantee that the operator preserves the L2 norm, giving factor one.
The preimage's reference scale is multiplied by that factor, so successive
applications propagate the scale instead of recomputing an output norm. For a
general operator `rtol` is therefore relative to that scale, not to the actual
`||A x||_2`; pass `unitary = true` for a Fourier transform, whose construction
error is accounted separately from the reconstruction bound.

The operator is applied exactly and the images stay separate, so a global direct
sum is never formed.

The transform stores frequency bit `t` at selected position `t` without an
output bit-reversal permutation. For contiguous output patches, supply
`patch_order = [r1, ..., rR]` with `PatchSplitStrategy::Sequential`.

### Level-coupled merge-refine scheduling

`reconstruction::schedule_merge_refine(&preimage, &center, &operator, &selection,
&subset, tolerance, &MergeRefineOptions)` instead runs the complementary
input-merge/output-refine trajectory of the patched Fourier algorithm. The
preimage must already be the dyadic input leaves of the `d` selected binary
indices, sharing identical spectator constraints. Each leaf fixes a contiguous
prefix of those indices, so leaf depths may differ as long as the leaves form a
prefix code; all `2^d` coordinate assignments must be covered by default, or a sparse
subset with `CoverageContract::ZeroForMissingLeaves`, where every omitted assignment
contributes exactly zero. Level zero applies the
complete transform once per leaf. Level `t` merges the input siblings by removing
the constraint on `k_(d-t+1)` and refines the output by fixing `r_t`, restricting
every contribution to its output region *before* adding it, so no sum over the
whole output domain is ever assembled and each computed object is reused by its
descendants.

`MergeRefineOptions::output_depth` stops after that many levels; `None` (the
default) refines every selected bit. `MergeRefineOptions::coordinate_groups` supplies
several coordinate axes explicitly: each `CoordinateGroup` gives that axis' selected
indices in input order and, separately, in output order, one level advances every
non-exhausted axis by one bit, and a two-axis level therefore combines four input
children and produces four output children. The default `None` treats `selection` as
one axis with the documented reversed placement. `MergeRefineOptions::target_bond_dim` is a
soft rank goal: `None` keeps the trajectory uniform and exact, while `Some(goal)`
makes it adaptive. An output region is then refined only when its retained terms
exceed the goal and refining lowers its maximum retained rank, a merged pair is
combined into one term only when combining pays off (otherwise both operands stay
as separate terms of the region's superposition), and every truncation candidate
must fit the item's share of the global allowance. An unaffordable or gainless
candidate is retained exactly rather than violating the accuracy contract, and
exceeding `MergeRefineOptions::max_terms` returns a resource-limit error.

`MergeRefineOptions::apply_options` opts into a truncating operator application.
The schedule then applies the operator with those options *and* exactly, measures
each leaf's deviation, and charges that sum before any scheduling compression, so
an approximate application is never a silent error; a measured application error
already above the allowance is rejected. Every other entry point applies exactly. `MergeRefineReport` records the schedule shape
(`level_count`, `applied_operator_count`, `additions`, `projections`,
`compression_attempts`, `compressions`, `work_items_per_level`, `peak_work_items`,
`refined_regions`, `stopped_regions`) next to the pinned reference scale, the
allowance, and the measured `error_bound`. Every measured component — an application
error, a truncation residual, or a dropped norm — is recorded once, at the level and
region where it was measured: components of one level live in disjoint regions and
combine by the Euclidean norm, while components of different levels can be nested and
add by the triangle inequality. The bound therefore never exceeds the allowance and
does not grow with the number of refined regions. It excludes operator-construction
error and any error of a caller-approximated operator outside the charged application
deviation. Per-input-branch refinement depths with lazy reconciliation and benchmark
evidence in the intended patched-input regime remain follow-up work. The transform is
never padded and its length never changes: the operator's input and output are the
same selected indices.

```rust
# use std::collections::HashMap;
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN};
# use tensor4all_treetn::{IndexMapping, LinearOperator};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
{{#include ../../../../crates/tensor4all-partitionedtreetn/examples/merge_refine.rs:merge_refine}}
# Ok(())
# }
```

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
