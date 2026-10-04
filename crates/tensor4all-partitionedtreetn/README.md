# tensor4all-partitionedtreetn

TreeTN-native partitioned tensor networks with eagerly masked subdomain
patches. The crate supports arbitrary named tree topologies, multiple site
indices per node, homogeneous `f64`/`Complex64` partitions, strict algebra, and
volume-proportional adaptive patching.

Adaptive patching is bond-cap-driven and independent of adaptive interpolation.

## Adaptive patched interpolation

`adaptive_interpolation::patched_interpolate` builds a partition directly from
a batch evaluator on an arbitrary named tree. It runs an interpolation engine
implementing `tensor4all_treetn::interpolation::TreeInterpolator` (for example
`tensor4all_treetci::TreeTciInterpolator`) on the whole domain, and otherwise
fixes the next site of `patch_order` and retries on every child. The crate
depends on the engine trait only, not on `tensor4all-treetci`.

The accuracy requirement is `PatchedInterpolationOptions::error_norm` with
`tolerance: ErrorTolerance { rtol, atol }`:

- `ErrorNorm::L2` (the default) is the unweighted discrete L2 error over the
  whole domain against `delta = max(atol, rtol * S)`, with `S` an L2 norm of
  the function. `L2Reference::Given(S)` is the usual choice;
  `L2Reference::Required` (the default) fails before any evaluation unless
  `rtol = 0` or the root has at most one site, and `L2Reference::MonteCarlo`
  is an explicit opt-in estimate that can loosen the allowance for localized
  functions. The allowance is split by patch volume (`rms_P(f - f~) <= tau`
  with `tau = delta / sqrt(|X|)`), and the driver measures every accepted and
  zero patch itself: exhaustively up to
  `VerificationOptions::max_exhaustive_points`, otherwise on fresh uniform
  samples with an independent audit. A failed measurement of a converged run
  reruns the engine with the worst points as pivots, then splits.
- `ErrorNorm::SampledMax` is the M2 criterion: the engine's sampled error
  estimate against `max(atol, rtol * max_reference)`. It is **neither** a
  certified bound nor a measured error, and makes no L2 claim.
- `ErrorNorm::MaxAbs` and `ErrorNorm::WeightedL2` are placeholders that fail
  with `UnsupportedNorm` before any evaluation.

Two optional bounds count patch size in generalized bits, one per active
(unfixed) site whatever its dimension. `min_patch_bits` stops splitting below
a minimum: a failing patch is then retained and reported as
`PatchStatus::ToleranceNotMet`, never certified, and the run reports
`GlobalL2Error::ToleranceNotMet`, whose error can exceed `delta`.
`capped_patches: CappedPatches::AcceptUpTo { bits }` accepts a patch that
reaches the bond cap when it has at most `bits` active sites and passes its
error check; larger capped patches split, and converged patches never split
for their size. Both default to the M3 behavior.

The opt-in `cache_candidates` starts every child patch from the largest
values of its inherited evaluation cache as well (no new evaluations). It
mitigates the corner-localized misses described below without excluding
them.

The report's `GlobalL2Error` states what an L2 run can claim. `Certified`
(every contribution exact or exhaustive, every patch within its allowance)
bounds the absolute error
`E <= delta (1 + GLOBAL_ROUNDING_MARGIN) + MEASUREMENT_ROUNDING_FACTOR * eps *
||f~||`, up to a calibrated (not proven) rounding model, and carries a conservative bound
on `E / ||f||` when `||f~||` exceeds `E`. `Audited` gives an estimate with a
standard error, never a bound; `AcceptanceOnly` (audits off) gives neither an
estimate nor a relative statement. `ToleranceNotMet` takes precedence when a
patch was retained without meeting its allowance; its `basis` says which of
the three applies to its measured value. Sampled measurements cannot bound
the L2 error of a black-box function.

Known limitation: the acceptance sample and the audit are both uniform, so
both can miss a localized feature that enters a patch only through a corner
or an edge, and the audit's standard error does not reveal such a miss (it is
computed from the same points). An `Audited` result can then underestimate
the error by orders of magnitude. Raising `VerificationOptions::samples`
makes this less likely but does not exclude it; only `Certified` results
(every contribution exact or exhaustive) are guarantees. See "Known
limitation: corner-localized misses" in
[`docs/design/tree-patching-error-contract.md`](../../docs/design/tree-patching-error-contract.md).

- Patches are processed sequentially in FIFO order; each has an evaluation
  cache, so no point is evaluated twice, and measured values reach the
  children of a split. Patches with at most one active site are evaluated
  exactly without the engine.
- A patch whose candidate samples are all exactly zero is a zero patch under
  `SampledMax`; under L2 the zero approximation is measured first. Zero
  patches are reported in `PatchedInterpolationReport::zero_patches` and
  omitted from the partition.
- Randomness comes from per-patch seeds derived from
  `PatchedInterpolationOptions::seed`. For a fixed seed, a deterministic
  evaluator, and a deterministic engine the report and every stored node
  tensor are identical across runs on fresh threads within one process,
  provided the measured network values are reproducible. This is tested for
  `f64` on trees with exactly one site per node. The cached evaluator's
  generic path (a site-free node, a node with several sites, or `f32`/`c32`
  data) can differ at rounding level between threads and processes, an open
  issue ([#795](https://github.com/tensor4all/tensor4all-rs/issues/795)). The
  intended scope is bitwise identical results on the same machine and build
  across threads, thread counts, and processes, with no cross-machine
  promise; it is not reached yet: generic-path trees need #795 fixed, and
  reproducibility across processes is not claimed until a two-process test
  passes. What is derived from the stored `TreeTN`s may still differ across
  runs ([issue #791](https://github.com/tensor4all/tensor4all-rs/issues/791)):
  materializing (`to_dense`, `contract_to_tensor`, `to_treetn`) in axis order
  and at rounding level, and the iteration order of `external_indices`,
  `site_space`, and `neighbors`.

The patch queue and pivot recycling derive from TCIAlgorithms.jl (MIT) through
the deprecated `tensor4all-partitionedtt`; this crate carries
`LICENSE-TCIALGORITHMS-MIT`.

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
the selected binary indices. Each leaf fixes a contiguous prefix, so leaf depths may
differ as long as the leaves form a prefix code; by default they must also cover all
`2^d` coordinate assignments, or a sparse subset under
`CoverageContract::ZeroForMissingLeaves`, where an omitted leaf contributes exactly
zero. `MergeRefineOptions` selects the output depth, the
work limit, an optional soft rank goal, and the retained term budget;
`MergeRefineReport` records the structural counters plus the pinned allowance and
the measured bound. With `target_bond_dim` unset the trajectory is uniform and
exact (`error_bound` zero). With a goal it is adaptive: a region is refined only
when that lowers its maximum retained rank, a merged pair is combined only when
combining pays off, and merged items are truncated only within an equal share of
the allowance, so the measured bound never exceeds it. A merged or unpaired
contribution is dropped only when its measured norm fits its share of the allowance,
with `dropped_terms` and `dropped_error` reporting it and a fully dropped region
omitted while its measured norm still counts. Every measured component is recorded
once, at the level and region where it was measured, so `error_bound` does not grow
with the number of refined regions. Exceeding `max_terms` returns a
resource-limit error. `MergeRefineOptions::apply_options` opts into a
truncating application whose measured deviation from the exact application is
charged to the same bound, and an application error above the allowance is
rejected. Several coordinate axes may be transformed in one synchronized level through
`MergeRefineOptions::coordinate_groups`, each stating its own input and output order;
the transform is never padded and its length never changes. Per-input-branch refinement
depths remain follow-up work.

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
