# Orthogonal-target reconstruction

## Decision

Implemented the general reconstruction layer before QFT integration, following
the agreed separation of target, fixed global L2 tolerance, and representation
policy. The target is immutable and orthogonality is established by disjoint
projector supports. Tensor-product targets retain independent factors on an
explicitly shared named topology and derive norms before product formation.
Other norms and arbitrary operator application are outside this change.

The [design record](../design/orthogonal-target-reconstruction.md) defines the
public surface, accounting, and subset-QFT follow-up contract. It incorporates
the supplied "Fourier transform of a patched QTT" algorithm note and the
existing [partitioned TreeTN contract](../design/partitioned-treetn.md). The
implementation reuses TreeTN partial contraction, strict addition, truncation,
and norms, and the existing projector and logical-parameter-count helpers.
No external implementation was translated.

## Tradeoffs

- Measure each accepted local approximation with an explicit difference-network
  norm. This costs an additional network residual calculation but does not
  assume repeated truncation errors are orthogonal or claim that a local SVD
  cutoff alone controls the final error.
- Add residuals within a region; combine disjoint-region bounds in quadrature.
  Candidate children are rebuilt from original region sources. Use conservative
  equal l1 child-budget allocation initially.
- Keep rank goals soft and retain unprofitable sums as lists. The initial
  deepest-first balanced pairing is deterministic, but does not search every
  possible pair or claim a globally optimal partition.
- Reuse the existing O(M²) projector-overlap validator for arbitrary product
  supports. This metadata cost is explicit; no product tensors are formed to
  establish orthogonality. More specialized dyadic scheduling is future work.
- Keep QFT subset selection independent of reconstruction split candidates.
  A QFT-owned target must preserve the validated preimage norm and account for
  transform approximation; overlapping transformed supports cannot be fed to
  the current disjoint-support target constructor as though they were disjoint.

## Verification

The focused integration matrix covers real/complex splitting, nonuniform
patch merging, no-gain list retention, no-gain splitting, soft rank/search
limits, zero/absolute tolerance, target norm snapshots, factorized products,
nested splitting, branched topology, multiple site indices per node, same-ID
indices with distinct prime levels, invalid options/identities/dimensions,
non-finite values, and a 48-site regression against accidental dense execution.
Small numerical checks materialize once and compare the whole dense result;
the long-network regression uses sampled values and rank/topology invariants.

At the initial snapshot, all 19 new integration tests passed, along with the
existing partitioned TreeTN tests. All 1,027 workspace doctests passed. The
executable guide example is included directly from its Rust source; its execution,
the mdBook HTML build, and the complete mdBook execution suite passed.
Changed-crate release Clippy
with all targets and warnings denied, workspace formatting, public-error-doc
validation, the refreshed API inventory, and deterministic repository-rules
review also passed.

Coverage impact reviewed: no test or behavior was removed; the shared logical
parameter-count helper remains exercised by existing adaptive-patching tests
and by the new split-candidate path. Hosted CI remains the coverage authority.

## Patching order follow-up (2026-09-15)

Reuse the existing `PatchSplitStrategy` and `patch_order` convention rather
than introducing a `PatchingOrder` enum. Reconstruction replaces its
`split_indices` field with `patch_order` and adds `split_strategy`, defaulting
to `ExactParameterGain` to preserve existing gain search. `Sequential` probes
only the next unprojected nontrivial index; no gain or insufficient region
capacity stops the region without bypassing that index. Global L2 accounting
and the soft rank goal are unchanged.

The QFT design now distinguishes significance from placement: k1 and r1 are
MSBs, while QFT without output bit-reversal correction places bits as
[rR, ..., r1].
Contiguous output intervals use `patch_order = [r1, ..., rR]` with `Sequential`.
This also applies within a selected subset, preserving spectator placement.
The input merge order k_d, ..., k1 and output refinement order r1, ..., r_d
are documented separately from the generic reconstruction engine.

Regression coverage now includes sequential vs gain selection on a no-gain
first index, an unaffordable first fanout, reversed output-bit placement, and
nested splitting under both strategies. Numerical regression checks run in
release mode to verify rank/gain decisions and residual bounds under optimized
factorization. No existing test tolerance was changed.

For this revision, all 93 changed-crate unit/integration tests (including 21
reconstruction tests) and 41 crate doctests passed. Changed-crate release
Clippy with all targets and warnings denied, the runnable example, full mdBook
snippet tests and HTML build, formatting, public error documentation checks,
and repository-rules dry-run review passed. The API inventory was refreshed.
Workspace-wide doctests were not rerun for this revision; the 1,027-test result
above belongs to the initial snapshot.

## Limits

The reported error bound uses floating-point norm measurements, not interval
arithmetic, and excludes backend roundoff. No AD guarantee or GPU execution was
validated. No runtime speedup is claimed. The complementary merge-refine
schedule, automatic zero-padding, and applying two selected indices that share
one tree node remain unimplemented.

## Subset QFT integration (2026-09-15)

Added `ReconstructionTarget::from_subset_operator`, which applies an existing
`LinearOperator` to an ordered subset of the preimage's full site indices. The
Fourier operator stays the caller's: `tensor4all-quanticstransform` is a
path-only cross-layer dev-dependency, so this crate keeps no simplett-stack
runtime dependency. Two selected indices on one node are rejected with repair
guidance rather than silently mis-bound; spectators may share a node with a
selected index.

The constructor accepts an arbitrary operator, which need not be unitary and
need not map the preimage's disjoint patches to orthogonal images, so it never
measures the transformed output norm. It neither sums the images into one network
(TreeTN addition adds bond dimensions and would restore the global-rank
bottleneck) nor computes a pairwise-overlap Gram sum (O(M^2) and
cancellation-prone). Instead `SubsetOperatorOptions::unitary` selects an
amplification factor: `false` (default) uses the selected-space operator's
Frobenius norm as an upper bound on its induced amplification, and `true` is a
caller guarantee of factor one. The preimage's reference scale is multiplied by
that factor, so successive applications propagate the scale instead of
recomputing an output norm. For a general operator the resulting scale is an
upper bound and `rtol` is relative to it, not to `||A x||_2`; the two coincide
for a unitary acting on an exactly known input scale. The target accessor and the
report field were renamed from `reference_norm` to `reference_scale` so their
names do not claim a measured output norm.

Application uses the local exact naive path and the entry point exposes no
truncating apply options, so the prepared target carries no application error
beyond backend roundoff. Exposing approximate application with its own retained
error allowance, and the QFT merge-refine schedule, remain follow-up work;
operator construction error stays separate from the reconstruction bound.

An integration test suite binds a real `quantics_fourier_operator` and checks
sign, normalization, and the documented no-permutation output placement against
a dense small-system oracle, plus a noncontiguous selection with spectators, an
index-aligned forward-then-inverse round trip, a multi-patch transform against a
dense reference, invalid selections, the multi-index node rejection, and the
scale contract: the Frobenius factor against the unitary factor of one, a
non-unitary factor, destructive interference whose true output is zero while the
scale stays the operator-based bound, propagation through successive
applications, and separate-term retention after preparation. That last check
observes only the retained representation; it cannot see a temporary network
built and discarded inside preparation, so it is not advertised as detecting a
global direct sum. The forward-then-inverse round trip
restores the original tensor on the original full indices: the inverse takes the
reversed operator-node-to-site selection to undo the forward transform's
bit-reversed output placement, and the result is compared index-aligned, not
against a bit-reversed oracle.
