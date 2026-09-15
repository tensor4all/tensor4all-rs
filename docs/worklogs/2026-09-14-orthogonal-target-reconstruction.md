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

All 19 new integration tests passed, along with the existing partitioned TreeTN
tests. All 1,027 workspace doctests passed. The executable guide example is
included directly from its Rust source; its execution, the mdBook HTML build,
and the complete mdBook execution suite passed. Changed-crate release Clippy
with all targets and warnings denied, workspace formatting, public-error-doc
validation, the refreshed API inventory, and deterministic repository-rules
review also passed.

Coverage impact reviewed: no test or behavior was removed; the shared logical
parameter-count helper remains exercised by existing adaptive-patching tests
and by the new split-candidate path. Hosted CI remains the coverage authority.

## Limits

The reported error bound uses floating-point norm measurements, not interval
arithmetic, and excludes backend roundoff. No AD guarantee or GPU execution was
validated. No runtime speedup is claimed. QFT itself, zero-padding semantics,
and its complementary merge-refine schedule are not implemented in this
general-reconstruction change.
