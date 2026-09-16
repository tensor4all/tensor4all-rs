# Issue #752 follow-ups: node merging and the uniform merge-refine schedule

Session: 2026-09-16. Base: `aedaa4d9` (squash of PR #753). Branch:
`feat/752-subset-operator-node-merge`.

## Scope and sources

Issue #752 tracks adaptive orthogonal-target reconstruction and subset-site QFT
integration. PR #753 shipped the reconstruction engine and
`ReconstructionTarget::from_subset_operator`. The maintainer's plan comment on
#752 lists the remaining work and a suggested PR sequence; this session implements
its steps 1 and 2 plus the operator-node merging item that PR #753 explicitly
deferred:

- operator-node merging, so two selected indices may share one tree node;
- the deterministic, serial level-coupled merge-refine schedule (uniform binary
  input tree, exact local application, structural counters, provenance).

Read back before implementing: `docs/design/orthogonal-target-reconstruction.md`,
`crates/tensor4all-partitionedtreetn/src/reconstruction/{mod,target,engine}.rs`,
`crates/tensor4all-treetn/src/operator/{apply,linear_operator}.rs`, and
`crates/tensor4all-treetn/src/treetn/{transform,restructure}`.

Left out deliberately, matching the plan: approximate operator application with a
retained error allowance; adaptive refinement, nonuniform input trees,
multi-coordinate groups, and benchmark evidence; automatic zero-padding, which the
plan requires to be a separate opt-in domain/embedding policy rather than part of
the scheduler.

## Operator-node merging

`from_subset_operator` previously rejected a selection with two indices on one
tree node, because `LinearOperator::rename_nodes` cannot map two operator nodes
onto one name. Instead of renaming, the constructor now builds the operator's
quotient site-index network (one node per preimage owner) and calls
`LinearOperator::restructure_to`, which fuses the group exactly and moves each
mapping to the node owning its internal index. Singleton groups reduce to the
previous rename behavior, so the ordinary selection path is unchanged in effect.

Rejected alternative: splitting the *state* node into one node per selected index
and merging the result back. That would also work but contradicts the documented
contract that a selected index keeps its node assignment, and it would need
truncation-free splitting decisions the schedule does not otherwise require.

Group connectivity is validated before restructuring. A group threaded through
another owner's node cannot be fused without absorbing sites that belong
elsewhere, so it is rejected with repair guidance instead of mis-binding a
multi-site node. `is_connected_group` checks induced-subgraph connectivity with a
name-based BFS, avoiding a new `petgraph` dependency in this crate.

## Preparation seam and provenance

`from_subset_operator` and the scheduler now share
`ReconstructionTarget::prepare_subset_images`, which returns each image together
with the preimage patch support it came from. The scheduler needs that input leaf
identity; the greedy path only needs the images. Sharing the seam keeps one
application path and avoids inferring ancestry from spectator-only projectors.

## Uniform merge-refine schedule

`reconstruction::schedule_merge_refine` implements `G(A, B) = P_B F P_A w` for a
uniform binary input partition: level 0 applies the complete transform once per
input leaf, level `t` merges the input siblings of `k_(d-t+1)` and restricts each
child to the output prefix region fixing `r_t`. Restriction happens before
addition, transforms are never summed over the whole output domain, and consumed
parents are released, so only two levels are live.

Geometry is explicit rather than inferred: input significance `k_j` is operator
node `j`, while frequency significance `r_j` sits on the selected index that
carried `k_(d+1-j)`, matching the documented no-permutation output placement of
`from_subset_operator`. Tests verify this against a dense subset-DFT oracle.

Validation before any application: non-empty binary selection, `output_depth` at
most the selection depth, a work limit covering `2^d`, and a preimage that is
exactly the dyadic leaves with identical spectator constraints. A missing or
repeated coordinate assignment, an unconstrained selected index, and disagreeing
spectator constraints are rejected; the last one is a `ProjectorMismatch`
because the strict subdomain addition used inside a region requires identical
projectors.

Report vocabulary is schedule-specific (`level_count`,
`applied_operator_count`, `additions`, `projections`, `work_items_per_level`,
`peak_work_items`, plus region/term/rank/parameter counts) rather than reusing
`ReconstructionReport`, whose `split_count`/`merge_count` describe greedy
acceptance decisions. The global accuracy contract is still shared: the pinned
`reference_scale` uses the same `SubsetOperatorOptions` amplification factor and
the allowance is `max(atol, rtol * reference_scale)`. The trajectory performs no
compression, so `error_bound` is zero and no compression-error ledger is claimed.

## Verification

- `cargo test -p tensor4all-partitionedtreetn --release`: 28 lib + 16 patching +
  13 partitioned_tree_tn + 16 qft_reconstruction + 21 reconstruction + 15
  subdomain_tree_tn tests, in addition to 51 doctests.
- The schedule test verifies every retained intermediate `P_B F P_A w` against a
  dense oracle for two- and three-bit selections at every executed depth, plus
  the fully refined strict partition against the dense normalized DFT.
- Structural counters are asserted, not inferred from the output: `M`
  applications, `M * d` additions, `M` live items per level, and a bounded peak.
- Release Clippy with `--all-targets --no-deps -- -D warnings`, `cargo fmt
  --check`, the runnable `--example merge_refine`, the mdBook snippet suite and
  book build, and the repository-rules dry-run review.

## Remaining risk

The reported counters are schedule shape, not runtime cost; no performance claim
is made and no benchmark was run. The intermediate-oracle test uses a small dense
reference (up to three bits) by design. Compression, adaptive stopping, and
nonuniform geometry are still unimplemented, and the error-transport derivation
for cached approximated parents is documented in the design record but not yet
exercised by code.
