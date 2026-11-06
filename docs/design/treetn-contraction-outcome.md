# TreeTN contraction outcome and early abort

## Status

Review candidate synchronized on 2026-11-06 with `097e2927`, following the
earlier two design reviews. The API below is proposed, not implemented. Part of milestone M6 (adaptive
patched contraction) of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md);
background in section 1 of
[tree-patching-findings.md](./tree-patching-findings.md). It adds public API
to `tensor4all-treetn` and must be approved before implementation, which is
scheduled with M6, not before the interpolation milestones.

## Problem

`tensor4all_treetn::contraction::contract` returns only a `TreeTN`. Callers
cannot learn the realized rank of each output edge, and cannot stop a
contraction whose output is already known to be too large.

`tensor4all-partitionedtreetn::contract_adaptive` needs both. It contracts
every compatible pair, compares the maximum bond dimension of each
contribution with the patch cap afterwards, and, if a group saturates,
discards the results and recomputes them on projected children. The
contractions that led to the discarded results are paid in full.

The achievable saving is bounded: zip-up processes edges from the leaves
toward the center, and the first edges usually carry small ranks, so an abort
typically happens partway through the sweep. The up-front copy,
canonicalization, and reindexing of both operands is always paid.

## Goals

- Report the final bond dimension of every output edge of a contraction, in a
  well-defined edge order, for every contraction method.
- Optionally stop a contraction once an output edge is known to reach a
  caller-given rank threshold, without changing which contractions are
  reported as saturated.
- Keep `contract` unchanged for existing callers.

## Non-goals

- Exact detection of whether a rank cap discarded nonzero weight. That needs
  core `factorize` to expose the discarded tail and is deferred to a core
  issue.
- Early abort on the chain zip-up path, for Src, for Naive, and for Fit in the
  first version (see Method coverage). The former Fit initializer dependency
  #656 landed through #981; it is no longer an external blocker. Fit's
  outcome/report coverage remains an unimplemented part of this proposal.
- Outcome variants of `partial_contract` and `hadamard`, and of the low-level
  SRC entry points `contract_src_with_rng` and `contract_src_with_rng_in`.
  Patched element-wise products are an M6 item and will reuse the entry point
  below.

## Public surface

```rust
/// Bond dimension of one output edge, oriented toward the report root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRank<V> {
    /// Endpoint farther from the root.
    pub child: V,
    /// Endpoint closer to the root.
    pub parent: V,
    pub rank: usize,
}

/// Final bond dimensions of the output network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionReport<V> {
    /// One entry per output edge, in `edges_to_canonicalize_by_names(root)`
    /// order of the output network (leaves toward `root`); see "Report root".
    pub edge_ranks: Vec<EdgeRank<V>>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ContractionOutcome<T, V> {
    /// The contraction finished below the threshold (or without one).
    Completed { network: TreeTN<T, V>, report: ContractionReport<V> },
    /// An output edge reached the threshold.
    Saturated { edge: EdgeRank<V> },
}

pub fn contract_with_outcome<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
) -> Result<ContractionOutcome<T, V>, TreeTNOperationError>;
```

- **Report root.** The root is `center` when the output network contains it.
  Otherwise it is the single node of the output network's canonical region.
  This case occurs on the chain zip-up path when the node `center` keeps no
  output site indices and is pruned; the path then seeds the canonical region
  at another node. If the canonical region is not a single node, the root is
  the smallest output node name. The rule is deterministic and applies to
  every method.
- **Saturation.** The outcome is `Saturated` exactly when some edge of the
  output network that `contract(tn_a, tn_b, center, options)` would return has
  a bond dimension `>= abort_at_rank`. Equality counts, matching
  `partitionedtreetn`: a capped result cannot distinguish an exact rank equal
  to the cap from a truncated one. With `None` the outcome is always
  `Completed`.
  This describes the stored result's link dimensions, not the tensor's
  mathematical Schmidt rank, discarded weight, convergence, or accuracy.
  An edgeless result has an empty report and is always `Completed`, including
  with a threshold of one; the adaptive caller's separate rank-one predicate
  remains necessary.
- **Which edge is named.** With post-hoc detection, `edge` is the first edge
  at or above the threshold in report order, with its final rank. With early
  abort (tree zip-up path only, where `center` is always kept), `edge` is the
  first such edge in the processing order,
  `edges_to_canonicalize_by_names(center)` of the input network `tn_a`. The
  definition does not rely on the two orders coinciding; a test checks that
  they agree today.
- **Independence from truncation.** `abort_at_rank` is separate from
  `ContractionOptions::max_bond_dim`. The truncation cap limits what the output
  may keep; the threshold is the caller's probe budget. `contract_adaptive`
  sets the threshold to the patch cap while its contraction options may carry
  a different cap or none.
- **No invalid state.** `NonZeroUsize` makes a zero threshold
  unrepresentable, so no validation error path exists for it. All other
  option validation happens before any shortcut, as in `contract`.
- **Unchanged options.** `ContractionOptions` gains no field. Existing
  `contract`/`contract_in` and the outcome entries share an internal execution
  seam. Ordinary calls do not build a report or pay a new per-edge report
  allocation. Outcome entries alone collect the final-edge metadata.
- **Why `Saturated` carries no network.** An early-aborted contraction has no
  network, and for post-hoc detection every current caller discards it; one
  shape for both keeps callers simple and frees memory early.
- `#[non_exhaustive]` on the enum follows the existing precedent in the
  workspace (`EvaluationHint` in treetn, `FactorizeResult` in core).

### Execution and RNG parity

The public entry above is the existing generic compatibility route with a
report. Its proposed `contract_with_outcome_in` counterpart accepts
`TreeTN<IdxTensor, V>` inputs and the caller's `ExecutionContext`, matching
`contract_in`. Both return `TreeTNOperationError`, preserve source chains,
and reuse current option/context/layout validation. A foreign context or an
invalid dense-reference limit must not be hidden by a rank shortcut. No new
backend, host readback, dense fallback, or AD-detaching path is introduced.

Randomized dispatch already exists: RSVD for Zipup/Fit and seeded SRC. The
outcome route must preserve that dispatch rather than treating every Zipup
as deterministic SVD. Caller-RNG companions, proposed as
`contract_with_outcome_with_rng` and `contract_with_outcome_with_rng_in`,
follow `contract_with_rng*`: consume the supplied stream directly, never
derive a private seed. Seed-based entries use the same named RNG and seed
fields as their current counterparts. Randomized methods complete before
threshold detection in this version, so the same seed/stream produces the
same full result and RNG consumption as the matching ordinary entry.

The first implementation establishes post-hoc reports on the existing
dispatch; the second adds early abort only to deterministic SVD on the
general-tree zip-up path. The abort hook is selected at the schedule boundary,
shared by compatibility and context routes. It may skip numerical work after
the decisive edge, so errors that would occur only in that unexecuted work
are not promised. Required input validation precedes the abort; an error from
any operation actually executed is propagated, never converted to saturation.

### Report shape by method

Edge order and orientation are always taken from the returned network, so the
report never disagrees with the network. Networks from different methods may
differ in shape:

- Zip-up drops the nodes of subtrees without surviving site indices; a
  rank-one factorization stays in the output as a dimension-one link.
- Fit keeps such nodes, joined by the dimension-one links of its
  topology-preserving initializer.
- Naive collapses a scalar result to one node, so its report is empty.

## Method coverage

| Method | Report | Early abort in this proposal |
|---|---|---|
| Zipup, deterministic SVD, tree path | Final link dimensions | Yes. Each retained edge's factorization rank is its final link dimension (no truncation pass follows the edge loop), so the contraction stops after the first retained edge whose rank reaches the threshold |
| Zipup, chain path | Final link dimensions | No; saturation is evaluated after completion. The edge loop runs in a non-orthogonal gauge and a final truncation sweep follows, so per-edge factorization ranks are only upper bounds on the final dimensions; aborting on them would report saturation that `contract` does not show |
| Zipup/Fit with RSVD | Final link dimensions | No; preserve current RNG consumption and detect saturation after completion |
| Src | Final link dimensions | No; evaluated after completion |
| Naive | Final link dimensions | No; preserve the explicit dense-reference guard and current SVD decomposition with `max_bond_dim` and `svd_policy`. The reference helper rejects any supplied `qr_rtol` |
| Fit | Final link dimensions | No; evaluated after completion on the current initializer/sweep path |

Post-hoc detection saves no work but gives the same answer as early abort
would, so callers see one contract for every method. Chain-path early abort
based on the conservative factorization ranks is possible future work; it
would need a separate opt-in because it changes which contractions saturate.

## Adoption in `partitionedtreetn`

The former #788 prerequisite is resolved. Adoption preserves the existing
ordinary and selected baselines described below.

- Today `contract_adaptive` contracts selected compatible pairs before grouping
  them by output projector, so no contraction can be skipped at the top
  level. The adoption groups pairs first: the output projector of a pair is
  computed from the operands (projector intersection filtered to the surviving
  output indices, as `SubDomainTreeTN::contract` does), without contracting.
- For each group with `PatchSplitStrategy::Sequential`, the split candidate
  is computed from the group's output indices and projector before any
  contraction. If a candidate exists, pairs are contracted with
  `abort_at_rank = patching cap`. A contribution counts as saturated when its
  outcome is `Saturated` or, for a `Completed` network, when
  `SubDomainTreeTN::max_bond_dim() >= cap`. The second clause keeps today's
  predicate for outputs without edges, whose maximum bond dimension is
  reported as one (so they saturate a cap of one). The first saturated
  contribution stops the group, and it splits at that candidate. If no
  candidate exists, pairs are contracted without a threshold, because the
  fallback returns the full group sum.
- With `ExactParameterGain` the split decision needs the full group sum, so
  contributions are contracted without a threshold, as today.
- Layout and value preservation is checked against the baseline, rather
  than inferred from rank predicates alone. Keep the existing logical
  ordering and content-digest tie-break, exact `SumEngine`, selected top-level
  projector matching, and project-first child recomputation. Selecting a
  parent group includes its split descendants; requesting only a descendant
  projector does not select an unmatched ancestor group.
- Chain-shaped inputs that keep output sites always take the chain zip-up
  path, which detects saturation after completion, so they gain only the
  skipped remaining pairs of a group, not a shortened contraction. Tests of
  the early-abort saving therefore use branched trees.

## Tests and documentation

- `contract_with_outcome(..., None)` equals `contract` on chain and branched
  trees (dense comparison), and its report equals the returned network's link
  dimensions in the documented order.
- A chain case in which `center` is pruned from the output: the report root
  is the canonical-region node and the report still covers every output
  edge.
- On the tree path, a threshold reached at an early edge returns `Saturated`
  naming that edge, with fewer factorizations than a full run (asserted by a
  test-only counter).
- On a branched tree where two branches both reach the threshold, the early
  abort's `edge` equals the first saturating entry of the `Completed` report
  for the same inputs without a threshold.
- For every method, `Saturated` is returned exactly when `contract`'s result
  has an edge at or above the threshold (equality included), including the
  chain path where the answer comes after completion.
- Report shapes: Fit dummy links and the Naive scalar case.
- `partitionedtreetn`: on a branched tree, a Sequential group stops
  contracting after the first saturated contribution (counter), with
  unchanged partition layout and dense values; edgeless contributions with a
  cap of one keep today's behavior; the `ExactParameterGain` path and the
  no-candidate fallback are unchanged.
- Rustdoc: runnable, asserted examples for `EdgeRank`, `ContractionReport`,
  `ContractionOutcome`, and `contract_with_outcome`, with `# Errors` naming
  the failure conditions.
- Drift check: `skills/use-tensor4all-rs/` and the TreeTN guide pages that
  mention contraction options.

## Review choices and implementation order

The concrete candidate keeps `abort_at_rank: Option<NonZeroUsize>` as a
separate argument. Equality saturates, `None` completes normally, and the
existing contraction options and return types remain unchanged. These are
review choices for the first outcome slice; accepting this document does not
decide adaptive overlap, merge placement or the sibling-merge error ledger.

1. Preserve ordinary/selected adaptive baselines, including threshold
   equality and exhausted split orders; establish post-hoc report and
   context/RNG parity through the shared dispatch.
2. Add the deterministic tree zip-up abort hook. Tests compare its retained
   edge to the completed report and count executed factorizations. No skipped
   work is claimed for the chain or randomized paths.
3. Adopt lazy grouping in the Sequential adaptive path, preserving the
   selected and ordinary baselines. `ExactParameterGain` retains full probes.

### Existing adaptive baselines

`tests/adaptive_contraction_baseline.rs` uses a true degree-three tree with
analytically known values. Two contributions cancel from `3 F - 2 F` to
`F(w, x) = (w + 1) (2 |000> + 4 |111>)`; the complex case carries the same
nonzero phase through contraction without conjugation. The spectator and
contracted indices share an ID but differ in prime level.

For Sequential/ExactParameterGain and Binary/Nary exact sums, the baseline
pins these known outputs for both ordinary and selected calls:

| Patch cap / split availability | Ordinary partition | Selected `w = 1` |
|---|---|---|
| 2, split `x0` available | Four `(w, x0)` patches, rank one | Two descendants, rank one |
| 3 or no cap | Two `w` patches, rank two | One `w` patch, rank two |
| 2, only already-fixed `w` in the split order | Two `w` patches, rank two | One `w` patch, rank two |

Dense values are checked against the analytical tensor after one whole-result
materialization. Duplicate selections do not duplicate contributions. An
empty selection or unmatched descendant selection returns an empty partition;
invalid contraction options are still rejected before those shortcuts.
