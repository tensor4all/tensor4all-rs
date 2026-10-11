# TreeTN contraction outcome and early abort

## Status

Proposal, revised after two design reviews. Part of milestone M6 (adaptive
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
  first version (see Method coverage). Fit also waits for the open draft
  PR #656, which restructures `fit.rs`.
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
- **Unchanged options.** `ContractionOptions` gains no field. `contract`
  delegates to `contract_with_outcome(..., None)` and returns the network.
- **Why `Saturated` carries no network.** An early-aborted contraction has no
  network, and for post-hoc detection every current caller discards it; one
  shape for both keeps callers simple and frees memory early.
- `#[non_exhaustive]` on the enum follows the existing precedent in the
  workspace (`EvaluationHint` in treetn, `FactorizeResult` in core).

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
| Zipup, tree path | Final link dimensions | Yes. Each edge's factorization rank is its final link dimension (no truncation pass follows the edge loop), so the contraction stops after the first edge whose rank reaches the threshold |
| Zipup, chain path | Final link dimensions | No; saturation is evaluated after completion. The edge loop runs in a non-orthogonal gauge and a final truncation sweep follows, so per-edge factorization ranks are only upper bounds on the final dimensions; aborting on them would report saturation that `contract` does not show |
| Src | Final link dimensions | No; evaluated after completion |
| Naive | Final link dimensions | No; evaluated after completion. Naive ignores `max_bond_dim`, `svd_policy`, and `qr_rtol` and decomposes with default SVD options, so its ranks are untruncated and saturate under different conditions than the other methods |
| Fit | Final link dimensions | No; evaluated after completion (deferred, see non-goals) |

Post-hoc detection saves no work but gives the same answer as early abort
would, so callers see one contract for every method. Chain-path early abort
based on the conservative factorization ranks is possible future work; it
would need a separate opt-in because it changes which contractions saturate.

## Adoption in `partitionedtreetn`

This depends on #788 (PR #789) being merged.

- Today `contract_adaptive` contracts all compatible pairs before grouping
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
- Because saturation is defined on final bond dimensions, the partition
  layout and values are unchanged; only contractions whose results would have
  been discarded are skipped or cut short.
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

## Open questions for review

1. Name: `abort_at_rank` versus `rank_budget` or `saturation_threshold`.
