# TreeTN contraction outcome and early abort

## Status

Reporting and deterministic general-tree SVD zip-up cancellation are implemented
on 2026-11-06 against `097e2927`, following the approved implementation slices.
`contract_with_outcome` and its context counterpart can stop at a retained
tree edge; the chain path and other methods still classify completed results.
Caller-RNG entries remain post-hoc. Adaptive-driver adoption remains future work.
Part of milestone M6 (adaptive patched contraction) of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md);
background in section 1 of
[tree-patching-findings.md](./tree-patching-findings.md). The current public
API is documented in `contraction/outcome.rs`. Factorization counts establish
skipped work on test cases; no wall-time or downstream speedup is claimed.

## Problem

`tensor4all_treetn::contraction::contract` returns only a `TreeTN`. Callers
cannot learn the realized rank of each output edge, and cannot stop a
contraction whose output is already known to be too large.

`tensor4all-partitionedtreetn::contract_adaptive` needs both. It contracts
every compatible pair, compares the maximum bond dimension of each
contribution with the patch cap afterwards, and, if a group saturates,
discards the results and recomputes them on projected children. The
contractions that led to the discarded results are paid in full.

The achievable saving depends on the first qualifying retained edge. The
general-tree schedule reindexes both operands before processing edges, and
that preparation is still paid. It does not first canonicalize the operands;
the separate chain schedule performs canonicalization and final truncation.

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
  reporting is now implemented; early abort for Fit is still out of scope.
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
report. Its `contract_with_outcome_in` counterpart accepts
`TreeTN<IdxTensor, V>` inputs and the caller's `ExecutionContext`, matching
`contract_in`. Both return `TreeTNOperationError`, preserve source chains,
and reuse current option/context/layout validation. A foreign explicit context
or an invalid dense-reference limit must not be hidden by a rank shortcut. No new
backend, host readback, dense fallback, or AD-detaching path is introduced.

Randomized dispatch already exists: RSVD for Zipup/Fit and seeded SRC. The
outcome route preserves that dispatch rather than treating every Zipup
as deterministic SVD. Caller-RNG companions,
`contract_with_outcome_with_rng` and `contract_with_outcome_with_rng_in`,
follow `contract_with_rng*` for RSVD and `contract_src_with_rng*` for SRC:
consume the supplied stream directly, never derive a private seed.
Seed-based entries use the same named RNG and seed
fields as their current counterparts. Randomized methods complete before
threshold detection in this version, so the same seed/stream produces the
same full result and RNG consumption as the matching ordinary entry.

The first implementation establishes post-hoc reports on the existing
dispatch. The second slice adds early abort only to deterministic SVD on the
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

Only `Zipup` with `factorize_alg == FactorizeAlg::SVD` is eligible for the
early-abort hook. The topology dispatch then keeps the chain schedule post-hoc.

| Method | Report | Early abort |
|---|---|---|
| Zipup, deterministic SVD, tree path | Final link dimensions | Implemented. Stops after the first retained edge reaching the threshold; no truncation pass follows the edge loop |
| Zipup, chain path | Final link dimensions | No; saturation is evaluated after completion. The edge loop runs in a non-orthogonal gauge and a final truncation sweep follows, so per-edge factorization ranks are only upper bounds on the final dimensions; aborting on them would report saturation that `contract` does not show |
| Zipup/Fit with RSVD | Final link dimensions | No; preserve current RNG consumption and detect saturation after completion |
| Src | Final link dimensions | No; evaluated after completion |
| Naive | Final link dimensions | No; preserve the explicit dense-reference guard and current SVD decomposition with `max_bond_dim` and `svd_policy`. The reference helper rejects any supplied `qr_rtol` |
| Fit | Final link dimensions | No; evaluated after completion on the current initializer/sweep path |

Post-hoc detection saves no work but gives the same answer as early abort
would, so callers see one contract for every method. Chain-path early abort
based on the conservative factorization ranks is possible future work; it
would need a separate opt-in because it changes which contractions saturate.

## Literature basis and tree-specific argument

The references below already appear in the repository's
[provenance policy](../PROVENANCE_AND_CITATION_POLICY.md). They establish the
algorithm lineage; the API and tree-specific equivalence argument are our
own design and derivation, not a theorem attributed to those papers.

- [Grosso et al., arXiv:2602.22372v1, §4.3](https://arxiv.org/html/2602.22372v1#S4.SS3)
  motivates rank-cap refinement and explicitly mentions stopping a local
  contraction by monitoring zip-up SVD rank. It does not specify our TreeTN
  traversal, outcome API, or an equivalence theorem for our schedules.
- [Stoudenmire and White, NJP 12, 055026 (2010), preprint pp. 5–6](https://arxiv.org/pdf/1002.1305)
  describes zip-up and the return sweep. [Paeckel et al., Ann. Phys. 411,
  167998 (2019), §2.8.3](https://arxiv.org/html/1901.05824v3#S2.SS8.SSS3)
  also discusses final compression and limitations of local truncation.
  These support keeping our chain path post-hoc, not treating its intermediate
  ranks as final ranks or error certificates.

The following is an **independent tree-specific derivation** for
`TreeTN::contract_zipup_controlled`, under its existing valid-input contract:

1. The child-to-parent traversal contracts a node with its completed child
   remainders, then factors all external indices except the two original
   parent bonds. Factorization produces a new link of dimension `rank`.
2. A subtree with no surviving left indices is passed upward before
   factorization, so it creates no report edge and cannot trigger the hook.
   For a retained child, its new link is a left index at each subsequent
   parent factorization. Its presence prevents that parent from being pruned.
   The root is always constructed, even without an output site index.
3. Later operations consume the right factor at the parent; they neither
   replace the saved child factor nor resize its link. Result assembly
   connects the same link identities, and setting the canonical region
   only changes metadata. There is no final rank-changing sweep on this path.
4. Therefore each observed retained rank is the stored link dimension that
   a successful complete execution would return. Once it reaches the
   independent threshold, that complete network necessarily satisfies the
   same existential saturation predicate. The hook may discard the factors
   and stop without constructing a partial public network.

This proves the stored-dimension classification for this schedule only.
It says nothing about minimal Schmidt rank, discarded weight, accuracy,
termination of adaptive splitting, or cancellation between later patch sums.
Errors in skipped numerical work are not evaluated. Adding a return sweep,
recompression, topology-changing fusion or a different link representation
requires rechecking this argument before reusing the hook. The first
implementation's analytical baselines and post-hoc reports are software
oracles; they are not additional claims made by the papers.

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

1. Implemented: ordinary/selected adaptive baselines, including threshold
   equality and exhausted split orders; post-hoc reporting and context/RNG
   parity through the existing dispatch. Metadata tests include unequal
   stored ranks, root fallbacks and the first saturating report edge.
2. Implemented: deterministic tree zip-up abort hook, with public-entry
   factorization counts, first/later-edge comparisons to completed reports,
   scalar-pruning coverage and unchanged chain/randomized schedules.
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
