# Tree patching: verified findings

## Status

Facts about the current code that the milestones of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md)
depend on. Milestone dependencies, public patch APIs, and issue dispositions
were synchronized on 2026-11-06 against `main` at `097e2927`; older
representation/storage observations retain their exploratory scope. Each
section names the milestone that consumes it. This record contains no performance estimates;
decisions that need numbers are made in their milestone on real data.

## 1. Contraction outcome and early abort (consumed by M6)

- `tensor4all_treetn::contraction::contract` returns only a `TreeTN`.
  The `contract_with_outcome` family reports final stored edge ranks for all
  methods. General-tree SVD zip-up can stop at a retained edge; other paths
  classify after completion. It does not report discarded weight or whether
  `max_bond_dim` was binding.
- Zip-up applies `max_bond_dim` through core `factorize` at each edge.
  `tensor4all_core::FactorizeResult` exposes the realized `rank` and, for SVD,
  the retained singular values, but not the discarded weight or whether the
  cap determined the rank.
- The chain zip-up path runs a final truncation sweep after its edge loop, so
  its per-edge factorization ranks are upper bounds on the final bond
  dimensions. The tree path has no such pass; its factorization ranks are the
  final link dimensions.
- `partitionedtreetn::contract_adaptive` contracts every compatible pair,
  then compares bond dimensions with the patch cap; a saturated group is
  discarded and recomputed on projected children.
- Non-adaptive `PartitionedTreeTN::contract` now refines operands when their
  predicted output regions overlap (#1093), subject to the default
  4096-atom bound (#1143/#1144). `contract_selected` and both adaptive entries
  use the pair-selection path without that wrapper. M6 still needs a
  reviewed overlap rule for adaptive groups and paired distinct indices;
  it must preserve the existing bounded non-adaptive contract.
- Groups use `SubDomainTreeTN::exact_sum` with the selected `SumEngine`, then
  truncate once. Approximate sum engines (#1125/#1126/#1129) are not a
  prerequisite for the M6 outcome seam.

Design: [treetn-contraction-outcome.md](./treetn-contraction-outcome.md).
Exact detection of whether a cap discarded nonzero weight needs core
`factorize` to expose the discarded tail and is left to a core issue.

## 2. Element-wise product (consumed by M6)

- `tensor4all_treetn::hadamard(left, right, index_pairs, center, options)`
  multiplies two TreeTNs element-wise along paired external indices through
  `partial_contract` with `PartialContractionSpec::diagonal_pairs`.
- Each pair attaches a structured copy tensor (`IdxTensor::copy_tensor`,
  diagonal storage) and runs the contraction selected by
  `ContractionOptions`; nothing is materialized densely.
- `partial_contract_with_outcome` exposes ranks and saturation for the same
  specifications, including diagonal-only Hadamard operations. Its tree hook
  is limited to equal input topologies without requested output ordering.
- Patch compatibility for pairs of distinct index identities is not defined
  yet; that rule belongs to M6.

## 3. Fixed sites in interpolation output (consumed by M2)

- A `TreeTciGraph` vertex is one TreeTCI site with one local dimension.
  Fixing a site can be expressed as a vertex of local dimension one on the
  unchanged graph, which is valid on every tree.
- Removing a fixed vertex changes the tree. A fixed vertex of degree three or
  more is a junction: re-inserting it into the original topology after
  removal fuses bonds into a product bond. All patches must share one named
  topology for patch algebra, so nodes are never removed.
- `tensor4all_treetci::to_treetn` creates fresh site indices
  (`DynIndex::new_dyn`). The driver maps them back to the caller's `DynIndex`
  identities; active sites can use `TreeTN::replace_site_index_with_indices`.
- A TreeTN node may carry several site indices while a TreeTCI vertex has one
  fused local dimension; splitting on one index of such a node shrinks the
  fused dimension. The mapping is specified in the M1/M2 design records.
- Under the current `partitionedtreetn` invariant, a fixed site appears in a
  patch as its full index masked to the fixed coordinate. M4 decides whether
  that changes.
- TreeTCI has no evaluation cache; the driver provides one (M2).

## 4. Patch representation (consumed by M4)

### Current representations

- `partitionedtreetn` masks each projected index with `IdxTensor::mask_index`
  (an element-wise product with a dense one-hot vector) and rebuilds the
  TreeTN. No node is inserted; the projected index keeps its full dimension
  and its zero entries are stored.
- The removed chain crate embedded fixed middle sites with a compact
  copy-selector (`from_copy_selector`) and boundary fixed sites with unit
  bonds. Core factorization accepts dense storage only.
- `PartitionedTreeTN::to_treetn` sums all patches by direct sum, so bond
  dimensions add. This explicit conversion is not the compact representation
  candidate; reconstruction avoids a global direct sum.

### Compact candidate

A patch would store a TreeTN whose projected site indices are removed with
`select_indices` instead of masked, with the `Projector` recording them. Nodes
and edges stay unchanged, so all patches keep one topology; a node whose sites
are all fixed becomes a site-free node. Operations would slice the other
operand (`select_indices`) where one operand has an index fixed, and un-fixing
reattaches the site leg on its original node.

Verified TreeTN preconditions:

- TreeTN supports site-free nodes (covered by `operator/linear_operator`
  tests; restructure treats them as internal connectors).
- Automatic absorption of site-free subtrees happens only in explicit
  topology changes (`fuse_to`, `split_to`, restructure); partition algebra
  does not call them.
- `TreeTN::same_topology`, checked before zip-up, compares nodes and edges
  only, not site indices.

The site-free-leaf robustness issue is fixed in `tensor4all-treetn` by PR #799
(merged to `main` as 8379852e): every empty-side split (canonicalization,
truncation, fit, swap, topology-preserving zip-up) goes through one helper,
`factorize_allowing_empty_side`, and canonicalization replaces a site-free
leaf's bond by a fresh dimension-one link. The subsequently reported failures
of `inner`, SRC contraction, and `factorize_tensor_to_treetn` (#797) were
fixed in #986. That issue is closed, rather than an outstanding adoption
blocker. This does not select compact storage: M4 still requires representative
producer and consumer measurements.

No representation decision has been made; see the M4 status in the roadmap.

## 5. Avoidable overhead in patch algebra

| Overhead | Evidence | Consumer |
|---|---|---|
| `contract_group_project_first` builds and truncates the exact group sum before checking whether one contribution already reached the cap | `partitionedtreetn/src/patching.rs` | M6; the `Sequential` case was fixed under [#788](https://github.com/tensor4all/tensor4all-rs/issues/788) (closed) |
| Saturated probes are discarded and recomputed at every recursion level, with no early abort | same | M6 (contraction outcome API) |
| Default `ExactParameterGain` projects and truncates every candidate's children for each split decision, about `L * d` truncations when `patch_order` is empty | `split_child_parameter_count` | M5 |
| Eager masking makes contraction and factorization iterate over zero coordinates; each projection adds a `64 * eps` compression sweep | `mask_index`, `project_if_present` | M4 |
| Pairwise disjointness validation on every partition construction and `N_A * N_B` pair enumeration in contraction | `partitioned_tree_tn.rs` pairwise loop | M6, before M7 |
| `norm` and `norm_squared` clone and canonicalize on every call | `SubDomainTreeTN::norm` | M6 |
| TreeTCI has no engine-local evaluation cache; the M2 driver now caches fallible batches and transfers parent samples to children once | `adaptive_interpolation/cache.rs` | M2 resolved at the driver boundary |

## 6. Sparse and block-sparse storage (not adopted)

- `tensor4all-tensorbackend` provides `Dense`, `Diagonal`, and `Structured`
  (repeated axis classes) storage. Factorization accepts dense storage only,
  and the CUDA path requires dense axis classes.
- tenferro keeps its core dense and provides an extension mechanism for
  domain-specific representations with traced execution and AD. Its
  `ext/sparse` crate is a tutorial (COO sparse-sparse matmul, fixed
  structure, unpublished) without general einsum or SVD. No block-sparse
  crate exists in the tensor4all organization.
- `Structured` storage is a precedent inside tensor4all: its contraction is an
  einsum over compact payloads executed by tenferro, and AD follows. A
  block-sparse layer could follow that pattern in `tensor4all-tensorbackend`
  and core. How the tensor4all graph execution mode handles structured
  storage is not verified.
- Relative to the current partition (a list of patches), block-sparse storage
  of patches stores the same nonzero blocks; its advantage appears only when
  one network over many patches must be kept under block-preserving
  operations, which the patching workflow avoids.

Decision: not adopted for this roadmap. A block-sparse proposal can be made on
its own merits (for example quantum-number symmetries).

## 7. Interpolation checkpoint follow-ups

Issue states checked on 2026-11-06. These are recorded limitations or deferred
work, not newly approved implementation tasks. The checkpoint corrections are
recorded in the [work log](../worklogs/2026-10-04-interpolation-checkpoint-review.md).

| Work | Owner | State and constraint |
|---|---|---|
| Patch representation | M4 | Deferred; downstream-scale measurements of real interpolated patches and M6 consumers are required before choosing eager or compact storage. |
| Site-free-node operations, [#797](https://github.com/tensor4all/tensor4all-rs/issues/797) | M4 adoption / treetn | Closed by #986; not a remaining blocker for M4 or M6. |
| Generic cached-evaluator determinism | M7 qualification / treetn | The fresh-thread generic-path regressions remain ignored; interpolation lanes exist, but full bitwise parity is not established. This does not block serial M6 contraction. |
| Planner tie-breaking, [tenferro-rs#1963](https://github.com/tensor4all/tenferro-rs/issues/1963) | M7 qualification / upstream | Closed by upstream #2014 for default non-annealed options, with an annealing caveat. Downstream generic-path interpolation qualification remains separate. The two-process patch-contraction test does not cover that interpolation path. |
| Corner-localized misses | M9 global review | #1034 added the default residual-TT estimator and Monte Carlo fallback. The regressions now run as scheduled heavy tests; their ignore marker denotes suite cost, not an unresolved reproduction. Sampled estimates still are not certificates. |
| Thread control and scaling, [#670](https://github.com/tensor4all/tensor4all-rs/issues/670) | M9 / TreeTCI | Closed after the thread-budget diagnosis and recommendation; representative scaling measurements remain a separate M9 task. |
| Overpatching and optional selector | M9 / study deferred from M5 | Downstream-scale, matched-accuracy measurements remain outstanding; the edge-pivot proposal is unimplemented and not an approved selector. |
| Blocked, unconverged patch precedence | M5 provisional contract | Patch-size plan open issue 1 remains provisional: the current rule judges a previously unjudged retained run once and reports whether it met its allowance. No final user decision is inferred here. |
| Reference/default and norm reservations | M3 decisions | Error-contract questions 3, 6, and 7 remain with the user: automatic reference policy, the separate M3b scope, and placeholder norm selection. The implemented defaults remain unchanged. |
| Contraction outcomes, sibling merging, algebra error accounting | M6 / M3b | The checkpoint and #900 are complete. The [M6 proposal](./tree-patched-contraction.md) still needs review, including its overlap policy and multi-level merge ledger; M3b remains separate. |

## 8. M6 issue dependency review

The [M6 readiness review](./tree-patched-contraction.md#issue-dependencies-and-implementation-gates)
compares current issues and their comments with the synchronized implementation.
No existing open issue must be closed before the serial contraction outcome
work can start. The proposal's unresolved API, overlap, merge-budget and
validation decisions still gate their respective implementation steps.
An unassigned issue is not evidence of availability: #1026 is explicitly
claimed, the #1125/#1126/#1129 sum work is queued, and #627 has draft #1146.

Resolved history: [#791](https://github.com/tensor4all/tensor4all-rs/issues/791)
was fixed by #793 (stable TreeTN site/edge construction order), and
[#788](https://github.com/tensor4all/tensor4all-rs/issues/788) closed the
Sequential group-sum shortcut defect. Their remaining consumers and distinct
planner work are listed above rather than treating these issues as open.
