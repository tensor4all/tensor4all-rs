# Tree adaptive patching roadmap

## Status

Planning record. This document orders the long-term goal into medium-sized
milestones by their data dependencies. It is not an implementation contract:
each milestone that changes public API or algorithm semantics needs its own
design record (or an update of an existing one) and a review before
implementation.

Verified facts about the current code that later milestones rely on are kept
in [tree-patching-findings.md](./tree-patching-findings.md).

## Goal

Provide the complete adaptive-patching method of Grosso et al. on arbitrary
tree tensor networks, with shared-memory and distributed parallel execution:

- adaptive patched interpolation (the pQTCI algorithm) that produces a
  `PartitionedTreeTN` directly;
- patched and adaptive patched contraction, including element-wise products;
- patch-order selection and overpatching control;
- a coherent error contract across interpolation, algebra, and reconstruction;
- patch-level parallelism through Hataori (Rayon first, MPI later).

The chain tensor train is one tree topology, not a separate code path. A
topology counts as a branched tree only if some node has degree three or
more; any claim about trees must be checked on such a topology.

## References

- G. Grosso, M. K. Ritter, S. Rohshap, S. Badr, A. Kauch, M. Wallerberger,
  J. von Delft, H. Shinaoka, *Adaptive Patching for Tensor Train
  Computations*, [arXiv:2602.22372](https://arxiv.org/abs/2602.22372).
- G. Grosso, *Efficient Tensor Compression through Adaptive Patched Quantics
  Tensor Cross Interpolation*, M.Sc. thesis, LMU/TUM (2025).
- Julia lineage: TCIAlgorithms.jl (adaptive interpolation) and
  PartitionedMPSs.jl (patch algebra).

Record any new port, derivation, or algorithm origin in
`docs/PROVENANCE_AND_CITATION_POLICY.md` in the PR that introduces it.

## Current state on `main`

| Capability | Chain (`tensor4all-partitionedtt`, deprecated) | Tree (`tensor4all-partitionedtreetn`) |
|---|---|---|
| Projector, subdomain, disjoint partition | yes | yes (eagerly masked, arbitrary named trees, multiple sites per node) |
| `add_with_patching`, `truncate_adaptive` | yes (`rtol`) | yes (local discarded-weight `cutoff`) |
| `contract_adaptive` (project-first recursion) | yes | yes |
| Split strategies | `Sequential`, `ExactParameterGain` | same |
| Adaptive patched interpolation | yes (`adaptiveinterpolate`) | sequential, generic over the M1 engine trait (`adaptive_interpolation::patched_interpolate`, M2) |
| Global L2 reconstruction and merge | no | yes ([orthogonal-target-reconstruction.md](./orthogonal-target-reconstruction.md)) |
| Parallel patch execution | Hataori Rayon/MPI for interpolation ([adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md)) | none |

The deprecated chain crate is design lineage only, not a verification
baseline. Branch `feat/treetci-adaptive-patching` holds a sequential TreeTCI
patching prototype; it is reference material for M2, not a merge candidate.

## Decisions

1. **Placement.** No new crate and no dependency change.
   - The interpolation-engine trait lives in `tensor4all-treetn`, the layer
     that every engine and `tensor4all-partitionedtreetn` already depend on.
   - Each engine implements the trait in its own crate (TreeTCI in
     `tensor4all-treetci`; later engines likewise). Engines adapt to the
     contract; the driver never adapts to an engine.
   - The patch driver lives in `tensor4all-partitionedtreetn` and sees only
     the trait, so that crate still depends on no interpolation engine. The
     statement in [partitioned-treetn.md](./partitioned-treetn.md) that
     adaptive interpolation is outside that crate must be amended by the M1
     design record; its rationale (no TCI dependency) remains satisfied.
   - Rejected: a new orchestration crate that owns engine adapters (the driver
     side would change for every new engine and adds a public crate); engine
     dependencies inside `partitionedtreetn` (violates the migration record);
     engines depending on `partitionedtreetn` (inverts the layering).
2. **Engine scope.** Implement TreeTCI first. Adding TreeACI or RSI later
   means adding an implementation in that engine's crate only, without
   modifying the trait, the driver, or the TreeTCI implementation.
3. **Error norm.** The L2 error, measured by the driver, is the primary
   accuracy criterion of the public API. It is a guarantee only where it is
   certified (exact or exhaustive); a sampled measurement is an estimate.
   (Wording amended on 2026-10-03 by open question 1 of the
   [M3 record](./tree-patching-error-contract.md#open-questions-for-the-user):
   "verified" is not used for results.) Other norms (for example the
   sampled max-norm used by TCI) are user-selectable. A selectable norm
   without an implementation is a placeholder returning an explicit typed
   "not implemented" error; it never falls back silently to another norm.
4. **Chain crate retirement.** Out of scope. `tensor4all-partitionedtt` is
   left untouched until a future repository-wide restructuring.

## Ordering principle

Milestones follow the direction of data flow: a milestone that consumes
patches comes after the milestone that produces them, and every benchmark or
decision gate runs only on data from the real producer. Stand-in data (for
example patches cut from a dense decomposition) must not be used to decide
anything.

## Milestones

Sizes are relative: S (one focused PR), M (a few PRs), L (a design record plus
several PRs).

### M0. Architecture decisions (S, done)

The four decisions above are recorded. The next gate is the approved M1
design record.

### M1. Interpolation engine seam (M)

Outcome: one patch driver can run any tree interpolation engine.

Scope:

- a trait in `tensor4all-treetn` for "interpolate one patch": inputs are a
  batch evaluator restricted to the active sites, the tree topology,
  candidate pivots, a bond cap, and a tolerance; outputs are a `TreeTN`, a
  convergence verdict that distinguishes "converged" from "reached cap or
  iteration limit", an error estimate, the maximum sampled magnitude, and
  recyclable full-domain pivots;
- the TreeTCI implementation of the trait in `tensor4all-treetci`;
- the design record amends the scope statement of
  [partitioned-treetn.md](./partitioned-treetn.md);
- `partitionedtreetn` keeps no dependency on any interpolation crate.

Exit: the trait and the TreeTCI implementation are merged with tests on chain
and branched trees, and a test-only mock engine exercises the trait,
demonstrating that a second engine needs no change to existing code.

### M2. Sequential tree pQTCI (L)

Outcome: adaptive patched interpolation on arbitrary trees that returns a
`PartitionedTreeTN`. This is the producer of every patch that later
milestones measure or consume.

Scope:

- a driver in `tensor4all-partitionedtreetn`, generic over the M1 trait;
- FIFO patch queue keyed by `Projector` over full `DynIndex` identities;
- acceptance only for a converged patch within tolerance and strictly below
  the bond cap;
- fixed sites handled as described in the findings (dimension-one engine
  vertices, index mapping back to the caller's identities, fused-coordinate
  mapping for multi-index nodes); nodes are never removed;
- opt-in pivot recycling, per-patch deterministic seeds, sampled-zero policy,
  and a driver-level evaluation cache with one-pass transfer to children
  (TreeTCI itself has no evaluation cache).

Exit: accepted patches reproduce the source function within the requested
tolerance against dense references on small cases, on chain and branched
topologies; tests cover splits at leaf, internal, junction, and multi-site
nodes; the driver is deterministic for fixed seeds.

### M3. Error contract (M)

Outcome: one accuracy requirement with a reported, measured error for
interpolation and patched algebra; the measured L2 error is the default
(Decision 3). The measured error is certified (a bound) where it is exact or
exhaustive (and for patched algebra), and a statistical estimate, not a
guarantee, where it is sampled. A sampled estimate,
audited or not, can miss a localized feature that enters a patch only through
a corner or an edge
([known limitation](./tree-patching-error-contract.md#known-limitation-corner-localized-misses); fix deferred to M9).

Scope:

- a user-selectable error-norm option shared by interpolation and patched
  algebra, with typed placeholders for unimplemented norms;
- interpolation acceptance in the selected norm, with the reference scale
  pinned once for all patches instead of per-patch maximum samples;
- an optional global-budget mode for patched contraction and addition,
  measured with the difference-network norms already used by reconstruction;
- reports expose measured errors (certified bounds where exhaustive,
  estimates where sampled), not only requested tolerances.

Exit: the contract is documented in rustdoc and design records; tests check
reported errors (bounds and estimates) against dense references on small
problems.

Design: [tree-patching-error-contract.md](./tree-patching-error-contract.md)
(the interpolation side implemented; the patched-algebra mode, M3b, scoped for
a separate record and not implemented). Open questions 1, 2, 4, 5, 8, and 9
were decided by the user on 2026-10-03 and are recorded there; 3, 6, and 7
are still with the user.

### M4. Patch representation decision (M)

Status: **deferred; no decision recorded.** Earlier measurements used bond
caps of 2 to 64 (realized ranks at most 62). At those bond dimensions timing is
dominated by per-node overhead, so they were smoke tests that cannot support a
decision; their results were removed from the repository.

Scope (unchanged goal, corrected method):

- measure storage and runtime of both representations on patches produced by
  the M2/M3 driver at bond dimensions matched to real downstream use (realized
  ranks in the downstream gw-rs workloads are mostly 30-200 with a tail to
  about 500), with TreeTCI at tolerance around `1e-4`, on chains and on
  branched trees, under a protocol committed before any data is collected;
- the workload must be one that patching is meant for (localized features).
  The three-dimensional tight-binding spectral function used so far has a
  delocalized singular surface, so fixing high-order bits lowers the patch
  rank only slowly and it over-patches without M5;
- preconditions for adoption: the site-free-node failures of `inner` and SRC
  contraction (issue #797), since compact patches create site-free nodes; the
  M6 consumers (patched addition and contraction) measured as well as norm and
  truncation; and an amendment of the full-site-index invariant of
  [partitioned-treetn.md](./partitioned-treetn.md).

Ordering: re-measure after the patching performance defects found on
2026-10-03 (patch-cache key allocation and hashing, cache split, engine
adapter copy) are fixed, and with a workload chosen with M5's partition
experiment, so the producer is representative. On the patching branch the
cache lookup and the cache split are fixed, and the adapter copy is reduced:
the batch is passed through without a copy only when no vertex is site-free,
and is otherwise gathered without the per-site division, as on trees whose
fixed or internal nodes carry no active site (see the cache entries of the M2
implementation decisions and the batch translation entry of the M1 ones).

Exit: a recorded, scoped decision with raw measurements.

### M5. Split selection and overpatching control (M)

Outcome: the patch tree adapts its split sites and does not proliferate
redundant patches.

Status: **design open; nothing implemented.** The design notes and the open
questions for the user are in
[`tree-pqtci-split-selection.md`](./tree-pqtci-split-selection.md). The only
data so far is an exploratory fixed-depth partition study
([`2026-10-03-m5-fixed-depth-exploration.md`](../../benchmarks/results/2026-10-03-m5-fixed-depth-exploration.md)):
at matched accuracy, partitioning a narrow ridge on a branched tree saved
about 4 times the TCI time, while the chain control and a delocalized
spectral function only got more expensive.

Scope:

- a pivot-based split heuristic generalized from the chain algorithm to tree
  edge bipartitions;
- `ExactParameterGain` remains the algebra-side reference strategy; its cost
  (about `L * d` truncations per split decision when `patch_order` is empty)
  motivates a cheaper default for large patch counts;
- a minimum patch size option and an in-loop merge of sibling patches,
  reusing the reconstruction merge logic;
- capped outcomes (M3 open question 4, deferred here by user decision on
  2026-10-03): decide whether a `BondCapReached` patch may be accepted on its
  measured error, together with an early exit of the engine at the first
  saturated sweep. In the review of the M3 open questions, an emulation of
  capped acceptance saved only 1 to 3 patches, changed evaluations by −53% to
  +16%, and most capped outcomes offered for acceptance failed verification;
  the saturated sweeps (29% to 42% of the cost) run before the accept-or-split
  decision, so only an early exit saves them. The corner-localized misses
  were all `Converged` patches, so selection bias is not the reason for the
  deferral
  ([open question 4](./tree-patching-error-contract.md#open-questions-for-the-user)).

Note: split-site selection and the candidate rules for child patches must
take the M3 corner-localized misses into account
([known limitation](./tree-patching-error-contract.md#known-limitation-corner-localized-misses)): a split that cuts through a feature can leave
children that the feature enters only through a corner or an edge, which
uniform sampling and the engine can miss. The fix itself is decided in the
M9 global review.

Exit: measurements on M2 patches compare the heuristic with `Sequential` and
`ExactParameterGain`, and overpatching cases do not exceed the unpatched
parameter count by more than a documented margin.

### M6. Adaptive patched contraction (L)

Outcome: the contraction side of the method is complete on trees. This line
works on existing `PartitionedTreeTN` values and does not depend on M1 or M2;
it is ordered after them to keep the interpolation line first.

Scope:

- the contraction outcome API with early abort
  ([treetn-contraction-outcome.md](./treetn-contraction-outcome.md)) and its
  adoption in `contract_adaptive`;
- patched element-wise products on top of the existing `hadamard`, including
  the projector rule for paired distinct indices;
- refine only the input patches that contributed to unconverged outputs,
  recompute only those outputs, then merge converged neighbors;
- the contraction-side overhead items listed in the findings (prefix-tree
  projector index, per-operation norm caching); the `Sequential` group-sum
  shortcut is [#788](https://github.com/tensor4all/tensor4all-rs/issues/788).

Exit: tests cover the worst, best, and general patch layouts on branched
trees; a benchmark reproduces their qualitative ordering.

### M7. Shared-memory parallel execution (L)

Outcome: interpolation and contraction run patch-parallel on one node with
reproducible results.

Scope:

- Hataori Rayon domains supplied explicitly by the caller, following
  [adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md);
  Hataori becomes an optional dependency of `tensor4all-partitionedtreetn`;
- dynamic scheduling of patches with very different costs;
- parallel contraction over independent output-projector groups;
- the pinned reference scale of M3, so acceptance does not depend on
  execution order;
- a documented outer/inner parallelism policy.

Prerequisite: the evaluator determinism fix of
[#795](https://github.com/tensor4all/tensor4all-rs/issues/795), in the order
decided under M3 open question 8: (c) raw kernels of `TreeTNCachedEvaluator`
for site-free nodes, then (a) pairwise positional contraction on the generic
path; (b), upstream tie-breaking in omeco or tenferro, is long term. Until
then, L2 measurements on trees with a site-free or multi-site node, or with
`f32`/`c32` data, are not reproducible across threads.

Exit: parallel and sequential runs produce identical partitions for fixed
seeds and deterministic callbacks; scaling is measured on M9 workloads.
Determinism scope (M3 open question 9): bitwise identical on the same machine
and build across threads and thread counts, and across processes once a
two-process CI test passes; no cross-machine promise.

### M8. Distributed execution (L)

Outcome: opt-in MPI execution for large workloads: a versioned wire format for
`TreeTN`/`IdxTensor` patches, patch ownership by projector prefix, and
collective entry points matching the Hataori MPI conventions. The M3
determinism scope makes no cross-machine promise, so M8 states its own
determinism contract.

Exit: an MPI smoke test and a multi-rank benchmark.

### M9. Validation and benchmarks (continuous)

- Correctness against dense or independently converged references.
- Workloads matched to real downstream use; every runtime comparison is made
  at matched measured accuracy.
- Each benchmark starts only after the component producing its inputs exists
  on the branch.
- Known risk to track: downstream TreeTCI runs on topologies with junction
  nodes have been observed to be much slower than on chains at low
  temperature; this affects tree pQTCI and must be profiled once M2 exists.
- Corner-localized misses: global review and fix decision. Uniform-sample
  acceptance and the audit of M3 can miss a localized feature that enters a
  patch only through a corner or an edge, underestimating the error by orders
  of magnitude without a warning
  ([record](./tree-patching-error-contract.md#known-limitation-corner-localized-misses); reproduction: the ignored test
  `corner_localized_ridge_is_not_missed_by_sampled_acceptance`). The fix was
  deferred to this review by user decision on 2026-10-03. Candidate remedies:
  - a reject-only screen on points already in the patch cache: no new
    evaluations of `f`; it may only reject a patch, never contribute to an
    estimate;
  - a larger default `samples`;
  - boundary-aware or recycled candidates for child patches: the parent's
    feature points just across the split face;
  - split-site rules that avoid cutting through a feature at a corner (M5);
  - stratified or importance verification.

### M10. Bindings (deferred)

C API and Tensor4all.jl exposure after the Rust API stabilizes, tracked in
their own issues.

## Dependencies

```text
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
                      │
                      └──────────► M7 ──► M8
M6 (independent of M1/M2; scheduled after M2) ──► M7
M9: continuous, each item gated on its producer
M10: deferred
```

## Non-goals

- Changing the TreeTCI or TreeACI algorithms themselves.
- Topology changes between patches; all patches share one named tree
  topology.
- Weighted or non-L2 norms in reconstruction.
