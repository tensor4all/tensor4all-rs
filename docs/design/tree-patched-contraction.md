# Adaptive patched contraction on trees

## Status

Proposal for milestone M6 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md). It
needs review and the answers to the [open questions](#open-questions-for-the-user)
before implementation. It builds on, and does not replace, the
[contraction outcome proposal](./treetn-contraction-outcome.md), which adds the
early-abort entry point to `tensor4all-treetn`; that record is still a
proposal and its approval is a prerequisite of step 2 below.

No code has been written for M6. The facts below were checked against `main`
at `4d4fb6fc`.

## Goal

Complete the contraction side of adaptive patching on trees:

- contraction and element-wise products of two `PartitionedTreeTN` values
  whose output patches stay below the bond cap, refining only where an output
  needs it;
- a post-processing merge that undoes overpatching on any partition,
  interpolation or contraction output alike;
- removal of the patch-algebra overheads that the findings attribute to M6,
  so that M7 parallelizes an algorithm without quadratic bookkeeping.

The roadmap exit applies: tests cover the worst, best, and general patch
layouts on branched trees, and a benchmark reproduces their qualitative
ordering.

## Current behavior

`contract_adaptive` (`crates/tensor4all-partitionedtreetn/src/patching.rs`)
already implements the project-first rule of `PartitionedMPSs.jl`:

1. Every pair of left and right patches is contracted
   (`SubDomainTreeTN::contract`, `N_A * N_B` calls). An incompatible pair
   returns `None` after validation; a compatible pair is contracted in full.
2. Contributions are grouped by their exact output projector (the
   intersection of the operand projectors, filtered to surviving site
   indices).
3. Each group runs `contract_group_project_first`:
   - with `Sequential`, if one contribution already reaches the cap and a
     split index exists, the group splits without forming the sum (#788);
   - otherwise the contributions are added exactly, truncated once with the
     contraction options, and checked against the cap (equality counts);
   - a saturated group chooses a split index among the unprojected output
     indices, projects the original operands of its pairs to each child
     coordinate (`project_if_present`, with a `64 * eps` compression), and
     recurses on the child groups.
4. The result is validated as a partition and passed through
   `truncate_adaptive` with volume-proportional cutoffs.

So the paper's steps "refine only the contributors of unconverged outputs"
and "recompute only those outputs" are already met: the recursion is per
group and projects only that group's operands. What is missing:

| Gap | Effect |
|---|---|
| All pairs are contracted before grouping | Saturated contributions are paid in full, then discarded |
| No early abort inside a contraction | Each saturated probe runs to completion |
| Overlapping output projectors are rejected | `from_subdomains` returns `OverlappingProjectors` whenever two groups' outputs overlap without being equal, for example a left partition split on its output index `x` and a right partition split on its output index `y` |
| No element-wise product | `tensor4all_treetn::hadamard` exists, but there is no partitioned or adaptive version and no projector rule for paired indices of distinct identity |
| No merging | Overpatched outputs keep all their patches |
| Quadratic bookkeeping | Pair enumeration is `N_A * N_B`; `validate_candidate_set` checks disjointness pairwise on every construction |
| Repeated norms | `SubDomainTreeTN::norm_squared` clones and canonicalizes on every call; one operation can ask for the same patch's norm several times |

`PartitionedTreeTN::contract` (non-adaptive) shares the pair loop and the
overlap rejection.

## Reference implementations

- `PartitionedMPSs.jl` (`src/adaptivemul.jl`, `src/contract.jl` at
  `1072bf46`): `adaptivecontract` builds lazy contraction tasks per output
  projector without contracting, fuses overlapping output projectors with
  `add_result_patch!` (`existing | proj`, the coordinates common to both),
  then contracts each task and replaces a saturated one by its projected
  children along the patch order (`patch_contract!`). Same-patch results are
  summed with `_add_patching`. It never merges patches.
- `TCIAlgorithms.jl` (`src/adaptivematmul.jl` at `e501032`, the version used by
  the paper): `adaptivematmul` follows the same lazy, project-on-saturation
  loop, then `_mergesmallpatches` merges bottom-up over the patch tree: a
  node's merged children and its own patches are summed with truncation to the
  cap, and the sum is kept only if its largest bond stays strictly below the
  cap.
- The paper (Grosso et al., arXiv:2602.22372, section on patched MPO-MPO
  contraction) states the compatibility rule (internal indices projected by
  both factors must agree; external projections are inherited), the
  early-stop observation, the optional neighbor merge, and the cost model of
  the best, worst, and general layouts.

Both Julia packages loop over all `N_A * N_B` pairs.

## Design

### 1. Projector rule for a pairwise operation

One rule covers contraction and element-wise products. A pairwise operation
is described by its index pairs between the left and the right operand:

- **contracted pairs**: summed over, neither index survives. For
  `contract_adaptive` these are the site indices shared by both operands
  (same identity), as today;
- **diagonal pairs** `(a, b)`: only equal coordinates contribute, and the left
  index `a` survives (the semantics of `hadamard` and `partial_contract`'s
  `diagonal_pairs`);
- every other site index belongs to one operand and survives unchanged.

For a left patch with projector `P_L` and a right patch with `P_R`:

- **Compatibility.** For every pair `(a, b)` of either kind, if `P_L` fixes
  `a = v` and `P_R` fixes `b = w`, then `v == w`; otherwise the product is
  zero and the pair is skipped before any tensor work. An index fixed by only
  one side imposes nothing.
- **Output projector.** An unpaired index keeps its owner's coordinate. For a
  diagonal pair `(a, b)` the output index `a` is fixed to `v` if `P_L` fixes
  `a`, else to `w` if `P_R` fixes `b`, else not at all. Contracted indices do
  not appear.
- **Refinement.** Splitting an output region on output index `o` at
  coordinate `x` projects each operand on its own index for `o`: the owner of
  an unpaired index, and both `a` (left) and `b` (right) for a diagonal pair.
  An operand without that index is unchanged, as in `project_if_present`.

For shared identical indices this is today's rule
(`is_compatible_with`, `intersection`, `filter_indices`); the new part is the
translation of the right index `b` to the surviving `a`. The rule is computed
from projectors only, so every pair's output projector is known before any
contraction.

Contraction pairs of distinct identity (`partial_contract`'s
`contract_pairs`) follow the same rule but get no public entry point in M6;
see [Non-goals](#non-goals).

### 2. Early abort in the probe

Adopt [treetn-contraction-outcome.md](./treetn-contraction-outcome.md) as
specified in its "Adoption in `partitionedtreetn`" section, with two
amendments:

- **Element-wise products.** `partial_contract` contracts through
  `contraction::contract` after attaching the diagonal copy tensors
  (`partial_contraction.rs`), so an outcome variant is a thin wrapper. This
  record adds `partial_contract_with_outcome` with the same `abort_at_rank`
  semantics, which reverses the outcome record's non-goal for
  `partial_contract` and `hadamard` (that non-goal deferred exactly this
  case to M6). The SRC entry points stay out of scope.
- **Grouping before contraction.** Section 1 makes the output projector of a
  pair computable from projectors, so groups (section 3) are formed without
  contracting, and a group's contributions are contracted lazily, in
  canonical pair order, with `abort_at_rank = cap` whenever the group can
  split. This is the lazy-task structure of both Julia packages.

Saturation keeps its current meaning: final bond dimension `>= cap`. The
partition layout and values therefore do not change for inputs whose output
projectors do not overlap; only discarded work is cut short. A golden-data
test frozen before the change checks this.

### 3. Overlapping output projectors

Today two groups whose output projectors overlap make the result invalid. The
proposal follows `PartitionedMPSs.jl`: overlapping output projectors are fused
into their common restriction (`Projector::common_restriction`, the
coordinates both fix with equal values) and their pairs form one group.
Fusing repeats until the group projectors are pairwise disjoint; the result is
independent of the visiting order, which a permutation test checks. The
contributions of a fused group are still masked to their own regions, so
their exact sum is the correct product on the fused region; if it saturates,
the project-first recursion refines it along the patch order like any other
group.

This replaces a recorded maintainer decision of the partition migration
("contraction output regions are not refined; overlapping outputs remain a
rejected, documented limitation",
[partitioned-treetn.md](./partitioned-treetn.md)), so it is
[open question 1](#open-questions-for-the-user). The alternative is to keep
the rejection, which leaves `contract_adaptive` unusable for operands
patched on different output indices, the general layout of the exit
criterion.

The non-adaptive `PartitionedTreeTN::contract` has no cap and cannot refine;
under the proposal it fuses the same way and returns the fused groups' sums
truncated with its options.

### 4. Patched element-wise products

```rust
pub fn hadamard_adaptive<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    index_pairs: &[(DynIndex, DynIndex)],
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
```

- Semantics of `tensor4all_treetn::hadamard`: the left index of each pair
  survives; every unpaired index passes through.
- Both partitions share one named topology (as for contraction), and each
  pair's two indices sit on the same node with equal dimensions; violations
  are typed errors before any work. The output's site set is the left set
  plus the right operand's unpaired indices.
- `patch_order` and split candidates refer to output indices; section 1
  translates them to the operands.
- The probe and the recursion are those of `contract_adaptive`; both public
  functions call one internal pairwise driver parameterized by the pair
  description of section 1, so the grouping, early abort, fusion, recursion,
  and final `truncate_adaptive` exist once.

A non-adaptive `PartitionedTreeTN::hadamard` is added alongside for parity
with `contract`.

### 5. Sibling merging

A post-processing function on any partition:

```rust
pub fn merge_patches<V>(
    partition: &PartitionedTreeTN<V>,
    center: &V,
    options: &MergeOptions,
) -> Result<(PartitionedTreeTN<V>, MergeReport)>
```

`MergeOptions` holds the bond cap, the merge tolerance, the merge order (a
full-index order like `PatchingOptions::patch_order`; empty means the
deterministic full-index order), and an optional set of projectors that must
not be merged. `MergeReport` lists each accepted merge: the new projector,
the member projectors, the rank, and the measured residual.

Algorithm, bottom-up as in `_mergesmallpatches`:

1. **Candidate region.** For a patch with projector `P`, its parent region `Q`
   drops the coordinate of `P` that comes last in the merge order. Patches are
   visited deepest first (most fixed coordinates), in canonical order.
2. **Members.** All patches whose regions lie inside `Q`
   (`is_subset_of`). The merge is not attempted if any other patch overlaps
   `Q` without lying inside it, so the result stays a disjoint partition, or
   if any member is in the excluded set.
3. **Probe.** The exact direct sum `S_Q` of the members (`SubDomainTreeTN::add`)
   is truncated to `T_Q` with the cap. The merge is kept only if
   - the largest bond of `T_Q` is strictly below the cap (equality is unsafe,
     as in the reference), and
   - the exact residual `t_Q = ||S_Q - T_Q||`, measured on the difference
     network (`TreeTN::axpby`, then `norm`, as in
     `reconstruction/engine.rs`), is within the region's share of the merge
     budget: `t_Q^2 <= delta_m^2 * |Q| / |X|`.
4. Accepted merges replace their members, and the pass repeats until no merge
   is accepted.

Bottom-up order keeps each probe small: a merge usually combines the children
of one split (`dim` of the split index, typically two).

**Error accounting.** Members are disjoint and each residual is supported in
its own region, so the residuals of all merges add in squares:
`||F_merged - F|| <= ||F_partition - F|| + delta_m`, where the first term is
whatever the partition's own error is (measured by M3 for interpolation
outputs, best effort for algebra outputs). The merge does not re-measure
against `f`; it reports `sqrt(sum t_Q^2)`. `delta_m` comes from an
`ErrorTolerance` (`rtol`, `atol`) resolved against the partition's norm,
computed once per call. How `delta_m` relates to an interpolation run's
`delta` is [open question 3](#open-questions-for-the-user).

**Patch-size bounds.** A merged patch is strictly below the cap, so the
capped-patch size bound of M5 does not apply to it. Whether a
`ToleranceNotMet` interpolation patch may be merged is undecided in M5
([patch-size bounds item 18](./tree-pqtci-patch-size-bounds.md)); the
excluded set lets the caller keep such patches as they are, and the proposal
is to exclude them by default when merging an interpolation result
([open question 4](#open-questions-for-the-user)).

**Composition.** Merging is a separate call, not an option of
`contract_adaptive` or `patched_interpolate`, so each producer keeps one job
and the caller decides when to pay for it. The guide shows the composition.
`TCIAlgorithms.jl` merges inside `adaptivematmul`; folding the merge into the
producers is the alternative ([open question 2](#open-questions-for-the-user)).

### 6. Bookkeeping overheads

- **Projector index.** A crate-private index of projectors, a trie over the
  canonical index order with a branch for "not fixed", answers "which stored
  projectors overlap this one" and "which are compatible on these indices"
  without scanning all entries. Uses: compatible-pair enumeration
  (section 1, indexed on the paired indices), group fusion (section 3),
  disjointness validation in `PartitionedTreeTN` construction and `append`,
  and merge membership (section 5). The worst case stays `O(N)` per query
  (every patch fixes nothing that the query fixes); the gain is in the
  typical case where patches fix the query's indices. It is adopted only if a
  measurement on M9 workloads shows a gain, per the performance rules; the
  quadratic loops stay as the test oracle.
- **Norm reuse.** Norms are computed once per operation and carried in the
  operation's working records (as `PatchStats` already does in
  `truncate_adaptive`), not cached inside `SubDomainTreeTN`: a hidden cache
  would need invalidation on `truncate(&mut self)` and would change `Clone`
  and `PartialEq` costs of a public type.

### 7. Determinism and the path to M7

Pairs, groups, fused groups, and merge candidates are visited in canonical
projector order, and every sum is formed in that order, so results do not
depend on `HashMap` iteration. After fusion, groups are independent and their
recursions share no state; M7 can schedule them as tasks without changing
results. The cached-evaluator determinism prerequisite of M7 does not affect
M6, which evaluates no networks pointwise.

## Non-goals

- The global error budget for patched algebra (M3b). The local
  discarded-weight `cutoff` stays best effort with no whole-network bound;
  merging adds a measured residual but does not change that contract for
  contraction outputs.
- Split decisions that use the volume-proportional budget. The cap check in
  the recursion uses the contraction options' truncation; the budget is
  applied afterwards by `truncate_adaptive`, as today. Using it earlier needs
  the output norm before the output exists, which belongs to M3b.
- Public entry points for contracted pairs of distinct identity, for
  contraction between different named topologies, and for mixed dtypes.
- Changes to `ExactParameterGain`, which needs the full group sum and gains
  only the skipped pairs.
- Parallel execution (M7) and the compact patch representation (M4).

## Tests

- Golden data: `contract_adaptive` outputs (layout and dense values) on the
  existing test cases, frozen before any refactor, are unchanged after steps
  1, 2, and 6.
- Projector rule: compatibility and output projectors for shared, diagonal,
  and unpaired indices, with the right index fixed and the left not, both
  fixed equal, and both fixed different.
- Early abort: on a branched tree a saturated group stops contracting after
  its first saturated contribution (test-only counter), with unchanged layout
  and values; the edgeless cap-one case keeps today's predicate.
- Fusion: two partitions split on different output indices contract without
  `OverlappingProjectors`; the result matches a dense reference; fusion is
  invariant under input permutation; a three-way overlap chain fuses to one
  group.
- Element-wise products: `hadamard_adaptive` against the dense element-wise
  product, for partitions patched on paired indices (equal and different
  coordinates), on unpaired indices, and mixed.
- Merging: a deliberately overpatched partition (children of a low-rank
  parent) merges back to one patch with residual near zero; a merge whose sum
  reaches the cap is refused; a merge whose residual exceeds its share is
  refused; a region partly covered by an outside patch is not merged; excluded
  projectors are never merged; dense values change by at most the reported
  residual.
- Layouts, on a branched tree (one node of degree three, from the gw-rs
  topologies): patches only on contracted indices (best), only on external
  indices (worst), and on both (general), each checked against a dense
  reference, with the compatible-pair count asserted (`N`, `N^2`, and in
  between).
- Typed errors for every validation branch of the new public functions.

## Benchmark

The exit benchmark reproduces the qualitative ordering best < general < worst
in wall time at matched output accuracy, with realistic bond caps (50 to 100)
and patches produced by tree pQTCI (M2), not cut from dense decompositions.
Proposed workloads ([open question 5](#open-questions-for-the-user)):

- element-wise product of two pQTCI partitions of the Matsubara `G0` on the
  gw-rs `tt_nblock` and `cttn` topologies;
- a QTT matrix product with operands interpolated under patch orders that
  place the patched bits on contracted indices, on external indices, or on
  both.

The merge is benchmarked on the same outputs: patch count, storage, and the
reported residual before and after.

## Implementation plan

Each step is one PR on `feat/tree-patching-m6-m10`, after review of this
record:

1. Golden data for `contract_adaptive`; norm reuse; the internal pairwise
   driver with the section 1 rule (no behavior change).
2. `contract_with_outcome` and `partial_contract_with_outcome` in
   `tensor4all-treetn` (the outcome record plus the amendment above), and
   their adoption in the driver.
3. Output-projector fusion in `contract_adaptive` and
   `PartitionedTreeTN::contract`.
4. `hadamard_adaptive` and `PartitionedTreeTN::hadamard`.
5. `merge_patches`, `MergeOptions`, `MergeReport`.
6. Layout tests and the benchmark; the projector index only if the benchmark
   shows a gain.

Public-surface updates (rustdoc examples, the partitioned TreeTN guide,
`llms.txt`, `skills/use-tensor4all-rs/`) go with the step that adds the API.

## Provenance

The fusion rule and the lazy project-on-saturation loop follow
`PartitionedMPSs.jl`, which the crate already credits as inspiration; the
merge follows `_mergesmallpatches` of `TCIAlgorithms.jl`, whose MIT license
and derivation notice the crate already carries for the interpolation driver.
If the implementation is written while reading those sources, it is recorded
as derived in the affected files and in
`docs/PROVENANCE_AND_CITATION_POLICY.md`; how to credit it is confirmed with
the user in the PR that adds it.

## Open questions for the user

1. **Overlapping outputs.** Replace the recorded rejection of overlapping
   contraction outputs with fusion into the common restriction (section 3,
   as in `PartitionedMPSs.jl`)? Recommended: yes, since the general layout of
   the exit criterion produces such outputs.
2. **Merge placement.** A separate `merge_patches` call (recommended), or a
   merge step inside `contract_adaptive` (as in `TCIAlgorithms.jl`) and
   optionally inside `patched_interpolate`?
3. **Merge budget.** The merge needs its own tolerance `delta_m`, and for an
   interpolation result the final error bound becomes the run's measured error
   plus `delta_m`. Should `delta_m` be an explicit, required option
   (recommended), or a default fraction of the run's `delta`?
4. **`ToleranceNotMet` patches.** Exclude them from merging by default
   (recommended), or allow merging them?
5. **Benchmark workloads.** Are the `G0` element-wise product and the QTT
   matrix product the right workloads, or is there a downstream contraction
   (for example a bubble in gw-rs) that should be the benchmark instead?
6. **Outcome record.** The [contraction outcome proposal](./treetn-contraction-outcome.md)
   still has its naming question open (`abort_at_rank` versus
   `rank_budget` or `saturation_threshold`) and has not been approved. Approve
   it with the amendment in section 2?
