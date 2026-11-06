# Adaptive patched contraction on trees

## Status

Proposal for milestone M6 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md). It
needs review and the answers to the [open questions](#open-questions-for-the-user)
before implementation. It builds on, and does not replace, the
[contraction outcome record](./treetn-contraction-outcome.md), which documents
the implemented post-hoc reporting API and proposes the early-abort hook
for `tensor4all-treetn`.

The post-hoc outcome API and its context/caller-RNG companions are implemented.
Early abort, adaptive adoption, partitioned Hadamard and sibling merging are
not implemented. This record was
originally written against `4d4fb6fc` and synchronized on 2026-11-06 against
`origin/main` at `097e2927`. Existing adaptive contraction and subsequently
merged refinement, selection and exact-sum work are reusable foundations;
their presence does not approve the new contracts below.

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

1. `PartitionedTreeTN::contract_pairs` validates the operand templates,
   enumerates pairs in logical patch order, and predicts each output
   projector before tensor work. Incompatible pairs and unselected outputs
   are skipped. Remaining compatible pairs are contracted in full.
2. Contributions are grouped by their exact output projector (the
   intersection of the operand projectors, filtered to surviving site
   indices).
3. Each group runs `contract_group_project_first`:
   - with `Sequential`, if one contribution already reaches the cap and a
     split index exists, the group splits without forming the sum (#788);
   - otherwise `SubDomainTreeTN::exact_sum` uses the caller's `SumEngine`,
     the result is truncated once with the contraction options, and checked
     against the cap (equality counts);
   - a saturated group chooses a split index among the unprojected output
     indices, projects the original operands of its pairs to each child
     coordinate (`project_if_present`, with a `64 * eps` compression), and
     recurses on the child groups.
4. The result is validated as a partition and passed through
   `truncate_adaptive` with the selected volume or norm allocation.

`contract_adaptive_selected` shares this implementation and filters the
top-level output groups before contraction (#1076). Non-adaptive
`PartitionedTreeTN::contract` additionally predicts overlapping outputs and
refines its operands first (#1093), subject to
`AUTOMATIC_REFINEMENT_LIMIT = 4096` (#1143/#1144). The selected and adaptive
entries do not call that automatic-refinement wrapper. `add_with_patching`
now projects original addends before splitting (#1132), and non-adaptive
`add` is exact with a separate `add_truncated` entry (#1142).

So the paper's steps "refine only the contributors of unconverged outputs"
and "recompute only those outputs" are already met: the recursion is per
group and projects only that group's operands. What is missing:

| Gap | Effect |
|---|---|
| Every selected compatible pair is contracted before grouping | Saturated contributions are paid in full, then discarded |
| No early abort inside a contraction | Each saturated probe runs to completion |
| Adaptive groups have no shared overlap-resolution policy | Adaptive/selected output assembly can still reject unequal overlapping projectors; non-adaptive `contract` already has bounded operand refinement |
| No element-wise product | `tensor4all_treetn::hadamard` exists, but there is no partitioned or adaptive version and no projector rule for paired indices of distinct identity |
| No merging | Overpatched outputs keep all their patches |
| Quadratic bookkeeping | Pair enumeration is `N_A * N_B`; `validate_candidate_set` checks disjointness pairwise on every construction |
| Repeated norms | `SubDomainTreeTN::norm_squared` clones and canonicalizes on every call; one operation can ask for the same patch's norm several times |

Non-adaptive `contract` shares the selected-compatible-pair contraction loop,
but its bounded refinement must not be removed or silently replaced by the
unapproved fusion proposal.

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

Saturation keeps its current meaning: final bond dimension `>= cap`. Layout
and value preservation on inputs already accepted by the baseline is an
acceptance condition, not a consequence proved by the threshold alone. Lazy
grouping must retain the existing content-derived contribution order and the
exact-sum-then-truncate contract. Frozen baseline tests check both the
ordinary and selected adaptive entries before any skipped-work claim.

### 3. Overlapping output projectors

Adaptive output groups can still overlap. Non-adaptive `contract` already
handles this class through bounded operand refinement. For the adaptive
path, the original proposal follows `PartitionedMPSs.jl`: overlapping output projectors are fused
into their common restriction (`Projector::common_restriction`, the
coordinates both fix with equal values) and their pairs form one group.
Fusing repeats until the group projectors are pairwise disjoint; the result is
independent of the visiting order, which a permutation test checks. The
contributions of a fused group are still masked to their own regions, so
their exact sum is the correct product on the fused region; if it saturates,
the project-first recursion refines it along the patch order like any other
group.

This is still [open question 1](#open-questions-for-the-user), now a choice
between reusing the shipped bounded refinement for adaptive groups and
introducing coarser fusion. #1088's earlier rejection/opt-in discussion was
superseded for non-adaptive operations by #1093 and #1143; it is not a reason
to restore the old rejection. Fusion changes region layout and summation
order, so it requires an explicit reviewed contract and a resource bound.
Whichever adaptive policy is chosen must preserve `contract_selected`'s
documented selection semantics and typed `RefinementTooLarge` diagnostics.
This synchronization makes no change to non-adaptive dispatch.

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
3. **Probe.** The exact direct sum `S_Q` of the members
   (`SubDomainTreeTN::exact_sum` with an existing exact `SumEngine`)
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

**Error accounting: unresolved implementation gate.** Residuals of disjoint
merges in one level add in squares, but a later parent merge overlaps the
regions merged at earlier levels. The original claim that all accepted
`t_Q` values add in squares is invalid: two nested aligned residuals of norm
`e` can accumulate to `2e`, exceeding `sqrt(2) * e`. Step 3's local test alone
therefore does not establish a global merge allowance.

The required contract is a reported bound `E_m >= ||F_after - F_before||`
with `E_m <= delta_m`. Candidate designs are a ledger that adds level bounds
linearly (using squared sums only within disjoint levels), or a final
difference-network check against the original partition. The latter must
also specify rejection/rollback and peak-memory behavior. This is
[open question 7](#open-questions-for-the-user); do not implement the old
`sqrt(sum t_Q^2)` report. A multi-level cancellation/alignment regression is
required before claiming a bound. The existing reconstruction scheduler
already uses squared accumulation within disjoint levels and linear
accumulation across levels (`reconstruction/schedule.rs`, the final
`level_bounds` reduction); reuse that accounting pattern where applicable.
The reconstruction engine measures final disjoint regions against their
original target. Neither path justifies summing nested merge errors in squares.

`delta_m` comes from an `ErrorTolerance` (`rtol`, `atol`) resolved against the
partition's norm, computed once per call. The composed error is bounded by
the input partition's actual error plus `E_m`; a sampled interpolation error
remains an estimate. How `delta_m` relates to an interpolation run's `delta`
is [open question 3](#open-questions-for-the-user).

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

Pairs, groups, fused groups, and merge candidates retain the existing logical
patch order and content-digest tie-break (#1087/#1100/#1101), rather than
introducing raw index IDs or `HashMap` iteration as an ordering rule.
Post-group recursions can become independent tasks, but contraction fan-out
and bitwise validation remain later M7 work. Interpolation lanes and MPI
already exist (#1005), so M6 must leave their shared driver unchanged. The
cached-evaluator interpolation qualification does not block serial M6,
which evaluates no networks pointwise.

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
  residual. Nested parent/child merges with aligned and cancelling residuals
  check the final bound against the original partition, not just the local
  truncation ledger.
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

Proposed slices on `feat/tree-patching-m6-m10`, after review of this record;
pushes and PR creation still require the user's approval:

1. Golden data for ordinary/selected `contract_adaptive`; reuse current
   logical ordering, `exact_sum`, and the section 1 pair metadata. Norm reuse
   is a separate performance candidate and needs evidence before adoption.
2. `contract_with_outcome` post-hoc reporting is implemented. Next come the
   tree zip-up early-abort hook, `partial_contract_with_outcome`, and adaptive
   adoption, with the outcome record plus the amendment above.
3. The reviewed adaptive overlap policy, reusing bounded refinement or
   introducing explicitly reviewed fusion; preserve shipped non-adaptive
   refinement and selected-output semantics.
4. `hadamard_adaptive` and `PartitionedTreeTN::hadamard`.
5. `merge_patches`, `MergeOptions`, `MergeReport`, after the multi-level
   error-ledger gate is settled.
6. Layout tests and the benchmark; the projector index only if the benchmark
   shows a gain.

Public-surface updates (rustdoc examples, the partitioned TreeTN guide,
`llms.txt`, `skills/use-tensor4all-rs/`) go with the step that adds the API.

## Issue dependencies and implementation gates

Review date: 2026-11-06, baseline `097e2927`. All 25 open repository issues
and the four open PRs were inventoried; relevant comments were read for scope,
decisions and ownership. An empty assignee field is not evidence that work
is unclaimed. This is a dependency assessment for serial M6, not a request to
take over the issues below.

**No currently open issue is a mandatory prerequisite to the first serial
M6 outcome slice.** The proposal itself still has design gates. Dependencies
of optional optimizations, MPI, or M3b must not become blanket prerequisites
for the outcome seam.

| Issue or work | Relationship to M6 | Disposition |
|---|---|---|
| [#1088](https://github.com/tensor4all/tensor4all-rs/issues/1088), refinement | Non-adaptive unequal-layout contraction is already implemented in #1093. Its outstanding size-bound follow-up #1095 closed through #1143/#1144. | No external blocker. Reuse the bound; decide the adaptive policy in question 1. Do not restore the obsolete rejection contract. |
| [#1125](https://github.com/tensor4all/tensor4all-rs/issues/1125#issuecomment-6616854981), [#1126](https://github.com/tensor4all/tensor4all-rs/issues/1126#issuecomment-6616855437), [#1129](https://github.com/tensor4all/tensor4all-rs/issues/1129#issuecomment-6616856683), approximate sums | The maintainer has queued these in dependency order. Existing exact `SumEngine` / `SubDomainTreeTN::exact_sum` suffices for the baseline and merge probes. | Not prerequisites. Do not add intermediate approximate sums to an exact M6 probe or duplicate the queued fitting work. |
| [#1130](https://github.com/tensor4all/tensor4all-rs/issues/1130#issuecomment-6616857097), tolerance-driven merge-refine; [#777](https://github.com/tensor4all/tensor4all-rs/issues/777#issuecomment-6259153774), reconstruction follow-ups | These concern the reconstruction scheduler, not the proposed general sibling post-processing call. They share the need to account for errors across levels. | No outcome blocker; do not merge the APIs or change exact-mode semantics. Resolve question 7 before claiming a bound for M6 merging. |
| [#549](https://github.com/tensor4all/tensor4all-rs/issues/549), dense-growth audit | New copy/diagonal helpers must stay structured and resource growth must be checked. The open GSE, quantics and TCI2 items are separate paths. | A constraint on the changed paths, not a requirement to close the whole audit. An unbounded dense bridge in a proposed M6 path would block that path. |
| [#696](https://github.com/tensor4all/tensor4all-rs/issues/696#issuecomment-6234670890), SRC state copying; [#691](https://github.com/tensor4all/tensor4all-rs/issues/691), parallel SRC | The first early-abort implementation targets tree zip-up. SRC stays post-hoc in the outcome proposal. | Not prerequisites; no SRC capacity redesign or parallel sketching is part of this slice. |
| [#859](https://github.com/tensor4all/tensor4all-rs/issues/859), execution frontends | Caller-owned engines and the interpolation lane/session route have shipped (#1009/#1005), although the umbrella remains open. | Use the current backend/context seams and preserve AD behavior. Do not wait for the entire umbrella or introduce an ambient fallback. |
| [#1023](https://github.com/tensor4all/tensor4all-rs/issues/1023#issuecomment-6556027238), typed errors | The staged migration policy and callback/source exceptions are recorded in `AGENTS.md`. | New M6 APIs return the owning crate's typed errors; no prerequisite whole-workspace migration. |
| [#1046](https://github.com/tensor4all/tensor4all-rs/issues/1046), per-call overhead | Existing call overhead matters to the exit benchmark. | No functional dependency. Norm caching and projector indexing need workload evidence before implementation. |
| [#1020](https://github.com/tensor4all/tensor4all-rs/issues/1020), [#1026](https://github.com/tensor4all/tensor4all-rs/issues/1026#issuecomment-6619123346), [#1027](https://github.com/tensor4all/tensor4all-rs/issues/1027), [#899](https://github.com/tensor4all/tensor4all-rs/issues/899) | Cache/interpolation performance can affect M2-produced benchmark inputs. #1026 is explicitly claimed by shinaoka. | Not prerequisites for serial contraction on existing partitions. Record the producer revision; do not absorb these tasks into M6. |
| [#854](https://github.com/tensor4all/tensor4all-rs/issues/854), [#627](https://github.com/tensor4all/tensor4all-rs/issues/627), [#148](https://github.com/tensor4all/tensor4all-rs/issues/148) | TreeACI, simplett cache and solver/API work are outside the first M6 slice; #627's key work already has draft #1146. | No dependency on those features or their active PRs. CG/BiCGSTAB/ALS PRs #1147–#1149 also are not prerequisites. |
| #848, #1025, #1119, #1120, #1121 | CI efficiency, releases and review-bot follow-ups. | No M6 algorithm dependency; normal repository validation still applies. |

Resolved historical prerequisites: #797 (site-free nodes) closed via #986,
#656 (Fit initialization) landed via #981, and #670 (thread-budget diagnosis)
is closed. Upstream tenferro-rs#1963 is closed for the default non-annealed
planner. Interpolation's downstream generic-path determinism qualification
remains open in the tests; serial M6 does not depend on that pointwise
evaluator. The earlier interpolation checkpoint is complete.

Implementation gates that remain inside M6:

1. Post-hoc reporting and the frozen ordinary/selected baselines are in place.
   Review the early-abort hook and method coverage before schedule changes
   (question 6).
2. Review adaptive overlap handling (question 1) against the shipped bounded
   refinement and selected-output contracts before changing layouts.
3. Settle merge placement, tolerance and excluded-patch behavior (questions
   2–4), and the multi-level error ledger (question 7), before implementing
   `merge_patches`.
4. Fix the representative benchmark workload/protocol before collecting
   timing evidence (question 5). Small dense references establish correctness,
   not the M6 performance exit.

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

1. **Overlapping outputs.** Reuse the shipped bounded operand-refinement
   machinery for adaptive groups, or adopt the coarser fusion of section 3?
   Preserve non-adaptive refinement, selection semantics and resource bounds;
   the former "replace rejection everywhere" question is obsolete.
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
   has an implemented post-hoc API: `abort_at_rank`, stored-rank semantics,
   context/RNG parity and analytical baselines. Review the deterministic
   tree zip-up abort hook with the amendment in section 2 before changing
   the numerical schedule.
7. **Multi-level merge error.** Choose a conservative level-wise ledger or
   a final residual check against the original partition, with explicit
   rejection/rollback and resource behavior. The old sum of squared local
   residuals cannot certify nested merges (section 5). This gates merging,
   not the independent contraction outcome slice.
