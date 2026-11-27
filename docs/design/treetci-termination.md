# TreeTCI stopping diagnostics

`optimize_default`, `optimize_with_proposer`, and
`optimize_with_proposer_with_rng` return `TreeTciOptimizationResult`.
`crossinterpolate2` and `crossinterpolate2_with_rng` return `TreeTciRunResult`,
which also owns the materialized `TreeTN`. Both results expose `ranks`,
`errors`, `nglobal_pivots`, `termination` and `local_threshold_factor`; callers use fields
rather than tuple destructuring.

## Stopping contract

The reasons describe the existing optimization loop, in its decision order:

| Reason | Condition | Consequence |
| --- | --- | --- |
| `MaxBondDimension` | The maximum rank is at the configured cap throughout `ncheck_history` trailing iterations. | Checked before global search and convergence, even if the recorded error is small. It does not mean all edges are saturated. |
| `Converged` | The trailing `ncheck_history` errors are strictly below `tolerance`, the final maximum rank satisfies the configured `rank_slack` condition, and no global pivots were accepted in that window. | Describes the recorded-error criterion of the stop rule, not an exhaustive error bound. |
| `DegenerateCandidateSet` | The `Converged` condition holds on the recorded errors, but the most recent update of at least one edge could not measure its bond error. | The run stops where it would have reported `Converged`. The approximation may be inaccurate. |
| `LocalThresholdFloor` | `Sampled` only, trees with E >= 2 edges only: the local threshold factor was tightened to its floor `1 / E` and stayed there in all `ncheck_history` trailing sweeps, and all their recorded errors are at or above `tolerance`. | Never reported under `PerBond`, nor with E <= 1, where the floor would be 1 and the threshold cannot descend. Checked after `MaxBondDimension` and before the convergence test, and before that iteration's pivots are injected, so the returned state is swept. The approximation may be inaccurate. |
| `MaxIterations` | The requested iterations completed without any earlier stopping condition. | A small final error alone does not establish convergence. |

### Configurable confirmation window and rank slack

`TreeTciOptions::ncheck_history` is a positive window length, default 3,
shared by convergence, cap saturation and the sampled threshold-floor stop.
A window longer than `max_iter` is valid for raw optimizers and ends in
`MaxIterations`; `TreeTciInterpolator` rejects that configuration because its
adapter contract requires a possible convergence window.

`rank_slack` is finite, nonnegative and defaults to 0. The rank test is
`last <= min(window) + rank_slack * max(window)`. At zero the comparison stays
in integers and is exactly the old `last == min(window)`, including a shrinking
history such as `220, 218, 217`. A value of 0.03 permits small rank fluctuations
near the tolerance without changing the recorded-error or global-pivot checks.
A larger slack can stop while a new mode is still growing; sampled errors do
not certify full-network accuracy, so use an independent error check before
adopting a nonzero slack ([#1264](https://github.com/tensor4all/tensor4all-rs/issues/1264)).

Both result types expose one `nglobal_pivots` entry per completed iteration.
It counts pivots found by that search, including discoveries discarded at a
threshold-floor stop; disabled/skipped searches and a zero search limit record
zero. A zero therefore does not imply an independent residual check occurred.

### Reproducing the rank-slack acceptance families

The `rank_slack` example runs the three-softened-Coulomb-bump target from
#1264. `chain` defaults to 8 bits per axis (24 sites, tolerance 1e-6), and
`comb` to 20 bits per axis (60 sites, tolerance 1e-8). Each requested seed
runs slack 0 and 0.03 against the same 20,000 independent sample points and
prints histories, termination, evaluation counts and relative L2/max errors.
It reports no timing or speedup. An optional final argument reduces the bits
per axis for a smoke check.

```bash
ai-run build --mem 6G -- cargo build --locked --release -p tensor4all-treetci --example rank_slack
ai-run build --mem 6G -- env RAYON_NUM_THREADS=1 ./target/release/examples/rank_slack chain 1 20 per-bond
ai-run build --mem 6G -- env RAYON_NUM_THREADS=1 ./target/release/examples/rank_slack comb 1 1 per-bond
```

Use the effective Cargo target directory if configured elsewhere. Pass
`sampled` instead of `per-bond` to exercise the original issue's former stop
semantics; the current default changed in #1405, so historical iteration
counts are not promises for either current path. The example intentionally
keeps expensive acceptance runs outside the ordinary test suite. A small dense
adversarial regression separately proves that a large slack can stop with a
missing diagonal mode even though the sampled errors are zero.

### Stop rules

`TreeTciOptions::stop_rule` selects what is recorded and what the global
search accepts ([#1405](https://github.com/tensor4all/tensor4all-rs/issues/1405)).
Global search uses the absolute tolerance `abs_tol = tolerance *
max_sample_value` in normalized mode and `tolerance` otherwise, and the margin
`threshold = abs_tol * tol_margin_global_search`.

| | `PerBond` (default) | `Sampled` |
| --- | --- | --- |
| Recorded error | Largest bond pivot error | Largest of the bond pivot errors and the residuals the global search measured in that iteration ([#1352](https://github.com/tensor4all/tensor4all-rs/issues/1352)) |
| Ordinary walks accepted | Above `threshold` | Above `threshold` first; when none cross it, above `abs_tol` |
| Fallback stages accepted (probes, restarts, stage-3 enumeration) | Above `threshold` | Above `abs_tol` |
| Local truncation threshold | `abs_tol` in every sweep (factor exactly 1) | `abs_tol` times a factor that halves after each iteration whose recorded error fails `tolerance`, down to `1 / E` ([#1371](https://github.com/tensor4all/tensor4all-rs/issues/1371), see below) |
| `LocalThresholdFloor` | Never reported | Reported when `ncheck_history` sweeps at `1 / E` all fail `tolerance`, for E >= 2 only |

`PerBond` is the rule of TensorCrossInterpolation.jl, TreeTCI.jl and xfac:
`tolerance` is a per-bond pivot threshold. `Sampled` adds a pointwise check on
the sampled points. With one shared threshold for truncation, search
acceptance and the stop, it stopped late on smooth targets, because the
pointwise error behaves like a sum of bond residuals that are each below that
threshold ([#1371](https://github.com/tensor4all/tensor4all-rs/issues/1371));
the adaptive local threshold below is `Sampled`'s answer to that. Neither
rule bounds the pointwise error. The rule enters the loop at two places: where
the search residual is or is not folded into the recorded error, and where the
local threshold factor is created (adaptive under `Sampled`, fixed at 1 under
`PerBond`). It enters the search at one binding that computes the acceptance floor of the ordinary walks
and the one threshold of every fallback stage; the walks, the enumeration and
the preference for discoveries above `threshold` are the same on both rules. The
public `find_global_pivots*` functions keep the `Sampled` acceptance.

With a positive sampled scale and normalization enabled, the recorded error is
divided by `max_sample_value`; otherwise it is used unnormalized. The observer
receives this same value after search, along with evaluation accounting
including the search. New pivots are processed in the next sweep.

If ordinary starts yield no pivot, edges with a dimension-one endpoint supply
retained tail pivot pairs and restarts that preserve one observed subtree
while randomizing its complement, on both rules. Each extra family is bounded
by `nsearch` walks. A walk ends after the first sweep whose error exceeds ten
times its threshold; for these walks that threshold is `abs_tol` under
`Sampled` and `abs_tol * max(1, tol_margin_global_search / 10)` under
`PerBond`, which is `abs_tol` bit for bit up to a margin of 10 and keeps a
larger margin reachable. If those families find
nothing either, a third stage enumerates around the same retained tail
pivots: every configuration of each maximal edge side with at most 32 local
configurations, the other sites held at the pivot, without random draws
([#1433](https://github.com/tensor4all/tensor4all-rs/issues/1433)). The points
are evaluated in batches of at most 2^20 index entries (points times sites;
a batch always holds at least one point), which bounds the index buffer on
large trees and changes neither the residuals nor the selected pivots. A walk
needs a nonzero residual one coordinate move from its start; the enumeration
reaches a defect several moves away as long as it lies in such a side. On
binary quantics chains that side is the aligned block of width at most 32
around the pivot's value of one variable; wider blocks are not searched. These
checks can escape sparse support that uniform random starts miss; they do not
exhaust the domain or turn convergence into an error bound. Under `PerBond` a
walk error between `abs_tol` and `threshold` qualifies nothing, so the
fallback stages run in iterations where `Sampled` would have kept that error
instead.

The third stage runs in a search that finds no pivot only when all of the
following hold:

- `nsearch > 0`. `nsearch` bounds the retained tail pivots used as anchors, so
  `nsearch = 0` leaves no anchor and disables the stage, even when
  `global_search_starts` are supplied and walked.
- `max_nglobal_pivot > 0`. With zero the search returns before evaluating
  anything, so no stage runs.
- The tree has a vertex of local dimension 1. The anchors are retained tail
  pivots at edges with such an endpoint; without one there is no anchor.
- Some edge side has a product of local dimensions between 2 and 32.

`Converged` needs `ncheck_history` trailing iterations without an accepted global pivot.
So when those iterations all searched with the stage enabled, none of
their searches saw a residual above the fallback threshold of the rule
(`threshold` under `PerBond`, `abs_tol` under `Sampled`) inside an enumerated
side around a retained tail pivot. This is a bounded guarantee: it says
nothing about blocks wider than 32, it does not hold in the configurations
above, and an iteration whose search was skipped (see below) contributes no
search.

Global search is skipped after the final requested iteration and at a
bond-saturation stop. An injected pivot must be processed by a later sweep:
immediate materialization of an unswept pivot set can have inconsistent ranks
across an edge ([#692](https://github.com/tensor4all/tensor4all-rs/issues/692)).
Disabled or skipped searches contribute zero to the convergence window. Thus
`Converged` can be reported on the final iteration without searching again;
it does not assert that an independent final residual search took place.
The same applies when search is enabled but `nsearch` or `max_nglobal_pivot`
is zero. If the convergence window is not yet available, the result is
`MaxIterations`, including a one-iteration call on an exactly representable
function.

## Adaptive local threshold on the sampled stop path

This section applies to `StopRule::Sampled` only. Under `PerBond` (the
default) the factor below stays exactly 1, so every sweep truncates at the
absolute tolerance, `local_threshold_factor` is reported as `1.0`, and
`LocalThresholdFloor` is never reported: that rule's trajectory is the one it
had without the adaptation. The reason is that `PerBond` records the bond
pivot error alone. With a `max_bond_dim` cap that error can stay above
`tolerance`, and keying the factor on it would tighten capped runs whose
error the local updates cannot reduce.

The local rrLU truncation of each edge update, the global-search acceptance
and the stop all start from the same absolute tolerance,
`tolerance * max_sample_value` (or `tolerance`). A point whose error is
spread over several bonds can then sit above `tolerance` while each of its
per-bond Schur residuals is below it. The local updates drop such a point on
every sweep, the search finds it again, and the run reaches a fixed point that
stops only when `ncheck_history` consecutive random searches miss every such point
([#1371](https://github.com/tensor4all/tensor4all-rs/issues/1371)).

Only the local truncation threshold adapts:

- Each optimization call starts with factor 1, so the local threshold is the
  absolute tolerance.
- After each iteration whose recorded error fails the stop comparison (it is
  not strictly below `tolerance`), the factor is halved. It never grows again
  and never drops below `1 / E`, where E is the number of tree edges. The
  inference behind the floor is that the pointwise error behaves like a sum of
  up to E bond residuals.
- The search acceptance and the stop tolerance keep the untightened value.
  There is no extra walk, recovery pass, retained-pivot validation or sample
  history; the next scheduled sweep uses the new factor.
- A run whose recorded error stays below `tolerance` keeps factor 1 and takes
  the same path as without adaptation. With `tolerance == 0` the factor is not
  used.
- If the `ncheck_history` trailing sweeps all ran at `1 / E` and all their recorded
  errors fail the tolerance, the run ends with `LocalThresholdFloor` instead
  of continuing to `MaxIterations`. That iteration's search pivots are not
  injected. The recorded error history is in `errors`.
- The floor stop is armed only for E >= 2, where the floor `1 / E` is below 1
  and reaching it takes at least one halving. With one edge (a two-node tree)
  or none (a single node) the floor would be 1: the threshold cannot descend,
  the factor stays 1, and failing sweeps are not a reason to stop. Such
  a run ends with `MaxIterations` or `Converged` exactly as without the
  adaptation, and keeps injecting its search pivots until the final
  iteration.

`local_threshold_factor` on both result types reports the factor applied by
the last sweep. `1.0` always means the untightened threshold: under `PerBond`,
under `Sampled` before any failing iteration, and with E <= 1. For E >= 2 it
equals `1.0 / E as f64` exactly when that sweep ran at the floor. Under `Sampled` the recorded error
includes the global-search residual, which is what can fail `tolerance` while
every bond residual is below the local threshold; a `Sampled` run whose bond
residuals and search residuals both stay below `tolerance` leaves the factor
at 1. The skipped search on the final iteration is unchanged.
`TreeTciInterpolator` always runs `Sampled`, so the adaptation applies to it.

## Unmeasured bond errors

An edge update is marked unmeasured when a candidate side has only one entry,
or the selected pivots exhaust the candidates at an endpoint of local dimension
1, while both full subtrees admit more configurations than the stored rank.
The dimension-one endpoint has no local variation to probe beyond the
neighboring pivots. The LU then records a zero residual because no candidate
row or column remains, even if unsampled configurations would increase the rank. Such an error does not count toward
`Converged`; the run reports `DegenerateCandidateSet` instead. If the rank
reaches the full configuration count of either subtree, candidate exhaustion
is legitimate: the bond cannot grow further. A rank below both candidate counts
also has a residual to measure. At endpoints of local dimension greater than
one, exhausting the local expansion alone does not invalidate the ordinary
stopping criterion; this diagnostic is not a global error certificate.

The rank-1 case occurs around a vertex of local dimension 1 whose adjacent
bonds have rank 1: the junction-side candidates are the product of the other
bonds' single pivots, so none of those bonds can grow
([#1153](https://github.com/tensor4all/tensor4all-rs/issues/1153)). The same
problem occurs at larger ranks on chains containing a dimension-1 site. Its
neighbor-product candidates can stop at rank 14 even though the full
bipartition admits rank 16; neither a zero local error nor a global search
that finds no new pivots establishes convergence
([#1352](https://github.com/tensor4all/tensor4all-rs/issues/1352)). The diagnostic
qualifies the recorded error; it does not enrich candidates or guarantee that
the returned approximation meets the requested tolerance.

The diagnostic fires only when the candidates are fully used. A unit vertex can
also lock its bonds while a few spare candidates remain. In the three-branch
table of #1352, the centre has local dimension 1, and the x and z bonds into it
capped each other at rank 11 (or 10) below the true rank 12. The bond into the
third branch had rank 1, so each side offered only the other bond's pivots.
The #1159 candidates kept the count just above the rank. Every retained pivot
pair had residual near 1e-16, and the run stopped `Converged` with a dense error
of 6.3e-10 at tolerance 1e-10.

`DefaultProposer` therefore makes a unit vertex transparent while it is locked.
On the unit side of an edge, let r be the edge's current rank. If the product of
the other bonds' pivots holds fewer than 2r candidates, the side also offers a
second set: the other bond with the most pivots has those pivots replaced by its
neighbour's own side candidates, expanded over the neighbour's local index. This
is what a two-site update across the unit vertex would see.

- The threshold 2r is the slack a binary site would add. It is not tuned.
- The largest-rank bond is the one entangled with the edge. A nearly separable
  bond has small rank and cannot carry the missing direction.
- Above the threshold, the plain product is used. High-rank junctions, such as
  the #1153 comb, therefore pay nothing extra after their first sweeps.

The same lock can occur at a junction with local dimension greater than 1
when its local site is constant on the observed support ([#1436](https://github.com/tensor4all/tensor4all-rs/issues/1436)).
`DefaultProposer` also applies the expansion at a branching endpoint (degree
at least 3) if at least two retained pivots on that edge side agree at the
endpoint's local coordinate. It compares the product of incoming pivot counts
with 2r before expanding the local index; nominal local dimension is not
additional rank when that index behaves like a delta. The existing slack
threshold and largest-rank-neighbor choice stay the same.

Agreement is only a reason to explore farther, not proof that the local site
is constant. Both the ordinary and expanded candidates retain **every** local
value, including values absent from the current pivots. A single retained
pivot does not establish agreement. Non-unit chain vertices keep their
original candidates: the heuristic targets junction rank locking, not generic
support detection or a global error guarantee.

This restriction preserves the #1153 comb and #1433 box candidate paths:
their branching junction has dimension 1 (the existing rule), and all other
vertices have degree at most 2 (the unchanged ordinary rule). It avoids the
extra chain expansions of the broader predicate measured in the #1436 issue
while recovering the frozen rank-12 delta-junction table for all 11 real seeds
under both stop rules.

To let those bonds grow, each optimization call first enriches the pivots
around vertices of local dimension 1, following xfac's
`TensorTreeCI::addPivotsAllBonds(enrich_pivot1())`: it collects the vertices of
local dimension greater than 1 adjacent to a vertex of local dimension 1 and,
for every unordered pair of them, adds the base pivot with those two sites set
to every pair of local values. The base pivot is the first pivot pair of the
first edge adjacent to a vertex of local dimension 1. The enrichment runs only
when some bond around such a vertex has a side with at most one pivot.

A function that is exactly rank 1 around the vertex collapses back to rank 1
after the first update of each bond. If the enriched pivots lived only in the
pivot sets, the junction side would again be a single candidate and the run
would end in `DegenerateCandidateSet` although rank 1 is the right answer
([#1159](https://github.com/tensor4all/tensor4all-rs/issues/1159)). xfac
avoids this because its candidate set is
`set_union(I0[{from,to}], kronecker(from,to))`, where the history `I0` holds
the enriched pivots. tensor4all-rs keeps no pivot history; instead the first
call that builds the enrichment pivots records their projections onto the
junction side (the side holding the vertex of local dimension 1) of each
adjacent bond, whether or not the pivots are added, and `DefaultProposer`
(and through it `TruncatedDefaultProposer`) appends them to that side's
candidates on every update. The junction side then keeps at least as many
candidates as there are distinct projections, so a rank-1 bond error is
measured. Later calls record nothing more, so the candidate count grows by a
fixed amount. A custom proposer that does not offer them can still reach
`DegenerateCandidateSet`.

## Continued optimization

A `TreeTCI2` can be optimized repeatedly with a larger `max_bond_dim` or with
its cap removed, for the same function, topology and local dimensions. Current
pivot sets and the maximum sampled magnitude carry over; bond errors are
refreshed during subsequent passes. Each pass visits each edge once; its pivot
sets remain unchanged until its own update, since other edges use distinct
canonical subtree keys. Proposers retain those current pivots directly. No
historical full pivot maps are copied or retained; `ijset_history` is removed.
A caller-supplied `EdgeVisitor`
must return a permutation of the graph's edges on every pass; the loop rejects a
missing, duplicated, or unknown edge before that pass updates any state, so the
stale-bond-error and false-convergence cases cannot arise from an invalid order.

Each call starts fresh ranks/errors/global-pivot diagnostics, a new convergence
window and an independent iteration budget. The final-iteration global-search
skip and stop precedence apply on every call. Several calls need not reproduce
one longer call, nor a fresh uncapped run's pivot trajectory or ranks at each
iteration. Increasing a cap permits growth; it does not guarantee convergence
or certify error over the whole network.

Seeded high-level calls restart their candidate and global RNG streams.
Caller-stream entry points advance the same generator when the caller reuses
it, including for random proposers. The [random-stream contract](treetci-random-streams.md)
defines that distinction. A failed evaluator can leave partial edge updates;
continuation is not a transactional rollback mechanism.

The #833 regression applies χ=4 then uncapped to real/complex three-arm trees
with a site-free junction, with global search enabled and disabled. It checks
all dense entries against the target, checks the capped result is inaccurate,
and compares the recovered result with a fresh uncapped run after aligning
physical indices. This fixture establishes recovery in that case, not a
universal equivalence between optimization schedules.

## Relationship to TreeTCI.jl

The Rust stopping logic follows the `TreeTCI.jl` convergence criterion cited
by the original implementation: branch `local-fix-convergence`, commit
`06563dd`. Rust applies an early-convergence break rather than exhausting
`max_iter` unconditionally, compares the error on the same normalized or
absolute scale as `tolerance`, and checks bond saturation before injecting
global pivots. The early-stop regression and capped chain/star regressions in
`crates/tensor4all-treetci/src/optimize/tests.rs` preserve those decisions.
This reference describes the implementation's recorded provenance; it is
not a claim about the behavior of every current upstream branch.

This document replaces the inaccessible local-file references reported in
[#834](https://github.com/tensor4all/tensor4all-rs/issues/834) and records the
public stopping diagnostics requested by
[#835](https://github.com/tensor4all/tensor4all-rs/issues/835).
