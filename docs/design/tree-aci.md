# Tree Alternating Cross Interpolation

`tensor4all-treeaci` implements Alternating Cross Interpolation directly on
tree tensor networks. It depends on `tensor4all-treetn` for the representation
and point evaluation, on `tensor4all-core` for dense matrix LUCI, and on the
core/backend crates for indices and numerical storage. It must not use the
train-specific ACI state or conversion bridges.

The implementation follows these invariants:

- Removing one edge of a validated tree produces two disjoint components.
  Component samples therefore glue to exactly one global physical assignment.
- Directed component samples are immutable records. A record stores its local
  physical coordinate and immutable sample IDs for incoming branches. Active
  pivot sets may change between generations without changing older records.
- Per-input frames are keyed by directed edge and immutable sample ID. A frame
  is the exact contraction of one input component with its physical assignment,
  leaving only the cut bond open.
- The conformance path materializes only a checked, edge-local, column-major
  matrix and calls the same owned dense LUCI contract as train ACI. It never
  materializes the full tensor represented by an input TreeTN.
- A deterministic minimum-retracing forward walk keeps consecutive local
  updates on adjacent edges. With an automatic root it starts at a diameter
  endpoint and has the graph-theoretically optimal open-walk length
  `2|E| - |P|`, where `P` is the selected spine. The return pass traverses
  `P` in reverse order, plus a re-commit excursion for each off-spine bond
  whose candidate set a Guard injection enlarged (below). Without such bonds
  the complete forward/return round has exactly `2|E|` directed edge updates
  and is an Euler tour of the bidirected tree.
- Traversal selection is a public, non-exhaustive strategy enum, while every
  strategy lowers into one internal `SweepPlan`. A common validator checks edge
  references, orientation, walk continuity, complete-round coverage, and the
  declared forward/return semantics before execution. Only
  `MinimumRetracingWalk` is implemented initially; no custom planner trait is
  public while frame-generation semantics are still evolving.
- Serial execution is the reference semantics. Parallel phases, if enabled
  later, use snapshot-isolated proposals and deterministic commit-time sample
  ID remapping.

For a path topology, `diameter = |E|`, so the forward and spine-only return
passes reduce to the two train sweeps without extra edge visits. In a branched
tree, branch edges outside the chosen diameter are visited in both directions
during the forward excursions, while the return visits only the diameter
spine. Thus every edge is used once in each direction per complete round;
the previous full inverse would have repeated the branch excursions. Visiting
branch edges in both directions during the forward walk remains required by
the current LUCI gauge invariant: a bond axis is ordered by its active
component samples, while a generic TreeTN canonicalization can change that
basis without updating the sample order.

That invariant is the **pivot basis**: each bond has one basis shared by both
adjacent cores, and bond axis `k` denotes the `k`-th retained candidate of
that bond. A commit writes `A[:, J] A[I, J]^-1` and `A[I, :]` on the existing
incoming bonds of its two cores and assumes every bond it reads is in this
basis; nothing records the basis, so a violation is not detected. The output is
therefore never canonicalized numerically, neither at initialization nor
between passes. A CI canonicalization of the whole output keeps the dense
tensor but re-selects its own pivots from the core values, which moves bonds
out of the pivot basis; the spine-only return pass then commits against them
and never rewrites the off-spine bonds. #1449 measured this after pass 1: a
run stopped `Converged` at the true ranks with a recorded error of 4e-16 and a
dense relative error of 1.58. The result carries the canonical region of the
last commit and no canonical form (`canonical_form()` is `None`); an initial
guess's form is cleared at initialization, because the run does not establish
it.

A Guard injection appends candidates to `candidates.ids` and leaves every bond
basis as it is. A later commit that reads an enlarged set as rows may select an
appended candidate, which its factor then drops (the prefix projection), so
that commit's bond leaves the pivot basis unless the enlarged bond was
re-committed first. A forward pass re-commits every bond child-first, so it
needs nothing more. The return pass re-commits, before each spine commit, every
off-spine bond into the commit's source node whose candidate set outgrew its
basis: a retracing excursion that re-commits that bond's enlarged child bonds
first, in neighbour order, and ends at the spine node, which stays the walk's
canonical centre. Bonds no commit of the pass reads as rows keep their basis:
an enlarged bond below a bond that did not grow, and the off-spine bonds of the
walk's last node. The next forward pass re-commits them. Before #1450 the
return pass re-committed only the spine; on G0 r = 5 (comb tree, seed 42,
tolerance 1e-3, one pass) the cleanup return pass then returned a tree with
max|M - I| = 0.76, where `M[c -> p]` is the `c`-side subtree evaluated at the
retained candidates of `c -> p`. The invariant is required on every returned
tree. A unit-test helper checks it; the library does not, because the check
evaluates every retained candidate.

### Unit vertices and locally fixed junctions (#1435)

A vertex of local dimension 1 (a node without sites, or with sites of
dimension 1 only) contributes no local coordinate to a commit's rows or
columns: its side of the local matrix is just the product of its other bonds'
candidate sets. Two bonds through such a vertex then cap each other. In the
#1435 table (three branches of 5 binary sites joined at a dimension-1 centre,
true centre ranks (12, 1, 12)), the x and z bonds stopped at (2, 1, 2) to
(10, 1, 10) on 8 of 11 seeds, each update recording a residual of ~1e-16 while
the dense error was 2.5e-8 to 0.33 (`main` `93a789ba`).

Each commit therefore makes a unit vertex transparent for one of its other
bonds, as TreeTCI's `DefaultProposer` does since #1434 (`crate::unit_vertex`):

- **When.** A side of the edge whose vertex has local dimension 1 and at
  least one other bond, or a non-unit branching vertex satisfying the local
  agreement predicate below, while the product of the other bonds' ranks is below
  `UNIT_VERTEX_EXPANSION_FACTOR = 2` times the edge's rank. The factor is the
  untuned #1434 constant; it was not tuned for tree ACI.
- **Which bond.** The largest-rank other bond `n -> a` (ties keep the first).
- **What.** The neighbour's own side samples, `local_dim(n)` times the
  candidate sets of `n`'s other incoming bonds, are appended after the current
  entries of `candidates.ids[n -> a]`, with their frames, through the growth
  path Guard injection uses (`TreeAciState::grow_candidate_sets`).
- **For how long.** One commit. The commit factors the enlarged local matrix,
  its factor is restricted to the bond-basis prefix as for Guard samples, and
  the set is cut back to its length before the expansion. A failed proposal or
  commit also removes the interned samples and their frames, and drops the
  memoized candidate frames that name them: the arena hands their ids out
  again (#1464).
- **Within the shape limits.** An expanded bond is charged against the limit of
  the side it enlarges (`max_candidate_rows` or `max_candidate_cols`, over the
  whole enlarged side) and against the local-matrix element ceiling. A bond
  that would exceed one is not expanded for that commit, which then has the
  shape it has without the expansion, so the expansion never adds a shape
  failure (#1464). The working-byte, frame-byte and sample-arena budgets are
  not checked ahead: an expanded commit over one of them fails with
  `ResourceLimit`.

The junction predicate reads the committed component samples' local coordinates;
the expansion reads the current candidate sets of the neighbouring bonds.
No retained pivot is evaluated again and no earlier matrix or history is
replayed (#1372). Without a qualifying unit vertex or locally fixed junction,
no expansion runs. The cut-back means no later step sees the expanded entries, so:

- the return pass's re-commit excursions (#1450) are planned exactly as
  without it, and a re-commit, being an ordinary commit, expands by the same
  rule. In the #1435 table it does on the bond from the off-spine branch into
  the centre;
- the stale-residual re-measurement (#1451) measures the plain local matrix of
  the current candidate sets, not the expanded one.

Unlike a Guard candidate, an expanded sample comes from the same product the
neighbouring bond's own commits read as rows. The pivot-basis deviation
max|M - I| was 0 at every pass boundary and on every returned tree in the
measured runs (the #1435 table at tolerances 1e-10, 1e-4 and 1e-3 and with
`max_bond_dim = 6`, 11 seeds each, and the #1449 three-branch fixture); within
passes it showed the same transient values (up to 1.4 after pass 1) as on
`main`. This is measured, not proven: a selected row outside the neighbour's
basis is not reproduced by the factor, which local factor construction
restricts to that bond's basis (the leading hyper-rectangle of the candidate
grid, `CandidateBasisProjection`).

The cut-back shortens the candidate set only. The samples an expansion
interned stay in the arena with their frames: the selected ones because the
commit's pivots name them, the others because they are interleaved with the
selected ones in the expanded bond's arena, so dropping only them would
renumber samples the commit's records name. A later expansion of the same sets
finds them again and reuses their records and frames. Per expanding commit
the retained samples are at most the enumerated side samples of the expanded
bond, within the candidate limit of the side, each costing one arena record
plus `bond_dim` scalars per input. Over a run they are bounded by the distinct
side samples the neighbouring candidate sets ever offer, so the retention stops
once those sets settle. Measured (#1464): on the #1435 table, in its default
runs (11 seeds) and over 40 passes (seeds 1, 4, 9, tolerance 1e-12), no
expansion interned a new sample, as every expanded sample was already in the
arena; on a 19-node tree with a degree-3 and a degree-2 unit vertex and random
cores of bond dimension up to 32 or 64 (reached as ranks), over 40 passes, 120
expanding commits left 156 or 183 unselected samples, at most 48 per commit, 44.5 KB or 79.4 KB
(9.8 % or 7.4 % of the arena and frame bytes), all within the first 5 or 3
passes and none after.

Measured on the #1435 table (tolerance 1e-10, 11 seeds): 8 seeds now stop
`Converged` at (12, 1, 12) with a dense relative error of at most 6.1e-16, for
1.22 times the oracle points of `main`. Seeds 2, 3 and 7 still returned the
zero output (mechanism 2 of #1435), which the third Guard stage below removes
(#1466). The same table with the unit
centre absorbed into a binary site (case B) and the G0 and star workloads
(213 runs) have no unit vertex and were bit-identical to that base before the
junction extension below.

#### Local agreement at non-unit branching vertices

Case B absorbs the centre into a binary site whose local coordinate is
constant on the support, so it has the same rank lock without a dimension-1
vertex. A non-unit endpoint now also qualifies when its degree is at least 3
and at least two of the edge's committed pivots agree at that endpoint's
local coordinate. This is the junction-scoped TreeTCI rule (#1436 / #1470),
with the predicate and slack accepted in #1435 unchanged from measurement.

The pivots are `state.pivots.per_edge[side / 2]`: entry 0 of each pair belongs
to directed edge `2e`, entry 1 to `2e + 1`. Agreement reads only the selected
sample records' `local_coordinate`, the combined local index for a multi-site
node. It needs no operator evaluation, tensor readout, allocation or history.
The scan is O(rank) at each branching endpoint per commit. The rank-product
slack, factor 2, largest-rank neighbour selection, shape limits, rollback and
one-commit lifetime are the same as for unit vertices.

Agreement is a reason to explore farther, not proof that a site is a delta.
Every value of the junction's local index is still crossed with the enlarged
incoming sets. Non-unit leaves and degree-2 chain sites never use this
predicate. The broader chain-site `value2` prototype was rejected.

In the [accepted measurement](https://github.com/tensor4all/tensor4all-rs/issues/1435#issuecomment-7181759972)
(base `794d9856`, tolerance 1e-10, 11 seeds), case B reaches (12, 1, 12) on
every seed, with dense relative error at most 6.1e-16 and 1.22 times the points
(259,746 versus 213,415). Case A, the 213 G0 and star runs, and the 180 G0
`min_sweeps = 1` runs were identical to the same base: the new predicate never
fired there. Case A's remaining zero-output misses are handled separately by
the third Guard stage below.

### Third Guard stage: small sides around the starts (#1466)

A Guard search has two stages: the points the previous search returned are
evaluated again, and a floating-zone walk runs from each of `nsearch` uniform
random starts. A walk moves one node at a time and only to a strictly larger
residual. From an output that is exactly zero, every residual off the target's
support is exactly zero, so a walk whose start is two or more node changes
away from the support never moves. On the #1435 table the target is `M[x, z]`
on the plane `y = 16` and 0 elsewhere; seeds 2, 3 and 7 drew no start of the
first two searches within one `y` bit of the plane, every residual of both
searches was exactly 0, and the run stopped `Converged` after 2 passes with
the zero output (dense relative error 1).

A third stage (`global_guard::side_enumeration`) runs when the first two
found nothing:

- **Trigger.** Only in a search whose revalidated retained points and walks
  all found nothing, and only on a tree with a vertex of local dimension 1, as
  #1440 restricts TreeTCI's stage. A tree without one never runs it and its
  runs are unchanged. The search is skipped as before when the Guard is off,
  when `nsearch_global_pivots` or `max_nglobal_pivots` is 0, and on a
  rank-limited pass.
- **What.** Around each of the search's own `nsearch` starts (no new random
  draw, no retained pivot read), every configuration of each maximal edge side
  whose product of local dimensions lies in 2..=N, with every other node held
  at the start. N = 32 (`SIDE_ENUMERATION_BUDGET`) is the #1440 bound,
  carried over untuned. A side is maximal when no other such side strictly
  contains it, as in TreeTCI's `maximal_enumeration_sides`. On the #1435 table
  the sides are the three 5-site branches, so enumerating the `y` branch puts
  each start's `(x, z)` on the plane.
- **Acceptance.** A point qualifies above the Guard's own threshold. At most
  `max_nglobal_pivots` points are kept, strongest first (equal residuals in
  evaluation order), and they are returned and injected like any Guard pivot.
- **Cost bound.** At most A · Σ_s c(s) ≤ nsearch · 2(n − 1) · N points per
  search, where A ≤ `nsearch` is the number of distinct starts, the sum runs
  over the maximal sides (at most two per edge of an `n`-node tree), and
  c(s) ≤ N is the number of configurations of side `s`. On the #1435 table
  (n = 16, three sides of 32) that is at most 480 generated points, of which
  up to 470 were evaluated (the rest are repeats); a search whose walks all
  miss costs 80 points there without the stage.
- **Streaming.** Points are generated in a fixed order (anchor, then side,
  then configuration with the side's first node fastest) and evaluated in
  chunks of at most ⌊2^20 / n⌋ points. A point is evaluated only from the
  first (anchor, side) pair that generates it; whether an earlier pair does
  is decided from the anchors and the sides alone, so the set of evaluated
  points is never stored (the concern of #1441 for TreeTCI). Chunking changes
  neither the points, their order nor the result.
- **Budgets.** The stage's largest chunk, with the search's retained starts,
  the kept points and the stage's own bookkeeping, is charged against
  `max_working_bytes` once, before the stage evaluates anything; a stage that
  does not fit fails with `ResourceLimit { resource: "working bytes" }`. The
  stage interns no sample and builds no frame, so the sample-arena and frame
  budgets apply only to the injection of what it returns, as for any Guard
  pivot. The evaluators' message caches stay within `message_cache_max_bytes`.
- **No history (#1372).** The stage reads no retained pivot and replays
  nothing.

**The guarantee is probabilistic.** The stage reuses the search's random
starts, so it covers only the sides around the starts drawn. On the #1435
table in the zero-output state (exact over all 2^15 starts, threshold the
absolute 1e-10 · 10), a start misses with its walk with probability
1 − 4,824 / 2^15 = 0.853, and misses with both its walk and the stage with
probability 8,496 / 2^15 = 0.259. Every walk from a start on the plane hits;
from a start off the plane the stage misses exactly when `M` is zero or at
most 1e-9 at the start's `(x, z)` (276 of the 1,024 pairs, 256 of them in the
eight zero rows), so a start is missed when that holds and its walk misses as
well. The first two searches draw 10 independent starts, so a run stays at the
zero output with probability 0.853^10 = 0.20 without the stage and
0.259^10 = 1.4e-6 with it. A miss is not detected: the run stops `Converged`.

Measured (#1466, base `794d9856`, points are `evaluated_points`):

- The #1435 table, tolerance 1e-10, seeds 1-11: 11 of 11 seeds stop
  `Converged` at (12, 1, 12) with a dense relative error of at most 6.1e-16
  (8 of 11 before), for 1.39 times the points (232,778 instead of 167,501).
  Seeds 2, 3 and 7 now take 5 or 6 passes; the other 8 seeds take 1.077 times
  their points and 52 passes instead of 58. Seed 4 now stops after 6 passes
  instead of 8, with the same exact tree.
- The 32 × 32 XOR-mask variants of that table with the plane at `y = 16`
  (11,264 runs): 0 wrong results instead of 768, at 1.166 times the points.
  With the plane at `y = 0`, where nothing was wrong before, every result is
  unchanged except the points, which grow 1.080 times: the stage runs in the
  converging searches and finds nothing.
- No dimension-1 vertex, bit-identical: the same table with the centre
  absorbed into a binary site, the G0 and star workload sweep (213 runs), and
  G0 r = 5 at tolerance 1e-3 with `min_sweeps = 1`, seeds 1-60, three layouts
  (180 runs). On such trees the stage costs one scan of the local dimensions
  per search that found nothing, and no point.
- One search where the stage runs costs up to 6.9 times the same search
  without it on the #1435 table (80 + 470 points against 80). On a tree with a
  dimension-1 vertex every `Converged` run has at least `min_sweeps` such
  searches.

Discontinuous minimum path covers and parallel paths remain mathematically
possible, but require sample-aware gauge transport or snapshot/merge semantics.
They must not be enabled by simply canonicalizing between paths: a binary-tree
regression showed that doing so can permute physical values while reporting tiny
local residuals.

Candidate products at a node scale with the product of incoming branch ranks.
The public preparation boundary therefore uses checked arithmetic and
caller-visible limits for candidate rows and columns, local matrix elements,
core and frame elements, arena retention, and working bytes. Low maximum degree
does not remove this requirement, but it keeps the rank exponent bounded in the
intended workloads.

Initialization deliberately leaves both generated outputs and explicit initial
guesses numerically uncanonicalized. Bootstrap uses only validated bond
dimensions and component coordinates; it does not read the guess's gauge. The
first pass replaces every core with interpolative factors in the pivot basis;
no canonicalization follows it. Canonicalizing a high-rank explicit guess
before that pass would be redundant and would make setup scale with a gauge
that ACI immediately discards.

For a node with one incoming component, all local physical slices are stacked
into one `(outgoing × local) × incoming` matrix. One backend multiplication then
contracts every physical coordinate and incoming sample column. This avoids a
separate small BLAS dispatch for every local coordinate in both bootstrap-frame
construction and local candidate evaluation.

Guard injection is capacity-based, not merely enabled/disabled per edge. Each
cut carries its remaining distance to the configured/algebraic rank limit, and
is removed from the active projection mask as soon as that capacity is used.
Thus one Guard scan offering several pivots cannot overshoot a cut that had
only one rank available.

Guard retains at most `max_nglobal_pivots` returned assignments across passes.
Each next search rechecks them against the current output in the starting
batch, with their targets counted once. Still-significant assignments keep
their slots; resolved assignments release them for new random discoveries.
Thus a random miss cannot erase an already established residual. The combined
retained/start/evaluation storage is checked against the working budget.
`global_pivots_found` includes retained assignments that still fail validation;
it is not the number actually injected into candidate sets.

Guard detection is independent of that injection capacity. Unless the local
residual already establishes a rank limit, an enabled search still validates
the output when every cut is saturated. Significant residuals without any
remaining headroom contribute to `RankLimited`; saturation cannot be treated
as a successful guard scan. A final cleanup pass runs only when pivots were
actually injected. Target values, reconstructed values,
residuals, and guard thresholds must be finite, including the exact single-node
shortcut. Callback and residual non-finiteness has a dedicated typed error.

Convergence requires a window of `min_sweeps` consecutive pass results. Its
first entry is the baseline: the first pass, or the last pass whose ranks grew
on any edge. The remaining `min_sweeps - 1` results must show no growth on
**any** output edge, so with `min_sweeps = 1` the rank condition cannot fail.
A rank decrease is allowed, but subsequent regrowth restarts the window. Network-maximum rank stability alone is insufficient: on the #741
low-temperature CTTN, the maximum stayed at 233 while smaller cuts were still
growing and the reconstructed output retained a large localized error.
The implementation retains one previous rank per edge and a stability counter,
requiring O(edges) storage and work per pass without extra contractions,
sampling, or per-pass allocation. Returned `max_ranks` remains a summary for
diagnostics, not the rank-stability decision input.

The local LUCI residual and enabled global-guard conditions still apply. One
private `TolerancePolicy` keeps their conversions consistent while preserving
the caller's selected units. With relative tolerance enabled, each local
matrix is divided by its largest sampled `|f|` and factored with the configured
tolerance as an absolute threshold on that normalized matrix. The sweep
compares `error / scale` with that tolerance, so the factorization and sweep
share a reference. A relative-to-largest-pivot rule would disagree whenever
Schur-complement growth lifts a pivot above the largest entry. Normalization
divides complex components by a real scalar rather than forming a complex
divisor's squared norm, avoiding overflow or underflow from squaring a
representable normalizer.
The output keeps all four supported scalar dtypes end to end.
When a rank ceiling ends rrLU, the reported local error is the remaining Schur
complement maximum, rather than the last accepted pivot. Thus a saturated cut
whose local matrix is already exact does not by itself imply `RankLimited`.
Normalization also makes the LUCI kernels' absolute `f64::EPSILON` pivot floor
a fixed relative round-off floor for this mode. With absolute tolerance selected, the
local matrix remains in raw output units and the configured threshold is
absolute in those same units. The global guard follows the same policy: in
relative mode it scales its threshold by the largest `|f|` among its random
starts **and** current edge-local matrices; in absolute mode it uses the
configured absolute threshold. A few random starts alone can underestimate a
heavy-tailed output by orders of magnitude; the guard then reports residuals
the next local update discards again, and a run whose output no longer changes
never converges (#776).
`Converged` is an algorithmic stopping condition, not a certified bound on the
full-grid maximum error: the tolerance controls edge-local residuals, and the
guard performs bounded randomized coordinate searches with its configured
acceptance margin. Independent validation remains necessary. Exhaustive
checks belong only in small tests or explicit diagnostic programs.

The recorded error (`max_errors`, and the local criterion of the stop) is the
per-edge LUCI residual of each edge's latest commit. Its contract is narrow.
When every bond a commit reads is in the pivot basis, the output at every point
of that commit's local matrix is exactly the product of the two new factors, so
the residual is the output's largest error over the whole local matrix at
commit time. It does not bound:

- the dense error away from the local matrices;
- the output after later commits, which change it on the same points;
- the output at all for pass-1 residuals: pass-1 commits read bonds that still
  carry the initial cores (the return pass revisits only the spine and the
  off-spine bonds a Guard injection enlarged).

The local values come from the input frames, not from the output, so the
residual cannot detect an output whose bonds left the pivot basis.

#### Re-measuring stale residuals before a stop (#1451)

Every commit rewrites the two cores of its edge, and every output value
contracts every core, so any commit can change the output on the points of
every earlier residual. The walk starts each commit where the previous one
ended (`run_phase_serial` checks the canonical centre), so after a pass every
edge except the one committed last is stale in this sense; "an adjacent core
was rewritten" selects the same edges. The state counts core-rewriting commits
(`output_revision`) and stores, per edge, the revision its residual was
measured on; a residual is stale when the two differ. Guard injection changes
candidate sets only and does not count.

#1451 showed what a stale residual hides. G0, r = 5, tolerance 1e-3,
`min_sweeps = 1`, seed 1 on `29baf4a9`: the pass-3 commit of edge 0 (x side)
recorded 2.2e-13, the next commit (0 to 3) rewrote core 0 and added the
candidates whose low x bits are `0000`, and the stop read the old residual,
so the run returned `Converged` with dense error 4.73 on the plane x = 16.

On a pass that meets every other stop condition (the local target on the
recorded residuals, rank stability, no Guard pivot in the window), the stop
now re-measures every stale residual first: the operator and the current output
are evaluated on the edge's current local matrix (the matrix its next commit
would read; the output through a frame store of the output), and the residual
becomes `max |f - output|` there. `max_errors` of that pass holds the
re-measured maximum, and the stop is refused when it exceeds the tolerance.
The rank-limit flag of the pass is not recomputed from re-measured residuals:
an edge at its algebraic bound reproduces its local matrix exactly on its next
commit, so a re-measured error there is not a rank limit. The re-measurement
touches only the stale edges and only their current local matrices; it does
not revisit retained pivots, Guard points or earlier matrices (#1372). Its cost
is one operator evaluation per entry of those local matrices, about one forward
pass, on each pass that reaches the stop test. A refused stop continues the
sweep, whose commits replace the re-measured residuals. A stop is accepted
only after Guard searches that found no pivot, so it never needs a cleanup pass,
and the cleanup pass does not re-measure.

Measured on G0 r = 5, tolerance 1e-3, seeds 1-400 per layout (#1451):

| layout | `min_sweeps` | `Converged` above 1e-2, before / after | largest dense error of a `Converged` run, before / after | oracle points, after / before |
|---|---|---|---|---|
| cttn | 1 | 13 / 0 | 12.4 / 1.18e-3 | 2.11x |
| swap | 1 | 6 / 0 | 12.4 / 6.0e-4 | 1.60x |
| nblock | 1 | 32 / 0 | 8.9e-2 / 1.9e-3 | 2.81x |
| cttn | 2 | 0 / 0 | 9.5e-4 / 9.5e-4 | 1.04x |
| swap | 2 | 0 / 0 | 6.0e-4 / 6.0e-4 | 1.37x |
| nblock | 2 | 0 / 0 | 3.4e-3 / 1.7e-3 | 1.33x |

On runs whose trajectory did not change, the re-measurement costs 1.21-1.61x
(median per layout). The rest is extra passes: with `min_sweeps = 1`, 34 cttn,
6 swap and 369 nblock runs no longer stop where they did, and 9 of the cttn runs
now end `MaxSweeps` at 50 passes on the known local-error plateau near 1.01e-3
(dense error 1.01e-3) instead of stopping at up to 5.8e-3. With the default
`min_sweeps = 2` every cttn and swap run keeps its termination, passes and dense
error.

Two alternatives were measured and rejected. Marking stale residuals as
uncertifiable makes `Converged` unreachable, since every pass ends with all
edges but one stale: all 44 runs of the 11-seed table ended `MaxSweeps` at 50
passes. Re-measuring the output on each edge's commit-time local matrix (cached
operator values, no new evaluations) does not see the error: on the diagnosed
run it gave 8.1e-4 after pass 3, and it removed none of the 19 wrong cttn and
swap stops below. Re-measuring after every pass instead of only at a stop gave
the same terminations, passes and dense errors on all 1,200 `min_sweeps = 1`
runs at 1.27-1.51 times the evaluations of re-measuring at a stop.

### Fresh local crosses and tolerance-boundary cycling

Every local update uses the fresh dense LUCI cross at the original tolerance
and rank ceiling. TreeACI does not retain a previous cross or run preferred-
pivot replacement searches. The retention mechanism introduced for #784 was
withdrawn because its full-matrix screening and repeated reconstruction caused
the severe G0 performance regression in #885.

Guard-injected candidate components are therefore available to fresh LUCI on
every following update, including neighboring saturated cuts. There is no
old-pivot preference or pending preference-refresh flag that can exclude a
Guard discovery; the unchanged #874 public regressions still validate this
behavior. Guard detection, revalidation, rank limits and cleanup remain in
force.

Tolerance-boundary rank oscillation and non-improving sweeps remain unresolved
under #784. The per-edge stability window, original tolerance and Guard
conditions still determine termination; reaching the pass limit reports
`MaxSweeps`. Withdrawal does not certify global accuracy or solve cycling.

## Caches

TreeACI owns three cache families. None outlives one `tree_elementwise` call;
the run boundary releases all of them.

| | sample arena | input/candidate frames | evaluator messages |
|---|---|---|---|
| owner | `TreeAciState` | `TreeAciState` | guard input/output evaluators |
| lifetime | one run | one run, incrementally extended after sample growth | input evaluators: one run; output evaluator: one guard scan |
| contents | immutable component-sample records and deduplication keys | exact directed component contractions plus optional candidate contractions | directed subtree messages reused across floating-zone batches |
| aggregate bound | `max_sample_arena_bytes`, 256 MiB | `max_frame_bytes`, 256 MiB | `message_cache_max_bytes`, 256 MiB shared across all evaluators |
| per-entry bound | -- | `max_frame_elements`, unset by default and then a quarter of `max_working_bytes` (`2^24` f64 elements at the 512 MiB default) | -- |
| accounting | `sample_arena_records`, `sample_arena_retained_bytes` | `frame_records`, `frame_retained_bytes` | opt-in `branch_diagnostics::snapshot()`, `query_cache` |

Mandatory arena and directed-frame bounds are checked before allocation, so an
over-budget run is refused instead of first reaching the configured peak.
Candidate frames and evaluator messages are optional accelerators: when their
budget is exhausted, evaluation continues without retaining new entries. If
base directed frames grow into space occupied by candidate frames, the
candidate cache is reclaimed first. Retained bytes are logical payload
accounting, not allocator or process measurements.

The per-entry and aggregate frame bounds are independent and neither implies the
other: the cache keeps one frame per input per directed edge, so a per-frame
ceiling sized to admit one frame still admits all of them.

The element ceilings (`max_local_matrix_elements`, `max_core_elements`,
`max_frame_elements`) and the working-byte budget measure the same physical
resource, so the ceilings are not independent constants: each is `None` by
default and then derived as a quarter of `max_working_bytes`, expressed in
elements of the run's scalar type and resolved once, in `prepare_problem`, for
every later enforcement site. Raising the budget raises them with it. Setting
one explicitly pins it, and a pinned ceiling no longer follows the budget in
either direction. The retention budgets (`max_frame_bytes`,
`max_sample_arena_bytes`, `message_cache_max_bytes`) are separate and do not
follow the working budget, because they bound what survives between local
updates rather than what one update may allocate. Within one preparation the
byte budget is checked before any ceiling derived from it, so an impossible
budget is reported as `working bytes` rather than as a derived ceiling.

Local preflight plans and charges both candidate lists before enumeration.
Frame contraction reserves their metadata together with its scalar buffers;
the LUCI phase uses Core's conservative logical working estimate plus retained
input values and candidate/pivot records. Provider-private workspace and
allocator overhead are outside this estimated logical budget; it is not RSS.
Guard's aggregate message budget is divided across evaluators, with floor
rounding keeping the aggregate within the configured bound; each evaluator
spends its portion as one total shared by its directed-edge caches. History capacity grows with completed passes rather than
reserving the entire requested stopping ceiling.

The two state-owned cache families report through `TreeAciDiagnostics`, in the
same logical-byte units. With the `diagnostics` feature, `query_cache` reports
message and prepared-slice entries, logical payload and owned-byte estimates.
These are whole-evaluator high-water observations attached to query-center
records, so do not sum them across nodes or output shapes. Estimates include
message vector capacity and hash-table buckets, but exclude backend allocator
arenas and dynamic allocations inside generic node labels.

## Per-node performance attribution

Enable `tensor4all-treeaci/diagnostics`, call `branch_diagnostics::reset()`
before a measurement window and read `snapshot()` afterward. Sorted records
distinguish `input:N:node` and `output:node`, and retain separate actual physical
and incident-bond shapes when output ranks change. `local_elements` is the
checked proxy `d * product(bond_dims)`, not a FLOP count. Message batches count
unique subtree assignments, frame batches count candidate samples, and query
batches count caller-supplied full points. Their denominators are not
interchangeable.

`guard_ns` excludes recursive child messages; batched `frame_ns` includes
candidate contraction and final packing, while scalar fallbacks measure each
candidate before outer batch packing. `query_ns` is inclusive of the whole evaluator
query and must not be added to message/frame time. Kernel snapshots are
thread-local, so setup, matrix multiplication, accumulation, gathering, and
prepared-slice hits/misses/refusals belong to the measured node. Scalar and
other uninstrumented work remains in the enclosing phase time. The legacy
`contraction_summary()` is process-wide and unsuitable for per-node attribution.

The observer stores aggregate rows rather than per-point histories. Its memory
scales with distinct operand/node/shapes in the chosen window; resetting releases
those rows. It adds no numerical cache and does not change contraction dispatch.
Instrumentation and cache-accounting overhead are included in measured wall
time; use a feature-disabled build for production timing, and measure observer
overhead separately. The reproducible experiment and dimension-adjusted model
are described in `benchmarks/README.md` (issue #732).

The crate exposes native TreeTN elementwise and simultaneous n-way Hadamard
entry points. Input message caches now persist across guard scans, while the
output cache is rebuilt per scan because the approximating output changes after
each sweep. Performance parity remains workload-dependent, so it is not yet a
drop-in train ACI replacement.

The guard evaluates all starting targets and approximations together and
retains their finite residuals. Coordinate walks reuse those initial errors;
operator-point accounting counts each start once. Retained starting errors
are included in the guard's logical working-storage estimate.

Bootstrap enumeration preserves the original descending node-position digit
order using compact component suffixes. Dimension-one axes have zero digits;
only enough nontrivial axes to represent the largest initial rank are kept,
at most `usize::BITS` per directed cut. Dependency-order propagation replaces
repeated full-component traversals and sorting. One projection scratch buffer
is reused across enumerated points, with each dependency overwritten before
use. These temporary structures are preflighted together before allocation;
rank-one bootstrap needs neither nontrivial suffixes nor projection scratch.

Message caches follow the later approved evaluator-lifetime policy from PR
#646: entries are never evicted to admit a later column of the same edge. The
budget is one total shared across the evaluator's directed edges, so a branch
that is reused may retain more than an equal split; when the shared total is
exhausted, whole columns are released from the least recently used other edge,
oldest first. Their logical
admission budget covers message payload;
owned-storage diagnostics additionally count spare column capacity, map
buckets, and heap payload owned by wide assignment keys. These estimates
exclude allocator headers and are not hard resident-memory ceilings.

This document records the durable repository architecture. The staged
implementation history, including the edge-order experiment and its verdict,
is in `docs/superpowers/plans/2026-08-14-treeaci-next-phase.md` and the
accompanying spec; the invariants those phases established are stated above
rather than left to the plan.
