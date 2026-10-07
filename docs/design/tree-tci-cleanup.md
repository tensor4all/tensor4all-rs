# Unassigned tree and TCI cleanup

This record specifies the remaining cleanup contracts after assignee screening.
It is a design record, not a claim that every tracked issue is resolved. Changes
are grouped by owning layer; already-assigned issues are not duplicated.

## Scope and external dependencies

Skip #779 (extreme-scale core arithmetic), #812 (tree global residual walk),
#824 (caller RNG), and #833 (warm starts), which have an existing assignee.
Assigned #834/#835 are already resolved through PR #837: consume their landed
API and stopping semantics without redesigning them here. Recheck assignment
and current-main status before starting an issue and before submitting a PR.

The shared numerical group covers unassigned #810/#818/#829: requested TCI
threshold plumbing, Lazy Rook zero fibers, and valid zero tensor factors.
Its stricter below-epsilon reconstruction acceptance case depends on #779.
Do not replace that dependency with a local normalization workaround, duplicate
the assigned core fix, weaken the test, or claim the case resolved. Independent
TreeTN and evaluator work can proceed while waiting for that prerequisite.

## Unassigned numerical contracts

User truncation thresholds and arithmetic-safety decisions are distinct.
Absolute-tolerance TCI updates do not impose an additional fixed relative
cutoff. Relative-tolerance updates use the maximum absolute oracle value
observed so far, fixing that absolute threshold at the beginning of each
half-sweep; new observations affect the next half-sweep. This is a sampled
scale, not a whole-tensor accuracy certificate. Core extreme-scale arithmetic
is an external prerequisite, not an implementation item in this cleanup.

Matrix decompositions may return mathematical rank zero. At the tensor
factorization boundary, zero selected rank becomes valid rank-one zero factors
with the original external indices, never a dimension-zero bond. The selected
canonical side keeps its unit diagonal; its partner carries the zero value.
The current
tensor LU/CI options do not expose a user tolerance: do not reinterpret SVD
options or add a tolerance API to manufacture an all-discarded tensor case.
Complete discard under an explicit tolerance is verified at existing matrix
selection entry points.

Lazy Rook distinguishes an unsuccessful trial fiber from a zero remaining
residual. It searches other residual fibers without full dense matrix
allocation; evaluation counts can still be large for zero-heavy inputs.
Reuse the existing external contributions for unassigned #810/#818, preserving
authorship. Leave the contribution associated with assigned #779 to its owner.

## Evaluation ownership and memory

Large batches remain accepted, but coordinate assembly and contraction
intermediates are processed in bounded chunks. Final candidate matrices and
returned values still occupy size-dependent memory. Chunk bounds and retained
cache bounds are different controls.

Persistent contraction caches belong to evaluators, use compact integer keys,
and enforce an evaluator-wide entry bound with LRU touch-on-hit eviction.
They provide clearing and honest entry/byte/hit/miss/eviction accounting.
Reported bytes estimate owned retained payload, not RSS; an entry cap is not
a strict byte budget. Pending batch values must remain valid independently
of persistent retention: eviction cannot invalidate a result or a packed slot
needed later in the same recursive evaluation. Tree message retention remains
opt-in; TT and oracle caches have configurable finite defaults.

Oracle-value memoization is an explicit reusable batch evaluator, not automatic
storage in TreeTCI. Reuse the core caller-driven cache, evaluate misses in
batches, preserve ordering, deduplicate repeated points, and do not cache failed
evaluations. Values must be stable during the wrapper's lifetime; changing
parameters requires clearing or recreation. Reusing the wrapper allows reuse
across optimization and materialization without redesigning the assigned warm
start API.

Lifetime oracle evaluation counts measure successfully evaluated unique misses
within each batch, accumulating again when an evicted point is reevaluated.
They are not currently retained entries or an unbounded lifetime coordinate set.

Selection-only TreeTCI updates keep the existing rrLU elimination and pivot
selection, omitting unused interpolation-factor construction and unnecessary
matrix copies through existing metadata APIs. They do not replace the assigned
global search or RNG paths, change stop reasons, or add threading controls.

## General TreeTN operations

Site-free nodes remain valid network constituents, including nontrivial
bond-only tensors. Empty-side splits use the existing shared factorization
seam, not independent operation-specific substitutes or chain conversions.
Inspect related solver routes, but distinguish reproduced failures from
suspicions.

Branched zip-up canonicalizes towards the actual contraction center by default,
without mutating inputs. An explicit skip option lets callers guarantee the
required canonical form. Skip never disables ordinary shape, index, topology
or truncation validation; noncanonical inputs under skip do not promise
default-path approximation accuracy. Investigate the natural-rank-cap failure
independently rather than assuming it has the gauge-related cause.

## Documentation and validation

Batch and memo APIs must be discoverable from use cases: many evaluation
points, expensive oracles, and repeated related runs. The guide, rustdoc,
`llms.txt` and usage skill provide memory/ownership contracts and runnable
value-checked examples. Do not teach per-point loops as ordinary repeated
network readout or disguise them inside a batch callback.

Cover zeros and small nonzeros within the available core contract, zero/one
cache capacity, mixed hits/misses, eviction, clearing, malformed callbacks,
site-free topology variants, and default/skip canonicalization. Preserve the
below-epsilon dependency explicitly. Performance evidence uses verified
effective single-thread settings and predeclared paired cases. Do not weaken
numerical tolerances, waive red gates, or publish assigned-scope checkpoints.

## Separate work

#670 parallelism remains deferred until tenferro-rs refactoring and subsequent
tensor4all-rs redesign. FIT initialization (#656), partitioned reconstruction
(#752/#777), general site removal (#605), shared-MPO sum-target FIT (#748), and
fused-layout unification (#821) remain separate. ACI accuracy/rank oscillation
(#572/#784) is rechecked only after its external numerical prerequisite lands;
remaining causes are separate investigations. Unreproduced guard repetition
(#794) is not a basis for implementation changes.
