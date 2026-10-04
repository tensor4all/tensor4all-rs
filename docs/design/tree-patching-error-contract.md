# Tree patching error contract

## Status

Implemented for the interpolation side of milestone M3 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md), on
`feat/tree-adaptive-patching`. It changes the public API and the acceptance
semantics of the M2 driver ([tree-pqtci-driver.md](./tree-pqtci-driver.md)).
The proposals were implemented as written; the decisions taken during
implementation, the deviations and their reasons, and the evidence gathered
for the open questions are recorded under
[Implementation decisions](#implementation-decisions). The patched-algebra
side (M3b) is scoped here and is not implemented
([Patched algebra](#patched-algebra-m3b)).

The two prerequisites: the frozen M2 golden outputs were committed before the
refactor ([Tests](#tests)); the evaluator fix of
[Determinism](#determinism) is still open, with its order decided (open
question 8, issue [#795](https://github.com/tensor4all/tensor4all-rs/issues/795)).
The cross-thread determinism test ran early: it passes on trees whose nodes
carry one site each and fails, as predicted, on the generic path, where it is
kept as an ignored test with that reason.

Every proposal was implemented as the default, and the evidence that bears on
each open question is recorded below. Open questions 1, 2, 4, 5, 8, and 9 were
decided by the user on 2026-10-03; each decision is recorded under its
question in [Open questions for the user](#open-questions-for-the-user). The
others remain with the user.

A known limitation of the sampled side of the contract was found by an
independent review after implementation: uniform-sample acceptance and the
audit can miss a localized feature that enters a patch only through a corner
or an edge, and the audited estimate can then be orders of magnitude too
small without any warning from its standard error. Only certified results are
guarantees. The fix is deferred to a global review after M9; see
[Known limitation: corner-localized misses](#known-limitation-corner-localized-misses).

Amended on 2026-10-04 by the M5 patch-size bounds: a minimum patch size can
retain a patch that misses its allowance, reported as `ToleranceNotMet`, and
capped patches up to a size bound can be accepted on their measurement. See
[Amendment: patch-size bounds](#amendment-2026-10-04-patch-size-bounds); the
sections below describe M3 and are superseded where that amendment says so.

## Goal

Give `patched_interpolate` one accuracy requirement with a measured, reported
error, following roadmap Decision 3: the L2 error is the default, the
sampled max-norm criterion of M2 stays available as an explicit choice, and a
norm without an implementation is a typed placeholder that fails before any
evaluation.

Concretely:

1. define what a measured L2 error can and cannot mean for a black-box
   function, and when it is certified;
2. split one global L2 allowance into per-patch allowances that stay valid
   when patches split;
3. add a user-selectable error norm to the options and report types without
   silently changing the meaning of an existing field;
4. add a verification step to the driver's acceptance flow that keeps the M2
   determinism guarantees and the per-patch evaluation cache;
5. keep the M1 engine contract and every engine unchanged.

## Findings that shape the design

Verified on `feat/tree-adaptive-patching` at `caf164f0` and rechecked at
`8db7b087`, after the #791 fix (#793) was merged; the cited line numbers hold
at both commits. API names were
checked against the `cargo run -p xtask --release -- api-dump` inventory
(`target/api-dump/*.md`) and the listed source.

- **M2 acceptance.** The driver accepts a patch only when the engine returns
  `InterpolationTermination::Converged` with a bond dimension strictly below
  the cap; any other verdict splits it
  (`crates/tensor4all-partitionedtreetn/src/adaptive_interpolation.rs`,
  lines 1029-1097). The engine's absolute tolerance is
  `rtol * reference_scale` (lines 1014-1021). The module documentation states
  that this is not a verified bound and makes no L2 claim (lines 43-47).
  `NoSplitIndexLeft` means "did not converge and no site of `patch_order` is
  left" (lines 595-604).
- **M2 reference scale.** Without `reference_scale` the driver pins the scale
  once, from the largest magnitude among the root patch's candidate samples
  (user pivots, recycled pivots, and random points), or from the exact values
  of an exact root (lines 991-1005, 1120-1122). It is a sampled lower bound on
  `max |f|`, in max-norm units. An all-zero exact root returns an empty
  partition with `reference_scale = 0`.
- **What the engine estimate is.** TreeTCI reports the maximum over edges of
  the last LU pivot error of each edge update
  (`crates/tensor4all-treetci/src/state.rs:193`, `update.rs:107-108`); the
  M1 engine runs with `normalize_error = false`, so the raw value is compared
  with the absolute tolerance. Its global pivot search looks for large
  `|f(x) - tt(x)|` from random starts with local coordinate optimization
  (`globalpivot.rs`, module header). Both are max-type quantities on points
  that the engine chose; neither estimates an L2 norm, and the points are not
  independent of the approximation.
- **Initial pivots in TreeTCI.** `TreeTCI2::add_global_pivots` adds every
  pivot's projection to the pivot sets of both subregions of every edge
  (`treetci/src/state.rs:117-160`). The M1 contract says nothing about how the
  number of initial pivots relates to the bond cap.
- **M1 contract.** `InterpolationOutcome::error_estimate` is the engine's raw
  error estimate, the quantity compared with the absolute tolerance
  (`crates/tensor4all-treetn/src/interpolation.rs`, lines 630-651). The
  outcome network carries only the active sites. The contract has no norm
  parameter, and the M1 record states that the driver computes the absolute
  tolerance from a reference scale "pinned for all patches in M3".
- **Exact small patches.** A patch with at most one active site is evaluated
  on all its points and built from those values with dimension-one links and
  one-hot factors; its record has `error_estimate = 0`
  (`adaptive_interpolation.rs`, lines 1102-1136).
- **Evaluation cache.** Every function value reaches the driver through the
  per-patch cache, which checks count and finiteness and is partitioned among
  the children in one pass on a split (`adaptive_interpolation/cache.rs`,
  `PatchSampler::sample` at line 185, `PatchCache::split` at line 127).
- **Randomness in the driver.** The driver implements SplitMix64 with
  Lemire's unbiased bounded draw. Each patch absorbs its path into a state and
  derives its candidate and engine sub-seeds with the stream selectors
  `CANDIDATE_STREAM` and `ENGINE_STREAM` (`adaptive_interpolation/sampling.rs`,
  lines 17-27 and `patch_seeds` at line 79). The driver offers only the seed
  API and documents the exception to the caller-owned `&mut R` rule.
- **Index IDs (checked, not the cause).** `DynIndex` IDs come from
  `generate_id()`, a per-thread, unseeded `rand::rng()`
  (`tensor4all-core/src/defaults/index.rs:413-427`), and
  `sort_indices_deterministic` breaks ties of equal dimension and prime level
  by `id()` (`tensor4all-core/src/index_like.rs:333-344`); earlier run-to-run
  differences in SRC contraction came from that (comments at
  `tensor4all-treetn/src/treetn/contraction/src_probe.rs:576-578, 786` and
  `src_tree.rs:622-625`). The evaluator path does not use it: a complete
  search of `cached_evaluator.rs` finds no use of `id()` or
  `sort_indices_deterministic`. According to the separate investigation
  cited below (not independently verified for this record), the core
  contraction it calls labels legs positionally, so the fresh-ID indices the
  generic path mints per message (lines 4388, 4759, 5592) do not affect the
  operation order. Index IDs stay in the determinism test as an excluded
  risk.
- **Contraction-path ties (the cause).** For an N-ary contraction the
  default planner of tenferro-einsum uses omeco's greedy planner, and omeco
  0.2.6 `tree_greedy` (`src/greedy.rs:170-307`) iterates
  `IncidenceList::vertices()`, which is `HashMap::keys()`
  (`src/incidence_list.rs:82-84`). Ties between equally cheap pairs are
  therefore broken by hash order, which differs across threads and
  processes. The chosen plan is cached per thread
  (`CONCRETE_EINSUM_PLAN_CACHE`,
  `tensor4all-tensorbackend/src/tenferro_bridge.rs:144, 276`), so repetitions
  on one thread reuse one plan and agree bitwise.
- **Norms of networks.** `SubDomainTreeTN::norm_squared` clones and
  canonicalizes the patch. `TreeTN::log_norm` (`treetn/ops.rs:120-188`)
  canonicalizes to the smallest node name and then takes
  `center_tensor.norm_squared()`, `sqrt`, and `ln`; `IdxTensor::norm_squared`
  (`tensor4all-core/src/defaults/idx_tensor.rs:4810`) accumulates with a
  scaled (Lassq) sum but returns the squared value, which overflows to
  `+inf` once the norm exceeds about `1.34e154` (`sqrt(f64::MAX)`). So
  `log_norm` is finite only below that norm, although its rustdoc
  (`ops.rs:91`) says it uses canonicalization "to avoid numerical overflow".
  That over-claim is a lower-layer discrepancy recorded here; M3 does not
  change `treetn` or `core` for it. `PartitionedTreeTN` stores
  patches in a `HashMap` (`partitioned_tree_tn.rs:45`) and `norm_squared`
  sums in its iteration order (lines 333-337). The bitwise reproducibility of
  the canonicalization path has not been audited; the hash-order sum is not
  reproducible.
- **Point evaluation of a network.** `TreeTNEvaluator::evaluate_batched`
  passes nodes without requested sites through unchanged
  (`treetn/evaluator.rs`, lines 294-345), but evaluates every point through a
  temporary network built with `TreeTN::from_tensors` and a full
  `contract_to_tensor`, one N-ary contraction per point.
  `TreeTNCachedEvaluator::evaluate_batched_typed::<T>`
  (`treetn/cached_evaluator.rs:2052`):
  - iterates nodes in sorted name order (lines 1452-1453) and treats a node
    without requested sites as having no entries (lines 764-768, 2329-2333);
    no test covers a site-free node;
  - uses its raw kernels only when every node has exactly one requested site
    (`can_use_raw_messages`, lines 2513-2557). A tree with a multi-site node
    or a site-free node, such as the M2 `quantics_tree` test tree, takes the
    generic `IdxTensor` path for every message;
  - on the generic path contracts N-ary operand lists through
    `contract_with_options` at a node with two or more children (line 4828)
    and at a center with two or more neighbors (line 5651), which reaches the
    planner ties above. A separate investigation measured relative
    differences up to `1.6e-13` in evaluated values across fresh threads and
    processes on such trees, including the M2 `quantics_tree`; on trees that
    take the raw kernels (one site per node, `f64`/`c64`) the values were
    bitwise identical across rebuilds, threads, processes, and thread counts
    under the controls prescribed in [Determinism](#determinism). These
    numbers come from that investigation; this record did not rerun it;
  - in the chain kernel chooses BLAS or a scalar loop from the composition of
    the batch (lines 2752-2775), and keeps message caches across calls, so a
    value depends at rounding level, deterministically, on the batch
    contents, the call history, the center, and the hint.
- **Reconstruction precedent.** Reconstruction supports only the unweighted
  discrete L2 norm and uses the allowance `max(atol, rtol * reference_scale)`
  with `ReconstructionTolerance { rtol, atol }` (default `rtol = 1e-6`), where
  `reference_scale` is an L2 norm (`reconstruction/mod.rs`, lines 37-68 and
  187). It measures a residual as the norm of an explicit difference network,
  `source.axpby(1, candidate, -1)` followed by `norm`
  (`reconstruction/engine.rs:366-367`), and combines disjoint regions with
  `hypot`. It derives its reference from the target and has no public
  override.
- **Algebra precedent.** `PatchingOptions::cutoff` allocates a local
  discarded-weight threshold in proportion to patch volume,
  `cutoff * ||F||^2 * volume_p / total_volume` (`patching.rs:61-68`), and is
  documented as best effort with no whole-network bound (lines 69-115); that
  was a recorded maintainer decision in
  [partitioned-treetn.md](./partitioned-treetn.md) (review of #655).
  `PatchingOptions` is not `#[non_exhaustive]`.
- **Extensibility.** `PatchedInterpolationOptions`, `PatchRecord`,
  `PatchedInterpolationReport`, and `PatchedInterpolationError` are
  `#[non_exhaustive]`; `PatchedInterpolationResult` is not. `Projector`
  implements `Debug`, `Clone`, `PartialEq`, and `Eq`.
- **Dense references.** `PartitionedTreeTN::to_treetn`,
  `TreeTN::contract_to_tensor` / `to_dense`, and `IdxTensor::{from_dense, sub,
  maxabs, norm}` exist for small dense test comparisons. REPOSITORY_RULES.md
  allows dense or exhaustive work in production only behind an explicit,
  caller-visible size limit.

## What a measured L2 error means

### The quantity

The domain `X` is the product of all site dimensions and `|X|` its number of
points. For a function `g` on a subset `P` of `X`,

```text
||g||_P^2 = sum over x in P of |g(x)|^2          (unweighted discrete L2)
ms_P(g)   = ||g||_P^2 / |P|,  rms_P(g) = sqrt(ms_P(g))
```

This is the norm of `TreeTN::norm`, `PartitionedTreeTN::norm`, and
reconstruction. For a uniform quantics grid, `||g||_X^2 * V / |X|` is a
Riemann sum of the continuum `||g||^2` over the grid domain of volume `V`, so
the discrete norm approximates the continuum norm times `sqrt(|X| / V)`, with
the discretization error of that sum.

The target quantity is the **absolute L2 error of the whole partition against
`f` over the whole domain**,

```text
E^2 = ||f - f~||_X^2 = sum over accepted patches P of ||f - f~_P||_P^2
                     + sum over zero patches Z   of ||f||_Z^2 ,
```

where `f~` is the returned partition. The equality is exact because accepted
and zero patches are disjoint and together cover `X` (an M2 report invariant).
Zero patches are part of the error: omitting a patch is an approximation by
zero, and it is charged like any other.

A relative statement needs `||f||`, which is not computable. It is bounded
from the computable side instead: `||f~||` is computable up to rounding from
the patch networks, and by the triangle inequality `||f|| >= ||f~|| - E`, so

```text
E / ||f|| <= E / (||f~|| - E)        whenever ||f~|| > E.
```

The driver forms this ratio conservatively: in place of `E` it uses the
certified upper value `E_up = E * (1 + GLOBAL_ROUNDING_MARGIN) + rounding`,
with the absolute rounding term of
[Measurement rounding](#measurement-rounding), and in place of `||f~||` the
computed value deflated by `GLOBAL_ROUNDING_MARGIN`. Its status follows that
of `E`:

- **certified run**: a bound on `E / ||f||`, up to that margin and the
  rounding model, independent of any reference norm;
- **audited run**: a plug-in estimate of the bound `E / (||f~|| - E)`,
  obtained by inserting the audited estimate of `E`; it is neither a bound nor
  an unbiased estimate of `E / ||f||`;
- **acceptance-only run**: no relative statement, because there is no
  estimate of `E`;
- **the denominator is not positive**, that is
  `(1 - GLOBAL_ROUNDING_MARGIN) * ||f~|| <= E_up` for a certified run or
  `||f~|| <= E` with the audited estimate of `E` for an audited run (in
  particular an empty or near-empty partition), or `||f~||` not computable
  (see [Records and report](#records-and-report)): no relative statement at
  all; the rustdoc says so.

The report encodes these cases in one enum (see
[Records and report](#records-and-report)).

### What can be computed without the dense function

| Quantity | How | Guarantee | Cost |
|---|---|---|---|
| `||f~_P||_P`, `||f~||_X` | `TreeTN::log_norm` of each accepted patch, combined by the driver in canonical path order | exact up to rounding while `||f~_P|| < ~1.34e154`, otherwise not computable; bitwise reproducibility unaudited | one canonicalization per patch, no evaluations of `f` |
| `f~(x)` at chosen points | TreeTN batch evaluators | exact up to rounding | one network evaluation per point |
| `||f - f~_P||_P` for a small patch | evaluate `f` and `f~_P` at every point of `P` | exact up to rounding (a certificate) | `|P|` evaluations of `f` (fewer with cache hits), `|P|` network evaluations |
| `||f - f~_P||_P` for a large patch | Monte Carlo on `n` fresh uniform points of `P` | a statistical estimate, see below; no bound | `n` evaluations of `f` (fewer with cache hits), `n` network evaluations |
| `||f||_X` | exactly only by evaluating all of `X`; otherwise a Monte Carlo estimate | a heavy-tailed estimate, see [Reference norm](#reference-norm) | `n` evaluations |

A held-out sample fixed in advance is the Monte Carlo row with a fixed point
set: its statement is exact on that set and statistical elsewhere.

For a black-box `f`, finitely many samples cannot bound `||f - f~_P||_P`: a
residual supported on a fraction `rho` of `P` is missed by all `n` uniform
samples with probability `(1 - rho)^n`, about `exp(-n rho)`, and its size is
unconstrained. Any claim of an L2 bound for a large patch is statistical,
never worst case.

### What a sampled measurement guarantees, before selection

Let `x_1, ..., x_n` be drawn uniformly and independently from `P` (with
replacement), independently of `f~_P` **and of any decision that uses them**,
and let `r = f - f~_P`.

- `m = (1/n) sum |r(x_i)|^2` is an unbiased estimate of `ms_P(r)`. Its
  standard error is `s / sqrt(n)` with `s^2` the sample variance of
  `|r(x_i)|^2` (so `n >= 2`). A confidence interval from it is asymptotic
  (central limit theorem) and can be badly optimistic when the residual is
  concentrated.
- Distribution-free: by exchangeability, a further uniform point exceeds
  `max_i |r(x_i)|` with probability at most `1 / (n + 1)`, so the maximum
  sampled residual is exceeded, in expectation, on at most a fraction
  `1 / (n + 1)` of the patch. This is a quantile statement, not an L2 bound.
- Exact on the sampled set: the residual at every sampled point is known.
- A standard error of zero carries no information. It arises whenever all
  sampled `|r|^2` are equal, in particular all zero, which is also what a
  residual concentrated on an unsampled set produces.

Independence of the choice of points from the approximation is essential: a
pivot of the engine has a residual near zero by construction. Values served
from the evaluation cache are the same values of `f`, so cache hits do not
break independence; only the choice of points matters.

### What it does not guarantee after selection

The driver accepts a patch when its measurement is small, using the same
sample. For an accepted patch the measurement is conditioned on acceptance,
and none of the statements above hold for it: `m` is not unbiased, and the
`1 / (n + 1)` statement does not hold for the reported maximum.

Counterexample: let `|r| = M` on a fraction `rho` of `P` and `r = 0`
elsewhere. The patch is accepted whenever no sample hits the support, with
probability `(1 - rho)^n`, and is then reported with `m = 0`, standard error
`0`, and maximum `0`, while the true error `sqrt(rho |P|) M` is arbitrarily
large. With `rho = 2/n` the acceptance probability is about `exp(-2)`, 0.13
for `n = 64`, and the true fraction above the reported maximum is `2/n`, about
twice the pre-selection bound.

- The bias depends on how concentrated the residual is, not on the distance
  to the allowance: it is small only for a residual spread over the patch and
  largest for a concentrated one, however large its norm.
- **Multiple testing.** A rerun after a failed verification may return a
  nearly unchanged approximation and draws a fresh sample, and each split
  gives the children new chances. `k` independent samples of an unchanged
  approximation accept the patch above with probability
  `1 - (1 - (1 - rho)^n)^k`.
- An upper confidence bound `m + z * SE` does not help against this: in the
  counterexample it is zero.

Valid statements about accepted patches need an **independent audit sample**,
drawn after the acceptance decision from a stream that no decision uses. The
audit estimate of an accepted patch is unbiased and carries the
`1 / (n + 1)` statement; it is still a statistical estimate, not a bound. The
proposal draws one by default for every sampled contribution
([open question 5](#open-questions-for-the-user)).

The audit removes the selection bias, not the miss: it is a uniform sample
too, so it misses a residual on a fraction `rho` of the patch with the same
probability `(1 - rho)^n`, and its standard error, computed from the same
points, is then small or zero. Unbiasedness is a statement about the average
over draws; a single audited run can underestimate a concentrated residual by
orders of magnitude, as measured in
[Known limitation: corner-localized misses](#known-limitation-corner-localized-misses).

### Definition

A patch error is **measured** when it has been computed from values of `f` at
points chosen independently of the approximation, by one of three methods,
which the report names per patch:

- **Exact**: the patch was built from all its values (the M2 exact small-patch
  path); its error is zero.
- **Exhaustive**: the residual was evaluated at every point of the patch; the
  measured error is exact up to floating-point rounding. This is a
  certificate, unaffected by selection.
- **Sampled**: the residual was evaluated at `n` fresh uniform points. The
  acceptance measurement is only a decision statistic. With an audit, the
  audit measurement is an unbiased estimate with a standard error, which can
  miss a concentrated residual (see above); without one, the report carries
  no estimate for the patch.

The run's global error is **certified** when every contribution is exact or
exhaustive (since the 2026-10-04 amendment, also within its allowance: a
retained `ToleranceNotMet` patch is never certified). Certification is a statement about the absolute error,
`E <= delta * (1 + GLOBAL_ROUNDING_MARGIN) + MEASUREMENT_ROUNDING_FACTOR * eps
* ||f~||` (see [Measurement rounding](#measurement-rounding)), where `delta` is
the allowance actually used. The report sets the flag `rounding_limited`
when the rounding term is at least the allowance (`delta` in L2 units, `tau`
in RMS units). When the reference norm was estimated, `delta` is itself random
(see [Reference norm](#reference-norm)), and only the a-posteriori relative
bound `E / (||f~|| - E)` is free of it. Otherwise the global number is either
an **audited** estimate, with its standard error and the certified fraction
of the domain, or, without a complete audit, an **acceptance-only** sum of
acceptance statistics, which is neither a bound nor an estimate of `E`. The
report distinguishes the three cases by type.

Exact and exhaustive measurements are **certified**. A sampled measurement
is measured, not certified: the acceptance measurement is a decision
statistic and the audit measurement an estimate, and neither is a guarantee
([Known limitation](#known-limitation-corner-localized-misses)). The word
"verified" is not used for results
([open question 1](#open-questions-for-the-user), decided).

## Budget

### Global allowance

The allowance is kept in root-mean-square units, so that no quantity scales
with `|X|` (which can reach the `f64` range, while `|X| * value^2` would
overflow much earlier):

```text
S_rms = S / sqrt(|X|)                          RMS value of the reference
tau   = max(atol / sqrt(|X|), rtol * S_rms)    RMS allowance
delta = sqrt(|X|) * tau = max(atol, rtol * S)  the same allowance as an L2 norm
```

Here `S` is a reference L2 norm of `f` and `S_rms` its RMS value; `delta` has
the form of the reconstruction allowance. `tau` is pinned once, before any
patch is accepted, and is the same for every patch, so acceptance does not
depend on processing order, as M7 requires. Values in L2 units (`delta`, `S`,
`E`, `||f~||`) are derived from RMS values for the report and are `None` when
the product overflows.

`rtol = atol = 0` is allowed and means `tau = 0`: a patch is accepted only if
every measured residual is exactly zero, so in practice the driver splits
down to exact patches (bounded by `max_patches`), as `rtol = 0` does in M2.

### Per-patch allowance

The allowance is split in proportion to patch volume, in squared norm, the
same rule as the volume-proportional local `cutoff` of `PatchingOptions`
(`patching.rs:61-68`):

```text
||f - f~_P||_P^2 <= delta^2 * |P| / |X|     equivalently     rms_P(f - f~_P) <= tau
```

Summing over the disjoint accepted and zero patches, whose volumes add up to
`|X|`, gives `E <= delta`, unless a patch was retained by the minimum patch
size without meeting its allowance (2026-10-04 amendment). The comparison is a root-mean-square test against
one constant.

- **Splits.** A patch's allowance depends on its volume only. The children of
  a split have volumes that sum to the parent's, so their squared allowances
  sum to the parent's: splitting never reallocates or borrows budget. A
  rejected parent approximation is discarded, so its error is never charged
  (as in reconstruction, the children replace it).
- **Zero patches** are charged `||f||_Z^2` against `delta^2 |Z| / |X|`.
- **Unused budget** of accurate patches is not redistributed; the guarantee
  holds without it, and redistribution would make acceptance depend on order.
- **Rounding.** `|P|` and `|X|` are inexact in `f64` above `2^53`, and the sum
  over patches rounds. These relative effects are covered by a named relative
  margin `GLOBAL_ROUNDING_MARGIN`, fixed at implementation and used by the
  report and the tests. The margin must also cover the approximation norm:
  `rms_P = exp(log_norm - ln(|P|) / 2)` turns an absolute error of about
  `eps * (|log_norm| + ln |P|)` in the exponent into a relative error of the
  same size, about `700 eps` for `|P| = 2^1000`. The rounding of the
  measurement itself is absolute and is treated separately below.

Rejected: an equal split by patch count. The final count is unknown while the
queue runs, so it needs reallocation and makes acceptance order-dependent.
Considered: a split proportional to each patch's own norm
([open question 2](#open-questions-for-the-user)).

### Measurement rounding

A measured residual `f(x) - f~(x)` carries the rounding of evaluating
`f~(x)`, about `c * eps * |f~(x)|` for a constant `c` that depends on the tree
and the contraction. That error is absolute: summed over a patch it is about
`c * eps * ||f~_P||`, independent of how small `E` is. Since tolerances near
or below roundoff are allowed
([Tolerances below roundoff](#tolerances-below-roundoff)), a relative margin
alone cannot express it. The certificate therefore reads

```text
E <= delta * (1 + GLOBAL_ROUNDING_MARGIN)
     + MEASUREMENT_ROUNDING_FACTOR * eps * ||f~||
```

for the true error `E`. The measured certified `rms_error` is at most
`tau * (1 + GLOBAL_ROUNDING_MARGIN)`, and the true RMS error differs from it by
at most `MEASUREMENT_ROUNDING_FACTOR * eps * approximation_rms` under the
model below. `MEASUREMENT_ROUNDING_FACTOR` is a named constant fixed at
implementation.

- The term is a first-order model for reporting, not a proven bound: with
  heavy cancellation in the contraction the evaluation error can exceed it
  (the same reason no rounding gate is used for decisions). The certificate
  is stated as holding up to this model.
- It never enters an acceptance decision; decisions compare the measured
  residual with `tau` as before.
- The report carries the term (`rounding_allowance_rms`) and a flag
  `rounding_limited`, set when the term is at least `tau`, so a user can see
  that the requested allowance is not resolved by the measurement.
- When `approximation_rms` is not computable (overflow, see
  [Records and report](#records-and-report)), both are `None` and the
  certificate is stated without a known rounding term.

### Engine tolerance

The engine still receives one absolute tolerance, now `tau`. If the engine's
pointwise criterion held everywhere on a patch, `|r| <= tau` would imply the
patch's allowance; verification checks what the engine did not. Whether a
smaller engine tolerance (a factor below one) lowers the total cost by avoiding
failed verifications is a measurement for M9 on M2 patches, not a knob in M3.

### Tolerances below roundoff

The driver does not predict an attainable residual from a fixed multiple of
machine epsilon and a patch value scale. A scale such as `max |f|` does not
identify whether a measured residual comes from roundoff, conditioning,
contraction order, or approximation error, so it cannot provide a reliable
universal rejection threshold.

The requested `tau` remains the engine tolerance and the acceptance threshold
for measured residuals. If verification exceeds `tau`, the driver follows the
same retry and split path at every scale. It may reach exact patches, or stop
with the existing verification or resource-limit error if its split order or
configured resources are exhausted. A very small tolerance can therefore cost
more work or reach a resource limit; it is not rejected before interpolation
or verification based on a predicted rounding floor. `tau = 0` follows this
same path and can reach exact patches.

With a tiny or zero `tau`, callers should set `max_patches`. With its default
`None`, the worst case splits down to exact patches of at most one active
site, about `|X| / d` patches for a last split site of dimension `d`, which is
as many evaluations as the whole domain.

### Reference norm

The two norms need references with different units:

| Norm | Reference | Units | Engine tolerance |
|---|---|---|---|
| L2 | `S`, an L2 norm of `f` | `sqrt(|X|)` times a function value | `tau`, about `rtol * rms(f)` |
| sampled max (M2) | `max_reference`, a sampled lower bound on `max |f|` | a function value | `max(atol, rtol * max_reference)` |

Because `rms(f) <= max |f|`, the L2 engine tolerance is never looser than the
max-norm one for the same `rtol` and exact references, and for a localized
function it is tighter by about `sqrt(rho)`. A max-norm reference passed as an
L2 reference would be wrong by `sqrt(|X|)`. The options and the report keep
the two references in separate, typed places (`ErrorNorm` variants and
`NormReport` variants) with different names, and the SampledMax field is
renamed `max_reference` so that it does not collide with reconstruction's
`reference_scale`, which is an L2 norm.

Where `S` comes from:

1. `L2Reference::Given(s)`: the caller's L2 norm (for a uniform quantics grid,
   approximately the continuum norm times `sqrt(|X| / V)`);
2. not needed when `rtol = 0` (then `tau = atol / sqrt(|X|)`);
3. the exact values of the root when the root is an exact small patch; an
   all-zero exact root returns an empty partition, as in M2, whatever the
   tolerance;
4. `L2Reference::MonteCarlo`, an explicit opt-in: `S_rms^2 = mean |f(x_i)|^2`
   over `verification.samples` uniform root points of a dedicated stream,
   pinned once and reported with its standard error;
5. otherwise `L2Reference::Required`, the default, fails before any
   evaluation with the remedy to give a reference norm, set `rtol = 0` with
   `atol`, or opt into the estimate.

The estimate is not a safe default, because it can loosen the tolerance
relative to `||f||`, not only tighten it. For a function of height `H` on a
fraction `rho = 1e-4` and `n = 64`: with probability about 0.994 no sample
hits the support and `S = 0` (a failure with `atol = 0`, or `delta = atol`);
with probability about 0.006 exactly one sample hits, and
`S_rms = H / 8` while the true value is `H / 100`, so `delta` is about 12
times too large and the run meets a tolerance 12 times looser than requested
without any sign of it except the a-posteriori relative bound. The default is
[open question 3](#open-questions-for-the-user); the previous revision of this
record proposed the estimate as the default.

## Public surface

All new items live in `tensor4all-partitionedtreetn`. Names are proposals.

```rust
/// The norm in which an accuracy requirement is stated and measured.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorNorm {
    /// Unweighted discrete L2 norm over the whole domain (default).
    #[non_exhaustive]
    L2 { reference: L2Reference },
    /// The engine's own sampled criterion against a max-norm reference: the
    /// M2 behavior, with no measurement by the driver.
    #[non_exhaustive]
    SampledMax { max_reference: Option<f64> },
    /// Placeholder: the maximum norm over the whole domain; for a black-box
    /// function it can be certified only exhaustively.
    MaxAbs,
    /// Placeholder: an L2 norm with caller-supplied weights.
    WeightedL2,
}
// Constructors: ErrorNorm::l2(L2Reference), ErrorNorm::sampled_max(),
// ErrorNorm::sampled_max_with_reference(scale).
// Default: ErrorNorm::L2 { reference: L2Reference::Required }.

/// Where the L2 reference norm comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum L2Reference {
    /// The caller's L2 norm of the function, finite and positive.
    Given(f64),
    /// Estimate it from uniform root samples (opt-in; see "Reference norm").
    MonteCarlo,
    /// No reference: allowed when rtol = 0 or the root is exact, otherwise
    /// an error before any evaluation (default).
    Required,
}

/// Accuracy requirement in the units of the selected norm:
/// allowance = max(atol, rtol * reference).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorTolerance { pub rtol: f64, pub atol: f64 }
// Default: rtol = 1e-8 (the M2 default value), atol = 0.

/// How the driver measures accepted and zero patches under ErrorNorm::L2.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VerificationOptions {
    /// Fresh uniform points per sampled measurement, at least 2. Default 64.
    pub samples: usize,
    /// A patch with at most max(max_exhaustive_points, samples) points is
    /// measured exhaustively. Bounds the exhaustive work and cache growth per
    /// patch. Default 1024; 0 means only patches with at most `samples`
    /// points.
    pub max_exhaustive_points: usize,
    /// Engine reruns of one patch after a failed verification, before the
    /// patch splits. Default 1.
    pub retries: usize,
    /// Draw an independent audit sample for every sampled contribution after
    /// its acceptance decision. Default true.
    pub audit: bool,
}
// Default, VerificationOptions::new() (same values), and with_samples,
// with_max_exhaustive_points, with_retries, with_audit builders.
```

`ErrorNorm` and `ErrorTolerance` are crate-level types because the
patched-algebra mode of M3b uses the same pair. The variants with data are
`#[non_exhaustive]` so that fields can be added; they are built through the
constructors. Placeholders carry no data; a placeholder that gains an
implementation may gain fields, a deliberate breaking change at that time.

`ErrorTolerance` is deliberately a plain struct without `#[non_exhaustive]`:
its two fields are the whole contract (`max(atol, rtol * reference)`), callers
build it with a struct literal as they build `ReconstructionTolerance`, and
the planned merge with that type in M3b keeps exactly these fields. A future
field would be a deliberate breaking change.

An enum, not a trait: the driver must know how a norm combines over disjoint
patches (Euclidean for L2, maximum for a max-norm), how its allowance splits,
and how it is measured. A trait would have to expose all three before a second
implemented norm exists to shape it. A trait can replace the enum when one
does.

### Options

| M2 field | M3 | Meaning |
|---|---|---|
| `rtol` | moved to `tolerance: ErrorTolerance` | relative to the reference of the selected norm |
| `reference_scale` | removed; `ErrorNorm::SampledMax { max_reference }` | unchanged inside that variant, renamed |
| (new) | `error_norm: ErrorNorm` | default `L2 { reference: Required }` |
| (new) | `tolerance.atol` | absolute floor of the allowance, default `0` |
| (new) | `verification: VerificationOptions` | used only by `L2`; validated under every norm |
| `max_bond_dim`, `patch_order`, `n_initial_pivots`, `recycle_pivots`, `seed`, `max_patches` | unchanged | unchanged |

Builders: `with_error_norm`, `with_tolerance`, and `with_verification` are
added; `with_rtol` and `with_reference_scale` are removed, so every caller
that set either one fails to compile and must choose a norm explicitly.

`verification` is validated under every norm because its checks do not depend
on the domain and an M2-equivalent call leaves it at its valid defaults, so no
input that M2 accepted is rejected; the domain-size check is L2 only (see
[Errors](#errors)).

### Records and report

Measurements are stored in RMS units, which cannot overflow for finite values;
L2-unit values are accessors returning `Option<f64>` (`None` on overflow).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasurementMethod { Exact, Exhaustive, Sampled }

/// One L2 measurement of a patch residual (under ErrorNorm::L2 only).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2Measurement {
    pub method: MeasurementMethod,
    /// Points measured: |P| for Exact and Exhaustive, the drawn sample count
    /// (duplicates included) for Sampled.
    pub points: usize,
    /// |P| as f64 (inexact above 2^53).
    pub patch_points: f64,
    /// rms_P(f - f~_P) over the measured points; exact for Exact/Exhaustive.
    pub rms: f64,
    /// Standard error of the mean square divided by the mean square; 0 unless
    /// Sampled, and 0 when the mean square is 0 (no information).
    pub mean_square_rel_std_error: f64,
    /// Largest |f - f~_P| over the measured points.
    pub max_residual: f64,
}
// error_norm() -> Option<f64> = sqrt(patch_points) * rms.

#[derive(Debug, Clone)]
pub struct PatchRecord {                  // #[non_exhaustive], as in M2
    pub projector: Projector,
    pub termination: InterpolationTermination,
    pub engine_error_estimate: f64,       // renamed from error_estimate
    pub max_sample_magnitude: f64,
    pub max_bond_dim: usize,
    pub retries_used: usize,              // new: engine reruns, 0 if none
    pub acceptance: Option<L2Measurement>,// new: the measurement that decided
    pub audit: Option<L2Measurement>,     // new: independent, Sampled only
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ZeroPatchRecord {
    pub projector: Projector,
    pub acceptance: Option<L2Measurement>, // Some under L2
    pub audit: Option<L2Measurement>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum L2ReferenceSource {
    Given,
    ExactRoot,
    /// Only when rtol = 0.
    NotNeeded,
    MonteCarlo { samples: usize, mean_square_rel_std_error: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MaxReferenceSource { Given, ExactRoot, MaxOfRootCandidates }

/// The global L2 error of a run, by what it can claim. RMS values are
/// E / sqrt(|X|). Bitwise reproducible under the Determinism prerequisite,
/// except the fields that depend on approximation_rms:
/// rounding_allowance_rms, rounding_limited, relative_error_bound, and
/// relative_bound_estimate (approximation_rms itself is exempt too).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GlobalL2Error {
    /// Every contribution is Exact or Exhaustive. The measured rms_error is
    /// at most tau * (1 + GLOBAL_ROUNDING_MARGIN), and the true
    /// E / sqrt(|X|) exceeds it by at most rounding_allowance_rms, up to the
    /// rounding model of "Measurement rounding".
    #[non_exhaustive]
    Certified {
        rms_error: f64,
        /// MEASUREMENT_ROUNDING_FACTOR * eps * approximation_rms; None when
        /// approximation_rms is None.
        rounding_allowance_rms: Option<f64>,
        /// rounding_allowance_rms >= tau: the allowance is not resolved by
        /// the measurement. None when approximation_rms is None.
        rounding_limited: Option<bool>,
        /// Conservative bound E_up / ((1 - GLOBAL_ROUNDING_MARGIN) ||f~||
        /// - E_up) on E / ||f||; None when (1 - GLOBAL_ROUNDING_MARGIN) *
        /// ||f~|| <= E_up or approximation_rms is None.
        relative_error_bound: Option<f64>,
    },
    /// Every Sampled contribution has an audit: an estimate, not a bound.
    #[non_exhaustive]
    Audited {
        rms_error_estimate: f64,
        /// Relative standard error of rms_error_estimate^2.
        mean_square_rel_std_error: f64,
        /// Plug-in estimate of the bound E / (||f~|| - E), with the audited
        /// estimate of E inserted; not an unbiased estimate of E / ||f||.
        /// None when ||f~|| <= the estimated E (no margins: an estimate) or
        /// approximation_rms is None.
        relative_bound_estimate: Option<f64>,
    },
    /// Some Sampled contribution has no audit: the combined acceptance
    /// statistics, which are neither a bound nor an estimate of E. No
    /// relative statement exists.
    #[non_exhaustive]
    AcceptanceOnly { acceptance_statistic_rms: f64 },
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct L2ErrorReport {
    /// |X| as f64.
    pub domain_points: f64,
    pub global: GlobalL2Error,
    /// Fraction of |X| whose contribution is Exact or Exhaustive.
    pub certified_fraction: f64,
    /// ||f~|| / sqrt(|X|) from TreeTN::log_norm per accepted patch, combined
    /// in path order; None when some patch's log_norm is not finite
    /// (||f~_P|| above about 1.34e154). Not covered by the bitwise
    /// determinism claim: the canonicalization it relies on is not audited
    /// for reproducibility.
    pub approximation_rms: Option<f64>,
}
// approximation_norm(), delta(), and error_norm() on GlobalL2Error return
// Option<f64> (None on overflow in L2 units).

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NormReport {
    #[non_exhaustive]
    L2 {
        reference_rms: Option<f64>,       // None only for NotNeeded
        source: L2ReferenceSource,
        tau: f64,                         // RMS allowance and engine tolerance
        error: L2ErrorReport,
    },
    #[non_exhaustive]
    SampledMax {
        max_reference: f64,               // 0 for an all-zero exact root
        source: MaxReferenceSource,
        engine_tolerance: f64,
    },
}

#[derive(Debug, Clone)]
pub struct PatchedInterpolationReport {   // #[non_exhaustive], as in M2
    pub tolerance: ErrorTolerance,        // new
    pub norm: NormReport,                 // new; replaces reference_scale
    pub accepted: Vec<PatchRecord>,
    pub zero_patches: Vec<ZeroPatchRecord>, // renamed from zero_projectors
    pub splits: usize,
    pub function_evaluations: usize,
    pub cache_hits: usize,
    pub measurement_evaluations: usize,   // new: part of function_evaluations
    pub audit_evaluations: usize,         // new: part of the above
    pub verification_failures: usize,     // new
    pub engine_retries: usize,            // new
}
```

Every acceptance measurement satisfies `rms <= tau`, so a certified
`rms_error` and the `acceptance_statistic_rms` never exceed `tau` (up to
`GLOBAL_ROUNDING_MARGIN`). The audited estimate can exceed `tau`; it is
reported as measured, because that is its purpose.

`approximation_rms` is computed without forming `||f~||^2` in L2 units: for
each accepted patch the driver takes `TreeTN::log_norm` of the patch data (a
clone, since the method canonicalizes), forms
`rms_P = exp(log_norm - ln(|P|) / 2)`, and combines
`approximation_rms^2 = sum over P of (|P| / |X|) rms_P^2` in canonical path
order. This avoids forming the global square in L2 units, but not the
per-patch overflow inside `log_norm`, which squares the center tensor's norm
(see [Findings](#findings-that-shape-the-design)): for a patch with
`||f~_P||` above about `1.34e154`, `log_norm` returns `+inf` (or fails), and
the driver then reports `approximation_rms = None` together with `None` for
the rounding term, the flag, and the relative fields. This is an overflow
outcome of the report, not an error of the run. A patch of norm zero
contributes zero.

### Errors

`PatchedInterpolationError` gains:

- `UnsupportedNorm { norm: ErrorNorm }`: validation, before any evaluation and
  before every other `InvalidInput` check, for `MaxAbs` and `WeightedL2`, with
  the remedy "use `ErrorNorm::L2` (the default) or `ErrorNorm::SampledMax`".
  It never falls back to another norm.
- `VerificationFailed { projector, measurement: L2Measurement }`: a patch
  converged below the cap, its measured L2 error still exceeded its allowance
  after the retries, and no site of `patch_order` is left to split it. Remedy:
  "list more sites in `patch_order`, raise `rtol` or `atol`, or check the
  reference norm". `NoSplitIndexLeft` keeps its M2 meaning (the patch did not
  converge).

and new `InvalidInput` branches, all before any evaluation: `rtol` or `atol`
negative or not finite; a given reference not finite and positive;
`verification.samples < 2`; under `L2` only, a domain whose point count is not
finite in `f64`, and `L2Reference::Required` when `rtol > 0` and the root is
not exact. A Monte Carlo reference that comes out zero with `atol = 0` is
`InvalidInput` after the root sample, as in M2. A non-finite value of a patch
network at a measured point is `Interpolation { source: Engine }`.
`SampledMax` accepts every input that M2 accepted.

### Explicit breaking changes

Early development allows them; each is deliberate:

1. `PatchedInterpolationOptions::new(cap)` now selects the measured L2 norm
   and requires a reference norm unless `rtol = 0` or the root is exact.
   Callers that relied on the M2 criterion pass `ErrorNorm::sampled_max()`
   and get the M2 behavior unchanged (checked against frozen M2 outputs, see
   [Tests](#tests)). Under L2 a run costs more evaluations.
2. `rtol` moves into `tolerance`, `reference_scale` becomes
   `ErrorNorm::SampledMax { max_reference }`, and their builders are removed.
3. `PatchRecord::error_estimate` is renamed `engine_error_estimate`, so it
   cannot be read as the L2 error.
4. `PatchedInterpolationReport::reference_scale` is replaced by
   `norm: NormReport`, and `zero_projectors` becomes
   `zero_patches: Vec<ZeroPatchRecord>`.

No field or variant keeps its name with a different meaning.

## Semantics

- **L2** (default): acceptance requires the M2 conditions (`Converged`,
  strictly below the cap, layout checks) and an acceptance measurement with
  `rms <= tau`; the 2026-10-04 amendment adds capped acceptance and retained
  patches. Zero patches require the same of the zero approximation. A
  certified run has
  `E <= delta * (1 + GLOBAL_ROUNDING_MARGIN) + MEASUREMENT_ROUNDING_FACTOR *
  eps * ||f~||`, up to the rounding model of
  [Measurement rounding](#measurement-rounding), with `rounding_limited`
  telling whether the second term is at least `delta` (in RMS units, at least
  `tau`); when `(1 - GLOBAL_ROUNDING_MARGIN) * ||f~|| > E_up`, also
  `E / ||f|| <= relative_error_bound`. An audited run gives an estimate
  of `E` with the reported standard error and a plug-in estimate of the
  relative bound; neither is a guarantee, and both can be far too small when
  a concentrated residual is missed
  ([Known limitation](#known-limitation-corner-localized-misses)). An acceptance-only run gives neither; its reported number is
  only the combined acceptance statistic. When the relative denominator is
  not positive (`(1 - GLOBAL_ROUNDING_MARGIN) * ||f~|| <= E_up` if
  certified, `||f~|| <= E` with the estimate if audited), or `||f~||` is not
  computable, no relative statement exists.
- **SampledMax**: exactly the M2 driver, with the engine tolerance
  `max(atol, rtol * max_reference)`; `atol = 0` reproduces M2. No measurement,
  and the rustdoc keeps the M2 statement that this is not a bound (worded
  "neither a certified bound nor a measured error" since open question 1).
- **Placeholders**: `UnsupportedNorm` before any evaluation.
- **Engines** keep the M1 contract: one absolute tolerance, their native
  criterion, and `error_estimate` in that criterion's units. They never see
  the norm.

## Algorithm

Changes to the M2 steps
([tree-pqtci-driver.md](./tree-pqtci-driver.md#algorithm)) under
`ErrorNorm::L2`. Everything not mentioned is unchanged.

1. **Validate** before any evaluation: the norm first (placeholders), then
   the tolerance, the reference, the verification options, and `|X|`.
2. **Pin** the reference at the root (given, not needed, exact root, or the
   opt-in estimate), then `tau`.
3. **Exact small patches** (at most one active site): unchanged. Their
   acceptance measurement is `Exact` with `rms = 0`: the network is built from
   the evaluated values and the one-hot factors multiply by exactly one, so no
   verification runs and no evaluation is added. An exact all-zero patch is a
   zero patch with an `Exact` measurement.
4. **Zero screening.** If every candidate sample is exactly zero, the zero
   approximation is measured on the zero-screen stream (exhaustively if the
   patch is small enough). If it passes, the patch is a zero patch, audited if
   the measurement was sampled. If it fails, the measured points with
   `f != 0`, largest `|f|` first and at most `max_bond_dim - 1` of them, are
   appended to the candidates (they are cached, so no evaluation is added),
   and the patch proceeds to the engine. An exhaustive zero screen leaves every
   value of the patch in the cache; later exhaustive measurements of it
   evaluate nothing. The zero screen does not consume a retry.
5. **Interpolate** (engine run `a = 0`) with the absolute tolerance `tau` and
   the M2 engine seed.
6. **Not converged** (`BondCapReached`, `IterationLimit`, or any future
   variant): split as in M2. No measurement runs on a patch that is not
   accepted anyway ([open question 4](#open-questions-for-the-user)).
   Superseded on 2026-10-04: capped-eligible and retained runs are measured
   ([amendment](#amendment-2026-10-04-patch-size-bounds)).
7. **Verify** the `Converged` outcome of run `a` after the M2 layout and cap
   checks, on the re-embedded patch that would be stored, so the measured
   network is the returned one:
   - if `|P| <= max(max_exhaustive_points, samples)`, evaluate every point of
     the patch in column-major order (first active site fastest), in chunks of
     a fixed size `MEASUREMENT_CHUNK`: `Exhaustive`;
   - otherwise draw `samples` points uniformly with replacement from the
     verification stream `a`: `Sampled`.

   Values of `f` come through the patch cache; values of the network come
   from one fresh evaluator per measurement, as specified under
   [Determinism](#determinism). Sums of squared magnitudes use scaled
   accumulation, so finite residuals cannot overflow.
8. **Accept.** `rms <= tau` accepts the patch with this acceptance
   measurement. For a sampled acceptance with `audit` on, draw the audit
   sample from the audit stream, measure it the same way, and record it; audit
   points never become pivots.
9. **Retry.** Otherwise, while `a < retries`, rerun the engine as run
   `a + 1`. Its initial pivots are the patch's **base candidates** (those of
   run 0: the M2 candidates plus any zero-screen points of step 4), followed
   by an **added list** built from the failed run `a` only:
   1. the measured points with `|r| > tau`, in descending order of `|r|`
      (ties in measurement order);
   2. then the pivots returned by the failed outcome, in their returned
      order.

   Points already among the base candidates or earlier in the list are
   dropped, and the added list is truncated to `max_bond_dim - 1` points.
   The outcome's pivots therefore count toward the cap and only fill what the
   worst points leave. Added lists do not accumulate across reruns: each
   rerun uses the base candidates and the list from the run just before it.
   The base candidates are not counted; they are bounded as in M2 (plus at
   most `max_bond_dim - 1` zero-screen points). A rerun that does not
   converge splits the patch as in step 6, and the children also receive the
   worst points of the preceding failed measurement, exactly as in step 10.
10. **Split** when the retries are exhausted, as for a non-converged patch
    (M2 step 10), with one addition: the **worst points** of the most recent
    failed measurement of the patch, meaning its measured points with
    `|r| > tau`, largest `|r|` first (ties in measurement order), truncated to
    `max_bond_dim - 1` points for the whole patch, are passed to the
    children, because they locate what the approximation missed and are
    already cached. Each child keeps the worst points inside its region (each
    is a full-domain point, so it belongs to exactly one child). This applies
    whenever a patch splits after at least one failed verification, whether
    the last engine run failed verification or did not converge. The worst
    points are independent of `recycle_pivots`.

    A child's candidates are then built by the M2 rule with one more source,
    in this order, each point kept only at its first occurrence: the
    compatible user pivots; the recycled pivots (if `recycle_pivots`); the
    worst points passed from the parent; then random points of the child up
    to `n_initial_pivots`. The worst points therefore count toward
    `n_initial_pivots` like recycled pivots: they reduce the random fill, and
    as in M2 no user, recycled, or worst point is dropped when these alone
    exceed `n_initial_pivots`. A patch that cannot split returns
    `VerificationFailed`.
11. **Report.** Records stay in canonical path order. The global error and
    the approximation norm (from `TreeTN::log_norm` per accepted patch, see
    [Records and report](#records-and-report)) are combined in that order, so
    they do not depend on processing order or on the hash order of the
    partition. The approximation norm feeds the report only, never a
    decision.

Failure handling in one line: a failed verification first reruns the engine
with the worst points as pivots (a missed feature that fits under the cap),
then splits (a feature that needs more rank). The same path applies when the
requested tolerance is below an apparent rounding scale; an exhausted
`patch_order` or `max_patches` is reported as `VerificationFailed` or
`ResourceLimit`.

Whether the retry pivots help is a measurement: `add_global_pivots` puts every
added point into the pivot sets of every edge, so they raise the starting rank
of the rerun and can make it reach the cap sooner. The cap on added points
limits that; its effect on retry success, rank, and evaluations is measured in
M9.

### Attempts, seeds, and streams

New stream selectors next to `CANDIDATE_STREAM` and `ENGINE_STREAM` in
`adaptive_interpolation/sampling.rs`; `s` is the patch path state of
`patch_seeds`, `R` is `verification.retries`.

| Stage | Engine seed | Measurement stream | Retries left afterwards |
|---|---|---|---|
| zero screen (only if every candidate is zero) | none | `mix(s ^ ZERO_SCREEN_STREAM)` | `R` |
| engine run 0 | the M2 engine seed | `mix(mix(s ^ VERIFY_STREAM) ^ 0)` | `R` |
| engine run `a`, `1 <= a <= R` | `mix(M2 engine seed ^ a)` | `mix(mix(s ^ VERIFY_STREAM) ^ a)` | `R - a` |
| audit of an accepted or zero patch | none | `mix(s ^ AUDIT_STREAM)` | none |
| reference estimate (root, opt-in) | none | `mix(s_root ^ SCALE_STREAM)` | none |

The engine run and its verification stream share the index `a`, so they stay
aligned whatever happens at the zero screen. Exhaustive measurements use no
stream. Each coordinate of a sampled point is drawn in active-site order with
the existing Lemire draw. The streams depend only on the root seed, the patch
path, and the stage, so a patch's measurements are independent of processing
order and of the engine's randomness. Unit tests pin the new streams against
an independent implementation, as for M2.

### Cache

- Measurement points go through the patch cache: cached values are reused,
  new values are cached, counted in `function_evaluations` and
  `measurement_evaluations` (audits also in `audit_evaluations`), and checked
  for finiteness like every other value.
- On a split the cache, including measured values, is partitioned among the
  children in the existing single pass; the children reuse those values
  without re-evaluation, and their own streams are independent of them.
- On acceptance or a zero verdict the cache is dropped after the audit, as in
  M2.
- Exhaustive measurement adds at most `max(max_exhaustive_points, samples)`
  entries to one patch's cache; that option is the explicit size limit the
  repository rules require for exhaustive work.

### Determinism

The M2 guarantee (identical report and bitwise identical stored node tensors
for a fixed seed, a deterministic evaluator, and a deterministic engine)
extends to L2 runs only if every measured network value is bitwise
reproducible, because an acceptance decision near `tau` could otherwise flip
between runs. The claim covers every report field except
`approximation_rms` and the fields derived from it, `rounding_allowance_rms`,
`rounding_limited`, `relative_error_bound`, and `relative_bound_estimate`;
those are reproducible only up to rounding, and the boolean
`rounding_limited` only away from its threshold (see
[Records and report](#records-and-report) and test 14).

**Root cause.** On trees where `TreeTNCachedEvaluator` takes its generic
`IdxTensor` path (a site-free node, a node with several sites, or `f32`/`c32`
data), its N-ary contractions at a node with two or more children and at a
center with two or more neighbors go through tenferro-einsum's default
planner, and omeco's greedy planner breaks ties between equally cheap pairs
in `HashMap` order. The chosen plan, and with it the rounding of the result,
can differ between threads and between processes. The plan is cached per
thread, so repetitions on one thread agree bitwise and cannot reveal the
problem. The M3 test tree (`quantics_tree`) takes this path.
`TreeTNEvaluator` contracts a whole network per point through
`contract_to_tensor` and is not used for the measurement.

**Checked and excluded.** Random index IDs are not the cause. This record
verified that the evaluator does not sort by ID (a complete search of
`cached_evaluator.rs`); that core labels legs positionally, so the fresh-ID
indices minted per message do not change the operation order, is a finding of
the separate investigation and was not independently verified here. IDs
remain in the test below as an excluded risk. The #791 fix (#793) makes
site-leg and edge order deterministic; it does not change the planner's
tie-breaking.

The gate is a code-level argument backed by a test: **the floating-point
operation order of a measurement is a function of the sorted node names, the
positional legs of the stored node tensors, and the batch contents only.**
The driver's part of the argument is fixed by this design:

- one fresh `TreeTNCachedEvaluator` per measurement, so no message cache
  carries values from another measurement;
- a fixed center (`CachedEvaluatorOptions::center` set to the smallest node
  name, so no greedy center search runs) and `EvaluationHint::default()` for
  every batch;
- points in a deterministic order (the stream order, or column-major for
  exhaustive), evaluated in chunks of the fixed size `MEASUREMENT_CHUNK`, so
  batch composition and call history are functions of the patch path and the
  seed. The BLAS or scalar choice of the chain kernel and the message-cache
  history then depend only on that fixed call sequence.

Under these controls the separate investigation found the raw-kernel trees
bitwise reproducible across rebuilds, threads, processes, and thread counts;
this record did not rerun that check. The evaluator's part
is not satisfied on generic-path trees. **Prerequisite**: a fix of the
contraction-path ties. It was planned before the M3 implementation, was not
done then, and is now a prerequisite of M7. The order is decided
([open question 8](#open-questions-for-the-user); treetn work tracked by
[#795](https://github.com/tensor4all/tensor4all-rs/issues/795)):

- (c) first: extend the raw kernels of `TreeTNCachedEvaluator` to site-free
  nodes. Driver patches keep at most one site per node, but the trees in use
  have site-free nodes (a site-free root, the junction of `quantics_tree`),
  and `can_use_raw_messages` (`cached_evaluator.rs`, lines 2530-2543) sends
  any tree with a site-free node to the generic path. (c) covers the
  driver's trees, is deterministic, and removes the generic-path overhead;
- (a) next: in `tensor4all-treetn`, fold the N-ary contractions of the generic
  message and center paths pairwise in positional order (children in sorted
  name order), so no planner runs there; this covers trees with a multi-site
  node and `f32`/`c32` data;
- (b) long term: upstream, deterministic tie-breaking in omeco or in the
  tenferro-einsum planner, which also covers every other N-ary contraction in
  the workspace.

The driver does not work around it.

**Scope (decided, [open question 9](#open-questions-for-the-user)).** The
target: for a fixed seed, a deterministic evaluator, and a deterministic
engine, results are bitwise identical on the same machine and build across
threads and thread counts, and across processes once a two-process CI test
(the same test binary run twice, comparing digests) passes. There is no
cross-machine promise; M8 (MPI) states its own contract. The current tested
state is narrower: fresh threads within one process, on `f64` trees whose
nodes carry exactly one site each (the raw kernels). Until #795 (c) and (a)
land, generic-path trees (a site-free node, a multi-site node, or `f32`/`c32`
data) are not reproducible across threads, and the two-process CI test does
not exist yet, so no cross-process claim is made.

The test (a unit test of the measurement): build patches on two trees, a
branched tree with a multi-site node and a site-free junction (generic path)
and a branched tree with one site per node (raw kernels). Run each
repetition on a **fresh thread** (so the thread-local plan cache starts
empty), and, for the decided cross-process scope, also in two separate test
processes (the two-process CI test, not yet added). In every repetition,
rebuild the patch from its raw node data with fresh IDs for every bond index
and a fresh
`TreeTN::from_tensors`, create a fresh evaluator, and measure the same points
with the same chunking, center, and hint; all repetitions must give bitwise
identical residuals. "The same patch" is never the same object. Repetitions on
the same thread are not a valid test of this property.

## Placement

Following roadmap Decision 1 (option E):

- **`tensor4all-treetn`: no change to the M1 contract.** The trait, problem,
  outcome, and termination stay as they are; engines still receive one
  absolute tolerance. The measurement uses the existing public batch
  evaluator. Fix (a) of [Determinism](#determinism) would change evaluator
  internals, not its API; fix (b) would change an upstream dependency.
  Rejected: a `TreeInterpolator` method that estimates the error (the
  measurement must be independent of the engine's sampling, and every engine
  would have to implement it, against Decision 2); a public residual
  estimator in `treetn` (its only consumer would be the driver; it can be
  promoted when a second consumer, for example single-network interpolation,
  needs it).
- **`tensor4all-treetci` and other engines: no change.**
- **`tensor4all-partitionedtreetn`:** `ErrorNorm`, `L2Reference`, and
  `ErrorTolerance` at the crate root (shared with M3b); the verification
  options, measurement, record, and report types and the driver changes in
  `adaptive_interpolation`; the measurement itself in a private
  `adaptive_interpolation/verify.rs`; the new streams in `sampling.rs`.
- **`tensor4all-core`: no API change.** `CachedFunction` still does not fit
  (M2 finding), and magnitudes use `CommonScalar::abs_val`.

No code reaches into another crate's internals; the driver uses only public
TreeTN and partition APIs.

## Patched algebra (M3b)

The roadmap's M3 scope also includes an optional global-budget mode for
`add_with_patching`, `truncate_adaptive`, and `contract_adaptive`. This record
fixes only its contract; the algorithm needs its own design record, because it
adds an alternative to the maintainer decision that `cutoff` is best effort
with no whole-network bound.

- The mode is opt-in; the local discarded-weight `cutoff` stays the default
  and keeps its meaning.
- It accepts `ErrorNorm::L2` with an `ErrorTolerance`; any other norm returns
  a typed unsupported error before any work. The reference is the operation's
  exact input norm, computed from the networks, with no caller override, as
  in reconstruction. `PartitionedTreeTN::norm_squared` is not bitwise
  reproducible (hash-order summation at `partitioned_tree_tn.rs:333-337`), and
  the reproducibility of the canonicalization it calls is unaudited, so a
  pinned reference must be summed in a canonical order and computed by a path
  whose reproducibility is established first.
- Every truncation is measured, not assumed: the residual is the norm of the
  explicit difference network against the untruncated source
  (`TreeTN::axpby` followed by `norm`, as in `reconstruction/engine.rs`).
  Stages within one output patch add by the triangle inequality; disjoint
  output patches combine in squares, with the volume-proportional split
  above. The result is an a-posteriori bound up to floating-point rounding,
  stronger than the interpolation side because no sampling is involved.
- For contraction the untruncated source is the exact product network, whose
  bond dimensions multiply; its cost, and the reuse of the M6 contraction
  outcome API, are the main questions of the M3b record.

Whether M3b is designed now or after M6 is
[open question 6](#open-questions-for-the-user).

## Tests

**Prerequisite, before the refactor.** A separate commit on the unmodified
M2 code adds a test that records the M2 driver outputs for the M2 test
scenarios as committed constants, in two classes:

- **discrete outputs**, compared exactly: projector keys and their canonical
  order, zero projectors, split count, function evaluations and cache hits,
  terminations, per-patch bond dimensions, and the ID-free leg layout of every
  node (sites by position, bonds by neighbor and dimension);
- **floating outputs** (per-node raw column-major data in that leg order, and
  the floating report values), compared within a tight named tolerance
  `M2_GOLDEN_RTOL` (relative to each tensor's maximum magnitude).

After the refactor the same scenarios under `ErrorNorm::sampled_max()` must
reproduce both. Bitwise equality of floating data is asserted only within one
test job, between two runs of the refactored code (test 14). Committing raw
floating data as bitwise constants would silently require cross-process and
cross-machine bitwise reproducibility, which
[open question 9](#open-questions-for-the-user) excludes (decided: same
machine and build only) and which the M2 evidence (one manual three-process
check on one machine) does not establish; committed bitwise golden constants
are therefore not planned.

The "exact" discrete outputs still rest on floating-point decisions (LU pivot
choices, an error estimate against a tolerance, a rank against the cap), so a
rounding difference on another machine can flip a decision near its
threshold. The golden scenarios are therefore chosen with **decisions well
separated from their thresholds**: scenarios on the dense test engine decide
on ranks of exactly representable data; for each TreeTCI scenario the
recording test also records the ratio of every patch's engine error estimate
to its tolerance, and the scenario is admitted only if every ratio lies
outside `[1 / GOLDEN_DECISION_SEPARATION, GOLDEN_DECISION_SEPARATION]`. The
M2 report does not contain these ratios: it has `error_estimate` only for
accepted patches and does not record the tolerance. The recording test
therefore wraps `TreeTciInterpolator` in a test-local engine that implements
`TreeInterpolator<T>`, delegates every call, and records
`problem.absolute_tolerance()`, the termination and the `error_estimate` of
each call; the driver is generic over the engine, so this needs no library
change. Only the final error-to-tolerance ratio of each call is screened.
TreeTCI's per-sweep error history, its rank-truncation decisions and its LU
pivot choices cannot be screened this way; a TreeTCI scenario
whose discrete output differs on another platform is evidence about
portability and is not silently re-recorded; question 9 (decided) makes no
cross-machine bitwise promise, but the tolerance-based golden check should
still hold. This is preferred over running the golden
check in a single pinned CI environment, because the check should hold
wherever the tests run, and a pinned environment would hide exactly that
portability question.

All tests live in `tensor4all-partitionedtreetn`, with the existing
driver-local dense test engine and TreeTCI through the path-only
dev-dependency. Topologies with a claim about trees use a node of degree three
or more, checked in the test.

1. **L2 guarantee against a dense reference, certified.** The M2
   `quantics_tree` (a site-free junction of degree three, two branches of three
   binary quantics sites, and a binary flag: `2^7 = 128` points), extended by
   one leaf node with two sites of dimensions 2 and 3, for `768` points in
   total; `max_exhaustive_points = 1024`, so every patch, the root included,
   is measured exhaustively. A localized function, TreeTCI, a given reference
   norm, and a sufficient cap so that some patches are accepted from the
   engine with rank at least two, and a tolerance well above roundoff. The
   global error must be `GlobalL2Error::Certified` with
   `rounding_limited == Some(false)`. Materialize the partition once
   (`to_treetn()?.contract_to_tensor()?`) and subtract the dense reference.
   The dense path rounds differently from the cached evaluator, so its own
   absolute term `R = MEASUREMENT_ROUNDING_FACTOR * eps * ||f~||` (with
   `||f~||` from `approximation_norm()`) is added once per evaluation path.
   Assert `diff.norm() <= delta * (1 + GLOBAL_ROUNDING_MARGIN) + 2 R`, that
   `relative_error_bound` is `Some`, and
   `diff.norm() / reference.norm() <= relative_error_bound + R /
   reference.norm()` (the bound already contains the margin and the
   measurement's own term). A separately named test,
   `rounding_model_calibration`, on the same problem, holds the explicit
   **calibration check of the rounding model**, which is not a contract
   assertion, so that a CI failure identifies itself:
   `|diff.norm() - error_norm()| <= GLOBAL_ROUNDING_MARGIN * diff.norm() +
   2 R`: it has no `delta` slack, so it holds only if
   `MEASUREMENT_ROUNDING_FACTOR` bounds the rounding of both evaluation paths
   on this network. The factor is set from the cancelling-network calibration
   of [Measurements needed later](#measurements-needed-later) with a recorded
   headroom factor, so that a correct M3 run does not fail this check
   spuriously; a failure means the model constant must be revisited, not that
   the certificate is wrong.
2. **L2 against a dense reference, sampled.** The same problem with
   `samples = 16`, `max_exhaustive_points = 0`, and a fixed seed, so a patch
   is exhaustive only with at most 16 points. Assert that some accepted patch
   has a `Sampled` acceptance measurement and an audit, that the true
   `diff.norm()` is at most a recorded constant times `delta`, and, when the
   audited relative standard error is positive, that the audited estimate lies
   within a recorded number of standard errors of the true error; when it is
   zero, only the first bound is asserted. Both constants are fixed-seed
   regression constants, not probabilistic claims. With `audit = false` the
   same run reports `GlobalL2Error::AcceptanceOnly` with no relative
   statement. A run whose `||f~||` does not exceed its `E` (for example only
   zero patches with a nonzero measured error within `atol`) reports its
   relative field as `None`.
3. **A missed feature is caught.** A dense test engine configured to drop a
   narrow feature returns `Converged`; exhaustive verification rejects it, the
   rerun receives the worst points, and the patch is accepted or split;
   `verification_failures` and `engine_retries` match. With `retries = 0` the
   patch splits immediately. A test engine that records its initial pivots
   and returns many pivots checks the added list of step 9 exactly: worst
   points first, then outcome pivots, duplicates of the base candidates
   dropped, at most `max_bond_dim - 1` added points, and no accumulation over
   two reruns.
4. **Sampled retry on a fresh stream.** A sampled verification that fails
   leads to a rerun whose measurement uses verification stream 1, not 0.
5. **Split-time candidates.** With a test engine that records its initial
   pivots, a patch that splits after a failed verification (once with the
   retries exhausted, once with a rerun that does not converge) gives each
   child exactly this candidate list: compatible user pivots, then recycled
   pivots (with `recycle_pivots` on; none with it off), then the parent's
   worst points inside the child in descending `|r|` (at most
   `max_bond_dim - 1` over all children), each point once, then random fill
   so that the total is `n_initial_pivots`, or no random fill when the
   earlier sources already reach it.
6. **Failure at the end of the order.** `VerificationFailed` when no split
   site is left after a verification failure (and `NoSplitIndexLeft`
   unchanged for a non-converged patch); `ResourceLimit` when `max_patches`
   is reached after a verification failure.
7. **Tolerance below roundoff.** A positive tolerance below a conventional
   machine-epsilon-times-scale estimate is not rejected at the root or after a
   failed measurement; retries and splitting proceed from the measured
   residual. With enough split resources, the case reaches exact patches; a
   constrained run reports its normal verification or resource-limit error.
   The zero-tolerance case follows the same path.
8. **Zero patches.** A region where every candidate is zero but the function
   is nonzero on half of the region fails the zero screen (fixed seed) and
   goes to the engine with the nonzero points as candidates, without
   consuming a retry; a truly zero region is a zero patch; accepted and zero
   patches still cover the domain.
9. **Exhaustive threshold.** A patch with exactly
   `max(max_exhaustive_points, samples)` points is exhaustive and one with one
   point more is sampled; an exhaustive measurement larger than
   `MEASUREMENT_CHUNK` is evaluated in several chunks with the same result as
   a single chunk would give on a patch built for the purpose.
10. **Exact small patches** carry `Exact` measurements with zero error and
    add no measurement evaluations.
11. **Budget arithmetic.** Every acceptance measurement has `rms <= tau`,
    and the global quantities combine as specified, within
    `GLOBAL_ROUNDING_MARGIN`. The overflow case is built on the exact
    small-patch path (a patch with one active site whose values are about
    `1e155` on enough points that `||f~_P||` exceeds `1.34e154`), or with
    the dense test engine, with `L2Reference::Given` so that forming the
    reference does not overflow first, so it exercises only the report's overflow
    handling: the run completes and reports `approximation_rms`, the rounding
    term, the flag, and the relative fields as `None`. TreeTCI's own
    behaviour at such magnitudes is out of scope for this test. A certified
    run with `tau` below `MEASUREMENT_ROUNDING_FACTOR * eps *
    approximation_rms` reports `rounding_limited == Some(true)`.
12. **Reference.** Given, not needed (`rtol = 0`), exact root, and Monte
    Carlo references; `Required` with `rtol > 0` and a non-exact root fails
    before any evaluation; a zero Monte Carlo estimate with `atol = 0` fails
    with its remedy; an all-zero exact root returns an empty partition under
    every tolerance.
13. **SampledMax.** The frozen M2 outputs are reproduced (discrete exactly,
    floating within `M2_GOLDEN_RTOL`); `atol > 0` raises
    the engine tolerance to `atol` when it exceeds `rtol * max_reference`; a
    domain too large for `f64` is accepted under `SampledMax` as in M2.
14. **Determinism.** Two L2 runs with the same seed, each on a fresh thread
    (and, for the decided scope of open question 9, in two separate
    processes, a CI test not yet added), give bitwise
    identical patches and reports, except the fields exempt from the bitwise
    claim: `approximation_rms`, `rounding_allowance_rms`,
    `relative_error_bound`, and `relative_bound_estimate` agree within
    `GLOBAL_ROUNDING_MARGIN`, and
    `rounding_limited` is compared only when `rounding_allowance_rms` is not
    within `GLOBAL_ROUNDING_MARGIN` of `tau` (a boolean derived from a
    rounding-level value near its threshold has no tolerance); otherwise it
    is not compared. On the branched tree with the dense engine and with
    TreeTCI, after the prerequisite fix. Plus the fresh-thread, fresh-ID
    measurement test of [Determinism](#determinism) on a generic-path tree
    and a raw-kernel tree.
15. **Cache.** Measurement points reuse cached values, a point is never
    evaluated twice, measured values reach the children, and audit points
    never appear among pivots.
16. **Streams.** The zero-screen, verification, audit, and reference streams
    follow the documented SplitMix64 and Lemire mapping.
17. **Errors.** `UnsupportedNorm` for each placeholder, ordered before every
    `InvalidInput` check (a placeholder together with an invalid `rtol`
    reports `UnsupportedNorm`), with an evaluator that fails the test if
    called; every new `InvalidInput` branch; a domain too large for `f64`
    under L2; a non-finite network value.
18. **Complex scalars** in the measurement (`Complex64`), with residual
    magnitudes by `abs_val`.
19. **Rustdoc** examples for every new public item, runnable and asserted,
    with `# Errors` naming the variants.

Inaccuracy from an insufficient bond cap is not a failure in these tests. In
the L2 mode an insufficient cap shows up as more splits, `VerificationFailed`,
`NoSplitIndexLeft`, or `ResourceLimit`; accuracy assertions use a sufficient
cap.

## Documentation surface

The M3 PR updates the rustdoc of the driver module and every changed type,
`crates/tensor4all-partitionedtreetn/README.md` and `src/lib.rs`, the guides
`docs/book/src/guides/partitioned-treetn.md` and
`docs/book/src/guides/tree-tn.md`, and
[tree-pqtci-driver.md](./tree-pqtci-driver.md) ("Error criterion"). Rustdoc
states what certified, measured, and estimated mean (open question 1), that a
sampled measurement is an estimate
only when audited and never a bound, that a certified error is absolute with
respect to the allowance used, that no relative statement exists for an
acceptance-only run or when the relative denominator is not positive, that
the audited relative value is a plug-in estimate of the bound, the absolute
rounding term of a certificate and the `rounding_limited` flag, that
`approximation_rms` is `None` above a patch norm of about `1.34e154`, which
report fields the bitwise determinism claim covers, and the units of each
reference. After the corner-miss finding, the module documentation,
`GlobalL2Error::Audited`, `MeasurementMethod::Sampled`, `L2Measurement`,
`VerificationOptions`, the crate root and README, the guide, the skill
reference, and `llms.txt` also state the
[known limitation](#known-limitation-corner-localized-misses): sampled
acceptance and the audit can miss localized features that enter a patch
through a corner or an edge, the audit's standard error does not reveal such
a miss, and only certified results are guarantees.

## Measurements needed later

None blocks the M3 implementation; the defaults below are provisional.

| Question | Measurement | When |
|---|---|---|
| Defaults of `samples`, `max_exhaustive_points`, `retries`, `audit` | measurement evaluations as a share of all evaluations, failure and retry rates | M9, on M2 patches of a real workload (TreeTCI, `rtol` near `1e-4`), chain and branched tree |
| Retry pivots and their cap | retry success, starting and final rank, evaluations per retry | M9, same workloads |
| `MEASUREMENT_ROUNDING_FACTOR` | evaluation error of the cached evaluator and of `contract_to_tensor` against exact values for networks of known values on the test trees, including cancelling ones; the constant is the largest observed ratio times a recorded headroom factor | M3 implementation, confirmed in M9; the value stays a model, not a bound |
| Engine tolerance below `tau` | total evaluations and patch count against the factor | M9, same workloads, at matched measured accuracy |
| Volume versus norm-proportional allocation | patches and evaluations on localized functions at matched measured error | M9, only if open question 2 selects both |

## Non-goals

- A worst-case L2 bound for a black-box function from finitely many samples;
  it does not exist.
- Changing the M1 contract or any engine's criterion.
- Implementing `MaxAbs` or `WeightedL2`.
- Weighted or non-L2 norms in reconstruction (roadmap non-goal).
- Truncating low-amplitude regions to zero patches on their measured error
  (zero patches still require exactly zero candidates first).
- The patched-algebra algorithm (M3b), parallel execution (M7), and
  split-site selection (M5).
- Merging `ReconstructionTolerance` into `ErrorTolerance`; it is proposed
  for M3b, where the algebra adopts the shared type (the defaults differ:
  `1e-6` there, `1e-8` here).

## Amendment 2026-10-04: patch-size bounds

M5 questions 2 and 4 of
[tree-pqtci-split-selection.md](./tree-pqtci-split-selection.md) add two
options, specified and argued in
[tree-pqtci-patch-size-bounds.md](./tree-pqtci-patch-size-bounds.md). Sizes
are generalized bits, one per active site, whatever its dimension. Both
defaults reproduce M3 exactly.

- **Capped acceptance.** With `CappedPatches::AcceptUpTo { bits }`, a
  `BondCapReached` run of a patch with at most `bits` active sites is checked
  (layout, re-embedding, bond at most the cap) and measured on the stream of
  its attempt. It is accepted when `rms <= tau` (the engine estimate under
  `SampledMax`). A failed capped measurement is not rerun. These acceptances
  meet their allowance by measurement, so they break no accounting
  statement; they widen the selection effects of sampled acceptance, and on
  binary layouts a bound of at most `log2(max(max_exhaustive_points,
  samples))` keeps them exhaustive.
- **Minimum patch size.** With `min_patch_bits = Some(m)`, a split whose
  children would have fewer than `m` bits is not made. The patch is retained
  with its last engine run: a judged run keeps its verdict without a second
  measurement; an unjudged one is checked like a capped run and measured once
  (by its estimate under `SampledMax`). Its record carries
  `PatchStatus::WithinTolerance` or `PatchStatus::ToleranceNotMet`; a sampled
  measurement of a retained patch is audited. An exhausted `patch_order`
  keeps its M3 errors; `VerificationFailed` now means that the last run was
  measured (converged or capped-eligible) and failed.
- **Report.** A `ToleranceNotMet` contribution turns the global error into
  `GlobalL2Error::ToleranceNotMet { measured_rms, unmet_fraction, basis }`,
  which takes precedence and whose `basis` carries the M3 classification
  (`ExactOrExhaustive`, `Audited`, `AcceptanceOnly`). `certified_fraction`
  excludes such patches, and `PatchedInterpolationReport::tolerance_met()`
  reports whether every patch met its allowance. The three M3 variants keep
  their promises, because they occur only when every patch did. The error of
  a run with a `ToleranceNotMet` patch can exceed `delta` without limit; the
  budget of accurate patches is not redistributed.
- **Validation and engine errors.** A capped bound below the minimum is
  rejected as `InvalidInput` before any evaluation. A capped-eligible or
  retained network above the cap or not matching the layout is an engine
  error, as is `Converged` at the cap.

## Open questions for the user

1. **Definition of "verified".** Exact and exhaustive measurements are
   certificates of the absolute error against the allowance used; sampled
   ones are decision statistics, and unbiased estimates only through an
   independent audit; a certified error does not cover an estimated
   reference, and a run without a complete audit is labelled acceptance-only
   and makes no error or relative claim. A certified error also includes a
   measurement-rounding term that is modelled (a calibrated constant times
   `eps * ||f~||`), not proven, so "certified" holds up to that model. Is
   this acceptable, or should the public API avoid the word "verified" for
   sampled measurements (for example "measured")?

   **Decided (user, 2026-10-03): "verified" is not used as an adjective for
   results or guarantees.** Exact and exhaustive results are "certified", the
   existing type name (`GlobalL2Error::Certified`), up to the rounding model
   above. Sampled results are "measured" or "estimated" and are not
   guarantees; see
   [Known limitation](#known-limitation-corner-localized-misses). The wording
   was changed accordingly in the rustdoc, the crate README, the Partitioned
   TreeTN guide, the skill reference, the roadmap, and this record (`llms.txt`
   already used "certified" and "estimated"). Process nouns such as
   "verification" stay, and so does "verified" where it means that someone
   checked something. The type and API names are unchanged:
   `GlobalL2Error::{Certified, Audited, AcceptanceOnly}`,
   `VerificationOptions`, and `PatchedInterpolationError::VerificationFailed`;
   the user has not approved renames. Renaming `Audited` to `Estimated` and
   `Verification*` to `Measurement*` is still open.
2. **Budget allocation.** Proposed: volume-proportional with one pinned
   `tau` (matches the roadmap's pinned reference, reconstruction's allowance,
   the local `cutoff` allocation, and the M1 record). It needs a reference
   norm, which is either given or an estimate that can loosen the tolerance
   (see question 3), and it over-refines localized functions by about
   `sqrt(rho)`, potentially requiring more exact patches or reaching a
   configured resource limit.
   Alternative: proportional to each patch's own norm,
   `ms_P(r) <= rtol^2 ms_P(f~_P) + atol^2 / |X|`, giving
   `E^2 <= rtol^2 ||f~||^2 + atol^2`. It needs no reference norm and no
   estimate, is order-independent, and avoids the over-refinement; its
   guarantee is relative to `||f~||` (relative to `||f||` up to
   `1 / (1 - rtol)`), it requires `rtol < 1`, with `atol = 0` it demands
   relative accuracy in low-amplitude regions, and it puts `||f~_P||` into
   the acceptance decision, which then needs a bitwise reproducible patch
   norm; the canonicalizing `log_norm`/`norm_squared` path is not audited for
   that, so the alternative would add a norm-reproducibility prerequisite
   like the evaluator one (question 8). The a-posteriori
   relative bound is available under either. Keep the proposal, switch, or
   offer both?

   **Decided (user, 2026-10-03): offer both, selected by the caller.**
   Volume-proportional stays the implemented default. Norm-proportional is
   added as a caller-selectable allocation once its prerequisite, a bitwise
   reproducible per-patch norm (canonical-order, audited for the same
   determinism scope as question 9), is in place; that prerequisite is shared
   with M3b. The M5 static-partition evidence (per-patch max-norm tolerances
   gave 2–17× larger L2 error than a monolithic run) supports a global L2
   budget but does not choose between the two allocations; the corner-miss
   behaviour of each allocation is part of its evaluation.
3. **Default reference norm.** Proposed now: none (`L2Reference::Required`),
   with the Monte Carlo estimate as an explicit opt-in, because the estimate
   is heavy-tailed for localized functions and can silently loosen `delta`
   (about 12 times with probability about 0.6% in the example above) as well
   as fail. The previous revision proposed the estimate as the default.
   Alternatives: the estimate as default, or the norm of the root's (possibly
   capped) engine network.
4. **Accept capped outcomes on their measured error?** A `BondCapReached`
   patch whose measured error fits its allowance could be accepted, saving
   splits. This changes the M1/M2 rule "only `Converged` is accepted", and
   with a sampled measurement it would widen the selection effects described
   above. Proposed: not in M3.

   **Decided (user, 2026-10-03): deferred to M5.** Capped outcomes are not
   accepted in M3. Whether to accept a `BondCapReached` patch on its measured
   error is decided in M5 together with an early exit at the first saturated
   sweep. Evidence from the review of the open questions (an emulation run
   for that review, not a committed diagnostic):
   - accepting capped outcomes that fit saved only 1 to 3 patches and
     changed the evaluations by −53% to +16%, and most capped outcomes
     offered for acceptance failed verification;
   - the saturated sweeps, 29% to 42% of the cost, run before the accept or
     split decision, so accepting afterwards does not save them; only an
     early exit at the first saturated sweep does;
   - the misses of the
     [known limitation](#known-limitation-corner-localized-misses) were all
     `Converged` patches, so selection bias is not the reason for the
     deferral.

   **Implemented in M5 (2026-10-04), without the early exit:** capped
   patches are accepted on their measurement only up to a size bound in
   generalized bits (`CappedPatches::AcceptUpTo`), off by default; see the
   [amendment](#amendment-2026-10-04-patch-size-bounds).
5. **Acceptance statistic and audit.** Proposed: accept on the point estimate
   of the acceptance sample, and draw an independent audit sample for every
   sampled contribution by default (`audit = true`), at the cost of `samples`
   more evaluations per sampled accepted or zero patch; without the audit the
   run is reported as `AcceptanceOnly`, with no error estimate and no
   relative statement. An upper confidence bound
   `m + z * SE` rejects more borderline patches but does not help against a
   concentrated residual (it is zero in the counterexample). Should the audit
   be on by default, and should a confidence-bound acceptance be offered?

   **Decided (user, 2026-10-03): keep both as implemented.** The audit stays
   on by default, and no confidence-bound acceptance is offered. This is
   consistent with the
   [known limitation](#known-limitation-corner-localized-misses): the audit
   removes the selection bias of the acceptance measurement but not a miss,
   so an audited result is an estimate, not a guarantee; an upper confidence
   bound `m + z * SE` does not help against a missed concentrated residual,
   whose sampled standard error is small or zero.
6. **Scope of M3.** Proposed: this record (interpolation) is implemented as
   M3; the patched-algebra global-budget mode gets its own record (M3b),
   designed after M6 so that it can use the contraction outcome API. Or
   design M3b now?
7. **Placeholders.** Proposed: `MaxAbs` and `WeightedL2`. Are these the norms
   you want reserved, or others (for example a Sobolev or a pointwise
   relative norm)?
8. **Evaluator determinism fix.** A prerequisite before implementation (see
   [Determinism](#determinism)); which one? (a) In `tensor4all-treetn`, fold
   the N-ary contractions of `TreeTNCachedEvaluator`'s generic path pairwise
   in positional order: local to one crate and quick to land, but it fixes
   only this evaluator and gives up the planner's cost optimization there.
   (b) Deterministic tie-breaking in omeco or the tenferro-einsum planner:
   fixes every N-ary contraction in the workspace, but needs an upstream
   change and a dependency update. Both, (a) first and (b) later, is also
   possible.

   **Decided (user, 2026-10-03): (c), then (a), then (b).** This is
   `tensor4all-treetn` work, tracked by issue
   [#795](https://github.com/tensor4all/tensor4all-rs/issues/795), and a
   prerequisite of M7.
   - (c), new and first: extend the raw kernels of `TreeTNCachedEvaluator`
     to site-free nodes. Driver patches keep at most one site per node, but
     the trees in use have site-free nodes (the site-free root, the junction
     of `quantics_tree`), and `can_use_raw_messages`
     (`cached_evaluator.rs`, lines 2530-2543) sends every tree with a
     site-free node to the generic path. (c) covers the driver's trees, is
     deterministic, and removes the generic-path overhead.
   - (a) next: pairwise positional contraction on the generic path, for
     trees with a multi-site node and for `f32`/`c32` data.
   - (b) long term: deterministic tie-breaking upstream in omeco or
     tenferro.

   Correction: the review of the open questions started from the premise
   that driver patches mostly take the raw kernels. They do only on trees
   without site-free nodes; the trees in use have such nodes and take the
   generic path (see the "Measured network" entry under
   [Algorithm](#algorithm)).
9. **Scope of the determinism guarantee.** Should identical seeds give
   bitwise identical results across processes (and machines with the same
   build), or only within one process? Within one process still means across
   threads once M7 runs patches in parallel, so the thread-local plan cache
   alone does not suffice for either scope. The choice sets whether the
   determinism tests also run in separate processes.

   **Decided (user, 2026-10-03): same machine and build.** For a fixed seed,
   a deterministic evaluator, and a deterministic engine, results are to be
   bitwise identical on the same machine and build across threads and
   thread counts, and across processes once a two-process CI test passes
   (the same test binary run twice). There is no cross-machine promise; M8
   (MPI) must state its own contract. This is the target; the current tested
   state is recorded in [Determinism](#determinism): fresh threads within one
   process, on `f64` trees with exactly one site per node. Until #795 (c) and
   (a) land, generic-path trees are not reproducible across threads, and the
   two-process CI test does not exist yet.

## Implementation decisions

Recorded during implementation. "Deviation" marks a place where the code
differs from the text above, with the reason; everything else fills a gap the
text left open.

### Public surface

- **Accessor placement (deviation).** The text puts `approximation_norm()`,
  `delta()`, and `error_norm()` "on `GlobalL2Error`", which does not know
  `|X|`. They live where the data is: `L2ErrorReport::error_norm()` (the
  global value of whichever variant, in L2 units) and
  `L2ErrorReport::approximation_norm()`, `NormReport::delta()` and
  `NormReport::reference_norm()` (`None` under `SampledMax`, without a
  reference, or on overflow). Added for convenience: `GlobalL2Error::rms_value()`,
  `NormReport::l2_error()`, `L2Measurement::error_norm()`, and
  `ErrorTolerance::allowance(reference)`.
- `L2ReferenceSource::MonteCarlo` is a `#[non_exhaustive]` variant, like the
  other variants with data.
- `GLOBAL_ROUNDING_MARGIN = 1e-8` and `MEASUREMENT_ROUNDING_FACTOR` are public
  constants of `adaptive_interpolation`; `MEASUREMENT_CHUNK = 256` is private.
- The new stream selectors are `ZERO_SCREEN_STREAM = "zeroscrn"`,
  `VERIFY_STREAM = "verifyst"`, `AUDIT_STREAM = "auditstr"`, and
  `SCALE_STREAM = "scalestr"` (ASCII as big-endian `u64`), pinned by a unit
  test against an independent Python implementation.
- `measurement_evaluations` counts the new evaluations of zero screens,
  verifications, audits, and the Monte Carlo reference estimate (the text did
  not classify the latter).

### Validation and pinning

- **Validation order (deviation in detail).** Step 1 lists "the norm, then
  the tolerance, the reference, the verification options, and `|X|`". The
  implemented order is: the norm (`UnsupportedNorm`, before everything,
  including the layout), the layout (`validate_layout`), `patch_order`, the
  tolerance, a given reference, the remaining M2 option checks, the
  verification options, the initial pivots, and under L2 the domain size and
  then `L2Reference::Required`. The layout and `patch_order` keep their M2
  place; the `Required` check needs the layout (whether the root is exact).
- References known before any evaluation (a given norm, `NotNeeded`, a given
  `max_reference`) are pinned before the queue starts; the exact root and the
  Monte Carlo estimate pin at the root. The Monte Carlo sample is drawn
  through the root cache before the root's candidates are sampled.
- The SampledMax root that cannot pin its reference keeps the M2 error, with
  the remedy now naming `ErrorNorm::sampled_max_with_reference`. An all-zero
  exact root without a given reference reports `max_reference = 0` with
  source `ExactRoot`.

### Algorithm

- **Measured network.** A measurement evaluates the stored re-embedded patch
  at full-domain points with every site requested (fixed sites at their
  coordinates). On a tree with one site per node every stored node then keeps
  exactly one site leg, so the cached evaluator's raw kernels apply to stored
  patches too; this was checked in the determinism test. A tree with a
  site-free node, such as the site-free root or junction of the trees in use,
  takes the generic path for every measurement (open question 8).
- **Zero screen.** The points added to the candidates are the distinct
  measured points with `f != 0`, largest `|f|` first (ties in measurement
  order), at most `max_bond_dim - 1`. They cannot duplicate a candidate,
  whose samples are all zero.
- **Worst points.** Duplicates (a sampled measurement draws with
  replacement) are removed before the truncation to `max_bond_dim - 1`.
- **Outcome pivots in a rerun.** They are checked for shape and range
  whenever a rerun uses them, independently of `recycle_pivots`; a malformed
  list is `Interpolation { source: Engine }`, as it is for recycling in M2.
- **Recycled pivots after a failed verification** come from the last engine
  run: the converged outcome whose verification failed, or the rerun that did
  not converge.
- **End of the order.** With no split site left, the error is
  `VerificationFailed` when the last engine run converged and failed its
  measurement, and `NoSplitIndexLeft` when it did not converge (even after an
  earlier failed verification).
- **Approximation norm.** A patch whose `log_norm` is `-inf` (norm zero)
  contributes zero; `+inf`, NaN, or an error makes `approximation_rms`
  `None`.
- **Exact patches** get an `Exact` measurement with `points = |P|` and add no
  evaluation; an exact all-zero patch is a zero patch with an `Exact`
  measurement.

### Tests and calibration

- **Golden outputs.** `tests/adaptive_m2_golden.rs` and
  `tests/golden/adaptive_m2.json`, recorded on the unmodified M2 driver and
  reproduced under `ErrorNorm::sampled_max()`. `M2_GOLDEN_RTOL = 1e-10`;
  `GOLDEN_DECISION_SEPARATION = 10`. The dense-engine scenarios use dyadic
  variants of the M2 functions, so ranks are decided on exactly representable
  data. The actual record goes through the same JSON text round trip as the
  committed one before comparison, because the default JSON float parser can
  differ by one unit in the last place; with that, the reproduction is
  bitwise on the recording machine. Re-recording is an ignored test that
  requires `T4A_RECORD_M2_GOLDEN=1`. The ten dense cases mirror
  `splits_at_the_junction`, `splits_at_a_multi_site_node`,
  `splits_until_every_site_of_a_node_is_fixed`,
  `reports_are_in_canonical_path_order`,
  `vanishing_regions_are_reported_as_zero_patches`,
  `one_site_patches_below_the_root_are_evaluated_exactly`,
  `recycled_pivots_seed_the_children`,
  `runs_are_deterministic_and_seeds_depend_on_the_patch`,
  `complex_patches_have_one_dtype`, and
  `iteration_limit_splits_like_the_bond_cap`. The two TreeTCI cases are the
  no-recycling and recycling runs of
  `treetci_patches_a_function_on_a_branched_tree_deterministically`.
- **A golden scenario was excluded (deviation from "the M2 test
  scenarios").** The M2 TreeTCI scenario on a seven-bit quantics chain fails
  the separation screen at every `rtol` tried from `1e-6` to `1e-12` (at the
  M2 `rtol = 1e-8`, two engine calls have error-to-tolerance ratios 0.22 and
  0.64). As the screen requires, it is not admitted. Both TreeTCI scenarios on
  the branched `quantics_tree` (with and without recycling) are admitted; their
  largest ratio is `1.4e-8`.
- **`MEASUREMENT_ROUNDING_FACTOR = 648`.** The ignored measurement
  `tests/adaptive_rounding_calibration.rs` compares the cached evaluator (as
  the driver uses it) and `contract_to_tensor` with double-double values for
  random networks (bond dimensions 2, 4, 8) on the extended `quantics_tree`,
  a raw-kernel tree, and `branched`, for cancelling networks `A - A'` with
  `A'` perturbed by 10% and 1% per entry, and for the TreeTCI patches of
  test 1. The ratio `||evaluated - exact|| / (eps ||exact||)` was at most 1.9
  for random networks, 13.2 at 10% cancellation, 161.23 at 1% cancellation,
  and 0.65 for the four TreeTCI patches of test 1's run (with its split
  order). The constant is 162 (the largest ratio
  rounded up) times a headroom factor of 4. Heavier cancellation exceeds it,
  as the text anticipates.
- **Test 14.** The fresh-thread run tests pass on the raw-kernel tree with
  the dense engine and with TreeTCI. The generic-path variants are ignored
  with a reference to open question 8. With TreeTCI on the extended
  `quantics_tree` they fail (the certified `rms_error` differed by 13 units
  in the last place between two threads). With the dense engine, whose patches
  there have rank one, they passed in three processes, which is not a
  guarantee.
- **Lower-layer defect, relied on by the report (fixed by #799).**
  Before #799, `TreeTN::log_norm`, and with it `TreeTN::norm`
  (`log_norm().exp()`) and `norm_squared`, overestimated the norm of a
  network in which a site-free leaf (a node without sites and with one
  neighbor) that is not the canonicalization center has a bond of dimension
  two or more. Minimal reproduction (the regression test
  `treetn_log_norm_of_a_site_free_leaf_with_a_wide_bond` in
  `tests/adaptive_l2.rs`): the chain `a - b - e` with `a = [1, 0]` on its
  binary site and a dimension-one bond to `b`, `b(y, k) = [[3, 0], [0, 4]]`
  over its binary site `y` and the bond `k` (dimension two) to `e`, and the
  site-free leaf `e = [1, 1]` represents `f(x, y) = [1, 0]_x [3, 4]_y` of norm
  5; the dense contraction gives 5, while the defective `log_norm().exp()`
  gave `7.0710678...` (`5 sqrt(2)`). A direct sum of patches has such leaves,
  so `PartitionedTreeTN::to_treetn().norm()` on `branched` patches gave
  213.09 against a dense 150.68.

  The M3 report relies on `log_norm`: `approximation_rms` comes from it per
  accepted patch, and a too large `||f~||` would make `relative_error_bound`
  and `relative_bound_estimate` anti-conservative. While the defect was open,
  the driver reported `approximation_rms` as `None` for a stored patch with a
  wide site-free leaf. #799 (merged to `main` as `8379852e`, and merged into
  this branch) fixed canonicalization, truncation, fit, swap, and zip-up on
  site-free nodes in `tensor4all-treetn`. The removal plan has been carried
  out: the regression test runs un-ignored; the driver guard and the rustdoc
  notes on `approximation_rms`, `relative_error_bound`, and
  `relative_bound_estimate` are removed; and the former guard test is now
  `a_site_free_leaf_with_a_wide_bond_reports_the_dense_approximation_norm`,
  which expects the approximation norm of such a patch to match the dense
  norm. The `branched` direct-sum test also checks `to_treetn().norm()`
  against the dense norm. Other site-free failures (`inner` and SRC
  contraction) remain open in issue #797.

### Review fixes

- **Overflowing exhaustive plans.** A patch is measured exhaustively only
  when its point count, computed with checked multiplication, exists and is
  at most `max(max_exhaustive_points, samples)`, and the point-list length and
  byte capacity fit in a `Vec`; otherwise it is sampled. Both exhaustive and
  sampled point-list constructors (the active-coordinate point lists only)
  use checked capacity arithmetic and fallible reservation; a reservation
  failure is `InvalidInput` naming `verification.samples` and
  `verification.max_exhaustive_points`, reported during the measurement and
  so possibly after evaluations (test
  `an_unreservable_measurement_point_list_is_invalid_input_after_evaluations`).
  `ResourceLimit` was not used: it reports a user-set cap that was reached,
  with the remedy to raise it, while here the remedy is to lower an option.
  The rest of a measurement still allocates infallibly: the full-coordinate
  point list and the values buffer of the patch network (`network_values`),
  the function values, and the growth of the patch cache. An exhaustive plan
  within a huge user-set `max_exhaustive_points` is therefore bounded only
  by the point-list reservation, not protected end to end.
  `verification.samples` whose list cannot fit a `Vec` is `InvalidInput`
  before any evaluation, a new validation branch.
- **Rerun list.** A rerun receives the distinct worst points of the failed
  measurement, without the base candidates, then the outcome pivots,
  truncated to `max_bond_dim - 1` after the base candidates are dropped
  (step 9); a split still passes the first `max_bond_dim - 1` worst points
  (step 10).

### Evidence for the open questions

Gathered with small release-mode runs on the real producer (TreeTCI through
the M3 driver), unless stated otherwise. Every number below is printed by a
committed test: the ignored diagnostics in
`crates/tensor4all-partitionedtreetn/tests/adaptive_l2_diagnostics.rs`
(command in its module documentation; all five were re-run together with
`--ignored --nocapture --test-threads 1`), and for OQ8 the stage-1 unit tests
`measurement_is_bitwise_reproducible_across_threads_on_*` in
`src/adaptive_interpolation/verify/tests.rs`, run with
`cargo test --release -p tensor4all-partitionedtreetn --lib
measurement_is_bitwise_reproducible -- --include-ignored --nocapture`.
Patch, split, and evaluation counts are deterministic; the OQ8 counts depend
on thread scheduling and vary between runs, so they are quoted as observed
ranges.
The runs used a branched quantics
tree with two six-bit variables on the branches of a site-free junction and a
binary flag (8192 points, junction of degree three), `seed = 1`, an MSB-first
interleaved split order, and two functions: a smooth one (`rms / max = 0.66`)
and a localized cusp `exp(-r / 0.03)` (`rms / max = 0.040`). These diagnostics
record evaluation counts and approximation errors, not elapsed time.

- **OQ2 (budget allocation; diagnostic, not a comparison of allocations).**
  The volume-versus-norm allocation question was not directly measured. The
  committed diagnostic (`oq2_volume_budget_local_tolerances`, cap 4) reports
  the effective local relative tolerance `tau / rms_P(f)` of the accepted
  patches, in units of `rtol`: for the smooth function from 0.80 to 1.41 at
  `rtol = 1e-4` and from 0.78 to 1.91 at `1e-6`; for the localized function
  from 0.072 (patches at the peak, close to a `sqrt(rho)` factor) to `6.4e5`
  (tails) at `1e-4` and from 0.072 to `1.9e6` at `1e-6`. It describes the
  volume allocation only; it does not compare it with a norm-proportional
  allocation. At `rtol = 1e-4` (`1e-6`), accepted patches and evaluations
  were: smooth L2, 16 (109) patches and 8192 (8192) evaluations; SampledMax,
  14 (86) and 6368 (8101); localized L2, 45 (100) and 6837 (8192);
  SampledMax, 22 (56) and 3531 (5745). The SampledMax numbers are a separate
  diagnostic of the change of norm at unmatched achieved accuracy: at
  `rtol = 1e-4` its localized run reached L2 relative error `1.2e-3`, while the
  L2 run reached `3.8e-5`. The small domain caps evaluations at 8192.
- **OQ3 (reference).** Every L2 test with a root that is not exact passes
  `L2Reference::Given`, many with a dense-reference computation that exists
  only to supply it; every runnable rustdoc and guide example with a
  non-exact root computes the norm from its dense values. (Earlier counts of
  these uses are not reproduced by a diagnostic and were removed.) The Monte
  Carlo estimate (`oq3_monte_carlo_reference_spread`: 64 uniform samples,
  10000 repetitions, the same estimator outside the driver) gave `S_MC / S`
  within 0.92 to 1.08 (1% and 99% quantiles) for the smooth function, and a
  median of 0.54 with 10% and 90% quantiles 0.12 and 1.66 for the localized
  one: too large by more than 1.5 times (a looser allowance) with
  probability 0.134, and below half with probability 0.469.
- **OQ4 (capped outcomes).** A test-local wrapper
  (`oq4_capped_outcomes_that_fit`) measured every `BondCapReached` outcome
  exhaustively against its `tau` (cap 4, `rtol = 1e-4`). Of 15 capped
  outcomes of the smooth run, 13 fit their allowance; of 44 of the localized
  run, 28 fit. The runs made 15 and 44 splits: every capped outcome was
  split, as the driver does by construction. The measurement was
  exhaustive; under the proposal a large patch would be sampled, with the
  selection effects described above.
- **OQ5 (audit).** `oq5_audit_overhead`, cap 6, `rtol = 1e-4`, defaults
  otherwise: the audit added 73 of 2937 evaluations (2.5%) on the smooth
  function and 48 of 5019 (1.0%) on the localized one. The audited
  relative-bound estimates were `2.25e-5` and `3.19e-5` against true
  relative errors of `2.76e-5` and `3.20e-5`; the acceptance-only statistics
  were `3.2e-5` and `1.5e-6` in RMS units. The agreement holds for these two
  runs only and does not make the audited estimate reliable in general: on a
  function whose feature enters some patches only through a corner, the
  audited estimate was `10^4` to `10^6` times too small
  ([Known limitation](#known-limitation-corner-localized-misses)).
- **OQ8 and OQ9 (determinism).** The stage-1 tests, six fresh threads per
  process, three processes: on the raw-kernel tree, all 128 measured values
  were bitwise identical across threads, and every thread of every process
  gave the same digest. On the generic-path tree, 4 to 193 of 768 values
  differed from the first thread (the count varies between runs), with a
  largest relative difference of `1.2e-16` of the largest magnitude
  (`6.0e-17` in one process); the digests differed between most threads and
  processes, with a few repeating (one digest occurred in two processes).
  `oq8_generic_path_runs_across_threads`, an L2 run with TreeTCI on the
  extended `quantics_tree` (cap 3, 83 patches, 80 splits) in two fresh
  threads, gave identical patch and split counts and identical stored node
  data (TreeTCI itself was reproducible), but the acceptance `rms` differed
  in 9 of 83 patches in this run (the count varies between runs, as the
  measured values do), because those residuals are at rounding level. The
  diagnostic does not print the size of these differences or compare the
  individual decisions; equal patches, splits, and node data imply that no
  decision flipped in this run. A decision can flip only when a measured
  residual lies within the evaluation's rounding (about `eps` times the
  network values) of `tau`.

### Known limitation: corner-localized misses

Found on 2026-10-03 by an independent review of the implemented driver, after
the evidence above was gathered. It is recorded here, not fixed.

- **Setup.** A three-dimensional Gaussian ridge along the diagonal of the unit
  cube, `f = exp(-d^2 / (2 w^2))` with `d` the distance from the line
  `x = y = z` and `w = 0.02`. Each variable has `R = 7` bits (`2^21` points);
  the three variables lie on the three branches of a site-free root of degree
  three, one binary site per node. `patch_order` is MSB-first and interleaved
  (`x0, y0, z0, x1, ...`). TreeTCI, cap 16, `rtol = 1e-4`, `ErrorNorm::L2`
  with `L2Reference::Given` (the dense norm), default verification
  (`samples = 64`, `max_exhaustive_points = 1024`, `retries = 1`, audit on),
  seeds 1 to 12. Every accepted patch was measured by sampling, and every run
  was reported as `GlobalL2Error::Audited`.
- **Result.** 8 of 12 seeds (1, 5, 6, 8, 9, 10, 11, 12) missed parts of the
  feature. Their true `E / ||f||` was `6.3e-2` to `10.8` (635 to `1.1e5`
  times `delta`), while the audited estimate of `E / ||f||` was `2.6e-6` to
  `8.3e-6`: an underestimate by `10^4` to `10^6`. The relative standard error
  of the audited mean square was 0.17 to 0.60, so the report gave no warning.
  In each of these runs one or two patches carried essentially all of `E^2`.
  The other four seeds reached `E / ||f||` of `4.9e-6` to `8.7e-6`, within the
  allowance.
- **Which patches were missed.** Patches with 2 or 3 fixed sites (a quarter or
  an eighth of the cube along the first bit levels), which the ridge enters
  only near a corner or an edge. There were two kinds:
  - (a) the engine never sampled the feature and converged at rank 1: the
    largest sampled magnitude was `1e-25` to `1.5e-11` times `tau`, and the
    true RMS residual about `1.8e3 tau`;
  - (b) the engine saw the feature (largest sample about `5e4` to `2e5`
    times `tau`, rank 4 to 7) and its own error estimate was below `tau`
    (0.4 to 0.9 `tau`), but the true RMS residual was `1.3e3` to `2.2e5`
    times `tau` (one further patch `3.4e2 tau`).

  In both kinds the acceptance and audit measurements were at most about
  `0.1 tau` in RMS: the 64 uniform points of each did not hit the small region
  where the residual sits. This is the concentrated-residual case of
  [What it does not guarantee after selection](#what-it-does-not-guarantee-after-selection);
  the audit removes the selection bias but not the miss.
- **What only partly helps.** `samples = 1024` reduced the misses to 2 of 12
  seeds (5 and 12, both kind (b), true residual `1.3e3 tau`); in seed 12 the
  audited estimate rose to `1.4e-3` against a true `6.35e-2`, still 45 times
  too small. `recycle_pivots = true` did not fix it: 6 of 12 seeds still
  exceeded `delta` (5 of them by 11 to 267 times), with audited estimates of
  `3.6e-6` to `4.4e-5`.
- **Opt-in cache candidates (M5 question 5, 2026-10-04).** With
  `cache_candidates = true`, every child patch also starts from the largest
  values of its inherited evaluation cache. On the reproduction below (seeds
  2, 3, 4) the true `E / delta` fell from 680, 880, and 880 to 51, 230, and
  16. No rank-1 patch of kind (a) was accepted any more, and the effect is
  not limited to kind (a): seed 2, which had none, improved because a kind
  (b) patch (true RMS `1.4e3 tau`) was no longer accepted. Other kind (b)
  patches, whose feature the engine saw but did not resolve and whose
  residual the 64-point acceptance sample and the audit missed, remained, so
  every seed still exceeds `10 delta`. These are three seeds of one
  workload, not a general guarantee. The option changes the start of the
  engine, not the measurement; the measurement-side remedies stay with the
  M9 review. The ignored test
  `cache_candidates_against_corner_localized_misses` records this.
- **What is not affected.** `Certified` results, in which every contribution
  is exact or exhaustive, are not affected: the limitation concerns only the
  sampled acceptance and the audit. `SampledMax` makes no L2 claim.
- **Reproduction.** The ignored test
  `corner_localized_ridge_is_not_missed_by_sampled_acceptance` in
  `crates/tensor4all-partitionedtreetn/tests/adaptive_l2_corner_miss.rs`,
  adapted from the review's scratch source, runs the M3 driver on the same
  ridge at the smallest size that shows both kinds of miss: `R = 6` (`2^18`
  points), `w = 0.02`, cap 16, `rtol = 1e-4`, default verification, seeds 2,
  3, and 4. It materializes the partition once against the dense reference
  and fails when the true `E` exceeds `10 delta`. Observed: true `E / ||f||`
  of `6.85e-2`, `8.79e-2`, and `8.80e-2` (680 to 880 times `delta`) against
  audited estimates of `4.3e-5`, `3.3e-5`, and `4.0e-4`; seeds 3 and 4 each
  accept a rank-1 patch with 2 fixed sites whose largest sample is at most
  `1.6e-15 tau` (kind (a)), and seeds 2 and 4 accept patches with 2 or 3 fixed
  sites that the engine resolved partly (kind (b)). It took 8.5 s in release
  on one core. At this size the audited relative standard errors were large
  (0.86 to 1.00); the 7-bit runs above show that a small one is no
  reassurance either. At `R = 6`, 11 of seeds 1 to 12 exceeded `delta`, 10 of
  them by more than ten times.
- **Decision.** The fix is deferred to a global review after M9 (user
  decision, 2026-10-03). Until then the public documentation states the
  limitation. The candidate remedies are listed under M9 in
  [tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md).
  Open question 1 (the naming of "verified") was not decided by this
  finding; it was decided afterwards by the user (see
  [open question 1](#open-questions-for-the-user)).
