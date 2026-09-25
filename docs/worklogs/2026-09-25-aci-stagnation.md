# ACI stagnation investigation

Issue: [#776](https://github.com/tensor4all/tensor4all-rs/issues/776).
Base: `076a75e0` (includes #774), unchanged at execution time.

## Decisions

- Investigate on a separate main-based branch, preserving the uncommitted
  RSI work. The [debug plan](../plans/2026-09-25-aci-stagnation-debug-plan.md)
  distinguishes normal rank/precision limitation from update, scale-policy,
  bookkeeping, termination and harness defects.
- Scope is TreeACI only. Chains are native path-shaped TreeTNs; the obsolete
  SimpleTT ACI is excluded. The author's separately maintained Julia code is
  an optional reference, not a correctness oracle or a quality ceiling; it
  was not needed for any conclusion below.
- Fix every root cause downstream, in TreeACI (maintainer direction). The
  absolute `f64::EPSILON` pivot floor lives in the shared `tensor4all-core`
  LUCI kernels, but TreeACI owns the matrices it passes them. Normalizing the
  local matrix to unit maximum makes that floor a fixed relative round-off
  floor for TreeACI without changing core behavior for other callers.
- Truncate with one absolute threshold on the normalized matrix (`rel_tol = 0`)
  instead of `rel_tol = tolerance`. The latter is relative to the largest
  pivot, while the sweep's convergence check divides by the largest entry.
  The new rule is never looser than the old one: it only keeps more pivots
  when Schur-complement growth lifts a pivot above the largest entry.
- Scale the guard threshold by the largest `|f|` among its starts and the
  current edge-local matrices. Every value involved is an exact operator value
  from the current state, so this is a lower bound of `max |f|` without
  cumulative history. A missed feature larger than anything sampled still
  produces a residual above threshold; the existing separated-feature rescue
  tests remain green.
- Rejected: replacing `global_pivots_found` with the accepted injection count
  in the convergence rule, relaxing tolerances, disabling the guard, or a new
  stagnation status. The traces showed injected pivots were accepted and then
  discarded, so counting accepted pivots would not have stopped the loop, and
  the cause was the threshold, not the bookkeeping.

## Evidence

Paper: ACI v2 (arXiv:2604.00037v2), SHA-256
`1eb0ab6047034d5a4e3155a385d3201df2e36a27412670f0c07f8d8856ff7369`.
Algorithm 1, PDF line 21 stops when ranks have not increased and
`max εℓ ≤ τ` with an absolute `τ`; the paper has no global guard (its `ε∞` is
measured on 1e3 random samples for evaluation only). The guard and its scale
are TreeACI policy.

Historical fixtures (hashes verified, copied to scratch storage, unit-RMS
rescaled as in the original lane; not committed): `rand_chi64.json`
`7a7b8c5b…d71122`, `gauss_0.1_0.9.json` `84e990d7…a56fe`. Committed
regressions use a self-contained splitmix64 generator instead.

| Claim | Kind | Locator | Evidence |
|---|---|---|---|
| Guard scale underestimates `max \|f\|` | observed | `global_guard.rs` `find_global_pivots`, scale from 5 starts | `rand_chi64`: guard scale `8e-3 … 3.5` per pass, true `max\|f\|` `249.8` |
| Guard pivots are discarded by the next update | observed | trace: residual at the same coordinates before injection and after the next pass | `rand_chi64` tol `1e-3`: e.g. `7.74e-2 -> 7.74e-2`; output relFrob `4.609e-3` in every pass from pass 1 to 199 |
| Guard scale is the cause for `rand_chi64` | observed, one factor varied | diagnostic scale = local maximum or exact `max\|f\|` | all tol `1e-3 … 1e-14`, seeds 1–3, caps 128/none: 200-sweep `MaxSweeps` → 3-pass `Converged`, same ranks and accuracy |
| Local reference mismatch blocks convergence | observed | `local_update.rs` LUCI call vs `schedule.rs` `error / scale` | complex 12-site chain: rank 43/64, `last/max_pivot = 9.977e-4`, `last/max_entry = 1.163e-3`, max pivot `6.91e7` > max entry `5.93e7`; 17 idle passes |
| Absolute pivot floor makes accuracy scale-dependent | observed, homogeneous rescaling | `matrixluci/dense.rs`, `matrixlu.rs` `f64::EPSILON` | gauss tol `1e-14`: local metric pinned at `5.76e-14 = EPS / max\|f\|`; input ×1e-6 → relmax `7.3e-8`; ×269 or ×1e6 → `1.2e-14`; guard on/off identical |
| Fix preserves accuracy within the guard contract | observed | frozen matrix re-run (82 configurations) | no termination got worse; relmax changes: `rand_chi64` tol `1e-4` 200 sweeps `8.45e-5 → 9.21e-5` (both < tol; base gained it from 197 extra sweeps), gauss tol `1e-8` seeds 1–2 `9.53e-9 → 1.33e-8` (< margin); gauss tol `1e-14` `1.1e-13 → 1.2e-14` |

## Classification

| Case | Class |
|---|---|
| `rand_chi64`, all tolerances | Scale-policy inconsistency bug (guard scale), fixed |
| Complex chain at tol `1e-3` | Scale-policy inconsistency bug (local reference), fixed |
| Gauss at tol `1e-14` | Scale-policy inconsistency bug (absolute floor on raw values), fixed downstream; afterwards `RankLimited` at cap 12 (normal cap limitation) or `Converged` |
| Gauss at tol `1e-3`/`1e-4` | Normal; relmax `2.2e-3`/`2.2e-4` is within the guard margin, unchanged |
| Constant-rank passes with rank oscillation at tol `1e-14` (gauss, base) | Consequence of the floor; not reproduced after the fix |

## Verification conclusions and constraints

- Regressions `heavy_tailed_hadamard_converges_without_idle_passes` (chain
  f64, binary tree f64, chain c64) and
  `local_truncation_is_invariant_under_homogeneous_rescaling` (f64, c64) fail
  on the base and pass with the fix; the invariance test also fails when only
  the normalization is removed.
- `Converged` remains an algorithmic stopping condition, not a certified
  full-grid bound. Several fixed cases converge with relmax slightly above
  `tolerance` (up to `1.3e-8` at `1e-8`), inside `tolerance *
  global_tolerance_margin`, as before.
- The diagnostic harness (`stagnation_diagnostics.rs`, ignored test) and its
  per-pass trace are test-only. Timings seen during tracing are not
  performance evidence; the only timing comparison made (release
  `g0_convergence`, single run, 28.0 s fixed vs 29.4 s base) shows no
  regression and is not an improvement claim. A performance protocol for
  fixed-iteration per-sweep overhead was not run; the change's runtime effect
  is fewer passes, not faster passes.
- The core LUCI kernels keep their absolute floor; other callers passing raw
  small-magnitude matrices (for example tensorci/treetci) are not covered by
  this fix and were not audited here.
