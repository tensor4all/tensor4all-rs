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
  LUCI kernels, but TreeACI owns the matrices it passes them. Relative mode
  normalizes the local matrix to unit maximum, making that floor a fixed
  relative round-off floor; absolute mode keeps the matrix in raw output units
  and preserves an absolute threshold. Neither mode changes core behavior for
  other callers.
- Use `rel_tol = 0` and one absolute RRLU threshold in the units selected by
  `scale_tolerance`: normalized matrix units in relative mode, raw output units
  in absolute mode. This matches the sweep's corresponding normalized or
  absolute error check. A threshold relative to the largest pivot can disagree
  with the sweep when Schur-complement growth lifts a pivot above the largest
  entry. The relative rule only keeps more pivots in that case; the absolute
  rule retains the caller's raw-unit cutoff.
- Centralize local normalization, local threshold conversion, sweep error
  comparison, and the global threshold in a private `TolerancePolicy`, so the
  three decisions cannot drift into different units again.
- In relative mode, scale the guard threshold by the largest `|f|` among its
  starts and the current edge-local matrices. Every value involved is an exact
  operator value from the current state, so this is a lower bound of `max |f|`
  without cumulative history. In absolute mode, keep the configured threshold
  absolute. A missed feature larger than anything sampled still produces a
  residual above the relative threshold; the existing separated-feature
  rescue tests remain green.
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
- A temporary ignored diagnostic harness and per-pass trace were used to
  establish the causal chain above. Their test-only source was removed after
  the investigation; timings collected with tracing are not performance
  evidence.

## Performance

Protocol frozen before timing: standalone release binaries without
`cfg(test)` instrumentation for base (`076a75e0`) and fix, shared dependency
builds, only the `hadamard_many` call timed, one thread pinned to one core,
one warmup per process, 7 repetitions with alternating arm order, relmax
checked on every run. A comparison is invalid if either arm's
`(max - min) / median > 0.30` or, end to end, relmax exceeds
`tolerance * global_tolerance_margin`. All comparisons were valid.

| Case | End to end: base → fix (median s) | Fixed 10 passes: base → fix | Fixed 10 passes, guard off |
|---|---|---|---|
| `rand_chi64` cap 128, tol `1e-3` | 0.174 MaxSweeps → 0.062 Converged (0.36×) | 0.093 → 0.162 (1.74×) | 1.011× |
| same, tol `1e-8` | 0.211 → 0.063 (0.30×) | 0.108 → 0.200 (1.85×) | 1.005× |
| same, tol `1e-14` | 0.322 → 0.078 (0.24×) | 0.165 → 0.236 (1.43×) | 1.009× |
| gauss cap 12, tol `1e-4` (control) | 0.0362 → 0.0365, both Converged in 3 (1.01×) | 1.07× | – |
| gauss cap 12, tol `1e-12` | 0.143 → 0.043, Converged 10 → 3 passes (0.30×) | 1.03× | – |
| gauss cap 12, tol `1e-14` | 0.255 MaxSweeps → 0.006 RankLimited (0.02×) | 0.13× (RankLimited earlier) | – |
| generated chain, tol `1e-3` | 0.121 MaxSweeps → 0.029 Converged (0.24×) | 1.10× | 1.015× |
| gauss cap 12, tol `1e-3` (control) | 0.0360 → 0.0359 (1.00×) | 1.10× | 1.018× |

- The end-to-end gain comes from fewer passes (earlier, justified
  termination), not from faster passes. The gauss `1e-14` rows compare an
  honest `RankLimited` at cap 12 with a `MaxSweeps` run; the accuracy there
  also improved (relmax `7.6e-14 → 1.3e-14`).
- Local updates cost 0.5–1.8% more per pass (normalization and factor
  rescaling) with identical evaluated points and accuracy.
- Fixed-iteration passes cost up to 1.85× more, entirely in the guard. With
  a correct threshold a search that finds nothing walks to its local maximum;
  on the base the underestimated threshold ended every walk after its first
  sweep on a false positive. The fix evaluates fewer points in total
  (`rand_chi64` tol `1e-3`: 39,760 vs 54,512 over 10 passes) but through more
  small guard batches. This is the ordinary cost of a guard that verifies,
  already paid on the base whenever the guard legitimately found nothing. It
  matters only for runs forced past convergence (large `min_sweeps`).
  Reducing per-search guard cost (#686/#728 area, or searching only when the
  local criteria would accept, as #608 proposed) is a separate follow-up.
- The core LUCI kernels keep their absolute floor; other callers passing raw
  small-magnitude matrices (for example tensorci/treetci) are not covered by
  this fix and were not audited here.

### Centralized scale-policy follow-up

After the stagnation fix, `TolerancePolicy` became the single private source for
TreeACI's local matrix units, absolute RRLU argument, sweep metric, and guard
threshold. Relative mode normalizes the local matrix; absolute mode leaves it
raw. The paired end-to-end check compared base `5218dad0c03dee0a398b33e645bd8cfd14b196b0`
with the candidate source (binary SHA-256 `2e72e894a39072095d3be14809ae41fc5380f3441404c5192a401a491e254d26`);
both used release builds of the same benchmark, lockfile SHA-256
`111c1302d1763d98c5424c1de4963be60b28266ccb3dd0e32628abcff2f7728d`, Rust
1.98.1, `tenferro-cpu-faer`, tolerance `1e-8`, seed 732, and one thread pinned
to CPU 2 (`RAYON_NUM_THREADS`, BLAS, OpenMP, and tenferro threads all set to 1).
The matrix covered f64/c64, chain-like degree 2/profile 1 and branched degree
4/profile 2, and both absolute and relative tolerance modes (`d=2`). Each case
used 15 paired process runs with 5 operation repetitions per arm.

The A/A run first used the same release binary on both arms. It had no validity
failures; the largest relative MAD was 2.22%, and the widest paired-ratio 95%
CI upper bound was 1.0621, supporting the predeclared 7% regression cap used
for A/B. Paired ratios are candidate/base; values below 1 are faster.

| A/A case | Paired ratio | 95% CI |
|---|---:|---:|
| f64, chain-like, absolute | 1.0143 | [0.9968, 1.0351] |
| f64, chain-like, relative | 0.9976 | [0.9679, 1.0089] |
| f64, branched, absolute | 0.9924 | [0.9727, 1.0051] |
| f64, branched, relative | 0.9968 | [0.9925, 1.0049] |
| c64, chain-like, absolute | 1.0075 | [1.0024, 1.0621] |
| c64, chain-like, relative | 1.0012 | [0.9742, 1.0172] |
| c64, branched, absolute | 0.9981 | [0.9925, 1.0021] |
| c64, branched, relative | 0.9968 | [0.9850, 1.0182] |

The paired A/B run passed its 7% cap with no validity failures. Every case's
upper CI bound was at most 1.04. Median operation times and paired-ratio CIs
were:

| A/B case | Base → candidate (ms) | Paired ratio | 95% CI |
|---|---:|---:|---:|
| f64, chain-like, absolute | 0.446 → 0.446 | 0.9860 | [0.9694, 1.0057] |
| f64, chain-like, relative | 0.442 → 0.437 | 0.9873 | [0.9750, 1.0134] |
| f64, branched, absolute | 4.001 → 3.918 | 0.9852 | [0.9724, 0.9880] |
| f64, branched, relative | 4.027 → 3.959 | 0.9850 | [0.9712, 1.0019] |
| c64, chain-like, absolute | 0.451 → 0.448 | 1.0112 | [0.9744, 1.0195] |
| c64, chain-like, relative | 0.449 → 0.455 | 1.0108 | [0.9951, 1.0400] |
| c64, branched, absolute | 4.058 → 4.060 | 1.0053 | [0.9857, 1.0099] |
| c64, branched, relative | 4.065 → 4.031 | 0.9906 | [0.9771, 1.0054] |

The maximum relative output error over all A/B runs was `3.47e-15`; evaluated
points were unchanged at 480 for chain-like cases and 1,280 for branched cases.
The end-to-end change is effectively neutral, with one small f64 branched
improvement and no regression near the 7% cap. This refactor does not support
a claim of a general TreeACI speedup.
