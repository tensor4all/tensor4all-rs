# TreeACI pivot retention at the tolerance boundary

## Problem and decisions

Issue [#784](https://github.com/tensor4all/tensor4all-rs/issues/784) records a
DMRG Hadamard product whose near-threshold local crosses keep changing. Its
original 50-site rank/error history cycles, although the local residuals stay
near the requested tolerance. Accepting an established sufficient cross helps
that witness, but complete-cross retention alone leaves the larger family at
the 20-pass limit. Captured chi=60 local matrices independently confirm that
rejected old crosses actually exceed tolerance: they cannot simply be accepted.

- Fresh dense LUCI determines the rank ceiling. Reconstruct an established
  cross, or complete its surviving axis, only in the current Cartesian
  candidate matrix. Measure the entire local residual at the same tolerance.
- When that cross is insufficient, restore individual surviving pivots in the
  fresh cross. Each accepted exchange keeps its rank, protects selected old
  pivots, and strictly increases overlap. Screen with the rank-one cross
  residual update, try feasible positions by largest absolute determinant
  ratio, and validate each accepted trial through full reconstruction.
- Keep cross reconstruction, axis completion, and partial restoration in Core's
  generic matrix APIs. TreeACI maps component samples to coordinates and owns
  the acceptance decision. Unioning historical samples into candidate axes
  would also require changing adjacent CI gauges and is not implemented.
- Pin the interpolative factor to exact identity at its selected axis, then
  measure the residual. Solve round-off in those algebraic identity entries
  otherwise changes subsequent nested gauge choices on branches.
- Complete retention and partial restoration have separate optional working
  reservations. The latter reserves another conservative Core LUCI estimate.
  Tight budgets keep ordinary LUCI, or sufficient complete retention alone,
  without introducing a new resource-limit failure.
- Partial restoration reuses solved coefficients after rejected candidates;
  accepted replacements invalidate them. The local matrix and residual are
  bounded by the existing local size gates. No full physical grid is constructed
  by either production algorithm. Extra local work means fewer passes do not
  by themselves establish a speedup.
- Canonical pivot sorting, limiting interpolation norms to fresh LUCI, removing
  the fresh-rank ceiling, restoring only the strongest exchange position, and
  preferring the smallest predicted residual did not resolve the whole family.
  These policies are excluded. Tolerance, Guard, candidate dimensions, RNG,
  per-cut stability counters, and the default pass limit are unchanged.
- The test-only skeleton reference contracts a site-free scalar network using
  explicit inverse gauges. Its former enumeration of rank^(2E) products
  amplified cancellation in the binary-tree edge-order regression. The deleted
  recursive helper had no production callers; existing skeleton, orientation,
  malformed-state, and branch tests retain coverage and unchanged thresholds.

## Numerical evidence

The public `hadamard_many` replay uses d=3, 50 sites, relative tolerance 1e-8,
seed 1, default Guard, and the original 20 directional-pass limit. A pass is
49 local edge updates. Pristine main e0a63a31 supplies the DMRG baseline;
cd8a1f8a has identical relevant Core/TreeACI implementation. The branch also
synchronizes subsequent main documentation changes before publication.

| Input chi | Rank cap | Pristine termination | Repaired termination | Baseline relative Frobenius error | Repaired relative Frobenius error |
|---|---|---|---|---|---|
| 40 | 400 | MaxSweeps, 20 | Converged, 6 | 3.00e-6 | 3.17e-6 |
| 60 | 600 | MaxSweeps, 20 | Converged, 9 | 1.43e-6 | 1.60e-6 |
| 80 | 160 | MaxSweeps, 20 | Converged, 6 | 4.54e-6 | 4.58e-6 |
| 80 | 640 | MaxSweeps, 20 | Converged, 6 | 4.44e-6 | 4.50e-6 |
| 100 | 200 | MaxSweeps, 20 | Converged, 6 | 7.71e-6 | 7.73e-6 |
| 100 | 800 | MaxSweeps, 20 | Converged, 12 | 7.71e-6 | 8.21e-6 |

Independent TT inner products contract the output norm, product/output overlap,
and four-leg target norm; they do not enumerate the 3^50 physical grid.
Cancellation limits their observed relative-error resolution to about 1e-7.
These results establish comparable error scales, not improved global accuracy
or a full-grid maximum-error certificate. In particular, chi=60 and chi=100 at
cap 800 have a small measured increase. Local admissibility is not a global
Frobenius tolerance guarantee. Three-pass and final histories reproduce the
full-run rank/error histories as exact prefixes in every family replay.

The conditioned chi=40 windows of 26/28 sites both converge at 4 passes, and
an additional chi=60 suffix of 33 sites converges at 7. Independent relative
Frobenius errors are 1.61e-6, 1.72e-6, and 7.38e-7 respectively.

R=10 NBlock W uses absolute tolerance 1e-4, seed 0, and cap 4096. It changes
from MaxSweeps at 20 to Converged at 8. Independently contracted 1,086 samples
satisfy the existing 10*tolerance diagnostic gate; maximum sampled absolute
error is 2.41e-4. This is bounded sampling evidence.

The archived author quantics-well Dxx inputs (16 sites, d=2) are also replayed
at cap 120, relative tolerance 1e-8, seeds 1/2/3, and default Guard. Raw and RMS
input variants both converge. Baseline passes are 6/4/14; repaired passes are
6/5/6. One dense contraction per TT yields an independent comparison over all
65,536 values: raw relative Frobenius errors change from
8.83e-9/9.43e-9/7.25e-9 to 8.69e-9/1.05e-8/9.56e-9. RMS errors agree at the
printed precision. The original supplementary report lacks its seed and full
parameters; these replays do not reproduce its reported non-convergence and
are recorded as supplementary validation, not an exact reproduction.

An additional raw-well search at seeds 0 and 4 through 12 finds convergence
on both versions in every case. Some repaired runs take more passes (seed 4:
4 to 9); this search provides no exact reproduction of the unspecified old
well seed and no general performance improvement claim.

The final public replays use an isolated release build of clean production
commit e08b4972a00782372e0e6574835154e36122c906. All 15 primary replays converge;
the six DMRG and six raw/RMS-well output-core files are byte-identical to the
prototypes independently checked above. The final W replay reproduces its
8-pass history and sampled residual. A shared-target executable whose SHA
matched the pristine baseline was rejected before replay; its invalid manifest
is retained with the local evidence. The separate build avoids mixed artifacts.

All dataset cores come from the reference repository pinned at
153b25a8aa059d0147b45955d0842b2f32fa5d1d or its archived precomputed inputs.
No RSI algorithm is executed. Numerical replays use one algorithm thread;
independent DMRG reference contractions use four BLAS threads on disjoint CPU
sets to reduce reference cost. No overall timing claim is made.

## Validation and limits

Core regressions cover all four scalar types, both factor orientations,
partial/empty preferences, invalid seeds and preferences, zero exchange
coefficients, nonfinite residual predictions, singular trials, rank preservation, protected pivots,
determinant-ratio ordering, actual residuals, and coefficient-cache reuse.
TreeACI regressions cover both edge directions and tolerance modes, all scalar
types, candidate remapping, rank reduction/zero output, one-axis completion,
and separate complete-retention/partial-restoration working-budget tiers.
Existing numerical thresholds are unchanged.

Local checks include the affected Core/TreeACI libraries, train ACI and TreeTCI
dependents, affected doctests, Clippy including production panic-path lints,
rustdoc, provider-inject compilation, API inventory, public error docs, crate
boundaries, and the repository-rules dry run. Both G0 regressions pass, including
the R=9 smaller-cut growth test; release is used for that impractically costly
unoptimized contraction workload. Hosted CI remains authoritative and is
reported separately on the pull request.

This fixes the reproduced #784 family without changing convergence criteria.
It cannot guarantee convergence for every input, monotone global error, or an
optimal preferred cross. The separate mixed-capacity Guard investigation
[#794](https://github.com/tensor4all/tensor4all-rs/issues/794) remains open.
