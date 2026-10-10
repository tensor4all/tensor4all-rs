# TreeACI tolerance-boundary pivot recurrence

Tracking: [#784](https://github.com/tensor4all/tensor4all-rs/issues/784),
under [#854](https://github.com/tensor4all/tensor4all-rs/issues/854).
Base: `0e1c49ba485f95ddda54b153bb1568ee3c1654e0`. This is investigation
evidence, not a numerical repair or a performance claim. #784 remains open.

## Result

The original 50-site, physical-dimension-3 DMRG Hadamard fixture still ends
with `MaxSweeps` after 20 directional passes. It does not falsely report
`Converged`. A bounded, test-only probe at the owning local-update boundary
captured all 980 cut updates: sampled scales, complete pivot-error spectra,
matrix dimensions, and exact selected recursive component-sample keys.

All 20 public rank/error/guard histories, termination, final per-cut ranks,
and all 1,532,976 output core scalars equal the earlier `5a6d330a` baseline
exactly. The schedule-sharing candidate and probe therefore preserve this
fixture's numerical result; no tolerance or numerical decision was changed.

After pass 3, 8–19 cuts change rank per pass, with 3–10 cuts growing each
time and maximum changes of 2–5. Such growth deliberately resets the
per-cut stability window introduced by #742. Across these 223 rank-changing
updates, 185 (82.96%) have a weakest accepted pivot within 10% above the
cutoff, and 189 (84.75%) have a rejected residual within 10% below it. This
is evidence of near-cutoff turnover, not proof that a proposed hysteresis or
alternative truncation policy would preserve accuracy.

The exact ordered row and column component-sample keys on every cut recur
at passes 10/18, 11/19, and 12/20. The complete restored pivot spectra and
sampled scales also match bitwise. Candidate dimensions match for 11/19 and
12/20; some column counts differ for 10/18, before selecting the same pivots.
This establishes recurrence of selected pivot states eight passes apart,
rather than merely recurrence of ranks. It does not prove indefinite cycling:
later randomized guard searches could still discover another pivot.

## Error scales and stopping policy

Every pass has a largest local sampled magnitude near
`7.92481632187613e-7`. The enabled guard's actual threshold is therefore at
least `7.92481632187613e-14` at tolerance `1e-8` and margin 10; random starts
can only increase its scale. Independent Born-weighted sampling and
coordinate searches found maximum absolute residuals of `3.0950541e-15`
after pass 3 and `2.3207147e-15` after pass 20, below that lower bound.
These searches do not certify the maximum residual or exclude a missed
feature elsewhere, but they provide no above-threshold guard counterexample.

Independent TT inner products give relative Frobenius errors
`3.0396351641e-6` and `2.9982033537e-6` after passes 3 and 20, with about
`1e-7` resolution. Those norm-relative quantities use a different normalizer
from the peak-relative local/guard thresholds. Their ratio to `1e-8` is not
evidence of a violated guard contract. Furthermore,
`||A3-A20|| / ||A3|| = 2.3593547741e-6`: nearly flat accuracy does not mean
the approximations are identical.

TreeACI's current-error convergence gate differs from TreeTCI's trailing
error window. The ACI provenance in
[the stagnation work log](2026-09-25-aci-stagnation.md) cites Algorithm 1's
current local residual condition; the original implementation and earlier
work logs preserve that policy. It is not an established missing TCI repair
and cannot explain this fixture, whose local errors are all below tolerance.

## Reproduction and limits

The fixture is `datasets/itensor_dmrg_mps/n50_system/psi_maxdim40_n50.h5`
from [Recursive-Sketched-Interpolation](https://github.com/zmeng137/Recursive-Sketched-Interpolation/tree/153b25a8aa059d0147b45955d0842b2f32fa5d1d)
at `153b25a8aa059d0147b45955d0842b2f32fa5d1d`. Only its MPS data is used;
no RSI algorithm runs. The column-major converter and public quality driver
are recorded in #784 and [#783](https://github.com/tensor4all/tensor4all-rs/issues/783).
Use two identical inputs, f64, cap 400, relative tolerance `1e-8`, seed 1,
and the default enabled guard, minimum 2 passes, and maximum 20 passes.

For the per-edge trace, the temporary owning-module patch and probe are
published in [the follow-up evidence](https://github.com/tensor4all/tensor4all-rs/issues/784#issuecomment-6077801651).
Apply the patch to
`local_update.rs` and its private tests, install the probe as
`src/local_update/tests/quality_trace.rs`, and run the explicitly selected
`quality_784_original_dmrg_pivot_trace` test with the original converted
fixture path in `TREEACI_784_FIXTURE` and an output path in
`TREEACI_784_TRACE`. Select `--ignored --nocapture` in a release build and
set `T4A_TREEACI_USE_OWNED_LOCAL_MATMUL=1` so the private test follows the
production owned-matmul route. Set Rayon, BLAS, and OpenMP thread counts to 1.
The probe explicitly fails beyond 4,096 updates or 401 pivot errors per
update; it never silently truncates. No probe, skipped test, or trace hook
ships in the library.

The remaining decision is algorithmic: determine whether candidate/pivot
replacement can avoid the observed recurrence while preserving the error
and guard contracts. Arbitrary stagnation termination, maximum-rank-only
convergence, disabling the guard, and relaxing tolerance are not justified
by this trace. No defect is declared repaired and no closure is proposed.
