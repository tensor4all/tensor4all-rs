# Strong pivot hints for TreeACI tolerance-boundary cycling

**Status: partial repair.** The original DMRG family and the registered G0
cases improve, but the newly supplied R10/seed 2026 relative-tolerance workload
still reaches MaxSweeps/50. This branch does not establish complete closure
of the convergence work in #784/#854.

## Decisions

- Address #784 within #854's remaining convergence work. Keep the original
  tolerance, per-cut rank-stability window, Guard, RNG and pass limits.
- Reuse surviving cross coordinates as hints inside one threshold-complete LU
  elimination. Accept a hint only if it meets the original tolerances and is
  at least half the current complete-pivot maximum; otherwise use that maximum.
  Always stop and report residuals from the full remaining Schur complement.
- Keep numerical selection in Core and nested-sample remapping in TreeACI.
  Charge the hint buffer with the other local working storage. Empty hints
  use the existing specialization without inverse-permutation allocation.
- Do not restore #870's full-cross reconstruction, full-matrix screening,
  partial replacement trials, rank ceiling or refresh flag. The new path
  performs no extra factorization and introduces no rank floor.
- `Converged` describes local residuals, rank stability and bounded Guard
  searches. It does not certify global Frobenius or pointwise relative error.
  A high-dynamic-range seeded witness keeps held-out points and a fixed
  reference peak so the sampled normalization has an explicit meaning.

## Verification conclusions and constraints

The original converted DMRG fixture matches issue #784's SHA256. Current main
reproduces all 20 original maximum ranks and local errors. The six original
DMRG family/cap settings change from MaxSweeps/20 to Converged/6–7, with the
same default Guard and exact three-pass history prefixes. Independent TT
inner products check output error without materializing the physical grid.
No global accuracy improvement or universal convergence is implied.

The smaller seeded regression reproduces smaller-cut regrowth beneath a stable
maximum rank and now converges at the unchanged stopping conditions. Core
controls cover shapes, scalar kinds, factor orientations, weak/unavailable/used
hints, permutations, zero rank, caps, tolerances and invalid/nonfinite inputs.
Existing #874 Guard and other TreeACI regressions retain their assertions.

Representative G0 performance uses the preregistered
[paired protocol](../experiments/treeaci-784-hints-protocol.json), with the
same current dependencies on both arms. The complete corrected study
[passes](../experiments/treeaci-784-hints-results.json): the sum of six
per-case construction medians changes from 37.783s to 13.636s (ratio 0.3609).
Every candidate converges and all sampled-accuracy and host-noise gates pass.
R8 nblock is 4.9% slower; the other five cases improve. Low-T R9 nblock changes
from MaxSweeps/50 to Converged/12, and warm R9 from MaxSweeps/50 to Converged/10.
Three repetitions give descriptive medians without confidence intervals.
These are current-revision comparisons, not historical attribution for #896.

The [initial complete comparison](../experiments/treeaci-784-hints-initial.json)
is INCONCLUSIVE: both arms fail the original sample-normalized maximum-error
gate for R8. Its six cases and all three repetitions remain recorded. The
reference sample set missed the large analytic peak. The revised complete
protocol adds that independently computed peak to all 1,093 original points,
keeps both numerical thresholds and every case, and repeats both arms in full.
The analytic peak comes from the bounded 2D momentum grid at the smallest
Matsubara magnitude; this does not certify errors over the full 3D grid.

The [DMRG evidence](../experiments/treeaci-784-hints-dmrg.json) retains every
family/cap setting, histories, independent errors and final output hashes.
Chi40's relative Frobenius error changes from 3.00e-6 to 2.91e-6. Chi100/cap800
increases from 7.71e-6 to 9.28e-6, about 20.4%; local admissibility and stable
ranks do not imply better global accuracy. The other cases remain near their
baseline error scales. Final clean-source replays are byte-identical to the
independently checked output cores, including the three-pass controls.

Local validation passes the existing Core library suite (751 passed, one
existing ignored), eight focused Core hint controls, the existing TreeACI
unit/integration suites (248 passed, eleven existing ignored), the added
remapping and seeded-convergence controls, and the affected cold-G0 fixture
controls. All 436 Core/TreeACI doctests, strict changed-crate Clippy, formatting,
API inventory, crate boundaries and deterministic repository-rules review pass.
The changed production files pass the public-error-doc audit; the unscoped
audit also traverses unrelated nested worktrees and reports existing boilerplate
there. Nextest is unavailable, so Cargo's test harness runs the local suites.
Hosted CI and the scheduled heavy R9 CTTN regression remain unrun locally.

#854 remains open for its other items. This change does not establish #794's
mixed-capacity disposition, attribute #896's historical timing gap, validate
the separate NBlock W witness, or guarantee convergence for every input.
No RSI algorithm is executed.

## R10 counterexample and scope correction

The [M6a probe records and controlled follow-ups](../experiments/treeaci-784-r10-counterexample.json)
retain all four user-supplied runs and verified binary hashes. At R10,
T=0.01, tt_nblock, seed 2026, cap 4000 and 50 passes, relative tolerance 1e-4
still returns MaxSweeps/50. Absolute tolerance 0.003183 converges only at 46.
The single-run construction observations are 38.636→38.432s for relative
mode and 38.254→34.646s for absolute mode, not the earlier six-case aggregate
improvement. Their original host-noise measurements were not available, so
these observations are not a new qualified performance study.

The importance-sampled relative L2 estimates increase from 2.752e-3 to
3.542e-3 in relative mode (28.7%), and from 3.104e-3 to 3.347e-3 in absolute
mode (7.8%). These use 5,000 importance samples with reported standard errors;
they are a different validation measure from the earlier 1,094-point checks.

The relative tolerance matches the previous R9 protocol. Matched controls on
the same M6a fix binary give R9/seed 42: 12 passes; R9/seed 2026: 30 passes;
R10/seed 42: 15 passes. The seed 42 rank histories match this branch's own
harness at both R values. Thus increasing R is not a sufficient explanation:
seed sensitivity also remains. The M6a path uses three exact inputs and an
explicit session, while the original harness uses four inputs (one is zero)
and the global route; the nonbinding rank caps are 4000 and 2000 respectively.
The controls do not identify a unique root cause.

`min_sweeps` remains 2 and checks growth on every edge. A constant maximum
rank does not establish that all edge ranks are stable, nor that Guard's
window is clear. The supplied logs lack per-edge rank and Guard histories;
the R10/seed 2026 case needs those traces and surviving/accepted hint counts
before another selection-policy change. No tolerance, stopping rule, pass
limit, rank floor or Guard setting is changed in response to this report.
