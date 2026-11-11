# Strong pivot hints for TreeACI tolerance-boundary cycling

## Decisions

- Fix #784 within #854's remaining convergence work. Keep the original
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
same current dependencies on both arms. Results and final local validation
are recorded after the complete comparison. #854 remains open for its other
items; this change does not establish #794's mixed-capacity disposition or
attribute #896's historical timing gap. No RSI algorithm is executed.
