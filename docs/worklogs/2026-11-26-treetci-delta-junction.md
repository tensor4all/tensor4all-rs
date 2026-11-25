# TreeTCI delta-junction candidates (#1436)

A branching site can have local dimension greater than one while its observed
support fixes that local coordinate. The neighboring bonds can then cap each
other below the true rank, despite nominal slack from the local dimension.
On the frozen reopened #1352 table with a binary junction, seed 5 previously
reported `Converged` with a dense relative residual of 6.120420004111999e-6
at tolerance 1e-10.

The default proposer now extends its existing unit-vertex expansion to a
branching endpoint whose side has at least two retained pivots agreeing at
the local coordinate. The same incoming-product slack threshold (fewer than
twice the current rank) and largest-rank neighboring bond select the extra
candidates. Both original and expanded candidates include every local value:
pivot agreement does not prove a support constraint, and eliminating other
values would be unsound. Dimension-one vertices keep their existing path.

The rule is deliberately limited to degree at least three for non-unit sites.
The broader prototype measured in the issue also expanded ordinary chains,
increasing comb and box oracle costs and changing unrelated pinned trajectories.
The #1153 comb and #1433 box topologies have unit branching junctions and
ordinary chains elsewhere, so every candidate path on those topologies remains
identical. This structural preservation avoids a new matched-accuracy cost
tradeoff on those workloads. No timing or speedup claim is made.

All eleven real seeds of the frozen delta-junction case converge under both
stop rules at ranks (12, 1, 12), with maximum dense relative residual
5.112028384439839e-16 (tolerance 1e-10). Complex seeds 5 and 9 exercise the
same contract, with maximum residual 1.1246462445767646e-15. The ordinary
changed-crate suite passes,
including comb, box, dimension-one, externally implemented proposer, and pinned
trajectory regressions. Candidate tests cover nonzero fixed local values,
insufficient pivot evidence, varying local coordinates, adequate slack, and
unchanged chain behavior.

This is an exploration heuristic, not exhaustive support detection or an error
certificate. It does not add support detection to non-unit chain sites or change
the global search. The analogous TreeACI value-dependent case remains tracked
in #1435; this change stays within TreeTCI.
