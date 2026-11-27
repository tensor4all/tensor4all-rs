- `tensor4all-treeaci`: two bonds through a vertex of local dimension 1 no longer
  cap each other (#1435, mechanism 1). A commit whose side has such a vertex, while
  that vertex's other bonds offer fewer than twice the edge's rank (the untuned
  constant of TreeTCI's #1434), also reads, for that one commit, the neighbour's
  side samples of the largest-rank other bond. On the #1435 table (three branches
  joined at a dimension-1 centre, tolerance 1e-10, 11 seeds) 8 seeds now stop
  `Converged` at the true centre ranks (12, 1, 12) with a dense relative error of at
  most 6.1e-16, where they stopped `Converged` at (2, 1, 2) to (10, 1, 10) with
  dense errors of 2.5e-8 to 0.33; the table costs 1.22x the oracle points (167,501
  instead of 137,084). Seeds 2, 3 and 7 still return the zero output, a separate
  mechanism that stays open in #1435. Runs on trees without a dimension-1 vertex
  are unchanged: the same table with the centre absorbed into a binary site and
  the G0 and star workload sweep (213 runs) are bit-identical. Runs on trees with
  one change their trajectory: in the non-heavy dense-oracle audit, 14 of 92 cases
  change, all on the `mixed-axes` geometry, with the same `Converged` terminations,
  the same or fewer passes and dense errors of the same size. The stale-residual
  re-measurement of #1451 measures the plain local matrix of the current candidate
  sets, not the expanded one. The return pass's re-commits (#1450) are planned as
  without the expansion and, being ordinary commits, expand by the same rule; on
  the #1435 table they do on the bond from the off-spine branch into the centre.
