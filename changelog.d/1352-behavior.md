- `tensor4all-treetci`: `DefaultProposer` (and through it
  `TruncatedDefaultProposer`) expands a vertex of local dimension 1 whose
  candidates are locked (#1352). Such a vertex has no index to expand, so its
  side of an edge offered only the product of its other bonds' pivots, and
  two bonds through it could cap each other below the true rank. A frozen
  three-branch table then stopped `Converged` at tolerance 1e-10 with a dense
  error of 6.3e-10 (seed 4) and 2.5e-8 (seed 6). When that product is below
  twice the edge's current rank, the largest-rank other bond is now replaced
  by its neighbour's own candidates, expanded over the neighbour's local
  index. Above that threshold nothing changes. On the #1153 comb the
  oracle-point cost at matched accuracy is 0.88–1.07x the previous one, and
  the reopened table needs about 25% fewer points. Trajectories on trees with
  such a vertex change, including the pinned #1153 junction ranks.
