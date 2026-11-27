- `tensor4all-treeaci`: the global Guard has a third stage (#1466), for the zero
  output of #1435 (mechanism 2). On a tree with a vertex of local dimension 1, a
  Guard search whose retained points and walks found nothing now also evaluates,
  around each of its own `nsearch_global_pivots` random starts, every configuration
  of each maximal edge side with 2 to 32 local configurations (TreeTCI's #1440
  bound), at most `nsearch_global_pivots * 2 (n - 1) * 32` points, and returns up to
  `max_nglobal_pivots` of them above the Guard threshold. It draws no new random
  numbers and its points count in `TreeAciDiagnostics::evaluated_points`. On the
  #1435 table (tolerance 1e-10, seeds 1-11) seeds 2, 3 and 7, which stopped
  `Converged` after 2 passes with the zero output, now stop `Converged` exact, so
  11 of 11 seeds are exact; the table costs 1.39x the oracle points (232,778
  instead of 167,501). Over the 32 x 32 XOR-mask variants of that table (11,264
  runs) the wrong results fall from 768 to 0 at 1.166x the points. The guarantee
  stays probabilistic: on that table the first two searches of a run miss with
  probability 1.4e-6 (exact over all starts; 0.20 before). Runs on trees without
  a dimension-1 vertex are unchanged: the same table with the centre absorbed into
  a binary site, the G0 and star workload sweep (213 runs) and G0 r = 5 at
  tolerance 1e-3 with `min_sweeps = 1` (seeds 1-60, three layouts) are
  bit-identical. Runs on trees with one can change their trajectory: the pinned
  #1435 seed 4 now stops `Converged` after 6 passes instead of 8, with the same
  exact tree. A stage whose evaluation chunk does not fit `max_working_bytes`
  fails with `TreeAciError::ResourceLimit` before evaluating anything.
