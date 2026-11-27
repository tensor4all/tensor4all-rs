- `tensor4all-treetci`: the global pivot search has a third stage (#1433).
  When the uniform walks, the retained tail-pair probes and the subtree
  restarts all find nothing, it evaluates, around the same retained tail
  pivots, every configuration of each maximal edge side with at most 32 local
  configurations, without random draws (in batches of at most 2^20 index
  entries, which does not change the selected pivots). On binary quantics
  chains that is the aligned block of width at most 32 around a pivot's value
  of one variable, so a run can no longer stop `Converged` while a residual
  above the tolerance lies in such a block in each of its three trailing
  searches. Simon's box at `R = 7`, which stopped `Converged` with a dense
  error of 0.75–0.875 on 9 of 11 seeds, is now exact on all 11. The budget is
  fixed: the width-64 box at `R = 8` is still missed, and this is not a fix of
  the zero-fiber defect class. Trees without a vertex of local dimension 1 are
  unaffected, and so are runs with `nsearch = 0` or `max_nglobal_pivot = 0`,
  where the stage does not run. On the #1153 comb, accuracy, ranks and termination are
  unchanged and the oracle points grow by 1.32% at `delta = 1` (53,153 to
  53,855) and 0.02% at `delta = 4` (1,125,824 to 1,126,042).
