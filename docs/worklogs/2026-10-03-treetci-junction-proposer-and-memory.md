# Issues #804, #800, #801: TreeTCI at branching vertices

## Decisions

- #804: `TruncatedDefaultProposer` budgets each edge side at
  `max(d, 2) * r` (`d` the endpoint's local dimension, `r` the edge rank).
  `d * r`, as in TreeTCI.jl, equals `r` at a site-free vertex and pins the
  bond. Rejected alternatives: a budget from the summed incoming pivot-set
  sizes pins the bond again whenever the edge rank exceeds that sum, which is
  legitimate (the rank is bounded by their product); the full Kronecker count
  disables truncation at junctions, where it matters.
- #804: the sampler also keeps the previous-pass pivots of the edge and
  samples only the remaining budget. With the budget fix alone, the 3-arm
  quantics tree (site-free degree-3 root, R=10, eta=0.3, rtol 1e-4) still ran
  into the 20-iteration limit at a sampled error of 1.7e-2: at a junction a
  uniform sample of `2r` out of about `r^2` candidates drops nearly all
  previous pivots, so each update restarted from a fresh random subset.
  Keeping them alone (old budget) also failed (1.0e-2, 20 iterations). With
  both, it stops after 8 iterations at 3.3e-4, against 2.4e-4 after 5 for
  `DefaultProposer`. The site-carrying junction goes from 9.3e-3 (16
  iterations) to 4.0e-4 (7). The optimization loop records the pivot sets
  at the start of every pass and visits each edge once per pass, so the kept
  set is always the edge's current pivots: every truncating update changes
  its sample (and RNG use) relative to TreeTCI.jl, on chains and on vertices
  with sites as well. Only updates whose candidates fit the budget are
  unchanged. A truncating chain case (eta=1.2) converged in 7 instead of 9
  iterations at the same error level. The branch for a kept set larger than
  the budget is unreachable from the built-in loop (budget `>= 2r`, kept set
  `r`) and only guards hand-edited histories.
- #800 and #801 share one crate-private helper, `evaluate_points_chunked`,
  which fills and evaluates at most 65,536 points per evaluator call through
  one reused buffer. Materialization walks the pivot product with a
  mixed-radix counter instead of allocating per point and per combination.
  Chunking materialization as well (not only flattening it) keeps its buffer
  bounded at junctions, where a root tensor has `r^3` points.
- `assemble_global_point` and `assemble_points_column_major` lost their last
  production user with #801 and nothing else in the workspace (including
  `tensor4all-py` and the C API) calls them, so they were removed with their
  tests rather than kept as unused public API. Coverage of the removed paths
  was reviewed: the tests removed with them exercised only those functions;
  `TreeTciError::IndexOutOfBounds` stays covered through `graph.rs` and the
  partition checks of `fill_tensor_values`, and `OwnedGlobalIndexBatch` keeps
  its doctests (it now has no production user either and was left in place).
  The materialization test oracle assembles its points inline.
- `fill_tensor_values` keeps `TreeTciError::IndexOutOfBounds` for its site
  partition failures (out-of-range, duplicate or missing sites), as the
  per-point assembly did; these arise only from an inconsistent internal
  state. A wrong value count from the evaluator names the failing call's
  point range and the total, since one matrix or tensor may now take several
  calls.

## Verification conclusions and constraints

- Release, one thread, pinned CPU, on the 3-variable quantics tight-binding
  function (scratch harness, not committed). Before and after produce
  identical evaluation counts, per-update candidate hashes, ranks, error bits,
  pivot sets and materialized tensor bits; only the number of evaluator calls
  grows. Site-free degree-3 root, R=10, cap 128: optimization 26-27 s to
  16-17 s, candidate assembly 7.5-8.1 s to 1.7-2.0 s, all `to_treetn` calls
  4.7-5.5 s to 0.9 s, peak RSS 1877 MB to 154 MB. Cap 64: 3.3-3.5 s to
  1.8-1.9 s, 277 MB to 52 MB. Chain, cap 128: 1.4-1.5 s to 1.2-1.3 s, 81 MB to 58 MB.
- The #804 regression test uses a 6-bit-per-axis Lorentzian (eta=0.1) on a
  19-vertex three-arm tree, with global pivots disabled: on a smaller tree
  the global search alone restores full rank, so the old proposer only missed
  the iteration limit at an error of about 1e-16. Each half of the fix is
  needed at this size. With the old budget the site-free junction bonds stay
  at rank 1. Without keeping the previous pivots both junction variants stop
  at a relative dense residual of 1.7e-8 (site) and 1.1e-7 (site-free),
  against the 1e-10 tolerance; the fixed proposer reaches about 1e-14 there,
  and stayed below 4e-12 over five other proposer seeds.
- The convergence evidence for the truncated proposer covers one function
  family on three topologies; it is a sampling heuristic and can still miss
  features that `DefaultProposer` would find.
