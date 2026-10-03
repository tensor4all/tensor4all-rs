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
  iterations) to 4.0e-4 (7). Without history, or when the candidates fit the
  budget, the output is unchanged, so chains that do not truncate are
  unaffected; a truncating chain case (eta=1.2) converged in 7 instead of 9
  iterations at the same error level.
- #800 and #801 share one crate-private helper, `evaluate_points_chunked`,
  which fills and evaluates at most 65,536 points per evaluator call through
  one reused buffer. Materialization walks the pivot product with a
  mixed-radix counter instead of allocating per point and per combination.
  Chunking materialization as well (not only flattening it) keeps its buffer
  bounded at junctions, where a root tensor has `r^3` points.

## Verification conclusions and constraints

- Release, one thread, pinned CPU, on the 3-variable quantics tight-binding
  function (scratch harness, not committed). Before and after produce
  identical evaluation counts, per-update candidate hashes, ranks, error bits,
  pivot sets and materialized tensor bits; only the number of evaluator calls
  grows. Site-free degree-3 root, R=10, cap 128: optimization 26-27 s to
  16-17 s, candidate assembly 7.5-8.1 s to 1.7-2.0 s, all `to_treetn` calls
  4.7-5.5 s to 0.9 s, peak RSS 1877 MB to 154 MB. Cap 64: 3.3-3.5 s to
  1.8-1.9 s, 277 MB to 52 MB. Chain, cap 128: 1.4-1.5 s to 1.2-1.3 s, 81 MB to 58 MB.
- The #804 regression test uses a 4-bit-per-axis Lorentzian on a 13-vertex
  three-arm tree. The old proposer fails it on the site-free junction; the
  site-carrying junction case passes before and after, so the history
  retention is pinned by a proposer unit test instead.
- The convergence evidence for the truncated proposer covers one function
  family on three topologies; it is a sampling heuristic and can still miss
  features that `DefaultProposer` would find.
