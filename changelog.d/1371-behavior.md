- `tensor4all-treetci`: under `StopRule::Sampled`, the local rrLU truncation
  threshold of each edge update now tightens when the recorded error fails
  the tolerance (#1371). The default `StopRule::PerBond` is unaffected: its
  factor stays exactly 1, `LocalThresholdFloor` is never reported, and its runs
  are those before this change (its recorded error is the bond error alone,
  which a `max_bond_dim` cap can keep above `tolerance`, so tightening on it
  would change capped runs). The pinned `PerBond` runs match `main` in
  termination, ranks and points exactly and in the recorded errors within
  1e-9 relative; they were bit-identical in local measurements on Linux
  x86_64. Under `Sampled`, local
  truncation, global-search acceptance and the stop all used
  `tolerance * max_sample_value`; a point whose error is spread over several
  bonds sat above `tolerance` while every per-bond residual was below it, so
  the sweeps dropped it every time and the run stopped only when three random
  searches happened to miss it. The local threshold now starts at that value,
  is halved after each iteration whose recorded error is not below
  `tolerance`, never loosens, and never drops below `1 / E` of it (E tree
  edges). The search acceptance and the stop tolerance are unchanged. A
  `Sampled` run whose recorded error never reaches the tolerance takes the
  same path as before. `TreeTciInterpolator` always runs `Sampled`, so the
  change applies there too. New: `TreeTciTermination::LocalThresholdFloor`
  (`Sampled` only), reported when three trailing sweeps at the `1 / E` floor
  all record an error at or above the tolerance (instead of continuing to
  `MaxIterations`; it maps to `InterpolationTermination::IterationLimit`). It
  is armed only for trees with at least two edges: with one edge or none the
  floor would be 1, the threshold cannot descend, and such runs end
  `MaxIterations` or `Converged` as before. Also new is
  `local_threshold_factor` on `TreeTciOptimizationResult` and
  `TreeTciRunResult`, the factor the last sweep applied (`1.0` under
  `PerBond` and for trees with fewer than two edges); see "Breaking Rust API
  changes". On the #1371 reproducer under `Sampled` (16-site chain, seeds
  1-6, `max_iter` 30) all 18 runs end `Converged` in 4-6 iterations (before:
  5-30, two `MaxIterations`) with a dense error of 0.22-0.50x the tolerance
  (before: up to 2.04x), at final factor 1/4 (one run 1/2); the oracle points
  sum to 540,563 / 647,757 / 892,792 at 1e-4 / 1e-6 / 1e-8, 0.40x / 0.30x /
  0.54x the previous `Sampled` cost and 1.25x / 1.02x / 1.19x that of
  `PerBond` (433,152 / 634,331 / 753,077, as at `501ebb4a`). On a 60-site
  slow-decay comb at 1e-8 (seeds 1-3) the `Sampled` runs end `Converged` in
  5 / 5 / 8 iterations with 7.6M / 7.3M / 14.9M points (before: `MaxIterations`
  at 30 with 59.9M / 59.3M / 61.8M; `PerBond`: 6.4M / 6.1M / 8.4M, so 1.19x /
  1.19x / 1.78x) and a sampled error of 0.39-0.57x the tolerance (before:
  1.26-2.28x). Trajectories of `Sampled` runs whose recorded error exceeds the
  tolerance in some iteration change, including the pinned 16-site chain in
  `full_runs_reproduce_recorded_ranks_and_errors` (4 iterations and 13,754
  points instead of 6 and 22,073; `PerBond` takes 4 and 12,720).
