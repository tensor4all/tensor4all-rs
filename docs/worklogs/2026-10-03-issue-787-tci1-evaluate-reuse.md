# Issue #787: reuse the TensorCI1 normalized tensor train

## Decisions

- Cache the normalized tensor train as derived state on `TensorCI1`
  (`RefCell<Option<SimpleTensorTrain<T>>>`) and build it lazily. `evaluate`
  and `to_tensor_train` share the cache, so the documented pointwise API no
  longer pays `L` linear solves (`right_solve` per site) plus a full tensor
  train construction per point, which PERFORMANCE_TIPS "Per-Call Reconstruction
  Inside Loops" forbids.
- `evaluate` evaluates on the borrowed cached train instead of cloning it: a
  clone per call still cost ~5.7x the reference path on the recorded workload
  (26.6 ms versus 4.6 ms debug-profile for 200 points), even though it is
  already ~10x faster than the baseline.
- Invalidate at the dependency writes only, never at the entry of a public
  mutator: every write of `t_tensors` or `pivot_matrices` goes through the
  private accessors `t_tensors_mut` / `pivot_matrices_mut`, which clear the
  cache. Invalidating on entry would throw away a valid cache when a pivot
  request is rejected (duplicate or already-exact pivot) and would defeat reuse
  inside `add_global_pivot`, which evaluates its own probe point.
- The cache carries an explicit policy in its field documentation (owner: the
  state itself; capacity: one normalized train; clearing: dependency writes;
  memory: one copy of the site cores plus tensor-train metadata; no
  configuration), and `TensorCI1` documents that it is now `Send` but not
  `Sync`.
- Rejected alternatives: maintaining the normalized train eagerly in the
  mutators (pays `L` solves on every pivot insertion, i.e. inside the
  interpolation loop, to serve callers who may never evaluate pointwise);
  documenting `to_tensor_train()` as the only reuse path (leaves the documented
  per-point API wasteful); a separate evaluator object (would duplicate
  `to_tensor_train` for a single consumer).
- A test-only thread-local build counter (`cfg(test)`, thread-local so
  concurrent tests stay isolated) pins the reuse contract; timing assertions
  would be flaky.

## Verification conclusions and constraints

- Paired release measurement, one thread and one pinned CPU (the probe reads
  `backend.num_threads()` and reports 1), 200 distinct points on a 6-site
  rank-2 TCI1. Six baseline and six candidate runs: the warm loop median was
  59.20 us per point before and 0.69 us after (**86x**), and 200 points
  end-to-end including the one-time build fell from 11.94 ms to 0.31 ms
  (**38x**). All twelve runs printed the same digest of the 200 values
  (`0xd0acc6492ed53e3a`) and `pointwise max|diff| = 0`, so the baseline and
  candidate values agree. The candidate's warm cost was 0.54-0.99 us against a
  0.36-0.80 us prebuilt-tensor-train reference on the same runs. Conditions,
  the probe, and the per-run table are in
  [the benchmark report](../../benchmarks/results/2026-10-03-tensorci1-evaluate-reuse.md).
- Behaviour is unchanged for existing tests and doctests (`tensor4all-tensorci`
  113 tests pass, 21 of them in the `tensorci1` module). The new tests pin:
  one build across repeated evaluation and both entry points, an independent
  uncached reconstruction as the value oracle, invalidation after both
  `add_pivot` and `add_global_pivot` with exactly one rebuild, no rebuild at
  all for a rejected or duplicate pivot request, the unavailable-state error
  and index errors before and after the first build, and that a clone's pivot
  mutation changes the clone while the original keeps its own values and its
  own cache.
- Constraints and limits: `TensorCI1` is now `Send` but `!Sync` because of the
  `RefCell`;
  no in-tree code depends on that (no `TensorCI1` reference outside
  `crates/tensor4all-tensorci`, no `Send`/`Sync` bound mentions it), but
  downstream users holding a `TensorCI1` across threads would need to re-check.
  The cache doubles the memory of a state that has been evaluated pointwise.
  No multi-thread throughput or large-rank measurement was made; the recorded
  workload is small (6 sites, dim 4, rank 2).
- Issue #747 (quanticstci owned-vector cache keys) was intentionally left out
  of this change: it touches `quantics_tci.rs` and `batched/mod.rs`, the same
  files the seeded-RNG work for #796/#751 modifies, and its public
  `cachedata()` / `from_discretized(.., HashMap<Vec<usize>, V>)` surface needs
  its own design. It should be sequenced with the quanticstci RNG work.
