# Changelog

## Unreleased — 0-indexed grid indices (breaking)

Adopts the 0-indexed convention of `quanticsgrids` 0.2.0 (tensor4all/tensor4all-rs#584):
the five `±1` conversions at the TCI boundary are gone and the public surface is
0-indexed.

**Porting note for QuanticsTCI.jl scripts: subtract 1 from grid indices at the
call boundary.**

- Interpolation callbacks (`quanticscrossinterpolate_discrete`) now receive
  0-indexed grid indices as `&[usize]` instead of 1-indexed `&[i64]`.
- `QuanticsTensorCI2::evaluate` takes 0-indexed grid indices (`&[usize]`).
- `initial_pivots` are 0-indexed `Vec<Vec<usize>>`.
- `cachedata()` keys are 0-indexed quantics indices (`Vec<usize>`).

## Unreleased - memoized targets and cache statistics (breaking)

The quantics interpolation entry points now memoize the target function on the
quantics multi-index with `tensor4all_core::MultiIndexCache` (issue #747):

- **Removed**: `QuanticsTensorCI2::cachedata()` and
  `QuanticsTensorCI2::cachedata_origcoord()`. The record-only
  `HashMap<Vec<usize>, V>` they exposed is gone; port scripts that read
  evaluation points by evaluating the returned tensor train instead.
- **Changed**: `QuanticsTensorCI2::from_discretized` / `from_inherent` take
  `CacheStats` in place of the cache `HashMap`, and
  `quanticscrossinterpolate_multicomponent` no longer reuses one grid point's
  evaluation across output components (each component run memoizes its own
  points); the callback now receives exactly `n_points * n_components` values,
  which is validated exactly instead of accepting oversized results.
- **Added**: `QuanticsTensorCI2::cache_stats()`, `num_evals()`,
  `num_cache_hits()` and `cache_hit_ratio()`, and the public `CacheStats` type.
- **Behaviour**: the target is called only for points it has not been asked for
  in the same run, with the misses of one batch evaluated in a single batched
  call. The returned values and the exit tolerance are unchanged, and the target
  must stay deterministic and batch-independent because batch composition can
  differ from a run without memoization.
- Measured on the `benchmarks/results/2026-10-04-issue-747-quantics-target-memo.md`
  cases: 74-82% fewer evaluated points and 1.8x-4.1x faster end to end (pinned
  single thread).
