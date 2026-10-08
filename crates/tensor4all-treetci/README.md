# tensor4all-treetci

Tree Tensor Cross Interpolation — a Rust port of
[TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl) by Ryo Watanabe.

Computes tensor cross interpolation on tree-structured graphs, producing TreeTN output.

## Key Types

- `crossinterpolate2()` — high-level entry point for tree TCI
- `TreeTCI2` — algorithm state
- `TreeTciGraph` — graph structure definition
- `TreeTciInterpolator` — TreeTCI engine for the engine-independent
  `tensor4all_treetn::interpolation::TreeInterpolator` contract
- `TreeTciRunResult` — `treetn`, per-iteration `ranks` and `errors`, and `termination`
- `TreeTciOptimizationResult` — diagnostics from optimizing a caller-owned state
- `TreeTciTermination` — `Converged`, `MaxBondDimension`, or `MaxIterations`
- `TreeTciEvaluationStats` — oracle requests, successful evaluations and memo payload accounting

The entry points return named results rather than tuples. `Converged` means
the sampled stopping criterion passed; it does not certify the full-network
error. See the [stopping contract](../../docs/design/treetci-termination.md)
for the convergence window, global-search behavior, and limit precedence.

Global pivot search uses the same retained-coordinate floating-zone walk as
chain TCI: at most 100 sweeps per random start, stopping on no improvement or
an error above ten times the acceptance threshold. One cached tree evaluator
is shared across starts and coordinate scans. The walk is greedy and may
stall on flat zero fibers; sampled convergence remains a sampled criterion.

## Documentation

- [User Guide: Tree Tensor Networks](https://tensor4all.org/tensor4all-rs/guides/tree-tn.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_treetci/)

## Random streams

`PivotCandidateProposer::candidates_with_rng` and
`optimize_with_proposer_with_rng` consume caller-owned streams directly.
The optimizer shares that stream between candidate generation and global
searches; reuse it across continued calls to preserve the draw sequence.
`DefaultProposer` consumes no draws.

Seeded proposer calls use `ChaCha8Rng`. The high-level optimizer creates one
candidate stream from the proposer's seed per call and a separate global-search
stream from `TreeTciOptions::seed`. No stream is derived by hashing edges,
ranks or history length. Fixed-seed trajectories intentionally change from the
former `DefaultHasher`/`SmallRng` implementation. Custom proposers implement
`candidates_with_rng` and may override `seed` for the high-level entry point.
See the [stream contract](../../docs/design/treetci-random-streams.md).

## Continued optimization

Reuse a `TreeTCI2` and call an optimizer again with a larger `max_bond_dim`
or no cap. Keep the same function, topology and local dimensions. Current
pivots and sampled normalization scale carry over. Proposers retain the current
edge's pivots directly from `ijset`; no historical full pivot maps are copied.
Custom proposers retaining that edge's previous pivots now read `ijset`.
The former `ijset_history` field is removed.

Each call starts its own diagnostics and convergence window and skips global
search after its final iteration. Several calls therefore need not match one
longer run. Seeded calls restart RNGs; reuse a caller-owned RNG to advance its
sequence across calls. See the [continuation contract](../../docs/design/treetci-termination.md#continued-optimization).

## Optional target memoization

Set `TreeTciOptions::evaluation_cache_bytes` to `Some(256 * 1024 * 1024)`
for an expensive target whose value at each multi-index is fixed throughout
the call. The default is `None`: every request reaches the target. Callbacks
may borrow mutable state, including a `TreeTNCachedEvaluator`, and return errors.

The run owns one mixed-radix cache shared across initial pivots, edge updates,
global searches and final materialization. Successful misses are deduplicated
within each batch, evaluated in first-occurrence order and scattered back into
request order. Failed callbacks and incorrect output lengths insert nothing.
At capacity, further insertions are skipped; cached entries are not evicted.
`Some(0)` deduplicates within each batch but retains no values. `Some(usize::MAX)`
explicitly removes the practical payload bound. Packed keys support index
spaces up to 1024 bits; wider spaces remain usable with memoization disabled.

Inspect `result.evaluation`: `requested_points` includes all requests;
`evaluated_points` counts successful points passed to the target; `cache_hits`
and `cache_misses` count persistent lookups, so repeated cold-batch points
are misses even when evaluated once. `retained_bytes` counts logical keys and
values, excluding hash-table/allocator overhead; it is not a bound on RSS.
`dropped_inserts` reports budget-limited insertions. Small or cheap targets can
cost more with memoization; no speedup is assumed merely from fewer calls.

The cache is dropped when the call returns. Continued optimization starts a
fresh cache, and a later standalone `to_treetn` does not share the optimizer's
cache. Use the high-level `crossinterpolate2` to share through materialization.
