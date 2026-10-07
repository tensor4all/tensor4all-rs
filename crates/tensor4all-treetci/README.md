# tensor4all-treetci

Tree Tensor Cross Interpolation — a Rust port of
[TreeTCI.jl](https://github.com/tensor4all/TreeTCI.jl) by Ryo Watanabe.

Computes tensor cross interpolation on tree-structured graphs, producing TreeTN output.

## Key Types

- `crossinterpolate2()` — high-level entry point for tree TCI
- `TreeTCI2` — algorithm state
- `TreeTciGraph` — graph structure definition
- `TreeTciRunResult` — `treetn`, per-iteration `ranks` and `errors`, and `termination`
- `TreeTciOptimizationResult` — diagnostics from optimizing a caller-owned state
- `TreeTciTermination` — `Converged`, `MaxBondDimension`, or `MaxIterations`

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
