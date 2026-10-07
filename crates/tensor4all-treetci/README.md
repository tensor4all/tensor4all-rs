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

## Documentation

- [User Guide: Tree Tensor Networks](https://tensor4all.org/tensor4all-rs/guides/tree-tn.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_treetci/)
