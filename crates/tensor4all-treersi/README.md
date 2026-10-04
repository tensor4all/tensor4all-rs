# tensor4all-treersi

Experimental, CPU-only recursive sketched interpolation (RSI) for simultaneous
Hadamard products of compatible `TreeTN<IdxTensor, V>` inputs. Supports `f32`,
`f64`, `Complex32`, and `Complex64`, explicit execution contexts, caller-owned
RNGs, and supplied column-major probes. RSI has no ACI dependency, including
build and development dependencies; the comparison executable lives separately
in `benchmarks/tree-rsi`.

Start with the [guide](../../docs/book/src/guides/tree-rsi.md) and the runnable
[validated product example](examples/validated_product.rs):

```sh
cargo run -p tensor4all-treersi --example validated_product
```

Use `hadamard_many_in` for an explicit CPU context, or `hadamard_many` with the
`global-defaults` feature (enabled by default). Supply `max_bond_dim` or
`sketch_dim`. `rel_tol` controls local sketch pivots, **not output accuracy**.
The result can be inaccurate even when the true product has low rank. A
regression test preserves an explicit rank-one counterexample.

Binary scales accompany contractions, avoiding loss of the overall magnitude
on long chains. An error is returned when detected loss of floating-point
range prevents faithful block normalization or final output conversion. This
does not remove cancellation or guarantee arbitrary dynamic range. The block
allocation limit is per block; total caches and backend workspace are extra.

The mathematical starting point is Meng et al.,
[Recursive Sketched Interpolation](https://arxiv.org/abs/2602.17974v1).
The [tree derivation and implementation correspondence](../../docs/design/tree-rsi.md)
separate paper steps, repository extensions, and conditions needed for accuracy.
Arbitrary nonlinear maps, reciprocal solvers, and automatic differentiation are
not supported. This crate does not certify a downstream GW calculation.
