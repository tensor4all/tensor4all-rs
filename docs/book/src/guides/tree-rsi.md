# Experimental tree RSI products

`tensor4all-treersi` approximates simultaneous Hadamard products of existing
[tree tensor networks](tree-tn.md). All inputs must have the same labeled tree
and matching physical indices at each node. Input bond dimensions may differ.
The implementation supports four real/complex scalar types and CPU execution.

Use `hadamard_many_in` with the same execution context that owns the inputs.
The convenience `hadamard_many` uses the default CPU context and requires the
`global-defaults` feature. Set `max_bond_dim` or `sketch_dim`; start with a rank
budget appropriate to the application and validate actual product error.
`rel_tol` controls local sketch pivots and is not a global error tolerance.

The example below takes a nonconstant bond-two chain, uses sketched columns,
and checks the full squared tensor against a dense oracle. Dense validation is
appropriate for this small example. Materialize each tensor once.

```rust
{{#include ../../../../crates/tensor4all-treersi/examples/validated_product.rs:2:}}
```

For larger problems, choose held-out points independently of the sketch RNG,
compute each input there using `TreeTNCachedEvaluator::evaluate_batched`, and
compare their pointwise product to the returned tensor at those same points.
Keep the reference values outside any timing loop. A sampled relative L2
error is useful evidence, but does not bound error over all physical entries.
Use a fixed acceptance threshold; reject failed calls and nonfinite values.

The edge diagnostics describe local row selection. Small `relative_pivot`,
a low output rank, or successful return does not certify accuracy. The method
can miss even a rank-one true product for legal explicit probes. See the
[derivation and counterexample](../../../design/tree-rsi.md).

Binary scales accompany contractions to handle long chains and extreme overall
magnitudes. Detected loss of representable numerical range is an error.
`max_local_elements` limits individual dense blocks; total cache and backend
scratch memory are additional. Only products are supported: there is no
arbitrary nonlinear map or reciprocal API and no automatic differentiation.
