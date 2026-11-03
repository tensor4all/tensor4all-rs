# Fixed-rank randomized factorization

Use `tensor4all_core::randomized_svd` when a dense local tensor needs only a
small number of singular triplets. `randomized_range` returns an orthonormal
basis Q instead, so a caller can explicitly insert the bond projector Q Q†.
Both support `f64` and `Complex64` and consume a caller-owned RNG directly.

The example below is also a runnable TreeTN example and a checked doctest.
It reconstructs an exactly rank-one matrix and verifies a known MPO product.

```rust
{{#include ../../../../crates/tensor4all-treetn/examples/randomized_factorization.rs:2:}}
```

`RsvdOptions::new(rank)` requests `min(rank, rows, columns)` triplets. Its
`range` field controls oversampling (default 8) and stabilized power iterations
(default 1). More power iterations cost extra GEMMs and QR factorizations.
The range finder returns `min(rank + oversampling, rows, columns)` columns;
set oversampling to zero when the projector must have exactly `rank` columns.
Index splits compare the complete index, including prime levels and tags.

`FactorizeOptions::rsvd(rank)` selects the same algorithm through `factorize`.
Choose `Canonical::Left` or `Canonical::Right` as usual. The seeded convenience
call uses `rsvd_seed` (default zero); `factorize_with_rng` consumes a supplied
stream instead. Fit and Zipup select RSVD with `FactorizeAlg::RSVD` and a
positive `max_bond_dim`. Their `randomized` and `rsvd_seed` options configure
the convenience calls. `contract_with_rng` advances one supplied stream across
initialization, all sites and every sweep.

RSVD ignores global QR/SVD truncation defaults. An explicit `rtol` or `cutoff`
policy returns `FactorizeError::UnsupportedTruncationPolicy`, with no automatic
fallback. Specify a fixed rank or select deterministic SVD for threshold
policies. Singular values of the projected matrix omit the range-finding error
and therefore do not certify the full approximation error.

Caller-owned execution contexts use `randomized_range_in`,
`factorize_with_rng_in`, and `contract_with_rng_in`. Plain held sessions use
`randomized_range_on_session` or `factorize_with_rng_on_session`; tracked or
resident inputs are refused there. TreeTN context RSVD also refuses tracked
inputs because its current execution leaves publish plain factors. The implicit CPU path preserves AD through
the backend's supported operations, with a finite-difference regression for
partial-rank RSVD.

The current Fit remains site-canonical. Environment projection requires the
future inverse-canonical parallel Fit and is not performed automatically.
RSVD still materializes the local two-site tensor; it does not implement a
matrix-free contraction.
