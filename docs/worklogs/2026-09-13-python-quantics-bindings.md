# Quantics TCI Python bindings (issue #744, step 3)

## Summary

Added `quanticscrossinterpolate` and `quanticscrossinterpolate_discrete` to
`crates/tensor4all-py`, built on the batched Rust entry points added in the
previous step. Also factored the shared evaluator-callback helpers out of the
TreeTCI binding and relaxed the `'static` bounds of the batched quantics entry
points so bindings can wrap non-`'static` closures.

## What changed

- **`src/batch.rs` (new)**: shared batched-callback helpers
  (`batch_to_numpy`, `index_batch_to_numpy`, `extract_values`,
  `unexpected_result`, `stash`, `take_error`, `panic_error`, `probe_dtype`).
  `treetci.rs` now uses them instead of its own private copies.
- **`src/quantics.rs` (new)**: `quanticscrossinterpolate(evaluate, bits, lower,
  upper, ...)` for uniform continuous grids and
  `quanticscrossinterpolate_discrete(evaluate, sizes, ...)` for integer grids,
  plus the `QuanticsTCI` result class (`evaluate`, `sum`, `integral`,
  `to_numpy`, `rank`, `shape`). The evaluator receives a C-contiguous
  `float64 (n_points, n_dims)` coordinate array (or `int64` grid indices) and
  returns `(n_points,)` values; the value dtype is fixed by a probe of the
  initial-pivot batch, exactly as in the TreeTCI binding.
- **`tensor4all-quanticstci`**: dropped the unnecessary `+ 'static` bounds from
  the batched entry points and the `pointwise_*_batch` adapters (a relaxation:
  callers that satisfied them still do). `tensor4all-treetci` never required
  them, so nothing forced the constraint.
- Tests (`tests/test_quantics.py`, 20 cases), example
  (`examples/quantics_interpolation.py`), type stubs, and a README section.

## Decisions

- **Batched callback only**, as for TreeTCI: one Python call per batch of grid
  points. The Python names drop the `_batch` suffix because the batched contract
  is the only contract exposed to Python.
- **Grid construction from Python** takes `bits` (int or per-dimension),
  `lower`/`upper` (float, broadcast, or per-dimension), and `unfolding` as a
  string; the binding builds the `DiscretizedGrid` with the public builder API.
  Discrete grids take `sizes` and reuse the crate's own validation.
- **`random_init_pivots` defaults to 0** (Rust default: 5) so that Python runs
  are reproducible; the Rust default draws from OS entropy and there is no seed
  knob exposed by `QtciOptions`.
- **`to_numpy()`** materializes grid values by calling `evaluate` for every grid
  point in C order and refuses grids above 2**22 elements. `SimpleTensorTrain::
  full_tensor()` was not used: it returns the tensor over quantics *bit* sites,
  and reshaping that back to grid indices would duplicate the grid's unfolding
  logic.
- **Dtype probing without caching** differs from the TreeTCI binding, which
  caches the probed batch: the quantics entry point adds random initial pivots
  internally (with OS entropy), so the first evaluated batch is not reproducible
  from the binding. The probe therefore costs one extra small Python call. The
  binding defaults `random_init_pivots=0`, so in practice the probe batch is the
  same batch TreeTCI evaluates first.
- **`integral()` on a discrete grid is the sum** (the crate documents step 1 for
  inherent discrete grids); the README states this rather than raising.

## Verification

- `pytest tests`: **66 passed** (26 tensor/TreeTN, 20 TreeTCI, 20 quantics).
  The quantics cases cover the batch shape/dtype contract, one-point batches,
  axis order against an asymmetric reference on an asymmetric grid, the
  unfolding-scheme invariance of grid semantics, complex values, `sum`/`integral`
  relations, rank caps, the dense-size guard, dtype-change rejection, exception
  passthrough, and the invalid-configuration matrix.
- `examples/quantics_interpolation.py` passes; the continuous 1-D run reports 43
  callback calls with batch sizes `{1, 4, 8, 10, 16, 20}`, rank 2, error
  `3.2e-16`.
- `cargo clippy -p tensor4all-quanticstci --all-targets -- -D warnings -D
  clippy::missing_errors_doc -D clippy::missing_panics_doc`, `cargo test -p
  tensor4all-quanticstci` (40+20+1+1, doc 29), `cargo clippy --workspace
  --all-targets -- -D warnings ...`, and `cargo fmt --all -- --check` all pass.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt` in
  `crates/tensor4all-py`.

## Remaining risks

- `quanticscrossinterpolate_from_arrays` (non-uniform grids) and
  multi-component quantics results are not exposed yet.
- The probe adds one extra Python call with a small batch.
- `to_numpy()` costs one `evaluate` per grid point; it is intended for tests and
  small grids, guarded by the 2**22 element cap.
- The Python crate still has no CI job; it is exercised through the commands in
  its README.
