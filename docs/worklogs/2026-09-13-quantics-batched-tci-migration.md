# Batched quantics TCI entry points (deprecating point-wise Rust APIs)

## Summary

Made batched evaluation the primary (and documented) boundary of
`tensor4all-quanticstci`, added the missing batched entry points, and deprecated
the point-wise ones as thin wrappers over the batched path. Migrated the crate
docs, doctests, tests, the tutorial-code workspace member, and the relevant
mdBook pages to the batched API.

## Motivation

The point-wise entry points (`Fn(&[f64]) -> V`, `Fn(&[usize]) -> V`) are the
easiest thing for a language binding or an AI assistant to reach for, but they
call the target once per point, which is the wrong boundary for vectorized
functions and impossible to wrap efficiently from Python. `tensor4all-treetci`
and `tensor4all-treeaci` already use column-major batch views
(`GlobalIndexBatch`, `TreeElementwiseBatch`), so quantics TCI was the outlier.

## What changed

- **`src/batch.rs` (new)**: `QuanticsBatch<'a, T>` — a borrowed column-major
  `(n_dims, n_points)` view (point `p` is the contiguous block
  `data()[p * n_dims..][..n_dims]`), plus `pointwise_coordinate_batch`,
  `pointwise_index_batch`, and `pointwise_components_batch` adapters that turn a
  scalar/point-wise closure into a batched evaluator with a per-coordinate cache.
- **New batched entry points**: `quanticscrossinterpolate_batch`,
  `quanticscrossinterpolate_discrete_batch`,
  `quanticscrossinterpolate_from_arrays_batch`, and
  `quanticscrossinterpolate_multicomponent`.
- **`quanticscrossinterpolate_batched` → `quanticscrossinterpolate_multicomponent`**:
  the old name meant "several output components", not "a batch of points", and
  actively misled. The old name is kept as a deprecated alias.
- **Deprecated wrappers**: the three point-wise scalar entry points and the
  point-wise multi-component entry point now delegate to the batched path (via
  the adapters), so there is a single interpolation implementation. Only the
  per-point cache for the point-wise form moved into the adapters.
- **One interpolation body**: `run_treetci_batch` + `site_evaluator` replace the
  previous per-function `batch_eval` closures; the multi-component path now uses
  `quanticscrossinterpolate_batch` per component instead of a point-wise scalar
  call, so its `callback_error`/`Arc<Mutex>` machinery is gone.
- **Docs**: crate module docs, `README.md`, `prelude.rs`, the `QuanticsTensorCI2`
  struct examples, `docs/book/src/**` snippets, and `docs/tutorial-code/src/**`
  now use the batched API with the adapters.

## Decisions and rejected alternatives

- **Deprecate, do not delete.** The point-wise form remains available (with the
  same one-evaluation-per-point cache) so existing callers keep working while
  rustdoc and `-D warnings` steer new code to the batched path. Re-exporting the
  deprecated names is annotated with `#[allow(deprecated)]` so the warning fires
  at call sites, not at our own re-export.
- **Batch layout mirrors treetci/treeaci** (column-major `(n_dims, n_points)`).
  A C-contiguous `(n_points, n_dims)` NumPy array is memory-identical, so
  bindings need no transpose.
- **Adapters instead of a point-wise entry point per call.** `pointwise_*_batch`
  keep Rust ergonomics for scalar closures without a second interpolation code
  path.
- **`tensor4all-tensorci` (legacy TCI1/TCI2) is out of scope.** Its point-wise
  `f` is not a signature detail: it is called inside the optimizer in ~10 places
  with an *optional* batch path, so making it batch-only is an algorithm rewrite
  rather than an API migration. Its successor is `tensor4all-treetci` (already
  batch-only); deprecating the legacy crate in favour of it is a separate
  decision.
- **Legacy `docs/tutorial-code/docs/tutorials/*.md`** were left alone: the
  repository declares them not the online source of truth, nothing compiles or
  compares them, and `scripts/refresh-tutorial-artifacts.sh` does not regenerate
  them.

## Verification

- `cargo clippy -p tensor4all-quanticstci --all-targets -- -D warnings -D
  clippy::missing_errors_doc -D clippy::missing_panics_doc`: clean (this is what
  proves no internal caller still uses a deprecated item without an explicit
  `#[allow(deprecated)]`).
- `cargo test -p tensor4all-quanticstci`: 40 unit + 20 + 1 + 1 integration pass;
  `cargo test -p tensor4all-quanticstci --doc`: 29 pass.
- `cargo clippy --manifest-path docs/tutorial-code/Cargo.toml --all-targets --
  -D warnings`: clean.
- `cargo test --doc -p book-tests`: **47 passed, 0 failed**. An earlier run had
  21 link-time `lld`/LLVM crashes for unrelated snippets as well
  (`tensor_basics`, `compress`); those were caused by the machine being at 100%
  disk usage. After freeing build artifacts the whole book doctest suite passes,
  including every quantics snippet migrated here.
- New tests: `site_evaluator` conversion-failure/number-of-values/caching paths,
  and an `#[allow(deprecated)]` equivalence test asserting the deprecated
  point-wise entry point produces the same ranks, errors, and values as the
  batched one.

## Remaining risks

- `pointwise_*_batch` caches by coordinate, so `f64` NaN payloads collapse to one
  cache entry per bit pattern (irrelevant for interpolation targets).
- The multi-component batched contract accepts at least `product(output_dims)`
  values per point and ignores extra ones; fewer, or a non-divisible count, is an
  error.
- `tensor4all-tensorci` still exposes point-wise entry points.
