# TreeTCI Python bindings (issue #744, step 2)

## Summary

Added `crossinterpolate` to `crates/tensor4all-py`: a batched Python-evaluator
entry point for discrete tree tensor cross interpolation, built directly on
`tensor4all_treetci::crossinterpolate2`. Includes tests, an example, a type
stub entry, and a README section.

## Context read

- `crates/tensor4all-treetci/src/api.rs` (`crossinterpolate2` signature and the
  initial-pivot probe / `max_sample_value` bootstrap).
- `crates/tensor4all-treetci/src/batch.rs` (`GlobalIndexBatch` is column-major
  `(n_sites, n_points)`, `data[site + n_sites * point]`, one contiguous
  `n_sites` block per point) and the four evaluator call sites
  (`api.rs:130`, `update.rs:232`, `materialize.rs:254`, `globalpivot.rs:141`).
- `crates/tensor4all-treetci/src/optimize.rs` (`TreeTciOptions` defaults:
  `tolerance 1e-8`, `max_iter 20`, `max_bond_dim None`, global pivot search on
  with `nsearch 5`, `seed None`).
- `crates/tensor4all-treetci/src/materialize.rs` (`to_treetn` returns
  `TreeTN<IdxTensor, usize>` with node names = site numbers and the site leg
  first, so the existing `TreeTensorNetwork` wrapper is the result type).
- `crates/tensor4all-quanticstci/src/quantics_tci.rs` (point-wise
  `f(&[f64]) -> V`, with the batch adapter and cache held internally).
- Absence of rayon/threads in `treetci`/`quanticstci`, so a Python callback
  runs on the calling thread with the GIL held.

## Decisions

- **Batch-only evaluator.** All four evaluator call sites are batched (largest
  is a site-tensor fill, `rank_in x rank_out x dim`, order 10^3-10^4 points), so
  a per-point Python API would multiply interpreter overhead by the batch size.
  No point-wise adapter is provided.
- **Layout**: Python sees a C-contiguous `(n_points, n_sites)` int64 array. This
  is the same bytes as Rust's `(n_sites, n_points)` column-major buffer, so the
  batch axis is axis 0 and the site axis is axis 1 with no transpose and a
  single copy. The batch axis is always present (`n_points == 1` included) and
  the result must be 1-D `(n_points,)`. `local_dims`, `edges`, and
  `initial_pivots` stay plain Python sequences, so the layout rule lives in
  exactly one place.
- **Value type from the initial-pivot probe.** The binding evaluates the
  initial-pivot batch itself, decides `float64` vs `Complex64`, caches those
  values, and serves them to TreeTCI on its first call (matched by comparing the
  raw buffer, with a fall-through to a fresh Python call if the assumption ever
  breaks). This gives one Python call per batch, no restarted run, and a dtype
  that cannot change mid-run: a later mismatch is a `TypeError` rather than a
  silently truncated real approximation.
- **Error policy**: Python exceptions raised by `evaluate` are stashed and
  re-raised unchanged after `crossinterpolate2` returns its own error; shape,
  dtype, and length violations raise `TypeError`/`ValueError`; non-convergence is
  not an error (inspect `ranks`/`errors`); Rust panics inside TreeTCI are caught
  with `catch_unwind` and re-raised as `RuntimeError` (`pyo3`'s `PanicException`
  derives from `BaseException`, which `except Exception` does not catch).
- **Exposed options**: `tolerance`, `max_iter`, `max_bond_dim`, `seed`.
  `center_site`, `enable_global_pivots`/`nsearch`/`max_nglobal_pivot`, and
  proposer selection are deferred; adding them later is not breaking.
- **Scope**: discrete TreeTCI only. The Rust quantics entry points are
  point-wise, so a batched quantics contract needs a batched entry point in
  `tensor4all-quanticstci` rather than an adapter here.

## Alternatives rejected

- A point-wise Python callback (one Python call per point): defeats the purpose
  and would need per-point batching logic in the binding.
- Passing the batch as an F-contiguous `(n_sites, n_points)` view to avoid the
  int64 copy: same memory, but reverses the axis meanings and invites transposes.
- A zero-copy numpy view over the Rust batch buffer: the borrow does not outlive
  the call, the callback could mutate TreeTCI's internal buffer, and it needs
  `unsafe`.
- Restarting TreeTCI as complex when a later call returns `complex128`: wastes a
  run and permits a dtype that silently changes mid-run.
- Accepting array-likes for the result (`np.asarray` coercion): would silently
  accept `int64`/`float32` and, worse, `complex128` on the real path.

## Verification

- `cargo clippy --all-targets` and `cargo fmt --check` clean.
- `pytest tests` 46 passed (26 previous + 20 new). New coverage: batch shape and
  dtype on every call, single-point batch keeps its axis, points-are-rows
  mapping, explicit chain equals the default, non-chain star graph against an
  einsum-style reference, complex values, real stays real, evaluator exceptions
  propagate unchanged (first and later call), wrong length/dtype/non-array/2-D/
  scalar results, dtype change after the first call, zero initial pivot, pivot
  shape validation, invalid graph/options, seed reproducibility, `max_bond_dim`
  cap.
- `python examples/cross_interpolation.py` passes; the reported run makes 36
  callback calls with batch sizes `{1, 4, 6, 8, 12, 16, 24, 45}` and reaches
  error `3.3e-17` on `f(i,j,k) = i + 10j + 100k` with `dims = [2, 3, 4]`, which
  is the axis-order regression check.
- `scripts/check-crate-boundaries.py`, `scripts/check-public-error-docs.py`
  (no findings for this crate), and `scripts/repository-rules-review.py --base
  main --worktree --dry-run` pass.

## Remaining risks

- A Python callback holds the GIL for the whole interpolation, so no other
  Python thread runs during it. This is inherent while the callback is Python;
  TreeTCI itself never crosses threads.
- The cached probe batch assumes TreeTCI evaluates the initial pivots first. If
  that ever changed, the cache would miss and the callback would simply be
  called once more; correctness does not depend on it.
- `seed=None` is the Rust default, so runs are not reproducible unless a seed is
  passed.
- Quantics and the remaining TCI options are out of scope, as is CI wiring for
  this crate.
