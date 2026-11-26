# TCI2 matrix-valued interpolation (#1463, Step 0)

## Decisions

- Matrix indices are the two boundary sites of an ordinary chain: internal dims `[rows, d_1..d_L, cols]`, `f(i, x, j) = F(x)[i, j]`. Both entry points delegate to the private `crossinterpolate2_with_rng_linalg`, so pivot kernels, RNG use, convergence and normalization are the scalar ones and no implicit-engine site is added (the ratchet stayed at 314 sites; no `_on_session` variant).
- The callback is one infallible batched function `source(points, out)`. Layout is point-major blocks, column-major inside a block (`out[p*rows*cols + i + rows*j]`). The library allocates `out` filled with `T::default()`; the callback must overwrite every entry.
- One fit-owned default-limit `MultiIndexCache<Box<[T]>>` keyed on `x` serves initialization, sweeps, global search and finalization. No custom cache policy. Payload bytes are reported next to the table bytes because the cache limit counts table bytes only.
- Only entries the solver requests update the amplitude scale, so `tolerance` and `normalize_error` keep their endpoint-problem meaning.
- `evaluate(x)` is one whole-matrix pass with plain column-major loops (row slab, middle cores at `x`, column slab). It uses no engine, so it adds no engine site. Batches larger than the addressable size are split into several callback calls instead of failing.
- The scalar tensor train keeps its invariants (unit boundary bonds, `L+2` cores). Folding is out of scope.

## Verification conclusions and constraints

- Parity tests (f64 and Complex64, Full and Rook separately) compare the native entry point with `crossinterpolate2` on a test-local scalar endpoint adapter that supplies both `f` and `batched_f`: ranks, error history, termination, `max_sample_value` and every I/J pivot set are exactly equal, with no tolerance. The target is a non-symmetric rank-two 2x3 matrix function.
- Reconstruction is checked on every physical point against the dense target.
- Stats: `source_points` counts points passed to the callback, `entry_requests - cache_hits == source_points` holds when nothing is evicted. At-most-once evaluation holds only while points stay retained.
- Open for the maintainer: a fallible callback variant, a payload-aware cache limit in `tensor4all-core`, and an `_on_session` variant if parity with the scalar entry set is wanted.
- No timings or speedup claims in this step; benchmarks and folding follow separately.
