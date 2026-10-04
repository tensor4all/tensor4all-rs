# Issue #747: memoize the quantics TCI target with `MultiIndexCache`

## Decisions

- The rule violation was the owned-vector cache *keys*, and the quantics path's
  cache was record-only (it never looked a point up). Both are fixed by
  memoizing the target on the quantics multi-index instead of recording it:
  `site_evaluator` looks each requested point up in a per-run
  `tensor4all_core::MultiIndexCache`, converts and evaluates only the misses,
  and inserts the successful results.
- `CachedFunction` was rejected for this path. Its callbacks are
  `Fn(&[I]) -> V + Send + Sync`, with `'static` on the batch callback, and
  cannot return `Result`. Adopting it would have broken the Python binding
  (`Python<'py>` / `Rc<RefCell<..>>` captures, not `Send`, not `'static`), the
  in-crate `RefCell`-capturing pointwise adapters, and the multicomponent
  closure, and it would have forced an out-of-band error channel. `MultiIndexCache`
  keeps the encoding and the automatic key width (`u64`..`U1024`) but stores no
  callback: the driver evaluates, so failures propagate unchanged and a failed
  evaluation is never cached. Two independent pre-reviews agreed with this
  (recorded on issue #747).
- Per-point allocations were the first implementation's dominant cost: a
  `Vec<usize>` per requested point plus a linear scan to deduplicate repeated
  points inside one batch made the change 1.6x *slower* than the baseline on a
  cheap target. Reading the batch into one flat buffer and deduplicating misses
  through a `HashMap<&[usize], usize>` removed both; the same case is now 1.9x
  faster.
- The multicomponent path's coordinate-keyed
  `Rc<RefCell<HashMap<Vec<u64>, Vec<V>>>>` cache is deleted rather than
  converted: float coordinates are not a mixed-radix index space, and sharing
  one discrete-keyed cache across components needs the caller-owned cache seam
  that this change deliberately defers. Each component run now memoizes its own
  points. The lost cross-component reuse is recorded as a follow-up on #747.
- Introspection (`cachedata()`, `cachedata_origcoord()`) is removed rather than
  reimplemented. It was used only by in-crate tests and one CHANGELOG line, and
  enumerating a flat-keyed cache needs a key-decode extension; the tests now
  verify the coordinate mapping by comparing `evaluate()` against the target at
  grid points, which is the stronger oracle.

## Verification

- `cargo test -j 4 --locked -p tensor4all-quanticstci --features
  tensor4all-core/backend-tenferro`: 47 lib + 20 + 1 + 1 integration tests and
  36 doctests pass.
- CI-flag clippy, `cargo check --locked --workspace --all-targets` and
  `python3 scripts/check-public-error-docs.py` are clean.
- Measurement: `benchmarks/results/2026-10-04-issue-747-quantics-target-memo.md`.
  Pinned single thread (verified), release builds, paired repetitions,
  predeclared correctness gate (identical sample digest, sampling error, bond
  dimension and sweep count): 74-82% fewer evaluated points and 1.8x-4.1x
  faster end to end across two grid sizes and a cheap and a compute-heavy
  target.

## Deferred (recorded on issue #747)

- A caller-owned cache shared across separate interpolation calls.
- Key widths above 1024 bits.
- Cross-component target reuse in `quanticscrossinterpolate_multicomponent`.
