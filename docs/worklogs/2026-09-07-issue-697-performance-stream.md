# Issue #697 performance stream

## Session summary

Issue #697 is a routing umbrella for twelve findings from the broad
performance audit. Items 1--11 are routed to #734--#739; item 12 remains
deferred to the future variational MPO fitting work because its only production
consumer is still an unsupported stub. This work starts from the fresh
`perf/issue-697-subissues` branch at `origin/main`; the previously checked-out
TreeACI branch was already merged and is not part of this stream.

## Code and documents read

- Issue #697 and sub-issues #734--#739, including the 2026-09-06 re-verification
  and disposition comment.
- Open PR #656, which overlaps `crates/tensor4all-treetn/src/treetn/fit.rs`.
- `README.md`, `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md`, the applicable
  shared tensor4all common/Rust rules, and the generated API inventory under
  `target/api-dump/`.
- The owning backend, TreeTN fit, structured-index selection, LUCI, and
  TensorTrain norm implementations plus their existing tests/benchmarks.

## Ordering and parallelism decision

The first implementation wave is intentionally disjoint:

1. **#734**: low-risk redundant lookups/copies in the cached evaluator,
   backend matrix helpers, and binary-contraction metadata.
2. **#736**: the owned complete-pivoting LU seam and its single LUCI caller.
3. **#739**: the dedicated TensorTrain norm-squared GEMM rewrite.

These three can proceed concurrently: their production write sets are
`cached_evaluator.rs`/`matrix.rs`/`idx_tensor.rs`, `backend.rs`/LUCI dense
factorization, and `tensortrain.rs`, respectively. Every implementation agent
must benchmark and test the untouched `origin/main` baseline before editing,
then rerun the identical cases after editing.

The remaining work is sequenced:

- **#735** waits for coordination with or resolution of PR #656 because both
  change `crates/tensor4all-treetn/src/treetn/fit.rs` and its fit tests. Once
  that conflict is resolved, it is a medium mechanical cache-layout refactor
  with a required fit differential test and paired measurement.
- **#737** first performs its reachability/frequency gate. If the structured
  selection branch is exercised, it follows #734 because both edit
  `crates/tensor4all-core/src/defaults/idx_tensor.rs`; otherwise the finding is
  recorded as a cold-path deferral without a speculative rewrite.
- **#738** follows #734 because both edit `crates/tensor4all-tensorbackend/src/matrix.rs`.
  It gets a dedicated blocked-transpose benchmark before choosing the tile size
  and small-matrix threshold; a loop-nesting swap is not an accepted substitute.

Item 12 is not scheduled as an independent change. The cache representation
  should be chosen with the eventual `contract_fit` consumer under #182.

## Benchmark gate

For each performance-sensitive lane, record before editing:

- exact baseline commit (`origin/main`), source/benchmark revision, release
  profile, host/CPU and affinity;
- backend/provider and all thread settings (`RAYON_NUM_THREADS`,
  `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` where applicable);
- the complete case ladder, repetitions, correctness oracle, and every result.

Correctness is a gate, not a timing observation: the baseline and candidate
must both pass the same focused tests and value-dependent checks. Candidate
results must be measured with the same cases and settings. Noise, failed
correctness, or an incomplete case set is recorded as inconclusive; no speedup
claim is made from a favorable subset. The paired experiment and its remaining
risks will be added here before the stream is considered complete.

## Current risks and deferred decisions

- #735's implementation order depends on the open #656 branch and cannot be
  safely parallelized against it without an explicit reconciliation.
- #735 must preserve directed-entry `len()` semantics and `NodeIndex.index()`
  neighbor ordering; replacing tuple keys with nested maps is not sufficient
  by itself. Its neighbor metadata must be built from the actual fit state,
  especially if #656 changes the initializer topology.
- #737's severity is deliberately unknown until a real structured-storage
  workload is shown to reach the branch.
- #737's reachability audit found additional structured-storage constructors
  (`from_storage`, `from_structured_storage`, `from_copy_selector`, and
  `from_inner_with_axis_classes`), but current TreeTN/SRC/TreeACI fixtures are
  dense. Frequency measurement must therefore use temporary instrumentation or
  profiling and must not add counters to #734's shared files.
- #738 and #739 are algorithmic rewrites requiring differential tests and
  dedicated review; they are not drive-by cleanups.
- #738 needs a predeclared tile/threshold grid and both direct-transpose and
  end-to-end evidence; skinny, empty, scalar-type, and tile-boundary cases are
  part of the decision, not post-hoc exclusions.
- #736 changes a public backend surface and must preserve concrete errors,
  rustdoc examples, and downstream crate boundaries.

## First-wave validation

The first-wave candidate changes are currently uncommitted on
`perf/issue-697-subissues`, whose base is the fetched `origin/main` commit
`1059be68e4a068562eaffc1081aea08429ea13dd`. No push or PR was created.

Correctness and API checks completed after integration:

- `tensor4all-tensorbackend` library tests: 232 passed, 1 ignored.
- `tensor4all-core` structured tests: 9 passed; MatrixLUCI tests: 36 passed.
- `tensor4all-treetn` cached evaluator tests: 81 passed, 3 ignored.
- Full `tensor4all-itensorlike` library tests: 134 passed.
- TensorTrain inner tests: 2 passed; complex-inner tests: 3 passed;
  large-TT norm tests: 4 passed; packed backend/oracle differential test and
  single-site norm test passed.
- Core, tensorbackend, and itensorlike doc tests passed (317, 156, and 29
  tests respectively).
- API dump, `cargo fmt --all -- --check`, `git diff --check`, and focused
  clippy with the repository's required missing-doc lints passed.

The previously incomplete #734 temporary harness was rerun for the candidate
with the same release/thread settings as its recorded baseline. Median timing
screening results were:

| case | baseline | candidate |
|---|---:|---:|
| `try_from_vec2d` 128x128 | 76,053 ns | 69,902 ns |
| `try_from_vec2d` 512x512 | 2,942,548 ns | 2,359,259 ns |
| Hermitian eig 8x8 | 22,893 ns | 23,073 ns |
| Hermitian eig 16x16 | 34,836 ns | 33,583 ns |
| Hermitian eig 32x32 | 90,840 ns | 80,822 ns |
| structured binary contraction | 21,060 ns | 20,599 ns |

This is a screening result from one 30/12-repetition harness run, not a
formal speedup claim; the #734 evaluator Criterion comparison remains
inconclusive because its cold and warm results move in opposite directions
within a short noisy run. The #736 exact caller allocation check was 703 to
700 allocations with unchanged 176.29 KiB peak heap. The #739 norm ladder was
20/45/90 sites at 0.003/0.007/0.013 s baseline versus 0.003/0.005/0.011 s
candidate; its focused TT-inner benchmark is not a norm benchmark and remains
noise-sensitive.
