# TreeTCI bounded target memo (#802)

The run now accepts a borrowed `FnMut` oracle and optionally owns one bounded
mixed-radix target memo across initialization, updates, global searches and
final materialization. Memoization is disabled by default, because suppressing
calls changes stateful callback semantics and can cost more for cheap targets.
A continued optimizer call starts a new memo; a separately called `to_treetn`
does not reuse it. Logical payload accounting is not process RSS.

Batch miss deduplication lives in core's `MultiIndexCache::evaluate_batched`.
TreeTCI and quanticstci share that owner-level seam rather than maintaining two
implementations. Persistent and temporary dedup keys use the existing compact
integer machinery, never owned index-vector keys. Successful values preserve
request order and exact bits. Callback failure, invalid coordinates and wrong
output length retain no partial entries. At capacity, inserts are skipped and
reported. The default preserves phase-specific uncached length diagnostics.

The regression suite covers real/complex, both precisions, chain/branched
shapes, unit junction axes, zero/full/partial payload budgets, failure/retry,
extended keys through 1024 bits and a borrowed mutable TreeTN evaluator.
Public edge updates and materialization also cross the 65,536-point boundary.

## Previously implemented issues

#800 and the bounded assembly portion of #801 were implemented in merged
PR #808 (`3bc704e3`). Their existing point-order, chunk-size and bit-identity
regressions remain applicable. This branch adds public default-boundary
regressions; repeated target evaluations are handled by the opt-in #802 memo.
Historical #808 benchmark observations are existing evidence, not a new
measurement performed in this work log.

## Validation and experiment

The complete core suite and affected TreeTCI/quanticstci suites passed; the
new public-boundary/memo tests passed for all four scalar kinds. Relevant core
cache doctests (41), all TreeTCI doctests (35) and quanticstci doctests (36)
passed. Strict changed-crate Clippy, library panic audit (zero new/stale
findings) and deterministic repository-rules preview passed. Nextest is
unavailable, so changed-crate suites used `cargo test --lib --tests`.

The [predeclared paired experiment](../../benchmarks/2026-10-08-treetci-memo-protocol.md)
passed all gates: the expensive-oracle primary time ratio was 0.2925 (95% CI
0.2913–0.2942), about 3.42x faster, with 78–89% fewer actual target evaluations.
Default-disabled paths were unchanged within measurement uncertainty.
Cheap-oracle memo cases cost 2–5% more; opt-in behavior remains appropriate.
All declared trajectory/sample signatures and memory/noise gates passed.
[Complete results](../../benchmarks/results/2026-10-08-treetci-memo/README.md)
retain all cases, confidence intervals, manifests and host observations.
The oracle cost fixture is synthetic; no TreeTNCachedEvaluator throughput
claim follows from it. No cases were selectively retried or omitted.

Both burn worktrees initially shared a target. Switching package sources
between worktrees exposed stale Cargo API artifacts despite the build lock;
source timestamps were invalidated and the affected checks rerun. #849 now
owns an isolated target with copied reusable release dependencies. Benchmark
binaries were copied immutably and no owned build/test ran during measurement.

The root README crate map and current examples remain accurate. Python uses
`..Default::default()` for these options; the new memo is a Rust option and is
not advertised as a Python feature. No dependency, numerical test tolerance or
coverage threshold changes are introduced. Removed quanticstci scatter/dedup
paths were reviewed for shared-helper coverage impact: the existing tests and
new core batch tests cover mixed/all hits, duplicates, failure/retry, conversion
failure, inconsistent dimensions, wrong result lengths and empty batches.
