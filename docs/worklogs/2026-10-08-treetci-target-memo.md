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

Local validation and the predeclared paired experiment are recorded when
complete. The [protocol](../../benchmarks/2026-10-08-treetci-memo-protocol.md)
fixes cases, thresholds and noise gates before candidate execution. The oracle
cost fixture is synthetic; no TreeTNCachedEvaluator throughput claim follows
from it. The complete result manifest, rounds and confidence intervals belong
under `benchmarks/results/2026-10-08-treetci-memo/`.

The root README crate map and current examples remain accurate. Python uses
`..Default::default()` for these options; the new memo is a Rust option and is
not advertised as a Python feature. No dependency, numerical test tolerance or
coverage threshold changes are introduced. Removed quanticstci scatter/dedup
paths were reviewed for shared-helper coverage impact: the existing tests and
new core batch tests cover mixed/all hits, duplicates, failure/retry, conversion
failure, inconsistent dimensions, wrong result lengths and empty batches.
