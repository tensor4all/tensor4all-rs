# Index integration-binary consolidation

## Scope and lifecycle audit

Baseline: `2ce532a8336f299897ce5cef44113e768acf9757`.
The five `tensor4all-core` integration targets `common_basic`,
`common_duplicate_indices`, `common_index_ops`, `common_index_tags`, and
`common_tagset` contain 55 tests of index identity, replacement, tags, and
small strings. They contain no environment mutations, explicit global state,
backend configuration, filesystem effects, HDF5 calls, or per-binary setup.
Generated index IDs are tested for relational properties rather than exact
sequence positions; combining the harnesses does not introduce a dependency
on process initialization order.

The proposed `index_suite` retains each source file as a private module and
preserves every assertion for the build comparison. Nextest continues to run
individual cases in separate processes. Cargo's libtest runner also needs a
parallel run of the combined suite to check shared-process behavior.

## Measurement protocol

Use release optimization because the question is compilation/link performance.
Measure with unchanged warm dependencies, Rust 1.98.1 on aarch64-apple-darwin,
Cargo build jobs 2, and incremental compilation disabled. Touch only the five
test source files to force their compilation; do not clean shared artifacts.
Exclude dependency warmup, then collect one discarded harness warmup and three
retained samples per layout. Preserve Cargo timing reports so binary compilation
and build scheduling can be distinguished from test execution. This experiment
measures the changed integration targets, not whole-workspace CI latency.

The compiler wrapper is explicitly disabled (`RUSTC_WRAPPER=`), overriding the
user's global kache setting. Initial cache-served compiler measurements were
discarded. The machine has 10 logical CPUs and 24 GiB RAM; other task builds and
benchmarks were stopped during the retained samples.

| Layout | Retained wall times (seconds) | Median | Binaries |
|---|---|---:|---:|
| Separate targets | 1.545, 1.498, 1.427 | 1.498 | 5 |
| Module suite | 0.993, 0.993, 0.979 | 0.993 | 1 |

This is a 33.7% reduction for rebuilding these five targets, about half a second
on this machine. Cargo reports only the test units compiling; dependencies and
the core library remain fresh. Stable Cargo timings combine compilation and
linking here, so this is not a measurement of linker time alone. Complete
sample data and unit durations are in [the JSON record](ci769-index-suite.json).
The small local absolute saving is not extrapolated to the workspace or Linux.

All 55 original cases passed before and after grouping, including a libtest
run with four threads. The grouped sources were byte-identical to baseline and
the discovered names matched the exact original set after adding module
prefixes. After the mapped duplicate removal below, Nextest passed all 54 cases
with zero skips (0.071 seconds). Hosted full-suite and coverage validation are
recorded separately in the PR.

Use `cargo nextest run -p tensor4all-core --test index_suite` for the suite or
filter `common_index_ops::` for its module. The module prefix distinguishes
individual failures; the former `--test common_index_ops` target is replaced.

## Proven duplicate mapping

After the layout-only comparison, remove `common_basic::test_index_basic` in
favor of `common_basic::test_index_dyn`. Both call the same `new_dyn(8)`
implementation for `DynId` and assert dimension 8 and a positive generated ID.
One spells the inferred generic parameter explicitly; this does not exercise a
different runtime path. Neither has a private/shared test helper or protects a
recorded historical failure. The remaining constructor cases cover custom IDs,
layout size, tagged construction, shared tags, cloning, and index equality.
The removal changes the suite from 55 to 54 cases; no tolerance is changed.

No numerical, AD, process-global backend, CUDA, or HDF5 test is grouped in this
experiment. In particular, keep `context_foundation` separate because its
context lifecycle has different responsibilities from index/tag semantics.
