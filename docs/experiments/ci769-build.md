# CI build reuse and runner usage evidence

## Measurement boundaries

This record covers the `CI_rs` workflow for issue #769 after #782. Job elapsed
seconds include setup, cache restore/save, commands, and artifact upload. Aggregate
runner time is the sum of individual job elapsed times, including `rollup-rs`;
it excludes queue time, the separate review workflow, and Pages deployment.
The workflow span runs from the first job start to the final job completion.
Overlapping job durations are never added to claim a wall-clock saving.

`ci-build-diagnostics.py` records each command's elapsed time, Cargo's final
build-completion timestamp, and the remaining post-build window. That window
includes execution and reporting overhead; it is not called pure test time.
Nextest's own summary reports actual suite execution separately. Cargo timings
include compile/codegen/link work; stable Cargo does not isolate linker wall time
in this record. `--timings` HTML and timestamped logs are retained as CI artifacts.

## Artifact reuse diagnosis

The instrumented, otherwise unchanged baseline is revision `5e30a970` in
[run 36306444646](https://github.com/tensor4all/tensor4all-rs/actions/runs/36306444646).
Its initial Doctests job restored cache
`v0-rust-doctest-Linux-x64-6ff13d87-454199b2` with an exact match (862 MB).
Cargo reported 186 fresh units, including `faer 0.24.4`, `faer-traits 0.24.0`,
all four `strided-*` packages, and the `tenferro-*` dependency closure. Eighteen
workspace units rebuilt. This proves actual hosted dependency reuse, beyond the
cache action's hit message.

The first dirty source was `tensor4all-tensorbackend/src/storage.rs`: Cargo's
fingerprint log explicitly compared restored dep-info time `1790497461.088085837`
with checkout source time `1790497874.277190967`. The source has no changes between
`019a74ce` and this baseline. Downstream units reported `StaleDepFingerprint`.
Fresh checkout timestamps therefore explain the observed workspace rebuild;
the observed heavy dependencies were reused. A content-derived cache key alone
would not change Cargo's source timestamp check. No source timestamp rewriting
or speculative cache reset is used.

The preceding main run `36305989737` restored the old Doctests key ending in
`ed4d29c4` with a partial match, then saved the new reviewed-lockfile key ending in
`454199b2`. Its 8m36s Doctests job and the baseline's 4m29s job therefore have
different cache conditions. Their difference is not attributed to code changes.
Historical pre-#782 logs lacked fingerprint diagnostics; this experiment cannot
retroactively prove a specific cause for every earlier dependency rebuild.

The inspected [rust-cache v2 source](https://github.com/Swatinem/rust-cache/tree/49a0bdc70d2e1b713ca9e2869b211fcce03d3c1c/src)
constructs keys from toolchain/environment/manifests, not the
`cache-workspace-crates` input. Its save action returns immediately for an exact
hit. This means a hit is neither proof that a particular artifact exists nor
proof that Cargo accepts its fingerprint. The Cargo evidence above is the
acceptance check.

## Doctest and mdBook separation

The initial instrumented baseline used Rust 1.98.1
(`48a229cea`, LLVM 22.1.8), an Ubuntu 22.04 hosted runner exposing four CPUs
(AMD EPYC 9V74), and Cargo jobs 8 for workspace rustdoc. Cargo's build finished
at 53.500s; the post-build window was 122.505s (176.005s total command time).
mdBook preparation took less than one second and the entire chapter command
30.705s. Its log begins chapter execution at 0.024s and records no Cargo build.

CI reuses the exact `book-tests` rustdoc `--extern` arguments from the preceding
workspace log and forwards HDF5 native search paths from `pkg-config`. Standalone local invocations
prepare their own verbose `book-tests` doctest log when one is not supplied.
An explicitly supplied missing log is an error, avoiding silent fallback on a
broken CI handoff. Fake-command integration tests check both paths and preserve
the exact extern paths even when diagnostic logs have timestamp prefixes.

Pages deployment now installs the pinned prebuilt mdBook, restores Rust artifacts,
and uses the same workspace-rustdoc-to-mdBook handoff. This fixes the previous
standalone caller regression and avoids the old separate `cargo rustc` probe.

## Initial cache-population observations

These are diagnostics, not a controlled candidate comparison. Main `2ce532a8`
(run `36305989737`) and instrumented baseline `5e30a970` (run `36306444646`,
attempt 1) both used the old partial-match Test/Coverage caches. Only the latter's
Doctests job already had the refreshed exact-match cache.

| Run | Test job | Coverage job | Doctests job | Aggregate CI runner time | Workflow span |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main, cache population | 19m08s | 34m41s | 8m36s | 67m49s | 34m46s |
| Instrumented baseline, attempt 1 | 18m58s | 34m10s | 4m29s | 60m38s | 34m37s |

The baseline Test fingerprint log identified `faer` as stale because a dependency
was newer than its cached output (`1789584224.717435866` versus
`1789544336.403612421`); affected `tenferro-*` units had stale dependency
fingerprints. Some build-script fingerprints also changed type. The partial
cache therefore did not represent a self-consistent fresh build of the resolved
graph. Its Test command took 981.365s through the build marker and 80.560s in
nextest, with 3,475 passing tests and 28 existing skips.

Coverage reported missing fingerprints under the correct instrumented path,
`target/llvm-cov-target/release/.fingerprint/`, including `clap`, `proc-macro2`,
and the instrumentation variants of heavy dependencies. It logged 96 Compiling
lines and rebuilt `faer`, `strided-*`, and `tenferro-*`. The old partial cache was
338 MB; artifact-path evidence does not support the external-target mismatch
previously found in tenferro. The inspected rust-cache implementation recursively
walks nested profile directories, including `llvm-cov-target/release`.

The Coverage command's build marker was at 1,737.873s, nextest took 201.808s,
and report generation after nextest took 33.573s. All 3,428 tests passed with
24 existing skips, and all 253 file thresholds passed. This particular runner
exposed four Intel Xeon Platinum 8573C CPUs, whereas the Doctests runner exposed
four AMD EPYC 9V74 CPUs. Runner model differences are retained with the evidence;
no hardware-independent speedup is inferred from unlike samples.

`compiled` and `fresh` in diagnostic JSON are observed log records, not a complete
Cargo unit inventory. Rustdoc `-vv` emits Fresh lines; nextest `-v` need not.
For nextest, the Cargo timing units and Compiling records must also be inspected;
a zero Fresh-line count alone says nothing about cache reuse.

## Warm hosted baseline

Attempt 2 of run `36306444646` reran the entire unchanged revision `5e30a970`.
All jobs passed. Test and Coverage restored exact-match keys ending in
`454199b2`. Neither logged any Compiling record for `faer`, `strided-*`, or
`tenferro-*`; timing records for `faer`, `strided-kernel`, `tenferro-cpu`, and
`tenferro-internal-cpu-kernels` all had zero build duration. Together with the
Doctests Fresh records, this establishes dependency reuse in all three lanes.

| Lane | Job seconds | Command seconds | Through Cargo build marker | Execution / post-build | CPU model (4 exposed CPUs) |
| --- | ---: | ---: | ---: | --- | --- |
| Test | 823 | 764.390 | 683.107 | nextest 80.780s; total post-build 81.283s | AMD EPYC 7763 |
| Coverage | 907 | 842.533 | 718.600 | nextest 94.225s; report 27.582s; total post-build 123.933s | AMD EPYC 9V74 |
| Doctests | 293 | rustdoc 211.942 + mdBook 36.698 | rustdoc 67.305 | rustdoc post-build 144.637s; mdBook preparation <1s | AMD EPYC 7763 |

Maintenance scripts took 107s, Lint 63s, and rollup 3s. Aggregate runner usage
was **2,196s (36m36s)** and workflow span **915s (15m15s)**. Coverage was the
longest job at 15m07s. Test passed 3,475 tests with 28 skips; Coverage passed
3,428 with 24 skips and all 253 unchanged file thresholds.

Cargo jobs were 2 in Coverage, 8 in Doctests, and the four-CPU default in Test.
Nextest used its four-CPU default; the dedicated backend thread environment
variables were unset. The pinned backend's environment-derived CPU context
uses `RAYON_NUM_THREADS` when set and otherwise available parallelism (four on
these runners). Tests with explicit contexts can choose their own settings.
This is hosted throughput configuration, separate from the controlled 1T local
fixture experiments. No single-thread backend claim is made for this suite.

The remaining Coverage build is dominated by workspace compilation, including
monomorphization and linking. The largest compiler invocations were:

| Cargo unit | Duration (seconds) |
| --- | ---: |
| treetn library tests | 82.53 |
| quanticstci library tests | 50.46 |
| treeaci library tests | 48.78 |
| tensorci library tests | 40.15 |
| core library tests | 35.26 |
| capi library tests | 30.87 |
| capi library | 28.35 |
| treetn `linsolve` integration test | 28.02 |
| quanticstci `feature_test_physicist` integration test | 27.17 |
| partitionedtt library tests | 26.28 |

The last completed compiler units were core library tests at 716.98s and
backend library tests at 709.98s. These durations overlap; their sum is not
critical-path time. Stable Cargo timings do not isolate the linker portion of
these test compiler invocations. The five small core index integration binaries
selected for consolidation totaled 5.57 compiler-unit seconds in the warm Test
lane, so that consolidation alone cannot remove minutes from whole-workflow
latency. Heavy workspace test compilation remains a possible future target;
cache hits already reuse the external dependency kernels.

## Candidate comparison protocol

The candidate includes fixture reductions, integration-binary consolidation,
and Rust 2024 for the book harness and mdBook snippets. Manifest changes alter
the cache identity, so the first candidate run can populate a new cache. Repeat
the same candidate revision after it succeeds and compare its warm run to the
warm baseline above. Preserve all thresholds and numerical tolerances, and
inspect dependency Compiling records and Cargo timings in addition to cache-hit
messages. Record CPU models and label comparisons affected by runner variation.

Final candidate revisions, raw timing summaries, aggregate runner cost, and
hosted threshold results are recorded in
[PR #785](https://github.com/tensor4all/tensor4all-rs/pull/785), alongside its CI
artifact links. This keeps the measured candidate revision fixed while adding
its final measurement results to the review record.
