# Serial phase and resource diagnostic for #1497

## Result and scope

All fourteen processes in the declared six-case matrix are retained, and all
fourteen have `verdict=CLEAN`. Seven complete the transform and emit existing
accuracy/norm observations; six IR processes across two configurations are rejected
by the producer; one M6a process reaches
application and times out at its declared wall cap. Failures remain failures.
There is no parallel candidate, speedup claim or candidate acceptance result.

The repeated M6a R=6 case spends about 11% of Fourier time in leaf application
and about 88% in merge/refine, with one incoming region job per executed level.
Whole-region fan-out cannot parallelize that bottleneck. The IR R=8/cap-64
feasibility case has four leaves, but still one incoming region job per level.
Leaf applications are independent, and leaf-only execution remains an option;
its M6a opportunity is limited and its IR opportunity needs repeats. The next
investigation is finer merge work. Probe/parent granularity remains a separate
granularity question, not something these whole-region measurements establish.

## Method

Source, case list, gates and resource caps are in [protocol.json](./protocol.json).
The numerical baseline is `8ce72e4e`; its crates, benchmark source and lockfile
match the original `379868e0` design inspection. The [manifest](./raw/manifest.json)
records source/binary SHA-256, compiler, platform and build flag observations.
The [attempt ledger](./raw/attempts.json) retains every process.
The review branch was subsequently synchronized with `62c3b606`, including
interpolation/evaluator/cache changes. These observations remain pinned to `8ce72e4e`;
they are not fresh performance evidence for the synchronized review head.

The host is an AMD Ryzen 9 6900HX, 8 physical / 16 logical CPUs, with one 16 MiB
L3 domain. Work used CPU 8, a single logical CPU; `ai-run` reserved CPUs 8–15,
and the observer was confined to 0–7. Thread environment variables are one.
Each fresh process has identical two-second CPU priming, outside its internal
Fourier timer. The entire process has an 8 GiB cgroup cap and 300-second wall
cap, including priming, input construction and checking.

The release build has debug information disabled and no target-cpu override.
Shared tensor4all kernel dependencies match the workspace lock. The copied
IR producer also brings sparse-ir 0.15's older backend and faer dependencies;
these are input-construction dependencies, not a second tensor4all transform
implementation. Its separate [lockfile](./ir-Cargo.lock) is retained.

The temporary [profiling patch](./profiling.patch) adds only scopes and
end-of-scope `/proc/self/status` observations. It preserves numerical calls and
iteration order. Nested scopes overlap: leaf time is inside prepare time and
region time inside level time. The shares below divide individual nonoverlapping
leaf/region durations by the front end's own timer, then take the median of
per-process shares. Remaining time includes construction, target scale,
initialization, report and diagnostic overhead.

Three repeats are used for M6a R=4/R=6 and both IR R=6 configurations.
M6a R=10 and IR R=8/cap-64 each have one feasibility process; neither is a
timing distribution.
No candidate runs, selective retries, omitted failures or tolerance changes
occurred. Only attempt 1 was needed; the entire matrix was CLEAN.

## Phase distribution

Here `W_t` is the number of `OutputKey` entries in the incoming level map
before transition `t`: one spectator-cell/per-axis-prefix key, containing all
its input items and terms, is one whole region job. The profiler emits one
`schedule/region` event for each loop iteration. Count those events by their
`level`; `largest_region_count` is the maximum count, `max_t W_t`.
The count is not a final output-region, leaf, term or child-probe count, nor
does "whole" mean the tensors cover the full output domain. Completed cases
are processes that finish the transform; it is not a special region type.

The [design's asserted two-bit example](../../design/merge-refine-parallel-execution.md#what-the-baseline-establishes)
has incoming widths `[1, 2]` and four final regions. It validates the
definition, without establishing useful parallel cost.

| Case | Completed / processes | Fourier seconds | Leaf share | Merge/refine share | Leaves | Maximum incoming region jobs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [m6a-r4-cap8](./raw/attempt-1/m6a-r4-cap8/0/summary.json) | 3 / 3 | 0.853 | 60.3% | 32.3% | 98 | 1 |
| [m6a-r6-cap16](./raw/attempt-1/m6a-r6-cap16/0/summary.json) | 3 / 3 | 33.316 | 11.3% | 88.3% | 118 | 1 |
| [m6a-r10-production](./raw/attempt-1/m6a-r10-production/0/summary.json) | 0 / 1 | unavailable | unavailable | unavailable | unavailable | unavailable |
| [ir-r6-none](./raw/attempt-1/ir-r6-none/0/summary.json) | 0 / 3 | unavailable | unavailable | unavailable | unavailable | unavailable |
| [ir-r6-cap32](./raw/attempt-1/ir-r6-cap32/0/summary.json) | 0 / 3 | unavailable | unavailable | unavailable | unavailable | unavailable |
| [ir-r8-cap64](./raw/attempt-1/ir-r8-cap64/0/summary.json) | 1 / 1 | 96.663 | 37.6% | 62.4% | 4 | 1 |

M6a R=4 uses the original G0 generator at T=0.1, G0 cap 8 and transform goal 16,
with full 4096-point DFT checking. R=6 uses T=0.1, G0 cap 16 and transform goal
64, with 4096 sampled points on four exact FFT
slices. These are downscaled diagnostics, not evidence of production-resolution
efficiency at T=0.01. The existing independent relative L2 errors are about
1.75e-8 and 4.94e-8 respectively, with unchanged transform/input tolerances.
The original IR R=8 case uses beta=10, L=24, four cap-64 input patches, goal 64,
rtol=1e-6 and certified share 0.5. Its existing Parseval discrepancy is 4.26e-12;
this is not a full DFT error check.

The declared need screen is a 20% median front-end share plus usable independent
work (at least two independent numerical jobs in the same dispatch wave).
Application passes on the small repeated M6a cell, not the larger repeated
one. The single IR observation cannot establish that median gate. The complete
region scheduling unit has width one in all completed cells, so even its large
time share does not pass a parallel-width screen.

## Memory observations

| Completed case | Start RSS MiB | After applications MiB | After schedule MiB | Process HWM observed in schedule MiB | Whole-process peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| [m6a-r4-cap8](./raw/attempt-1/m6a-r4-cap8/0/summary.json) | 29.0 | 40.6 | 52.4 | 52.4 | 52.3 |
| [m6a-r6-cap16](./raw/attempt-1/m6a-r6-cap16/0/summary.json) | 40.2 | 111.7 | 856.5 | 888.8 | 888.8 |
| [ir-r8-cap64](./raw/attempt-1/ir-r8-cap64/0/summary.json) | 295.8 | 609.0 | 933.9 | 1285.7 | 1285.7 |

RSS is an end-of-scope process observation. HWM includes earlier input work and
does not attribute allocations exclusively to the current phase. Whole-process
peak RSS from `/usr/bin/time` also includes priming, producer and validation;
the cgroup's separately retained `mem_peak` measures a different resource.
Kernel RSS interfaces can also differ in accounting granularity.
None of these is an exact per-job live-allocation or factor-memory bound.

One OS thread is observed at M6a boundaries and two at IR boundaries. The
launch record shows CPU-8 affinity; OS thread count is not a numerical worker
budget or evidence of parallel kernel concurrency. No lane concurrency was
measured. A serial RSS sample cannot safely choose parallel admission or prove
that several applications fit together under 8 GiB.

## Failed and incomplete cells

- **M6a R=10, T=0.01, cap 64:** input construction produces 113 patches;
  preparation reaches application, but no leaf finishes before the process
  exits 124 at the declared cap. There is no completed Fourier duration or
  accuracy result. This is a process-feasibility observation with input
  construction included, not a 300-second transform measurement.
- **IR R=6, no cap and cap 32:** all six processes reject the input at depth 5
  with `the box ... still fails at depth 5; raise R or the degree`. No Fourier
  call begins. Resolution and degree are not retuned after the failure.

These limits narrow what can be concluded. A further campaign at valid
production inputs or finer merge units requires its own declared case grid,
resources and acceptance criteria; these cells are not replaced by better ones.

## Diagnostic harness correction

The unmodified M6a harness and both profiling settings first completed the R=4
schedule, then failed at `result.into_partition()` because a valid region
retained multiple terms. All three initial processes were CLEAN. Their original
[protocol](./protocol-preflight.json) and [outputs](./validation/summary.json)
remain as preflight evidence, not successful timing cells.

The diagnostic-only [lazy reader patch](./lazy-reader.patch) preserves the
superposition and evaluates each term through a public batch API after timing.
It neither globally materializes nor recompresses the output. With the same
case and tolerances, profiling off/on then produces identical reported
accuracy, error bound, output structure and work-item trajectory; only time
fields differ. [Validation records](./validation-reader/summary.json).
This checks reported observables, not every tensor coefficient or lane behavior.

No production source or public API change from either patch is part of the PR.
The related IR probe already handles overlapping terms explicitly. The
diagnostic reader issue should be considered when interpreting older M6a
failures; the underlying Fourier result is allowed to be a superposition.

## Reproduction and next decisions

Use a disposable checkout of the baseline and the pinned `lingrui96/gw-rs`
source recorded in the protocol. [prepare.py](./prepare.py) applies the two
diagnostic patches, copies the IR source and portable manifests/lockfile, and
copies the controller into that checkout. It does not build or run anything.

```bash
git worktree add --detach ../t4a-1497-baseline 8ce72e4e
python3 docs/experiments/1497-baseline/prepare.py \
  --checkout ../t4a-1497-baseline --ir-source /path/to/pinned/gw-rs
cd ../t4a-1497-baseline
ai-run build --mem 6G -- cargo build --locked --release -p tensor4all-benchmarks \
  --features patched-fourier --example benchmark_m6a_fourier
ai-run build --mem 6G -- cargo build --locked --release \
  --manifest-path target/1497-ir-harness/Cargo.toml --target-dir target --bin ft_probe
taskset -c 0-7 python3 docs/experiments/1497-baseline/run.py
```

The IR source is not vendored; access to that pinned repository is required.
The shared-machine `ai-run` tool is external and was not versioned or hashed
during this campaign. Its original CLEAN reports are retained, but reproducing
that exact guard implementation/verdict is outside the repository's evidence.
A new run must record its installed guard version or hash before measuring.

The reviewed [portable manifest](./ir-Cargo.toml) uses local path dependencies
into the detached baseline checkout, not git dependencies on moving `main`.
Its inherited source comment was corrected after measurement. The measured
harness manifest used machine-local absolute paths; the portable copy changes
those path spellings and comments, so it does not reproduce that manifest's
recorded byte hash. It resolves the same pinned baseline crates and lockfile.
The recorded Rust sources, workspace lock, standalone IR lock, protocol,
profiling/reader patches and controller remain byte-for-byte reproducible.
Choose appropriate observer/measurement CPU sets for another host, declare
that new protocol before running, and record its different environment. Do
not format or edit instrumented sources during a run. The raw command files,
hashes and guard reports describe this host's actual execution paths.

Before implementation, review the [design](../../design/merge-refine-parallel-execution.md)
and [dependency/rollout plan](../../design/merge-refine-parallel-integration-plan.md),
especially #1376, M7 determinism and #1407's evolving retained-work admission.
The baseline supports rejecting complete-region jobs as the initial merge
strategy. It does not establish that application fan-out solves the motivating
solver's bottleneck or that finer merge parallelism is worth its extra machinery.
