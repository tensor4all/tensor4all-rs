# Merge-policy acceptance comparison (#1130)

Status: approved, before measurement. The maintainer fixed the hard maximum
median paired wall-time ratio at **10.0x for each case** in the
[2026-11-19 decision](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7025360934).
This decision supersedes the earlier planning range. A ratio above 1.5x is a
soft review trigger, not a stop condition. A case above 10.0x requires redesign;
the hard limit will not be relaxed after timing results are known.

Publish the commit containing this plan, the machine-specific protocol, and
that decision URL in PR #1407 before starting any timed candidate run.

## Cases and arms

Both cases use the original #1130 seeded SplitMix64 input generator: nine
binary sites, input bond at most four, seed `0x5eed0006`, and three trailing
spectators. Partition into all 64 selected-coordinate leaves. Use `rtol=1e-8`,
`atol=0`, normalized forward transforms, exact leaf application, full maximum
output depth, and identical work/term limits in both arms.

| Case | Selected axes (zero-based site positions) | Baseline | Candidate | Maximum median paired wall-time ratio |
|---|---|---|---|---|
| Original six-qubit transform | `[0,1,2,3,4,5]` | `RankGoal(8)` | `Tolerance` | 10.0x |
| Two interleaved axes | `[0,2,4]`, `[1,3,5]` | `RankGoal(8)` | `Tolerance` | 10.0x |

The baseline and candidate are policies of the same implementation revision
and finished binary, so provider/dependency changes cannot bias the comparison.
Resolve the source commit through Git after validation and record it, the
harness source and binary SHA-256, Cargo feature/profile selection, dirty state,
hardware/affinity, and exact environment in the run metadata. The baseline's
numerical decisions are also covered by the unchanged exact/rank-goal tests.

## Protocol and gates

- Build once with `ai-run build`, release profile, platform CPU provider (faer
  on this Linux host). Time only the finished binary through `ai-run measure`.
- One CPU, with `RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` all set to `1`. Record the
  measurement CPU and keep observer work on the other CPUs.
- Use five paired repetitions per case, alternating baseline/candidate order.
  Every arm runs in a fresh process. Prime the CPU for one second with the same
  work in every process, outside the timed interval; do not warm the scheduler
  by executing additional transforms.
- Measure scheduler wall time including leaf application, excluding input and
  operator construction and the dense DFT oracle. Retain every raw sample and
  report the median paired ratio `T_tolerance / T_goal` for each case separately.
- Set each process's virtual-address limit to 5 GiB and the exclusive job's
  memory cap to 6 GiB. Report process peak RSS, including setup/priming, for both
  policies. For the original case, require tolerance peak RSS at most three
  times goal-mode peak RSS and below 5 GiB.
- Every measured cell must satisfy its reported numerical error bound (with an
  explicit `1e-9` absolute backend/DFT roundoff allowance), and the report must
  remain within the global allowance. Returned region keys must be disjoint.
  Tolerance must have one term per region at full input-merge depth and convert
  with `into_partition()`. The unchanged rank-goal baseline may retain multiple
  overlapping terms within a region; include all of them in its oracle check,
  storage count, and measurements.
- Every case must meet its pre-agreed slowdown limit; memory improvement does
  not waive a wall-time failure. No limit may be selected or relaxed after
  looking at candidate timings. Run the complete paired suite, not selected
  favorable cases. Trust only `verdict=CLEAN`; retain the gate JSON and raw
  results together, including any contaminated attempt that must be rerun under
  the machine policy.

The cap-32 Green's-function workload remains a separate diagnostic, with time,
RSS, and storage reported together if run. It is not substituted for either
case above. Functional release checks through `ai-run build` may proceed while
timing is gated, but their incidental timing/RSS is not comparison evidence.

## Harness

`benchmarks/rust/benchmark_merge_policy_1130.rs` is the executable specification:

```bash
ai-run build --mem 6G -- cargo build --release -p tensor4all-benchmarks \
  --features fourier --example benchmark_merge_policy_1130
```

The finished example takes `original|interleaved` and `goal8|tolerance`.
Without a third argument it performs functional validation and emits null
timing/RSS fields. After the gates above are satisfied, append `measure` and
run the finished binary under `ai-run measure`, the stated affinity/thread
settings, and the address-space limit. That mode primes the CPU and records
only scheduler wall time, with the dense oracle outside the timed interval.

## Registered host and execution details

[`measurement-protocol.json`](measurement-protocol.json) fixes the hardware,
source-code revision, CPU sets, environment, ordered case list, limits, and
host-noise gates before measurement. Both policies use the finished binary
built from this pre-registration commit; the runner resolves that commit through
Git and records it as both baseline and candidate. The earlier code revision is
also recorded. The runner and all Rust sources must remain committed and clean.

The measurement CPU is logical CPU 8 (physical core 4); observer processes use
CPUs 0-7 (physical cores 0-3). Each fresh process runs under its own exclusive
`ai-run measure --mem 6G --cpus 8` gate. The controller remains on observer CPUs
and holds a checkout-local lock across build and the complete paired suite.
The original agent-process affinities are recorded and restored afterwards.
Every process has a 5 GiB address-space limit. The runner builds once, copies the
finished binary to an immutable path, checks its hash throughout, and performs
no extra scheduler warmup calls.

Both case medians must be at most 10.0x. Report all five paired ratios, their
median, and a descriptive percentile bootstrap 95% interval (10,000 resamples,
seed 1130); the interval does not replace or relax the median gate. For original
case RSS, check each paired tolerance/goal ratio against 3.0 and report the
worst pair, as well as each policy's maximum process RSS.

A noise/validity failure makes the **whole paired experiment inconclusive**.
Retain the failed attempt and rerun the complete suite once under the same
protocol after the quiet gate; never replace selected samples. A second
inconclusive attempt stops the campaign. A numerical or performance failure
is retained and never retried to obtain a favorable result.

Run the committed controller from observer CPUs, with the published registration
comment URL and the commit resolved by Git:

```bash
taskset -c 0-7 python3 docs/experiments/issue-1130-implementation/run_acceptance.py \
  --expected-head <published-pre-registration-commit> \
  --registration-comment <PR-comment-URL> --output <new-scratch-directory>
```

The scratch directory must be outside the measured checkout. Copy its complete
results into the experiment directory only after measurement has ended.
