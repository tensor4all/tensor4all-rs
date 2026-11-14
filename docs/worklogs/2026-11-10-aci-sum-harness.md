# Reproducible ACI summation experiment harness

## Decisions

- Prepare a portable Linux experiment for #1127 using the existing batched
  elementwise API. Keep production summation API work gated on evidence.
- Compare identical original summands through ACI, both pairwise orders,
  density matrices, exact N-ary addition, and exact addition truncated once.
  Share the existing #1107 input generator with the density benchmark so the
  reproducer's streams remain consistent.
- Freeze a small tolerance/cutoff grid and accuracy thresholds before timing,
  retaining every configuration and error. Select the fastest accurate
  configuration within that declared grid, including pairwise order; do not
  infer globally optimal tuning from it.
- Require verifiable benchmark/observer CPU separation and host/provenance
  validity. The runner audits rather than reconfigures the machine. A full
  result requires all declared cells; a smoke run or an incomplete/invalid
  record cannot establish the gate.
- Treat the proposed 20% advantage and cancellation fraction as experiment
  settings for maintainer assessment. The original nonisolated observations
  remain historical and are not combined with this experiment.

## Verification conclusions and constraints

- The [benchmark instructions](../../benchmarks/README.md#aci-summation-gate-1127)
  define the portable execution path, workloads, selection and gate semantics.
- The isolated Linux performance experiment remains user-run. Local checks
  verify numerical behavior and recording/analysis logic, not a speed claim.
- Numerical checks cover all six method dispatches, exact/truncated rank
  behavior, known DFT values and bit order, original input streams,
  cancellation metrics and invalid configurations. Optimized small-case
  checks also validate each worker record against the recorder schema.
- Recorder tests cover frozen-grid completeness, accurate-arm selection,
  process cleanup/raw streams/RSS, source and external build configuration
  changes, CPU/thread/SMT placement, clock/noise observations and saved-run
  replay. Deliberate reverted faults exercise every new test.
- Runtime proof stores compact per-poll audit digests and worker placement,
  with full pre/post process snapshots. Audits still inspect every visible
  thread at every poll; trace size does not grow with a repeated full task
  list. Frequency and sibling/noise evidence are required during saved replay.
- The benchmark package tests, focused strict Clippy and optimized worker
  builds validate locally. The macOS checks use Accelerate; actual Linux
  faer execution and the complete isolated performance gate remain unrun.
