# Merge-refine parallel design and serial baseline (#1497)

## Decisions

- The [design](../design/merge-refine-parallel-execution.md) proposes leaf
  applications as the smaller first lane candidate, with preparation and the
  entire merge/refine/report tail serial on supplied engines.
- The [baseline](../experiments/1497-baseline/README.md) rejects whole-region
  fan-out as the initial merge strategy: every completed case has region width
  one, including cases dominated by merge time. Finer candidate/probe/parent
  units need a separate cost and ordering study before implementation.
- Keeping the tail serial avoids introducing an accounting event recorder
  prematurely. It also preserves #1407's evolving policy-specific ledger and
  canonical retained-work admission, which is a separate dependency from
  accuracy allocation.
- The [integration plan](../design/merge-refine-parallel-integration-plan.md)
  makes a relevant repeated need gate, consumed engine paths, determinism,
  placement and memory contracts explicit. Fixed engine teams cannot be
  borrowed opportunistically between phases.

## Verification conclusions and constraints

- All fourteen declared processes are CLEAN; seven complete and seven fail or
  time out. The failures are retained with their actual stage and resource cap.
  The IR feasibility observation does not establish a repeated timing gate.
- The temporary profiler preserves the numerical source calls and order.
  Profiling off/on produces identical reported non-time observables on the
  full-grid M6a check. This is not a coefficient-level or lane-equivalence proof.
- The diagnostic M6a reader handles the existing public lazy result instead of
  assuming `into_partition()` must succeed. Original preflight failures and
  both diagnostic patches are retained for review and reproduction.
- The reproduction helper reconstructs all fifteen recorded Rust source/lock
  files exactly in a disposable checkout; the portable IR manifest resolves
  there. Main-matrix binary/source hashes were checked before removing the
  task's instrumentation. Production Rust, root manifests and root lockfile
  are unchanged. The active task target is retained for reproducibility.
- Baseline API inventory completeness, source consistency, Python syntax,
  formatting, links and deterministic repository-rules checks are the local
  validation scope. No new lanes API, AD claim or parallel speedup is asserted;
  maintainer agreement is still required before implementation.
- The review branch includes later main updates through `f3d0df4c`. Historical
  baseline evidence remains tied to `8ce72e4e`; it is not re-timed evidence for
  those interpolation/evaluator/cache changes.
