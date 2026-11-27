# Merge-refine parallel design and serial baseline (#1497)

## Decisions

- The [design](../design/merge-refine-parallel-execution.md) defines incoming
  region jobs as incoming `(spectator, per-axis prefix)` map entries, not final
  outputs, leaves or terms. The smallest single-root/no-spectator two-bit Exact
  fixture asserts work-item counts `[1,2,4]`, four final regions and known DFT
  values `[0.5;4]`; its diagnostic trace records incoming widths `[1,2]`.
  It validates the definition, not a performance need.
- The [original baseline](../experiments/1497-baseline/README.md) has whole-region
  width one in every completed case. Defer that executor until a relevant
  repeated workload supplies both cost and independent width. Leaf-only
  execution remains optional rather than the committed next implementation.
- The [finer diagnostic](../experiments/1497-merge-granularity/README.md) at
  integrated #1407/#1511 source `5fbd73e5` passes the parent-work need screen:
  eligible addition-parent shares are median 30.0% in R4 and 74.8% in R6,
  with maximum same-probe widths 35/38. R6 also has two axis candidates in
  one region decision. These serial costs establish no speedup or safe
  concurrent memory admission.
- The [integration plan](../design/merge-refine-parallel-integration-plan.md)
  now prioritizes designing input-parent jobs inside a single child probe.
  It spells out one-pass grouping, original sibling folds, ordered scalar and
  counter replay, canonical errors, child/admission barriers, engine ownership
  and aggregate memory. Review that contract before extracting a serial seam
  or implementing lanes. Fixed engine teams cannot borrow idle peer engines.
- #1407 merged as `9ddffd17`, including #1511's typed output-region API. All three
  merge policies are tested through both scheduler and Fourier frontend;
  selected coverage counters share one private seam, and rejected probes do
  not add selected cuts. Tolerance retains canonical `RetainedWork` admission.

## Verification conclusions and constraints

- All fourteen original-baseline processes are CLEAN; seven complete and seven fail or
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
- The finer campaign retains twelve processes: six formal matrix processes,
  three accepted controls and three initial controls. Eleven are CLEAN; the
  first counts-only tiny fixture passed numerically but was CONTAMINATED
  (4.1% foreign CPU against 3%). The entire preflight was rerun unchanged
  before the formal matrix. All six formal processes are CLEAN and complete.
- Finer off/on controls match all reported non-time fields. All parent
  item/addition counts, public addition totals and nested durations reconcile.
  `verify.py` checks controls, all frozen controller/protocol/helper/patch pins
  and source/binary hashes. Both independent patch application and generation
  reproduce all eight captured source hashes at `5fbd73e5`. Owned disposable
  checkouts are removed after verification; the active build cache is retained.
- The review branch includes later main updates through `9ddffd17`. Historical
  baseline evidence remains tied to `8ce72e4e`; it is not re-timed evidence for
  those interpolation/evaluator/cache changes.


## Cross-provider review follow-up

The measured `5fbd73e5` and merged #1407 have identical consumed crates,
benchmarks and lockfile. The plan now inherits that settled policy contract.
All completed cases retain one final region: no split was selected, and output
keys never re-join. Wider cases require selected splits or initial spectator
cells; their realistic repeated cost remains unmeasured. The synthetic test's
work-item/final-value assertions are distinguished from the direct width trace.
The IR source repository is private, limiting public reproduction.
