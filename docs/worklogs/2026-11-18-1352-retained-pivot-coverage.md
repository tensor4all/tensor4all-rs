# TreeTCI known-sample validation and measured floors

## Candidate implementation and pending decision

The final meaning of `tolerance` is **not settled**.
[Hiroshi's latest reply](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6969875677)
explicitly defers that choice until the 18-run cost table after retained-pivot
seeding is available. The reported sampled maximum must remain honest, and
`Converged` must agree with it. The local budget allocation below is an
experimental candidate to compare, not a published policy decision.

- Implement the reported-error consistency requested in
  [Hiroshi's #1371 direction](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6968020404).
  His [follow-up](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6968958483)
  confirms that optional pointwise checking cannot substitute for consistency
  between the reported error and `Converged`.
  [Simon's opt-in second-tolerance proposal](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6968797961)
  defines a different default contract and is not adopted in this patch. Its
  plane-table failure threshold is 1e-9 at a requested 1e-10; this regression
  retains the requested 1e-10 check and includes his expanded seeds 1 through 16.
  [Simon's request to compare costs after retained-pivot seeding](https://github.com/tensor4all/tensor4all-rs/issues/1371#issuecomment-6969097780)
  remains relevant: compare oracle counts before treating the local budget
  allocation as a settled policy.
- Treat #1352 and #1371 as one stopping-contract correction. A measured
  pointwise violation must remain visible even when local LU updates discard
  its pivot. Known oracle samples are replayed against every completed
  approximation, including final and capped sweeps, without querying the oracle
  again. The maximum measured residual feeds diagnostics, observers and stops.
- History alone is insufficient for the reopened table: seed 6 had a known
  residual of 2.17e-8 while reporting 4.09e-16, but seed 4's sampled points were
  accurate while unsampled points were not. Confirm current retained pairs
  across the whole tree and restart each half of unit-endpoint cuts separately
  so randomization can preserve the sparse branch. This remains a heuristic,
  not an exhaustive error certificate.
  Identical full retained points across cuts share one deterministic walk,
  deduplicated with compact integer keys without consuming extra RNG draws.
- Share the initial local absolute budget across tree edges when global
  validation is enabled; keep the pointwise stopping threshold unchanged. A
  stable measured violation gets recovery without absolute truncation, still
  subject to the existing relative numerical cutoff and bond cap. A persistent
  measured stall is `SampledErrorFloor`, not apparent convergence or an
  indistinguishable iteration limit. No calibrated tolerance multiplier is used.
- Validation owns a bounded, per-call history independently of oracle memoization.
  Compact integer keys deduplicate samples; full points and values are replay
  payload. The default accounted budget is 32 MiB, configurable through
  `validation_cache_bytes` and visible in evaluation stats. If it cannot retain
  a new sample, `ValidationBudgetExceeded` prevents an incomplete-history success.
  The history is frozen before the requested final materialization and dropped
  at return; continued optimization starts a new history.
  The named adapter records its already evaluated initialization values after
  converting their coordinates to vertex order, without querying them twice.
  The quantics options converter inherits the new field's default while keeping
  its global validation disabled.
- Inject pivots only when an iteration will continue. Floor and budget exits
  leave a swept, materializable state. Preserve the floor in the canonical
  interpolation verdict and MPI wire representation; a cap-caused floor maps
  to `BondCapReached` at the common adapter's intentionally coarser boundary.
- Preserve Simon's attached table as exact f64 bits with its zeros, nonzero
  delta plane, starting points and initial scale. The regression checks one
  dense whole-result residual per run for real and complex scalars. Source and
  attachment links are beside the fixture. The chain finder was inspected;
  this change belongs to the TreeTCI owner and its common result boundary.

## Verification conclusions and constraints

- The reopened real table's sixteen seeds and complex seeds 4/6 converge
  accurately in the optimized regressions. The existing three-branch and
  unit-site regressions remain covered. No test tolerance was relaxed.
- The 16-site #1371 Gaussian at tolerances 1e-4, 1e-6 and 1e-8 converges in four or five
  iterations for all six seeds with dense errors below tolerance. Both recorded
  and actual errors are printed by the regression.
- Default-provider TreeTCI unit and integration suites pass: 126 unit tests and
  81 integration tests; the existing negative-control test remains ignored.
- The fixed full-run chain recording deliberately changes from six iterations
  at rank eight to four iterations at rank nine. The branched recording keeps
  ranks [8, 16, 16, 16]. The numerical assertion allowance is unchanged.
- TreeTCI's 51 doctests, TreeTN's 230 doctests and QuanticsTCI's 36 doctests pass,
  along with QuanticsTCI's 55 unit tests, 16 focused
  TreeTN interpolation tests and the MPI verdict round-trip and acceptance
  policy checks. Clippy for the three affected crates, formatting, API inventory,
  public error-doc checks and the deterministic repository-rules preview pass.
  `cargo nextest` is unavailable; the crate suite used `cargo test` instead.
- Optimized #1352/#1371 regressions pass after retained-point deduplication.
  The duplicate-point test also verifies unchanged RNG consumption, while the
  earlier-pair control keeps the non-tail search path covered. Replaced helper
  tests were reviewed for coverage impact: both unit orientations, every unit
  edge, ordinary cuts and missing/empty/mismatched tables remain exercised.
- Linux uses the default faer provider with rustc 1.98.1; macOS/Accelerate is
  unavailable. Confirmation and sample replay add documented work and bounded
  storage. Iteration counts establish convergence, not performance. A common
  cost reproducer compares oracle evaluations against 501ebb4a and a version
  with retained search/history but unchanged local LU truncation. These counts
  are functional diagnostics, not elapsed-time measurements. The unadjusted-LU
  ablation still reaches the iteration limit for several seeds (all six at
  1e-8); it does not establish that seeding alone removes the cost gap. The
  [Complete comparison and provenance](artifacts/2026-11-18-issue1371/README.md)
  are now available. The baseline sums reproduce Simon's rounded 433k/634k/753k.
  With known-sample validation and retained search but unchanged local LU, the
  sums are 2,045,022 / 2,822,777 / 5,936,335. The full candidate uses
  629,702 / 841,887 / 1,064,104: 1.454x / 1.327x / 1.413x baseline at
  1e-4 / 1e-6 / 1e-8, respectively. Seeding/history alone has not removed the
  gap. The full candidate resolves these targets but still buys higher accuracy
  with additional oracle calls and higher ranks; it is not a free cost recovery.

## Validation commands

All Cargo commands ran through `ai-run build --mem 6G --jobs 2`, in the
candidate's dedicated target. The baseline cost reproducer used a separate
target. No cross-worktree integration binary was reused for these conclusions.

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --locked -p tensor4all-treetci -p tensor4all-treetn -p tensor4all-partitionedtreetn --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo test --locked -p tensor4all-treetci --lib --test integration -- --test-threads=4
cargo test --locked --release -p tensor4all-treetci --test integration issue13 -- --nocapture --test-threads=2
cargo test --locked -p tensor4all-quanticstci --lib -- --test-threads=2
cargo test --locked --doc -p tensor4all-treetci -p tensor4all-quanticstci -- --test-threads=2
cargo test --locked --doc -p tensor4all-treetci -p tensor4all-treetn -- --test-threads=2
cargo test --locked -p tensor4all-treetn interpolation -- --test-threads=2
cargo test --locked -p tensor4all-partitionedtreetn --features hataori-mpi --lib every_known_termination_crosses_the_wire
cargo test --locked -p tensor4all-partitionedtreetn --features hataori-mpi --lib measured_floors_and_validation_limits_are_not_capped_successes
cargo run --locked -p xtask --release -- api-dump
```

Lightweight checks: `git diff --check`, `scripts/check-public-error-docs.py`,
changed Markdown relative-link checks, and
`scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth`.
Hosted CI and macOS/provider checks have not run for this unpublished branch.
