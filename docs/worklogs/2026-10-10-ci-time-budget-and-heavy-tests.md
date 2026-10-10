# CI time budget and heavy tests

## Decisions

- One total budget for the whole `CI_rs` pipeline, enforced once in the required
  `rollup-rs` gate, instead of per-job caps: the maintainer asked for a bound on the
  pipeline as a whole. The span is this attempt's first job start to its last job
  completion, which keeps overlapping jobs from being summed and keeps the gate's own
  scheduling wait out of the measurement. It is a completion gate, not a cancellation.
- A separate, tighter budget on suite execution (`--budget-seconds 300` in
  `scripts/ci-build-diagnostics.py`), because the total budget alone only reacts to a
  regression large enough to move the pipeline span.
- The r=9 low-temperature witness moved to `tests/heavy_g0_convergence.rs` marked
  `#[ignore]` and runs from a nightly workflow, rather than being repaired or dropped:
  its cost is real algorithm work, not a fixture accident, and it is a regression
  witness for the tolerance-boundary behaviour that #741 and #870 fixed.
- Example and bench targets left the default suite in the Test and Coverage jobs
  (`--lib --bins --tests`). They add no test execution but a quarter of the compile CPU;
  `cargo clippy --workspace --all-targets` per pull request and the nightly
  `--all-targets` build keep them compiling.
- kache (`kunobi-ninja/kache-action`) now restores compiled workspace artifacts. The
  previous cache could not: a fresh checkout makes unchanged sources newer than the
  restored dep-info, so workspace crates were rebuilt on every run (the dependency
  closure was already fresh), and the recorded policy forbids rewriting timestamps to
  make the hit indicator green. A content-addressed store sidesteps Cargo's timestamp
  check instead of fighting it.
- The scheduled workflow files or updates one rolling `ci-nightly` issue rather than a
  new issue per failing night, and reports from a separate job so a run that dies on the
  workflow's own timeout is still reported.

## Verification conclusions and constraints

- `CI_rs` run 38001393991 measured the regression: the r=9 test took 51.4 s before #870
  and 1369.9 s after it in the `ci` profile, and 5695.8 s of the 5759 s coverage test
  phase; the Test job's build stayed at 796 s (21 workspace crates compiled, all
  dependency artifacts fresh). The rest of the suite is 79 s.
- #875, merged while this change was in review, removed the cost that motivated the
  separation: the same witness now takes 30.8 s in the `ci` profile and 66.5 s under
  release instrumentation, and hosted `main` went from 36m27s to 17m39s (Test) and from
  112 min to 22m51s (Coverage). The nightly home is therefore a policy choice, not a
  necessity: keeping the witness in the default suite is defensible, and the 300 s suite
  budget would catch a recurrence (79 s + 1370 s = 1449 s). Reverting the move is a
  three-line change if the maintainer prefers the PR-time signal.
- kache on the first hosted runs: the cold build kept its previous cost (877 s) and
  saved a 1 GiB store; a re-run of the same job then restored compiled artifacts in
  12 s with a 100% hit rate over its 21 workspace crates, so the job wall clock fell from
  17m33s to 2m21s. A *new* push on the same branch still hit only 1.8% of 168 units
  (build 612 s, job 12m24s): the missing units are test binaries, which the default
  `cache-executables: false` leaves out, and kache's own report names them. Raising that
  input or investigating the misses with `kache why-miss` is the follow-up, not a change
  to make blind.
- Cache accounting: the repository now carries ~11.3 GiB of GitHub Actions cache
  (rust-cache entries plus the two kache stores), above the 10 GiB allowance, so eviction
  of the oldest entries is expected. Dropping `cache-workspace-crates` in the two kache
  jobs removed the part that was never reuse-eligible; watch the store sizes in the job
  summaries before adding a third store for the scheduled workflow.
- The measured pipeline span after the change is 17m01s on the first attempt (Test
  12m24s, Coverage 16m49s, Lint 58 s, Doctests 2m25s), against the 40-minute ceiling the
  gate enforces.
- Replaying that job's Cargo unit graph on four workers reproduces the measured build
  (812 s simulated against 796 s observed) and puts a core-layer-only workspace at
  155 s and the tree layer with a prebuilt core layer at 701 s. A repository split
  therefore would not speed up the churn-heavy tree layer, and the time budget targets
  the test surface and artifact reuse instead.
- What the budgets do not cover, by design: a regression that stays below both budgets,
  and a job that hangs until GitHub's own six-hour limit. The span budget is also only
  measured on a run's first attempt: a re-run carries the start times of jobs it did not
  re-run, so its span would charge re-run latency to the pipeline. The per-job suite
  budgets apply to every attempt.
- Coverage per-file thresholds are the remaining acceptance check on hosted CI. The
  paths the r=9 test exercised are also covered by `low_temperature_g0_full_grid`
  (r=3,4,5 over three topologies) and the TreeACI unit suites; the margin is expected to
  hold, but hosted coverage is authoritative.
- The first runs after this lands are the kache pilot: the build diagnostics artifacts
  record `build_finished_seconds`, `suite_seconds`, the compile and fresh line counts,
  and the kache store sizes appear in the job summary. Revert or retune from those
  numbers, not from the assumption that the cache is effective.
