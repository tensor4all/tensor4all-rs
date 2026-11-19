# Tolerance merge policy implementation (#1130)

## Decisions

- Implement the design merged in #1394: mutually exclusive `MergePolicy`
  variants, static prefix-key merge forecasts truncated at output depth,
  Euclidean allocation over disjoint regions, and serial minimum-storage
  candidate selection. Exact and rank-goal numerical decisions stay on the
  existing schedule path.
- The per-level report distinguishes the retained error from `budget_spent`.
  Tolerance spends only retained error. Rank-goal mode keeps its historical
  conservative charges for discarded probes; exposing both quantities avoids
  silently changing that mode's future allocations. The active-merge forecast
  is present only for tolerance mode.
- Keep the existing conservative SVD cutoff and measured residual helper.
  No final fitting pass or output-norm estimation is added.
- Keep the early prototype branch separate; its boolean opt-in and uniform
  refinement are not the accepted interface or algorithm.

## Verification conclusions and constraints

- Implementation and local functional validation are complete. The new policy
  probes every eligible axis even at zero allowance, selects the smallest
  retained partition with the specified tie order, and carries only its ledger.
  Metadata child expansion prunes requested regions incrementally and checks
  resource bounds and packed-prefix overflow before creating tensors. Resource
  errors identify whether work items or terms were the limiting budget.
- The original six-qubit SplitMix64 case and the two interleaved three-bit axes
  pass independent product-DFT checks in both policies, with a 5 GiB process
  address-space limit. Tolerance returns one strict region/term in each case.
  The original case has measured L2 error `2.989798202397942e-8`, reported bound
  `5.253295304462644e-8`, and global allowance `5.959933682156905e-6`. These are
  numerical errors, not performance measurements. Functional JSON and harness
  provenance are retained in
  [`../experiments/issue-1130-implementation/`](../experiments/issue-1130-implementation/).
- The interleaved rank-goal baseline retains nine overlapping terms in its one
  region, which its existing contract permits. The harness sums all those terms
  for the oracle and storage count; only tolerance promises strict conversion
  at full input-merge depth.
- No benchmark slowdown limit has been approved yet. The implementation PR
  must register and obtain agreement on the complete paired benchmark plan
  before timed candidate runs, as required by the accepted design. Numerical
  checks that do not report timings can proceed through `ai-run build`.

## Local checks

All builds and tests ran through `ai-run build` with an explicit memory cap
(6 GiB, except the 1 GiB compiler-version query). No test tolerances or coverage
thresholds were relaxed.

- `cargo fmt --all`.
- `cargo test -p tensor4all-partitionedtreetn --lib reconstruction::schedule`:
  22 passed; one pre-existing ignored timing experiment was not run.
- `cargo test -p tensor4all-partitionedtreetn --test integration reconstruction`:
  21 passed.
- `cargo test -p tensor4all-quanticstransform --features partitioned --test
  qft_reconstruction --test patched_fourier`: 49 + 13 passed, including the
  existing 44 exact/rank-goal reconstruction cases and five new tolerance cases.
- `cargo run -p tensor4all-partitionedtreetn --example merge_refine`: passed both
  exact and tolerance known-value assertions.
- Affected-crate doctests: 150 partitionedtreetn + 49 quanticstransform passed.
- `TENSOR4ALL_CARGO_PROFILE=dev ./scripts/test-mdbook.sh`: all chapters passed.
- Strict Clippy for both changed crates, all targets with `partitioned`, and
  the four affected benchmark examples with `patched-fourier`: passed.
- `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps` for
  both affected crates with `partitioned`: passed.
- Public error documentation audit, test-target inventory, `git diff --check`,
  and deterministic repository-rules dry run: passed. The latter reports only
  an advisory for growth of the existing schedule module.
- Seven production mutations were rejected by eleven test/mutation checks,
  covering every newly added test. Compiler failures do not count as rejection;
  each named test failed its assertion, and all sources were restored before
  the final green regression run. The
  [runner and evidence](../experiments/issue-1130-implementation/mutations/)
  record exact patches, commands, and failure output.

## Review notes

- Coverage impact of removed paths was reviewed. The obsolete runtime
  zero-rank-goal error is now unrepresentable through `NonZeroUsize`; the
  corresponding test checks that invariant. No shared numerical helper was
  removed. Existing exact/rank-goal tests and new tolerance tests exercise the
  shared merge, restriction, cancellation, and truncation helpers.
- The tolerance planner/selector and its unit tests live in a separate module.
  Growth remaining in `schedule.rs` belongs to its public policy/report types
  and level integration; keeping the old exact/rank-goal loop there makes its
  unchanged decisions reviewable without an unrelated refactor.
- The root README crate map and capability statements remain accurate. API
  migration examples, the affected crate README, guides, and portable skill
  references were updated together. Greedy `ReconstructionOptions` keeps its
  separate `target_bond_dim` field.
- The branch is based on current `origin/main` (`35174fdd` at the final fetch).
  Hosted CI, rules LLM review, macOS validation, and the agreed performance
  acceptance suite remain pending publication. No push or implementation PR
  creation has been performed.
