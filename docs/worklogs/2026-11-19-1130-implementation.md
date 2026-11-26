# Tolerance merge policy implementation (#1130)

## Current integration and output-layout limits (2026-11-26)

The implementation is rebased onto `373c246b`, including #1479's SRC component
join fix, with both sets of CHANGELOG entries retained. A range-diff confirms
that all four existing implementation/evidence patches are unchanged by this
synchronization. The scheduler algorithm is unchanged. Historical acceptance
measurements remain evidence for their recorded source and dependency revisions,
not new performance measurements of the synchronized branch.

The latest [consumer requirements](https://github.com/tensor4all/tensor4all-rs/issues/1130#issuecomment-7186152520)
ask for two transformed axes, spectator constraints, a prescribed shared output
layout and one term per region. Full-depth tolerance mode consolidates each
nonempty region; `refine_to` can then align outputs only onto a layout that
refines their existing restrictions. It cannot coarsen finer boundaries or
restore dropped regions. The guide, public Fourier documentation and
[design contract](../design/partitioned-treetn.md#report-and-acceptance) state
that limitation instead of promising arbitrary fixed layouts.

A small two-axis regression checks real-valued and complex inputs on the same
requested layout, including a spectator sharing a selected index's ID at a
different prime level. It checks strict conversion, exact refinement and a
dense product-DFT residual against the reported allowance. A coarser-layout
request explicitly retains the existing boundaries. The 64-by-64 input-patch
case and a hard peak-memory bound remain unverified and are not claimed.

Fresh focused validation on the synchronized source passes: 22 scheduler units (one
pre-existing timing test remains ignored), 21 reconstruction integration cases,
14 patched Fourier cases, 49 QFT reconstruction cases and 49 Fourier doctests.
The new test rejects an injected tolerance-merge mutation that retains both
operands instead of consolidating them; the production source was restored
before the passing Fourier suites. Strict affected-crate Clippy, formatting,
the implicit-engine ratchet and deterministic repository-rules review were
repeated and pass. Earlier rustdoc-link, API-inventory, public-error-documentation
and test-target-inventory evidence remains applicable: this synchronization
changes no public API, example or guide snippet. The mdBook executable suite
was not repeated locally. Hosted checks and implementation review must run on
the published rebased head.

## Implementation review follow-up (2026-11-26)

The six findings in the [implementation review](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7201649508)
are addressed without changing approximation tolerances or candidate tie order.
The branch now includes `main` through `de053a07`.

Candidate key metadata stays bounded before tensor work. Work/term admission now
uses the actual nonempty input items and terms after restriction and merging,
including earlier selected regions, rather than multiplying a parent upper
bound by every planned child. The resource error distinguishes planned key
expansion from retained work and reports the count that actually exceeded the
relevant bound. A missing input-parent shape is rejected directly.

Coordinate-group width is validated against `usize::BITS` before operator work,
so input decoding, requested output decoding and child expansion share the same
packed-prefix contract. Full-width prefixes remain valid, and several shorter
axes may together exceed one machine word. The API migration is recorded under
Breaking Rust API changes, including report literals. Refinement statistics
explicitly count selected decisions even when their children cancel or are
pruned; they do not claim to count retained nonempty regions.

A public three-site, two-axis identity-operator fixture fixes the actual output
projectors: one axis stops while the cheaper axis refines. It checks the dense
represented value as well as the structural decision. The resource regression
keeps a one-term result under a one-term budget even though two child keys were
planned. The reviewed standard-library-only zero-goal assertion is removed.
Mutation evidence is retained in
[`review-mutations.json`](../experiments/issue-1130-implementation/review-mutations.json).

Historical runtime/RSS results remain pinned to their original source, binary
and dependency revisions. These boundary, accounting and regression fixes are
not a new timing or peak-memory measurement, nor a claim about the unverified
64-by-64 consumer workload.

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
  planned-key bounds and packed-prefix overflow before creating tensors. Actual
  work-item and term limits are checked after restriction, with the exceeding
  retained count in the error.
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
- At the initial 2026-11-19 validation, no benchmark slowdown limit had been
  approved yet. The implementation PR
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
  scheduler policy tests cover its stored goal and default behavior. No shared numerical helper was
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

## Follow-up

The maintainer subsequently fixed 10.0x for both cases. The
[2026-11-20 acceptance worklog](2026-11-20-1130-performance-acceptance.md) records
the completed paired experiment and its soft review triggers.
