# Repository growth review

The file-size check from [#917](https://github.com/tensor4all/tensor4all-rs/issues/917)
runs in the existing repository-rules reviewer. The hosted workflow explicitly
enables it for same-repo PRs without a maintainer waiver, and appends advisory
findings to the existing updating review comment. The canonical policy and scope are in
[File Growth Dry Run](../../REPOSITORY_RULES.md#file-growth-dry-run).

The check combines NUL-delimited Git numstat with the destination file's physical
line count. Subtracting the net delta gives its merge-base size, including for
detected renames. No Rust parser, build, persistent backlog snapshot, or execution
of reviewed code is needed. The existing finding representation carries `warn`
results through text and JSON reports without changing pass/fail semantics.

Whole-file counts deliberately expose their inclusion of inline tests. Cutting
at the first test module would silently discard production items that follow it.
Conventional test paths and generated paths/headers are excluded heuristically;
these filters are not a complete classifier of Rust test or generated code.

The backtest driver calls the same check on the latest PR landings in first-parent
history. It records pinned base/head commits for each contribution and handles
both squash commits and merge commits. This is an offline analysis of what landed,
not a reconstruction of each original pre-merge PR head. Nonstandard PR landing
messages and PRs into other base branches are outside this sampling convention.

The rollout uses the maintainer-reviewed 1000/2000-line triggers and 50-line net
growth floor. The [first sample](../experiments/2026-11-03-file-growth-dry-run.md)
produces 5 reminders in 5/20 PRs; a
[second, disjoint sample](../experiments/2026-11-03-file-growth-rollout.md) produces
4 in 4/20. These observations support a monitored rollout, not a universal
optimality or false-positive-rate claim for the defaults.

The workflow still checks out only the trusted base and uses the PR head solely
as Git data. No new job, permission or comment-posting mechanism is needed.
Warnings leave the review passing; existing blocking checks retain their exit
status through the shell pipeline. Workflow changes take effect after they land
in the trusted base, not by executing this PR's workflow or scripts from its head.

Reuse matching, removal-attestation reminders, per-crate growth summaries and
scheduled reporting remain separate scope decisions. Unit-test extraction and
the inline-test gate are owned by #1104 / PRs #1105 and #1106; this rollout does
not duplicate their behavior or change the file-size counting algorithm.
