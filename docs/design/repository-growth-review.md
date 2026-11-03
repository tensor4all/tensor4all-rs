# Repository growth review

The first implementation of [#917](https://github.com/tensor4all/tensor4all-rs/issues/917)
is a manual file-size experiment in the existing repository-rules reviewer.
It supplies evidence for a threshold decision; hosted automatic comments remain
disabled for this check. The canonical policy and scope are in
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

Before enabling hosted comments, maintainers must assess the
[20-PR evidence](../experiments/2026-11-03-file-growth-dry-run.md), choose thresholds
and confirm rollout. The maintainer's review selected a configurable minimum net
growth of 50 lines for the already-large case, avoiding reminders on tiny edits.
First crossings retain their original semantics. Reuse matching,
removal-attestation reminders, per-crate growth
summaries and scheduled reporting are separate follow-ups to this first slice.
