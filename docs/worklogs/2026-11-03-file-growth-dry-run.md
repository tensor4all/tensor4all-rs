# File growth dry run (#917)

## Decisions

- Implement the file-size experiment first, following the maintainer's
  [accepted scope](https://github.com/tensor4all/tensor4all-rs/issues/917#issuecomment-6601222469).
  An explicit `--file-growth` opt-in is needed because CI already uses `--dry-run`
  and posts its report. Turning it on by default would bypass the requested
  evidence review before automated comments.
- Use the existing finding/report seam for advisory warnings and the same check
  for local previews and historical replay. The
  [design record](../design/repository-growth-review.md) explains baseline,
  whole-file counting and sampling choices. No new library dependencies or builds
  are needed.
- The maintainer's [threshold review](https://github.com/tensor4all/tensor4all-rs/pull/1103#issuecomment-6603046689)
  selects a configurable 50-line minimum net increase for already-large files.
  First crossings are unaffected; zero restores any-positive-growth reminders.

## Verification conclusions and constraints

- Real temporary Git repositories verify threshold boundaries, rename identity,
  diverged bases, worktree/staged/untracked data, binary and generated exclusions,
  JSON reports, and non-blocking outcomes. Net-growth boundaries, deletion-heavy
  edits and CLI overrides are covered. The helper suite has 109 passing tests.
- The [20-PR report](../experiments/2026-11-03-file-growth-dry-run.md) finds 5
  warnings in 5 PRs with provisional 1000/2000 limits and the 50-line floor,
  compared with 11 warnings in 8 PRs without the floor. First-crossing reminders
  alone still affect 3 PRs. These are reminder frequencies, not a measured
  false-positive rate or approval to enable automatic comments.
- No hosted growth comments are enabled and no thresholds are finalized. The
  report supplies the evidence for maintainer review on #917 before rollout.
