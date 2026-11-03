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

## Verification conclusions and constraints

- Real temporary Git repositories verify threshold boundaries, rename identity,
  diverged bases, worktree/staged/untracked data, binary and generated exclusions,
  JSON reports, and non-blocking outcomes. The helper suite has 106 passing tests.
- The [20-PR report](../experiments/2026-11-03-file-growth-dry-run.md) finds 11
  warnings in 8 PRs with provisional 1000/2000 limits; first-crossing reminders
  alone affect 3 PRs. Single-line growth of a large module illustrates why the
  already-large rule needs calibration. These are reminder frequencies, not a
  measured false-positive rate or an approved policy change.
- No hosted growth comments are enabled and no thresholds are finalized. The
  report supplies the evidence for maintainer review on #917 before rollout.
