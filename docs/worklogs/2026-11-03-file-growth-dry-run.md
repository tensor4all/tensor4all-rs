# File growth dry run (#917)

## Decisions

- Implement the file-size experiment first, following the maintainer's
  [accepted scope](https://github.com/tensor4all/tensor4all-rs/issues/917#issuecomment-6601222469).
  Keep local previews opt-in with `--file-growth`. After the reviewed prototype
  and a second sample, the hosted workflow passes that flag explicitly alongside
  its existing `--dry-run` and posts advisory findings in its existing comment.
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
  edits and CLI overrides are covered. A replay of the actual hosted shell step
  verifies warnings pass, existing blockers fail, and untrusted head scripts are
  never executed. The helper suite has 110 passing tests.
- The [20-PR report](../experiments/2026-11-03-file-growth-dry-run.md) finds 5
  warnings in 5 PRs with provisional 1000/2000 limits and the 50-line floor,
  compared with 11 warnings in 8 PRs without the floor. First-crossing reminders
  alone still affect 3 PRs. These are reminder frequencies, not a measured
  false-positive rate or approval to enable automatic comments.
- The [second sample](../experiments/2026-11-03-file-growth-rollout.md) is disjoint
  from the first and yields 4 reminders in 4/20 PRs with the same defaults. The
  rollout enables those advisory defaults and leaves recalibration open.
- Reviewed #1104, PRs #1105/#1106, and newer issues #1107/#1108 before extending
  the work. The inline-test extraction and gate are separate owned work; this
  change only enables the existing size check and verifies its workflow path.
