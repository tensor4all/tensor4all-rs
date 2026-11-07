# Review gate: hunk state and a head-bound waiver

Applies [tensor4all-meta#88](https://github.com/tensor4all/tensor4all-meta/issues/88)
and [#89](https://github.com/tensor4all/tensor4all-meta/issues/89) to this
repository's review gate, following the `tenferro-rs` application (#2509) of the
same `tensor4all-meta` reference implementation.

## Decisions

- **One parser, three call sites.** `added_lines_with_text`,
  `split_large_file_diff` and `sensitive_diff_location` now all walk one
  `parse_diff`, which spends the `@@ -a,b +c,d @@` line budget before deciding
  that any `+++ ` line is a header. Each helper keeps its previous signature and
  its previous result for well-formed input, so `deterministic_checks`,
  `select_rule_sections` and `main` did not change. Rejected: patching each loop
  separately. The three loops already disagreed about what a header is, and a
  per-loop fix would let them drift again.
- **The `--name-status -z` cross-check is included** as a second, blocking
  layer. It costs one local `git diff` call and cannot be influenced by the diff
  text, which is the point: the gate derives paths from the diff, and the diff is
  PR-controlled. Fail-closed is safe only because git's own vocabulary is
  accepted exactly — binary, mode-only, rename, delete and quoted non-ASCII
  paths are all admitted, and untracked worktree files are appended to git's list
  because `unified_diff` synthesises their diff itself.
- **Quoted paths are decoded as bytes, not through `json.loads`.** That was a
  real defect in the reference implementation, found by this repository's own
  verification: with `core.quotePath` on (git's default) the diff spells a path
  `"src/na\303\257ve file.rs"`, whose escapes are three-digit octal *bytes* of
  the UTF-8 name. `json.loads` rejects them, the fallback left the escapes in
  place, and the path then matched no `--name-status` entry — so the fail-closed
  check blocked every diff touching a non-ASCII path. `run_git` passes
  `-c core.quotePath=false`, which hid the bug in production, but the cross-check
  must not depend on that staying true. `_unquote_diff_path` now reassembles the
  bytes and decodes them.
- **The waiver is a comment naming a 40-character sha**, not a label. The
  composite action `.github/actions/verify-review-label` now delegates to
  `scripts/review-waiver.js`, the same shape the reference uses, so the rule can
  be replayed offline. Consequence to expect: applying the label without a
  `waive-review <sha>` comment now fails the job, and the failure message names
  the sha to post. `write` is deliberately not enough — a waiver removes a
  required check for one head, and widening who may do that is a policy change.
- **`AGENTS.md` still describes the old label semantics.** Its
  "Repository Rules Review Bot" section says `rules-review:waive` "Waives the
  review entirely" and "(reapply after the latest push)". Both statements are now
  wrong. The file is a protected agent-instruction file in this workspace, so the
  correction is recorded here and in the `review_bot.yml` header instead of being
  edited in the same PR; whoever owns that section should replace the two-row
  label table with the `waive-review <sha>` protocol.

## Verification conclusions and constraints

- The pre-fix loops are kept as `legacy_added_lines_with_text` and
  `legacyEvaluate` in the test files and asserted to accept the attack, so a
  regression is a failing test rather than a silent reopening.
- Over 213 per-file diffs from 30 consecutive commits on `main`, the five
  downstream helpers return byte-identical results before and after the change,
  with `split_large_file_diff`'s chunk limit forced low enough to split in 191 of
  them. `paths_consistent` also reports no mismatch on all 30 commits.
- The hosted gate command returns `pass` on 25 consecutive `main` commits before
  and after the change (0 block findings either way; 27 advisory `file-growth`
  warnings), so the fix changes no verdict on existing history. `--worktree` mode
  also passes with the new untracked fixtures present in the diff.
- Fail-closed width is verified against real `git` output for a binary file, a
  mode-only change, a rename, a delete and a non-ASCII path containing a space,
  under both `core.quotePath` settings.
- One deliberate difference from the pre-fix helper: a file that names a path but
  adds no line (binary, mode-only) now gets an empty bucket instead of no entry.
  Every call site tests `if not touched`, so it is inert; the shared fixtures in
  `tensor4all-meta` assume the entry exists.
- The block path of the path cross-check is covered by unit tests, not
  end-to-end: the gate derives both the diff and the name-status list from git,
  so only a hand-written diff can make them disagree.
- Not verified: the two GitHub Actions jobs are exercised by
  `node --test scripts/test-review-waiver.js` against recorded comment sets, not
  by a live pull request. The workflow has not been run on GitHub from this
  branch.
- Not changed, and worth a separate look: `review-bot`'s `if:` still requires
  `author_association` OWNER/MEMBER/COLLABORATOR, so an outside contributor's
  fork PR is `skipped` and `review-bot-gate` fails it — the comment above that
  gate claims fork PRs are no longer refused.
