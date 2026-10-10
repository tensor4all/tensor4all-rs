# Kache store persistence in CI

## Decisions

- The workflow persists the kache store itself (`actions/cache/restore` plus `actions/cache/save`)
  instead of using the action's GitHub cache backend. That backend keys the whole store as
  `{prefix}-{kacheVersion}-{platform}-{lockfileHash}`, and GitHub cache entries are immutable, so
  the entry freezes at the first run after a lockfile change: later runs restore it, find the key
  taken (`GitHub cache already holds key …; skipping save`), and publish nothing. A per-commit key
  with a prefix restore key keeps the newest snapshot reachable and lets each run publish its own.
  The action keeps installing kache and configuring `RUSTC_WRAPPER` (`github-cache: "false"`,
  `cache-dir: ${{ runner.temp }}/kache`), so no fork pin is needed.
- Only `main` publishes. Pull-request jobs restore the newest snapshot through the base-branch
  cache scope and never claim a key, so a PR cannot occupy the key the next main run wants to
  refresh.
- The save is skipped when the restore already matched this exact key
  (`steps.kache-restore.outputs.cache-hit != 'true'`), which is the idiomatic pattern and avoids
  reserving a key that an entry already holds.
- An upstream input would make this two steps into one line: kunobi-ninja/kache-action#42 adds
  `cache-key-salt`, which appends a salt to the exact key while the restore key stays prefix-only.
  That pull request is open and waiting on that repository's maintainer, and this change does not
  depend on it.

## Verification conclusions and constraints

- Frozen-key baseline (parent branch, Test job, 168 cacheable units): a new commit hit 1.8% of the
  units and rebuilt for 612 s, while an exact re-run of the same commit hit 100% and rebuilt for
  12 s. The difference is the freeze described above, not kache's compilation cache itself.
- Cold run under the new key scheme: 0% hits, 168 units compiled, and the run published a 994 MiB
  `ci` snapshot and a 1177 MiB instrumented `coverage` snapshot. The first run after the switch
  therefore pays a full build, as any cache does when the key scheme changes.
- Warm numbers in the final configuration (a new commit on the branch restoring the snapshot the
  previous commit published): the Test job restored the previous snapshot, kache reported 100.0% of
  168 units served from cache with 0 compiled, and `build_finished_seconds` fell to 18.6 s from
  612 s under the frozen key (suite 28.2 s, job 2m13s). Coverage likewise restored and reported
  100.0% of 166 units with `build_finished_seconds` 30.7 s and its per-file check at 258/258 files
  (suite 64.9 s, job 3m41s). The pipeline span was 4m17s against the 40-minute budget. A commit that
  changes Rust sources still misses the changed crate and its dependents; 18.6 s is the
  unchanged-units case, not a guarantee.
- Constraint: a snapshot is ~1 GiB per job and the GitHub cache budget is 10 GiB per repository, so
  a frequently merged `main` keeps the newest entries and evicts older ones. Watch
  `gh api repos/<owner>/<repo>/actions/caches` before adding a third store, and consider dropping
  the never-reuse-eligible `cache-workspace-crates` entries from the jobs that still set it.
- The measurement is only meaningful as a pair: the cold run establishes the snapshot, the next
  commit measures restoring it. Re-running the same commit measures neither.
