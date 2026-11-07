# #1150: line-independent panic-audit baseline keys

## Problem

`scripts/library-panics-baseline.json` was a flat list of `path:line:kind`. Any
change that inserted lines above a baselined assertion turned `Maintenance
scripts` red and asked the author to renumber JSON by hand (#1147: the assertion
was unchanged, `main:8285` → branch:8301). One exit status covered two different
events, so a moved entry read as a violation.

## Change

- A baseline entry is keyed on `(path, kind, enclosing item, occurrence index
  within that item)`, with the line recorded but never matched on. The enclosing
  item comes from the `syn` tree that `scan_sources` already parses: the
  innermost item whose span contains the finding's full source position, named
  by `quote`'s rendering of the item's own syntax, so qualification, generics,
  const generics, references and arrays all stay distinguishable
  (`Matrix<T>::from_elem`, `DenseMatrixSource<'a,T> as CandidateMatrixSource<T>::get_block`,
  `mod::item` for nested items, `Trait::method` for trait default methods), and
  `<file>` for include! fragments that do not parse. The occurrence index is the
  ordinal of that kind inside the item, ordered by position.
  `tools/library-panic-audit` now depends on `quote` directly; it was already in
  the dependency graph through `syn`, and `tools/api-dump` uses it the same way.
- Positions, not lines, are compared, so two items or two assertions written on
  one line stay distinct.
- The normalized assertion text is the secondary check. It is rendered from the
  token stream with a space only where two tokens would otherwise merge: between
  two word-like tokens, and between two standalone punctuation characters that
  spell a compound operator. Indentation, line breaks and `|&row|` /
  `| & row |` respacing are therefore invisible, a literal keeps its exact text
  (`"a + b"` and `"a+b"` differ), and `&&` stays distinct from `& &`. Pairing depends on the
  counts: with as many findings as entries the recorded ordinal decides, so a
  rewrite or a reordering inside one item is reported as `changed`; when the
  counts differ, equal text pairs first, so an insertion or a deletion reports
  the assertion that actually appeared or disappeared instead of blaming its
  neighbour.
- Signals are separated. `moved` (same key, different line or ordinal) is
  informational and the run stays green; `changed` (same key, different text),
  `added`, and `stale` fail as before.
- `--update-baseline` rewrites the recorded lines and text, prints `- old` /
  `+ new` per entry, and refuses in CI (`CI` set to anything but `0`/`false`) and
  whenever the set of keys changed, because that is not renumbering. It also
  refuses with `--config-name`, since a configuration audits a subset of the
  workspace.
- The baseline file holds one JSON object per entry
  (`{"content","index","item","kind","line","path"}`). The previous
  `path:line:kind` strings are still read for one release and are migrated while
  they still match the finding at their line; a legacy entry whose assertion
  moved is reported stale rather than guessed, because the line alone no longer
  identifies it. Keyed entries claim their identity before any legacy entry, so
  mixing both shapes in one file cannot depend on entry order.
  `scripts/library-panics-baseline.json` itself is migrated, so a feature branch
  only has to pick up the new file.

## Verification

- `cargo test -p library-panic-audit`: 51 passed, including fixtures that insert
  lines above a baselined assertion (green, one `Moved baseline entry`), add an
  assertion (red, naming only the new one), delete one (red as stale), rewrite
  one (red as changed), swap two inside an item (red), swap two assertions
  written on one line (red), move one into another item (red, with the entry of
  the item it left reported stale), and give an entry an ordinal that no finding
  has (red, and the updater refuses it). Separate assertions pin the resolved
  keys (`check#0`, `check#1`, `first#0`, `second#0`) so a scanner that collapsed
  every finding into one scope could not pass.
- `python3 scripts/test-audit-library-panics.py`: 23 passed. The suite needs
  `TMPDIR` on a non-symlinked path on macOS, because
  `test_wrapper_discovers_custom_target_artifact` compares a resolved path with
  an unresolved one; it fails identically on the unmodified wrapper otherwise.
- `python3 scripts/audit-library-panics.py` on `main` after migrating the
  baseline: exit 0, 19 entries matched, 0 added, 0 stale, and `--update-baseline`
  then rewrites 0 of 19 entries, so the committed file is the tool's own output.
- #1147 simulation on `main`: 16 lines inserted above
  `crates/tensor4all-core/src/defaults/idx_tensor.rs:8285` (assertion now at
  8301, the line named in the issue) → exit 0, one
  `Moved baseline entry: …:8301:debug_assert_eq [IdxTensor::fuse_indices#0]`.
  With the same assertion duplicated on that line instead → exit 1, one
  unbaselined finding. With a synthetic extra assertion above it → exit 1,
  naming only the new assertion. All edits were reverted.

## Known limits

- An item nested inside an expression block (for example a `fn` declared inside a
  closure body) is not collected, so its assertions key to the enclosing function.
  The key is still stable under line movement, but such a nested item is not a
  separate scope.
- A legacy entry can only migrate while its line still holds the assertion it
  described; a legacy entry whose assertion moved is reported stale. The
  committed baseline is already keyed, so this only affects a branch that edits
  the old file itself.
- A `--config-name` pass cannot see a deletion, because an unmatched entry may
  simply be outside that feature selection; that is the pass's documented
  contract. A swap alone still fails there, since the counts are equal and the
  ordinals decide; a swap combined with a deletion can only be printed as
  `Moved`.
