# Merge-refine leaf validation through the projector trie (#1325)

## Decisions

- `validate_leaf_geometry` now asks
  `projector::index::has_overlapping_pair` over the preimage patch projectors
  instead of scanning every pair of selected-prefix keys. The answer is the same:
  the per-patch pass has already rejected a non-contiguous prefix and shown one
  shared spectator constraint, so two patch projectors agree on every index both
  fix exactly when their per-axis selected prefixes nest, which is what the
  removed `prefixes_overlap` loop tested. Equal keys, the one case
  `allow_equal = false` treats differently, are rejected before the check, so
  `false` cannot change a reachable result. The error variant, message, and
  position in the check order are unchanged.
- `prefixes_overlap` stays: it still prunes output regions in `kept_children`,
  and it is the reference predicate for the tests.
- Rejected: dropping the check because `from_partition`, `from_tensor_products`,
  and `from_subdomains` already reject overlapping supports (a future
  constructor could reach it, and the issue asks for the safe minimal change);
  rebuilding one projector per leaf with `geometry.input_projector` (correct but
  allocates `N` projectors, while the patch projectors already exist); keeping
  the pairwise loop under a size threshold (a second path for one concern); a
  Criterion bench (it would need a public hook, and `benchmarks/README.md`
  prescribes ignored unit tests when the harness needs crate-private state).

## Verification conclusions and constraints

- Existing tests are unchanged and pass: `tensor4all-partitionedtreetn` (lib,
  integration, `routing_oracle`, doctests) and the `tensor4all-quanticstransform`
  `qft_reconstruction` target, which holds the merge-refine validation and
  spectator-mismatch cases.
- A new differential test pins the trie answer to the removed all-pairs scan on
  random two-axis split layouts with shared spectators, including a layout with
  an added ancestor region and the duplicate-key `allow_equal` case.
- Measured with the ignored unit test
  `leaf_validation_is_near_linear_at_ten_thousand_leaves` (release-equivalent
  `ci` profile, one CPU of one L3 domain, single-threaded code path), two
  pinned runs, one of them while an unrelated build ran on the host: at 4,096 /
  8,192 / 16,384 leaves the trie takes 14.6-18.6 / 33.1-49.3 / 87.7-114.1 ms and
  the removed scan 45.0-45.4 / 179.4-179.8 / 719.4-722.0 ms. The trie grows
  2.27x-2.66x per doubling; the removed scan grows 4.00x each time. The exact
  command, host, and affinity are in the PR.
- Limits: the overlap branch is not reachable through the public constructors
  today, so no end-to-end test drives that error; the differential test pins its
  predicate instead. A layout whose patches fix unrelated index sets can still
  visit every member per query (the worst case #1293 documents); dyadic prefix
  layouts sit near the trie's best case. The host has no HDF5 headers, so the
  workspace check excluded `tensor4all-hdf5` and `book-tests`.
- Independent review on GLM-5.3-flash: plan and diff verdicts `APPROVE`, no
  BLOCKER/IMPORTANT findings. Attribution is in the PR body.
