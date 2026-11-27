- `tensor4all-partitionedtreetn` validates merge-refine input leaves through
  that same projector trie instead of comparing every pair of selected prefixes
  (#1325). The check, its `InvalidOptions` message, and the error order are
  unchanged, and the cost drops from `O(N^2 d)` to `O(N d)` for `N` leaves of
  `d` selected bits, which is linear in the geometry the caller already supplies.
  16,384 full-depth dyadic leaves validate in under 120 ms instead of about
  720 ms (release profile, one core; two pinned runs), and each doubling costs
  2.3x-2.7x instead of 4.0x.
