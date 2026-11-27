- `schedule_merge_refine` and `patched_fourier_transform` accept differing
  spectator constraints (#1299). Disjoint groups are retained directly;
  overlapping constraints use a common spectator grid checked against work and
  term limits before tensor operations. Each original patch is transformed
  once. The obsolete `PartitionedTreeTNError::SpectatorMismatch` is removed.
