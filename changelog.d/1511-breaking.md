- `tensor4all-partitionedtreetn`: `MergeRefineOptions::output_regions` is now
  `Option<Vec<OutputRegion>>`, a static bit-prefix list, instead of
  `Option<Vec<Projector>>` (#1284, the contract decided on #1254).
  Migration: a region that was `Projector::from_pairs([(output_index_k, bit_k)
  for k in 0..depth])` on one coordinate group becomes
  `OutputRegion::new(depth, prefix)`, where `depth` is the number of fixed
  leading output indices and `prefix` is their bits, most significant first
  (`prefix < 2^depth`). With explicit `coordinate_groups`, use
  `OutputRegion::per_axis([(depth, prefix), ...])`, one pair per group in group
  order. No index identity appears in the option any more; the previous
  "non-contiguous", "foreign index" and "non-binary coordinate" rejections
  cannot be expressed, and the new rejections are a wrong axis count, a depth
  above the group's output count (or 64 or more), and a prefix that does not fit its depth.
  `MergeRefineReport` gains `pruned_regions`, the number of child regions
  the kept refinements excluded under the selection; it is a coverage count kept apart from `error_bound`.
  `OutputRegion` is re-exported from `tensor4all-quanticstransform`.
