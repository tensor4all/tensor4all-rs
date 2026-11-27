- `tensor4all-partitionedtreetn`: `schedule_merge_refine` now rejects a selection of
  `usize::BITS` or more selected indices across all coordinate groups up front with
  `PartitionedTreeTNError::InvalidOptions`. `CoordinateGroup` documented
  `usize::BITS` indices per group as the limit, but a selection of `usize::BITS` bits
  in total was always rejected later by the input-leaf check with
  `LogicalParameterCountOverflow`; the documented and enforced limit is now
  `usize::BITS - 1` selected indices in total. The rustdoc of
  `MergeRefineLevelReport::merging_levels_left` and `MergeRefineReport::refined_regions`
  / `pruned_regions` now states their existing semantics; no counter changes.
