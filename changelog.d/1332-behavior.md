- `tensor4all-partitionedtreetn`: `MergeRefineOptions::target_bond_dim` decides
  one coordinate axis at a time (#1328). A level previously probed the joint
  product of every refining axis and then fixed an output bit on all of them or
  on none; it now probes each axis on its own and fixes output bits only on the
  axes whose own children store no more parameters in total than the region's
  merged, unsplit candidate (the #1327 guard, applied per axis; a candidate that
  already fits the goal stops every axis without a probe). A level of
  an `N`-axis schedule therefore splits into `2^(refining axes)` children instead
  of `2^N`, and the stopped axes keep their region's shorter prefix while the
  region's input siblings keep merging on every axis. Single-axis runs are
  unaffected; `MergeRefineReport::stopped_regions` now counts one region-axis
  decision per stopped axis instead of one region, which is the same value for a
  single axis. A region with several axes can now stop every axis at a coarse level
  even though only the joint split would have paid, because no single axis pays on
  its own.
