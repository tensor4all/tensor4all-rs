- `tensor4all-partitionedtreetn`: `PartitionedContractionOptions` has the new
  field `group_sum_limit_bytes` (#1316). Complete struct literals must add it.
  `contract` and `contract_selected` return `GroupSumLimit` before building a
  sum whose estimate exceeds the limit. The default is 4 GiB, including when
  passing plain `ContractionOptions`. Use `usize::MAX` to disable the limit.
