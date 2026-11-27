- `tensor4all-partitionedtreetn`: the default group-sum limit is the crate-level
  `DEFAULT_GROUP_SUM_LIMIT_BYTES`, shared by `PartitionedContractionOptions` and
  `PatchingOptions`; `PatchingOptions::DEFAULT_GROUP_SUM_LIMIT_BYTES` is removed.
  Where `usize` cannot represent 4 GiB, the default is `usize::MAX`.
  `contract`, `contract_selected` and `add_with_patching` check every group
  before building any group sum. The `GroupSumLimit` message also suggests
  forming only the needed outputs.
