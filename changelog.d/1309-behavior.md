- `tensor4all-partitionedtreetn`: `contract_adaptive` and `add_with_patching`
  check each exact group sum before forming it (#1309). A sum whose estimated
  size exceeds `PatchingOptions::group_sum_limit_bytes` (default 4 GiB) returns
  the new `PartitionedTreeTNError::GroupSumLimit` instead of being allocated.
  A global operator, such as a Fourier transform on patches of the transformed
  variable, puts every pair of `contract_adaptive` into one group whose summed
  bond is the sum of all contribution bonds; it previously ran out of memory.
  Such an operator is applied with `schedule_merge_refine` instead.
