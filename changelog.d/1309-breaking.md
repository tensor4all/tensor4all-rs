- `tensor4all-partitionedtreetn`: `PatchingOptions` has the new field
  `group_sum_limit_bytes` (#1309). A complete struct literal must add it; one
  that ends in `..PatchingOptions::default()` is unaffected.
