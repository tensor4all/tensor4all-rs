- `tensor4all-partitionedtreetn`: `PartitionedTreeTN::iter`, `values`, `projectors`
  and both `IntoIterator` impls yield patches in canonical patch order (#1366):
  by index signature, then by the patch's projector coordinates. The order no
  longer depends on insertion order or hash-map state, so an order-dependent
  step on the patches, such as a truncated pairwise sum, is reproducible across
  runs. The `IntoIterator` impls now require `V: Ord` and return
  `std::vec::IntoIter`; each call sorts the patch list once.
