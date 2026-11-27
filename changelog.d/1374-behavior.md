- `tensor4all-partitionedtreetn`: a certified operator application refused by its
  memory gate is reported as the new `PartitionedTreeTNError::CertifiedMemoryLimit
  { operation, estimated_bytes, limit_bytes }` (#1374). It was a `ResourceLimit`
  whose `value` held the estimate and whose message called that estimate the
  limit, so a 9.7 GB estimate against the 4 GiB default read as "limit of
  9705853056". Callers matching `ResourceLimit { limit: "memory_limit (bytes)", .. }`
  match the new variant instead.
