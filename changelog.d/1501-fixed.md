- `tensor4all-treetn`: `TreeTNCachedEvaluator` on a tree mixing real and
  complex nodes no longer fails with `dtype mismatch: expected F64, actual C64`
  when a real node's message needs a complex child message that was already
  materialized as a tensor (#1501). The raw message kernels now decline such a
  child, and the generic path computes the message in the promoted kind. This
  affected flat batches (e.g. centre 0, or centre 1 hinted, on a chain with a
  complex leaf) and `evaluate_candidates_typed` across the same edges.
