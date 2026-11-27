- `tensor4all-treetn` composes operators in one scalar type (#1298).
  `compose_exclusive_linear_operators`, and the extension of a partial operator
  inside `apply_linear_operator`, built their identity gaps and bridges with
  `T::delta`, which is real. A complex operator therefore became a network
  mixing `f64` and `Complex64` tensors, which `PartitionedTreeTN` rejects. When
  any operator tensor is complex, the identities and the real operator tensors
  are now promoted to complex, an exact conversion; real operators are
  unchanged.
