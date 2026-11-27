- `tensor4all-treetn`: `TreeTNCachedEvaluator::evaluate_candidates_typed` and
  `evaluate_candidates_typed_with_parallelism` evaluate the candidate matrix of
  one tree edge from two sides (`CandidateSide`: sites and `MultiIndex`
  candidates), either all of it (`CandidateEntries::All`, column-major, left
  fastest) or given `(left, right)` pairs (`CandidateEntries::Pairs`) (#1481).
  The edge is inferred from the two site sets; values are the product of the two
  directed messages across it, through the existing message caches and their
  shared budget. Malformed requests report `CandidateInputError`.
