- `tensor4all-treetn`: `TreeTNOperationError` gains the variant
  `InvalidCandidates { payload: CandidateInputError, context }` (#1481), so
  exhaustive matches on it need a new arm. Under the 0.x policy this break
  ships in the next minor release. The payload `CandidateInputError` is
  `#[non_exhaustive]`, so later cases will not break callers again.
