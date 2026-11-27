- `tensor4all-core`: `IdxTensorError` has two new variants, `RuntimeMismatch` and
  `PlacementMismatch` (#1393). The enum is not `#[non_exhaustive]`, so exhaustive
  matches must add both arms. They are reported by the engine-taking operation entry
  that stage B2 of #1376 makes public; no current public operation returns them yet.
