- `TreeTN::inner` returns `1.0` instead of `0.0` for two empty networks,
  following the empty-product convention and matching `TensorTrain::inner`
  (#1383). Passing exactly one empty network still returns an error.
