- `tensor4all-partitionedtreetn`: `ErrorNorm::sampled_max()` pins its
  reference to the largest `|f|` that a greedy coordinate ascent reaches from
  the root patch's candidate samples (#1274), as `optfirstpivot` does in
  TensorCrossInterpolation.jl. It used to take the largest candidate sample,
  which for a peaked function was up to 2711 times too small. This changes
  the tolerance of `sampled_max()` runs.
  - A root whose candidate samples are all zero still fails as before.
  - `sampled_max_with_reference` is unchanged.
