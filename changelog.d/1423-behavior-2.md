- Core scalar multiplication, rank-zero extraction, and direct-sum publication
  keep caller-owned CPU runtimes when used by TreeTN Krylov vectors (#1423).
  `TensorConstructionLike::to_scalar` extracts rank-zero values without a
  foreign scalar-one allocation; `IdxTensor` retains its scalar AD graph.
  Untracked scalar coefficients may join the operand's runtime, while tracked
  foreign coefficients and initialization failures remain errors. Direct sums
  retain their existing non-differentiable host-copy semantics.
