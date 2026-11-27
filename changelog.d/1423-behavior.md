- Nonempty `TreeTN<T, V>` implements `TensorVectorSpace`, enabling the core's
  `gmres_with_truncation` and `restart_gmres_with_truncation` on branched trees
  (#1423). Sums remain exact until the explicit truncation callback; operands
  retain strict full-site-index identity. Trait norms leave the input's gauge
  unchanged; use `TreeTN::norm(&mut tree)` for in-place canonicalization.
  Elementwise extrema remain unsupported without explicit dense materialization,
  and existing single-precision QR/SVD restrictions still apply.
