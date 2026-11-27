- `tensor4all-tensorci`: `MatrixTCI2Result::to_folded` returns a `FoldedMatrixTT` (#1515),
  the same matrix-valued function with the row and column cores contracted into the
  first and last physical cores: `L` cores with the matrix indices as open legs. It
  draws no new samples and does not truncate. The `L + 2` layout stays the default;
  folding can grow storage slightly (the docs give the exact condition) and does not
  speed up readout. `MatrixTCI2Result::retained_bytes` reports the core bytes of the
  `L + 2` layout for comparison.
