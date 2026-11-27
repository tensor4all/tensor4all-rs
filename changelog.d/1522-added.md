- `tensor4all-simplett`: `Tensor::as_col_major_slice` borrows the host buffer as an
  immutable column-major slice (first index fastest, no copy) (#1522, #1513). The
  matrix-valued readout (`MatrixTCI2Result::evaluate`, `FoldedMatrixTT::evaluate`) and
  `AbstractTensorTrain::evaluate` read site slabs in place instead of copying one per
  site; values are bit-identical.
