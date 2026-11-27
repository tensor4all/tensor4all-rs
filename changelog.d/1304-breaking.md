- `tensor4all_core::MatrixLuciScalar` is sealed and
  `matrix_luci_factors_from_blocks_seeded` is a required method (#1300). The
  trait dispatches to crate-internal kernels, so a foreign implementation could
  not produce MatrixLUCI factors; the supported scalar types stay `f32`, `f64`,
  `Complex32`, and `Complex64`. Generic `T: MatrixLuciScalar` code is
  unaffected.
