- `tensor4all-tensorbackend`: the crate root no longer re-exports bridge and
  storage helpers that had no caller outside the crate (#1380). Removed:
  `contract_storage`, `make_mut_storage`, `min_dim`, `axpby_native_tensor`,
  `axpby_storage_native`, `contract_native_tensor`, `contract_storage_native`,
  `outer_product_native_tensor`, `outer_product_storage_native`,
  `permute_native_tensor`, `permute_storage_native`,
  `reshape_col_major_native_tensor`, `scale_native_tensor`,
  `scale_storage_native`, `sum_native_tensor`, `svd_native_tensor`,
  `native_tensor_primal_to_storage` and `tangent_native_tensor`.
  Migration: run native-tensor operations on an explicit session
  (`CpuExecutionContext::with_concrete_session`, then `Session::contraction`,
  `permute`, `reshape`, `svd`, `sum` or `einsum`), and build or combine
  tensors through `tensor4all-core` (`IdxTensor::from_dense`, contraction,
  `scale`, `axpby`) instead of `Storage`. `qr_native_tensor` and the
  `einsum_native_*` entry points remain.
