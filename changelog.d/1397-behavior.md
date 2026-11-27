- `tensor4all-tensorbackend`: `global-defaults` is no longer enabled by the default
  features (#1377). The default is now `explicit-context` plus the platform CPU
  provider, which also exposes `Matrix`, `Storage`, `TensorElement`, the native
  tensor bridge and the `*_in` linalg entries. Direct dependents that call a
  process-global function (`mat_mul`, `solve_matrix`, `svd_backend`,
  `einsum_native_tensors`, ...) enable `global-defaults` (or `backend-tenferro`).
  `tensor4all-core` still enables it for its dependents. New caller-context entries:
  `hermitian_eigendecomposition_in`, `lowest_hermitian_eigenpair_in`,
  `hermitian_exponential_first_column_in`, `einsum_native_tensors_in`,
  `einsum_native_tensors_owned_in`, `einsum_native_tensor_reads_in`,
  `NativeEinsumPlan::execute_reads_in`.
