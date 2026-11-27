- Native einsum tracing/profiling settings and core contraction/GMRES profiling
  settings are read once per crate on first use (#1387). Set the `T4A_PROFILE_*`,
  `T4A_TRACE_NATIVE_EINSUM_*`, and `T4A_GMRES_OP_PROFILE` environment variables
  before the first operation. The undocumented automatic cleanup switches
  `T4A_RESET_NATIVE_EINSUM_ENGINE_AFTER_CALL`,
  `T4A_RESET_NATIVE_EINSUM_BUFFER_POOL_AFTER_CALL`, and
  `T4A_RELEASE_ALLOCATOR_AFTER_NATIVE_EINSUM_CALL` are removed. Manage cleanup
  explicitly with `CpuExecutionContext::reset_graph_runtime`,
  `CpuExecutionContext::reset_graph_buffer_pool`, and
  `release_process_allocator_cached_memory` instead.
