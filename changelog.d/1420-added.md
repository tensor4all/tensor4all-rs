- `tensor4all-tensorbackend` adds `Runtime`, an engine-free runtime that several
  engines share (#1393, stage A2 of #1376). It holds the eager AD runtime, the graph
  compiler and the compiled-program cache, and no engine or engine lock; clones are
  the same runtime. `Runtime::with_eager(exec, f)` and
  `Runtime::run_graph(graph, inputs, exec)` borrow the caller's
  `exec: &mut dyn ExecutionEngine` for one call, so a value made on one `CpuEngine`
  runs on any other `CpuEngine` the runtime supports. `Runtime::check_engine` refuses
  an engine with another runtime id, engine type or allocation domain with
  `RuntimeError::UnsupportedEngine` instead of registering it lazily. With
  `tenferro-cuda`, `Runtime::for_cuda_engine` builds a CUDA runtime (no graph path),
  and `CudaEngine`/`CudaDeviceId` are re-exported. `ExecutionEngine` is re-exported.
  `CpuExecutionContext` and `CudaExecutionContext` are now built on a private
  `Runtime` each, with the same numerics and error messages. One difference: a CPU
  context now builds its graph and eager runtimes together on first use. Any first
  use (`compile_graph`, `run_graph`, `graph_cache_stats`, `eager_runtime`) can
  therefore report an initialization failure of either half (for example
  `eager_runtime` can report `graph CPU engine`), and the `Debug` output reports
  `graph_initialized` and `eager_initialized` together.
