- `tensor4all-core` no longer pins `tensor4all-tensorbackend/global-defaults` and no
  longer enables the process-global route by default (#1403). It has its own opt-in
  `global-defaults` feature (`backend-tenferro` is an alias); its default is the
  platform CPU provider only, and the provider features of the workspace crates no
  longer imply it. Without it, a context-free operation that would have adopted an
  owner-less value into the process-global eager runtime, or run on the process-global
  engine, returns the new `ExecutionContextEntryError::NoContext` (inside
  `IdxTensorError::EngineEntry` or the operation's own error) instead of choosing an
  engine; values created through a caller-owned context keep running on that context's
  engine. With `global-defaults` nothing changes. `tensor4all-treersi` drops
  `global-defaults` from its defaults (so `hadamard_many` needs it; `hadamard_many_in`
  does not); `tenferro-cuda` (in core, and so in every crate forwarding it) still implies it, because
  CUDA execution keeps host steps with no caller CPU context (core's CUDA-resident SRC
  readback, treersi's rank decision). `tensor4all-hdf5` enables it in its
  library, because its load API takes no context. Callers of the context-free API enable
  `tensor4all-core/global-defaults`. Explicit callers of
  `IdxTensor::src_error_estimate_in` and of structured contractions with host-only
  operands now run the estimator solve and the native einsum on their own context's
  engine instead of the process-global one. New: `src_error_estimate_general_in`.
  `TensorTrain::norm`/`norm_squared` run the packed-site backend kernel on the CPU
  context that owns the chain (new `IdxTensor::with_owning_cpu_engine`) in every build;
  only a chain with no CPU-owned value uses the process-global context (or refuses
  without `global-defaults`). `IdxTensor::scale` places an untracked `AnyScalar` that has
  no tensor of its own in the operand's runtime, so scaling context-owned values works
  without `global-defaults`. In every build, `From<anyhow::Error> for IdxTensorError`
  recovers a wrapped `ExecutionContextEntryError` as `IdxTensorError::EngineEntry`
  instead of `IdxTensorError::Operation`.
