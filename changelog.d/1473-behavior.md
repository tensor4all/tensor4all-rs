- `tensor4all_tensorbackend::with_runtime_engine` no longer substitutes the
  process-global CPU context when no live context owns the runtime (#1378). It
  returns the typed, transitional `ExecutionContextEntryError::NoLiveOwner` in
  every build. The legacy fallback remains only as the explicitly named,
  doc-hidden bridge `with_runtime_engine_or_global` (also
  `tensor4all_core::legacy::with_runtime_engine_or_global`), which core's
  context-free operations use, so their behavior is unchanged. Operations given
  an explicit context still run on exactly that context's engine. Breaking:
  `ExecutionContextEntryError` gains the variant `NoLiveOwner`, so exhaustive
  matches on it need a new arm.
