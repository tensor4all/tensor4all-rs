- `tensor4all-partitionedtreetn`: `MergeRefineOptions::apply_options:
  Option<ApplyOptions<V>>` is replaced by `MergeRefineOptions::application:
  OperatorApplication<V>` (#1286). `None` becomes `OperatorApplication::Exact`
  (the default) and `Some(options)` becomes
  `OperatorApplication::Measured(options)`; the new `Certified` variant is the
  third choice, so the three cannot be combined.
