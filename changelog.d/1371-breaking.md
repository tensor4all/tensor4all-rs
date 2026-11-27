- `tensor4all-treetci`: `TreeTciOptimizationResult` and `TreeTciRunResult`
  have a new public field, `local_threshold_factor: f64` (#1371). Neither
  struct is `#[non_exhaustive]`, so downstream struct literals and
  destructuring patterns without `..` must add the field. The new
  `TreeTciTermination::LocalThresholdFloor` variant is not breaking: that enum
  is `#[non_exhaustive]`.

