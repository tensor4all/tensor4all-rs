- `tensor4all-tensorci`: matrix-valued chain TCI2 (#1478, #1510).
  `crossinterpolate2_matrix` and `crossinterpolate2_matrix_with_rng` fit a function
  `x -> F(x)` returning a `rows x cols` matrix, with `MatrixShape`, `MatrixTCI2Result`
  and `MatrixSamplingStats`, all re-exported from the prelude. Use it for genuine
  matrix-valued functions whose entries share structure; for unrelated entries,
  separate `crossinterpolate2` fits can be cheaper. `tensor4all-core` adds
  `MultiIndexCache::with_value_payload_bytes`, which counts a declared per-value
  payload in `retained_bytes()` and the retained-byte limit; the default of 0 keeps
  the previous behavior.
