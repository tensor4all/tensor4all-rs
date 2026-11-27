- `tensor4all-core`: `direct_sum`, `direct_sum_many` and, through
  `TensorContractionLike`, `TreeTN::add` / `add_many` promote operands of
  different dense scalar types instead of refusing them. The lattice is the one
  the eager arithmetic already uses: `f32 + f64 -> f64`, `f32 + Complex32 ->
  Complex32`, `f64 + Complex64 -> Complex64` and `f64 + Complex32 -> Complex64`
  (so a real network and a complex-scaled copy of it add). Operands are cast
  before the block copy, so each block stays exact; a same-type call allocates
  nothing new. Tracked operands are still rejected (#1426).
