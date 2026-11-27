- `tensor4all-interpolativeqtt` constructors, `estimate_interpolation_error[_nd]`
  and `invert_qtt` are generic over the function's value type
  `T: InterpolativeScalar` (`f64` or `Complex64`) and return
  `SimpleTensorTrain<T>` / `Matrix<T>` (#1276). Real results are bit-for-bit
  unchanged and ordinary calls compile as before; only a call that names the
  closure type with a turbofish (`::<F>`) has to add `T` (`::<f64, F>`).
