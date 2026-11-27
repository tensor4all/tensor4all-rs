- `tensor4all-simplett`: `Tensor3Ops::slice_site` and `try_slice_site` copy a
  site slice from the tensor's column-major host buffer instead of reading it
  element by element, so the tensor-train readout no longer pays one indexed
  access per element. Values, order and the API are unchanged (#1513).
