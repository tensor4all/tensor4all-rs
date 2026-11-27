- `tensor4all-core` adds the required method `TensorVectorSpace::is_complex`
  (#1298), the complex counterpart of `is_single_precision`. An implementor
  outside this workspace must add it; a container reports `true` when any part
  is complex.
