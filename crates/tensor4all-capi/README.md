# tensor4all-capi

C-compatible FFI interface to the tensor4all Rust library, focused on the
minimal Julia-facing surface.

## Features

- **Index API**: Immutable index handles with constructor/getter-only access
- **Tensor API**: Dense `f64` / `Complex64` construction, export, and contraction
- **TreeTN API**: General tree tensor network accessors and core operations
- **QTT layout API**: Canonical binary QTT layout descriptors for transform materialization
- **Generated C header**: `include/tensor4all_capi.h` for downstream bindings
- **Error handling**: `StatusCode + out` constructors/clones plus `t4a_last_error_message`
- **Panic safety**: All exported functions catch Rust panics at the FFI boundary

## API Conventions

- Opaque pointers with lifecycle management (`*_new(..., out)`, `*_release`, `*_clone(..., out)`)
- Status codes: `T4A_SUCCESS`, `T4A_NULL_POINTER`, `T4A_INVALID_ARGUMENT`, etc.
- Column-major data layout for dense tensor data
- Complex buffers use interleaved doubles: `[re, im, re, im, ...]`

## Example (C)

```c
/* Create a tagged index with explicit plev. */
struct t4a_index *idx = NULL;
t4a_index_new(2, "Site", 0, &idx);

/* Construct a dense tensor in column-major order. */
double data[] = {1.0, 2.0};
struct t4a_tensor *tensor = NULL;
const struct t4a_index *indices[] = {idx};
t4a_tensor_new_dense_f64(1, indices, data, 2, &tensor);

/* Release objects */
t4a_tensor_release(tensor);
t4a_index_release(idx);
```

## Regenerating the Header

```bash
mkdir -p crates/tensor4all-capi/include
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

## License

MIT License
