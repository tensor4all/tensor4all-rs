# tensor4all-capi

C-compatible FFI interface to the tensor4all Rust library, enabling language bindings for Julia, Python, and other languages.

## Features

- **Index API**: Create and manipulate tensor indices with tags
- **Tensor API**: Dense and diagonal tensors for `f64` and `Complex64`
- **TensorTrain API**: Full MPS functionality (orthogonalize, truncate, contract)
- **Algorithm selection**: Configurable factorization and contraction algorithms
- **Error handling**: Status codes instead of exceptions
- **Panic safety**: All functions protected with `catch_unwind`

## API Conventions

- Opaque pointers with lifecycle management (`*_new`, `*_release`, `*_clone`)
- Status codes: `T4A_SUCCESS`, `T4A_NULL_POINTER`, `T4A_INVALID_ARGUMENT`, etc.
- Row-major data layout for tensor data

## Example (C)

> **Note**: A C header is not yet generated automatically. Declare the required
> function prototypes manually, or generate one with `cbindgen` from the crate source.

```c
/* Create an index with dimension 2 and a "Site" tag */
t4a_index* idx = t4a_index_new(2);
t4a_index_add_tag(idx, "Site");

/* Create a tensor train of zeros: 4 sites each of dimension 2 */
size_t site_dims[] = {2, 2, 2, 2};
t4a_simplett_f64* tt = t4a_simplett_f64_zeros(site_dims, 4);

/* Query properties */
size_t n_sites = t4a_simplett_f64_len(tt);

/* Compute norm; returns T4A_SUCCESS (0) on success */
double norm;
t4a_simplett_f64_norm(tt, &norm);

/* Release objects */
t4a_simplett_f64_release(tt);
t4a_index_release(idx);
```

## License

MIT License
