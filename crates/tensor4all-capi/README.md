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

```c
#include "tensor4all.h"

// Create an index
T4AIndex* idx = t4a_index_new(10);
t4a_index_add_tag(idx, "Site");

// Create a tensor
size_t dims[] = {10, 10};
T4ATensor* tensor = t4a_tensor_new_dense_f64(2, &idx, dims, data, 100);

// Create tensor train
T4ATensorTrain* tt = t4a_tt_new(&tensors, 3);

// Orthogonalize and truncate
t4a_tt_orthogonalize(tt, 1);
t4a_tt_truncate(tt, 1e-10, 20);

// Clean up
t4a_tt_release(tt);
t4a_tensor_release(tensor);
t4a_index_release(idx);
```

## License

MIT License
