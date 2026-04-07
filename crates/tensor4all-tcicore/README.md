# tensor4all-tcicore

> This is an internal crate. Most users should use `tensor4all-tensorci` or `tensor4all-quanticstci` instead.

Low-level TCI infrastructure: matrix cross interpolation, LUCI/rrLU algorithms,
cached function evaluation, and index set management.

## Key Types

- `MatrixLUCI` — matrix LU-based cross interpolation
- `CachedFunction` — thread-safe cached function evaluation
- `IndexSet` — bidirectional index sets for pivot management

## Documentation

- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_tcicore/)
