# tensor4all-tensorbackend

> This is an internal crate. Most users should use `tensor4all-core` instead.

Scalar types (`f64`, `Complex64`), storage backends, and tensor algebra primitives
backed by tenferro-rs.

## Key Types

- `AnyScalar` — dynamic scalar type (f64 or Complex64)
- `Matrix` — shared column-major dense matrix boundary for tensor4all crates
- `Storage` — dense/diagonal tensor storage
- `StructuredStorage` — axis-class-aware storage snapshots
- `svd_backend`, `qr_backend`, `solve_matrix`, `full_piv_lu_matrix` —
  tenferro-backed dense linear algebra entry points

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_tensorbackend/)
