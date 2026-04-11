# Conventions

This page collects important conventions that apply across the entire tensor4all-rs codebase.

## Dense Layout (Column-Major)

tensor4all-rs uses **column-major** (Fortran order) dense linearization internally. Flat dense
buffers, `reshape`/`flatten` semantics, the C API, and the HDF5 layer are all defined in terms
of column-major ordering.

This matches Julia, ITensors.jl, and tenferro-rs. When exchanging dense data with NumPy, use
`order="F"` when you need explicit control over flattening or reshaping.

## Indexing

- Sites are **0-indexed** in Rust (unlike ITensors.jl, which is 1-indexed).
- **Exception**: `tensor4all-quanticstci` grid indices are **1-indexed**, following the Julia
  convention for compatibility with QuanticsTCI.jl.

## Truncation Tolerance

tensor4all-rs uses `rtol` (relative tolerance). ITensors.jl uses `cutoff`. The conversion is:

```text
rtol = sqrt(cutoff)
```

| Library | Parameter | Conversion |
|---------|-----------|------------|
| tensor4all-rs | `rtol` | — |
| ITensors.jl | `cutoff` | `rtol = √cutoff` |

**Example**: ITensors.jl `cutoff=1e-10` corresponds to `rtol=1e-5` in tensor4all-rs.

## ITensors.jl Type Correspondence

| ITensors.jl | tensor4all-rs |
|-------------|---------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `ITensor` | `TensorDynLen` |
| `Dense` | `Storage::StructuredF64/C64` |
| `Diag` | `Storage::StructuredF64/C64` (diagonal `axis_classes`) |
| `A * B` | `a.contract(&b)` |

## Scalar Types

tensor4all-rs supports two scalar types:

- `f64` — double-precision real
- `Complex64` — double-precision complex (from the `num-complex` crate)

Generic APIs handle both types. Prefer writing generic code over scalar-specific variants (`*_f64`
/ `*_c64`) in library and test code. The C API uses scalar-specific names at the FFI boundary
where generic dispatch is not available.
