# C API Design Guidelines

This document describes the common design patterns and guidelines for C APIs in the tensor4all-rs project. These patterns apply to all crates that provide C-compatible FFI interfaces, including `tensor4all-capi` and any future C-API crates. These patterns ensure consistency, safety, and ease of use when creating language bindings (Julia, Python, C++, etc.).

**Scope**: This document provides general guidelines for C API design. Individual crates may have crate-specific naming conventions (e.g., `t4a_` prefix in `tensor4all-capi`), but the core patterns and safety requirements apply universally.

## Table of Contents

1. [Overview](#overview)
2. [Mandatory Rules](#mandatory-rules)
3. [Naming Conventions](#naming-conventions)
4. [Opaque Types](#opaque-types)
5. [Ownership Model](#ownership-model)
6. [Immutable vs Mutable Types](#immutable-vs-mutable-types)
7. [Lifecycle Management](#lifecycle-management)
8. [Error Handling](#error-handling)
9. [Panic Safety](#panic-safety)
10. [Buffer Management](#buffer-management)
11. [Data Layout (Column-Major)](#data-layout-column-major)
12. [Memory Contiguity Requirements](#memory-contiguity-requirements)
13. [Language Binding Normalization](#language-binding-normalization)
14. [Thread Safety](#thread-safety)
15. [Function Export](#function-export)
16. [Code Organization](#code-organization)

## Overview

C APIs in tensor4all-rs follow patterns inspired by `sparse-ir-capi` and common FFI best practices:

- **Opaque pointers** to hide Rust implementation details
- **Explicit lifecycle functions** for memory management
- **Status codes** for error handling
- **Panic protection** to prevent Rust panics from crossing FFI boundaries
- **Consistent naming** with crate-specific prefixes
- **Column-major data layout** for multi-dimensional data (Fortran/Julia style)
- **Contiguous memory requirements** - all data buffers must be contiguous
- **Boundary normalization** handled by language bindings when interoperating with row-major ecosystems such as NumPy

**Note**: While this document uses examples from `tensor4all-capi` (with `t4a_` prefix), the patterns apply to all C-API crates in the tensor4all-rs project. Each crate should choose its own prefix to avoid naming conflicts.

## Mandatory Rules

This section consolidates the rules that MUST be followed for all C API types and functions. These rules take precedence over any guidance in later sections.

### Rule 1: Type Mutability Classification

Every opaque type must be classified as either **Immutable** or **Mutable**:

- **Immutable types** (`Box<Arc<T>>`): Objects that are never modified after construction. `clone()` is cheap (Arc reference count increment). Only `inner()` is provided -- `inner_mut()` MUST NOT be defined.
- **Mutable types** (`Box<T>`): Objects that support in-place modification. `clone()` is a deep copy. Both `inner()` and `inner_mut()` are available.

#### Classification Table

| Type | Mutability | Wraps | `clone()` | `inner_mut()` | Notes |
|------|-----------|-------|-----------|---------------|-------|
| `t4a_index` | Immutable | `DynIndex` | Yes (cheap) | No | |
| `t4a_tensor` | Immutable | `TensorDynLen` | Yes (cheap) | No | |
| `t4a_qgrid_disc` | Immutable | `DiscretizedGrid` | Yes (cheap) | No | |
| `t4a_qgrid_int` | Immutable | `InherentDiscreteGrid` | Yes (cheap) | No | |
| `t4a_linop` | Immutable | `LinearOperator<TensorDynLen, usize>` | Yes (cheap) | No | |
| `t4a_qtci_f64` | Immutable | `QuanticsTensorCI2<f64>` | Yes (deep) | No | Clone possible (`QuanticsTensorCI2` derives `Clone`) |
| `t4a_treetn` | Mutable | `DefaultTreeTN<usize>` | Yes (deep) | Yes | Orthogonalization, truncation |
| `t4a_treetci_f64` | Mutable | `TreeTCI2<f64>` | Yes (deep) | Yes | Clone possible (`TreeTCI2` derives `Clone`) |
| `t4a_treetci_c64` | Mutable | `TreeTCI2<Complex64>` | Yes (deep) | Yes | Clone possible (`TreeTCI2` derives `Clone`) |
| `t4a_treetci_graph` | Immutable | `TreeTciGraph` | Yes (cheap) | No | |
| `t4a_simplett_f64` | Mutable | `TensorTrain<f64>` | Yes (deep) | Yes | Compression |
| `t4a_simplett_c64` | Mutable | `TensorTrain<Complex64>` | Yes (deep) | Yes | Compression |

### Rule 2: Lifecycle Functions

Every opaque type MUST provide:

- `t4a_<TYPE>_release` -- mandatory for all types
- `t4a_<TYPE>_clone` -- mandatory if `Clone` is possible; if `Clone` is impossible, document why and do NOT provide the function
- `t4a_<TYPE>_is_assigned` -- mandatory for all types

Prefer `impl_opaque_type_common!` macro which generates all three. When `Clone` is not available (e.g., `TreeTCI2`), implement `release` and `is_assigned` manually and add a comment explaining why `clone` is absent.

#### Current Status

| Type | `_release` | `_clone` | `_is_assigned` | Notes |
|------|:---:|:---:|:---:|-------|
| `t4a_index` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_tensor` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_treetn` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_qgrid_disc` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_qgrid_int` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_linop` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_treetci_graph` | Yes | Yes | Yes | `impl_opaque_type_common!` |
| `t4a_simplett_f64` | Yes | Yes | Yes | Manual impl |
| `t4a_simplett_c64` | Yes | Yes | Yes | Manual impl |
| `t4a_qtci_f64` | Yes | -- | Yes | Clone possible: `QuanticsTensorCI2` derives `Clone` (impl pending) |
| `t4a_treetci_f64` | Yes | -- | Yes | Clone possible: `TreeTCI2` derives `Clone` (impl pending) |
| `t4a_treetci_c64` | Yes | -- | Yes | Clone possible: `TreeTCI2` derives `Clone` (impl pending) |

### Rule 3: Thread Safety

- All opaque types MUST implement `unsafe impl Send` and `unsafe impl Sync`
- Concurrent reads on the same pointer are safe
- Concurrent writes to the same pointer are undefined behavior (caller's responsibility)
- Callbacks execute on the calling thread

### Rule 4: Naming Conventions

In addition to the conventions in the [Naming Conventions](#naming-conventions) section below:

- **In-place operations on Mutable types**: no special suffix needed, as `*mut` pointer already signals mutation
- **Scalar type suffix**: `_f64`, `_c64` for type-specialized functions (e.g., `t4a_simplett_f64_*`, `t4a_treetci_c64_*`)

### Rule 5: Complex Number FFI Convention

Complex64 values are passed as interleaved f64 pairs: `[re, im, re, im, ...]`.

- For callback results returning `n_points` complex values, write `2 * n_points` doubles to the result buffer
- Element at index `i` has real part at `results[2*i]` and imaginary part at `results[2*i + 1]`
- This convention MUST be documented in all c64 callback type definitions

### Rule 6: Type Definition Location

- All opaque type struct definitions MUST be in `types.rs`
- Type-specific logic (functions, tests) go in their own module files (e.g., `simplett.rs`, `treetci.rs`)
- Enum definitions for C API parameters belong in `types.rs`

## Naming Conventions

### Function Names

C API functions should use a consistent prefix pattern. Each crate should define its own prefix to avoid naming conflicts:

```
<CRATE_PREFIX>_<TYPE>_<OPERATION>
```

**Examples from `tensor4all-capi` (uses `t4a_` prefix):**
- `t4a_index_new()` - Create a new index
- `t4a_index_dim()` - Get index dimension
- `t4a_tensor_get_rank()` - Get tensor rank
- `t4a_tensor_new_dense_f64()` - Create dense f64 tensor

**Guidelines:**
- Choose a short, unique prefix for your crate (e.g., `t4a_` for tensor4all-capi)
- Use lowercase with underscores
- Keep function names descriptive but concise
- Group related functions by type (e.g., all `index_*` functions together)

### Type Names

Opaque types should use the crate's prefix:

**Examples from `tensor4all-capi`:**
- `t4a_index` - Opaque index type
- `t4a_tensor` - Opaque tensor type
- `t4a_storage_kind` - Storage kind enum

**Guidelines:**
- Use the same prefix as function names
- Keep type names descriptive
- Use `snake_case` for consistency with C conventions

### Status Code Constants

Status codes should use an uppercase prefix matching the crate's prefix:

**Examples from `tensor4all-capi` (uses `T4A_` prefix):**
- `T4A_SUCCESS` - Operation succeeded
- `T4A_NULL_POINTER` - Null pointer error
- `T4A_INVALID_ARGUMENT` - Invalid argument
- `T4A_BUFFER_TOO_SMALL` - Buffer too small
- `T4A_INTERNAL_ERROR` - Internal error

**Guidelines:**
- Use uppercase for constants
- Use the same base prefix as functions/types (uppercase version)
- Define common error codes (SUCCESS, NULL_POINTER, INVALID_ARGUMENT, INTERNAL_ERROR)
- Add crate-specific error codes as needed

## Opaque Types

All Rust objects exposed through the C API are wrapped in opaque pointer types to hide implementation details.

### Structure

```rust
#[repr(C)]
pub struct t4a_<TYPE> {
    pub(crate) _private: *const c_void,
}
```

**Key points:**
- Use `#[repr(C)]` to ensure C-compatible layout
- Use `*const c_void` to hide the internal type
- The internal type is stored in a `Box` and accessed via the pointer

### Implementation Pattern

```rust
impl t4a_<TYPE> {
    /// Create a new opaque wrapper from an internal type
    pub(crate) fn new(internal: InternalType) -> Self {
        Self {
            _private: Box::into_raw(Box::new(internal)) as *const c_void,
        }
    }

    /// Get a reference to the inner type
    pub(crate) fn inner(&self) -> &InternalType {
        unsafe { &*(self._private as *const InternalType) }
    }

    /// Get a mutable reference to the inner type
    pub(crate) fn inner_mut(&mut self) -> &mut InternalType {
        unsafe { &mut *(self._private as *mut InternalType) }
    }
}
```

### Drop Implementation

Always implement `Drop` to properly deallocate the internal object:

```rust
impl Drop for t4a_<TYPE> {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalType);
            }
        }
    }
}
```

### Send + Sync

If the internal type is `Send + Sync`, mark the opaque type as well:

```rust
// Safety: t4a_<TYPE> is Send + Sync because InternalType is Send + Sync
unsafe impl Send for t4a_<TYPE> {}
unsafe impl Sync for t4a_<TYPE> {}
```

## Ownership Model

**Core Principle: "Owned Objects with Explicit Lifecycle"**

The C API follows an **explicit ownership model** where objects created via `new` functions are owned by the caller and must be explicitly released. This model provides clear ownership semantics while allowing both immutable and in-place operations.

### Key Principles

1. **Explicit Ownership**: Objects created with `t4a_<TYPE>_new()` are owned by the caller
2. **Explicit Release**: All objects must be explicitly released using `t4a_<TYPE>_release()`
3. **No Ownership Transfer**: Functions never take ownership of objects; they operate on borrowed references
4. **Immutable by Default**: Objects appear immutable by default, but in-place operations are allowed when needed
5. **Copy Semantics**: Operations that modify objects either return new objects or modify in-place

### Ownership Rules

**Rule 1: Creation implies ownership**
```c
/* Caller owns the returned object */
size_t site_dims[] = {2, 2, 2, 2};
t4a_simplett_f64* tt = t4a_simplett_f64_zeros(site_dims, 4);
/* Must release when done */
t4a_simplett_f64_release(tt);
```

**Rule 2: No ownership transfer**
```c
/* Functions take const pointers - they don't take ownership */
t4a_simplett_f64* copy = t4a_simplett_f64_clone(tt);
/* tt is still valid and owned by caller */
/* copy is a new object owned by caller */
t4a_simplett_f64_release(tt);
t4a_simplett_f64_release(copy);
```

**Rule 3: In-place operations modify the object**
```c
/* In-place operations take *mut pointer — no special suffix needed (Rule 4) */
/* Example: hypothetical in-place scale */
/* t4a_simplett_f64_scale(tt, 2.0); */
/* tt is modified, still owned by caller */
t4a_simplett_f64_release(tt);
```

### Mutability Strategy

The C API provides both **immutable** and **in-place** operations:

#### Immutable Operations (Return New Objects)

These operations create new objects and leave the original unchanged:

```rust
/// Clone a tensor train, returning a new independent object.
///
/// The original tensor train is unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_clone(
    ptr: *const t4a_simplett_f64,
) -> *mut t4a_simplett_f64 {
    // Returns new object, original unchanged
}
```

**Use cases:**
- When you need to preserve the original object
- When you want to chain operations
- When the operation is expensive and you want to avoid copying

#### In-Place Operations (Modify Existing Objects)

These operations modify the object in place:

```rust
/// Example pattern for an in-place operation.
///
/// Modifies the tensor train in place. The `*mut` pointer already
/// signals mutation, so no `_inplace` suffix is needed (Rule 4).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_scale(
    ptr: *mut t4a_simplett_f64,
    factor: libc::c_double,
) -> StatusCode {
    // Modifies object in place
}
```

**Use cases:**
- When you don't need the original object
- When memory efficiency is important
- When the operation is cheap and doesn't require copying

### Naming Conventions

- **Immutable operations**: Use past participle or descriptive names
  - `t4a_simplett_f64_clone()` - returns a new copy
  - `t4a_simplett_f64_scaled()` - returns new scaled object (pattern)
  - `t4a_simplett_f64_compressed()` - returns new compressed object (pattern)

- **In-place operations**: No special suffix — the `*mut` pointer already signals mutation (Rule 4)
  - `t4a_simplett_f64_scale()` - modifies object in place (pattern)
  - `t4a_simplett_f64_compress()` - modifies object in place

### Example: Complete Ownership Lifecycle

```c
/* 1. Create objects (caller owns them) */
size_t site_dims[] = {2, 2, 2, 2};
t4a_simplett_f64* tt1 = t4a_simplett_f64_constant(site_dims, 4, 1.0);
t4a_simplett_f64* tt2 = t4a_simplett_f64_constant(site_dims, 4, 2.0);

/* 2. Clone tt2 (immutable operation — original unchanged, caller owns copy) */
t4a_simplett_f64* tt3 = t4a_simplett_f64_clone(tt2);

/* 3. Query properties */
double norm;
t4a_simplett_f64_norm(tt3, &norm);

/* 4. Release all objects */
t4a_simplett_f64_release(tt1);
t4a_simplett_f64_release(tt2);
t4a_simplett_f64_release(tt3);
```

### Benefits of This Model

1. **Clear Ownership**: Explicit ownership makes memory management predictable
2. **No Double-Free**: Each object is owned by exactly one caller
3. **Flexibility**: Both immutable and in-place operations available
4. **Thread Safety**: Immutable operations are naturally thread-safe
5. **Memory Efficiency**: In-place operations avoid unnecessary copies
6. **Familiar Pattern**: Similar to C++ smart pointers or Rust's ownership model

### When to Use Immutable vs In-Place

**Use immutable operations when:**
- You need to preserve the original object
- You're chaining multiple operations
- The operation is expensive (avoid unnecessary copies)
- You want thread-safe operations

**Use in-place operations when:**
- You don't need the original object
- Memory efficiency is critical
- The operation is cheap (simple scaling, etc.)
- You're performing a single modification

### Reference Counting Model (Future Consideration)

**Current Implementation (tensor4all-rs):**
- Objects are wrapped in `Box<T>` directly
- `clone()` creates a **full copy** of the object
- `release()` **immediately frees** memory when called
- No reference counting - each object is independent

**Alternative Model (sparse-ir-rs style):**
- Objects are wrapped in `Box<Arc<T>>` internally
- `clone()` **increments reference count** (cheap operation)
- `release()` drops the `Box<Arc<T>>`, but memory is **only freed when the last reference is dropped**
- Multiple references can exist, and memory persists until all are released

**Example of sparse-ir-rs model:**
```rust
// sparse-ir-rs style: Arc-based
impl spir_sve_result {
    pub(crate) fn new(sve_result: SVEResult) -> Self {
        let inner = Arc::new(sve_result);  // Arc for reference counting
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }
}

impl Clone for spir_sve_result {
    fn clone(&self) -> Self {
        // Cheap clone: Arc::clone just increments reference count
        let inner = self.inner_arc().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }
}
```

**When to consider Arc-based model:**
- Objects are large and expensive to copy
- Multiple references to the same object are common
- `clone()` operations are frequent
- Memory sharing is beneficial

**Current tensor4all-rs approach:**
- Simpler implementation
- Predictable memory behavior (immediate release)
- Suitable for objects that are not too large
- Full copy semantics are clear and explicit

**Note:** The current `Box<T>` model is simpler and sufficient for most use cases. Consider migrating to `Arc<T>` if profiling shows that `clone()` operations are a bottleneck and multiple references to the same object are common.

## Lifecycle Management

Every opaque type should provide three standard lifecycle functions:

1. **`t4a_<TYPE>_release()`** - Release (drop) the object
2. **`t4a_<TYPE>_clone()`** - Create a clone
3. **`t4a_<TYPE>_is_assigned()`** - Check if pointer is valid

### Using the Macro

Use the `impl_opaque_type_common!` macro to generate these functions:

```rust
impl_opaque_type_common!(index);
// Generates: t4a_index_release, t4a_index_clone, t4a_index_is_assigned
```

### Current Implementation Status (tensor4all-capi)

See the authoritative table in [Mandatory Rules -- Rule 2](#rule-2-lifecycle-functions).

### Manual Implementation

If you need custom behavior, implement manually following this pattern:

```rust
/// Release the object by dropping it
///
/// # Safety
/// The caller must ensure that the pointer is valid and not used after this call.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_release(obj: *mut t4a_<TYPE>) {
    if obj.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(obj);
    }
}

/// Clone the object
///
/// # Safety
/// The caller must ensure that the source pointer is valid.
/// The returned pointer must be freed with `t4a_<TYPE>_release()`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_clone(src: *const t4a_<TYPE>) -> *mut t4a_<TYPE> {
    if src.is_null() {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let src_ref = &*src;
        let cloned = (*src_ref).clone();
        Box::into_raw(Box::new(cloned))
    }));

    result.unwrap_or(std::ptr::null_mut())
}

/// Check if the object pointer is valid
///
/// # Returns
/// 1 if the object is valid, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_is_assigned(obj: *const t4a_<TYPE>) -> i32 {
    if obj.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    }));

    result.unwrap_or(0)
}
```

## Error Handling

### Status Codes

Use status codes (`StatusCode = libc::c_int`) for error handling:

```rust
pub type StatusCode = libc::c_int;

pub const T4A_SUCCESS: StatusCode = 0;
pub const T4A_NULL_POINTER: StatusCode = -1;
pub const T4A_INVALID_ARGUMENT: StatusCode = -2;
pub const T4A_TAG_OVERFLOW: StatusCode = -3;
pub const T4A_TAG_TOO_LONG: StatusCode = -4;
pub const T4A_BUFFER_TOO_SMALL: StatusCode = -5;
pub const T4A_INTERNAL_ERROR: StatusCode = -6;
```

### Function Return Pattern

Functions that can fail should return `StatusCode`:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_<operation>(
    ptr: *const t4a_<TYPE>,
    out_value: *mut SomeType,
) -> StatusCode {
    // Null pointer checks
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    // Wrap in catch_unwind for panic safety
    let result = catch_unwind(|| {
        let obj = unsafe { &*ptr };
        // ... operation ...
        unsafe { *out_value = value };
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}
```

### Constructor Pattern

Constructors that can fail should return `*mut t4a_<TYPE>` (null on error):

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_new(...) -> *mut t4a_<TYPE> {
    // Validation
    if invalid {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let internal = InternalType::new(...);
        Box::into_raw(Box::new(t4a_<TYPE>::new(internal)))
    });

    result.unwrap_or(std::ptr::null_mut())
}
```

## Panic Safety

**Always wrap FFI functions in `catch_unwind`** to prevent Rust panics from crossing the FFI boundary, which can cause undefined behavior.

### Pattern

```rust
use std::panic::catch_unwind;

let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
    // Your code here
    T4A_SUCCESS
}));

result.unwrap_or(T4A_INTERNAL_ERROR)
```

### For Constructors

```rust
let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
    let internal = InternalType::new(...);
    Box::into_raw(Box::new(t4a_<TYPE>::new(internal)))
}));

result.unwrap_or(std::ptr::null_mut())
```

**Note:** Use `AssertUnwindSafe` when you're certain the code is unwind-safe (which is usually the case for FFI wrappers).

## Buffer Management

When returning variable-length data, use the "query-then-fill" pattern:

### Pattern

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_get_data(
    ptr: *const t4a_<TYPE>,
    buf: *mut u8,           // Can be NULL to query length
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,  // Always written
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let obj = unsafe { &*ptr };
        let data = obj.get_data();
        let required_len = data.len();

        unsafe { *out_len = required_len };

        // If buf is NULL, just return the length
        if buf.is_null() {
            return T4A_SUCCESS;
        }

        // Check buffer size
        if buf_len < required_len {
            return T4A_BUFFER_TOO_SMALL;
        }

        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf, required_len);
        }

        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}
```

### Usage Pattern

Callers should:
1. First call with `buf = NULL` to get `out_len`
2. Allocate a buffer of size `out_len`
3. Call again with the allocated buffer

## Data Layout (Column-Major)

**All tensor data in the C API uses column-major memory layout.**

### Column-Major Order

Column-major order means that the leftmost index varies fastest in memory. For a tensor with
dimensions `[d0, d1, d2, ...]`, element at position `[i0, i1, i2, ...]` is stored at:

```
offset = i0 + d0 * (i1 + d1 * (i2 + ...))
```

**Example for a 2×3 tensor:**
```
Tensor dimensions: [2, 3]
Data layout: [a[0,0], a[1,0], a[0,1], a[1,1], a[0,2], a[1,2]]
```

### Why Column-Major?

- **Internal consistency**: tensor4all-rs uses column-major dense linearization internally
- **Julia / ITensors.jl alignment**: no conversion is needed at those boundaries
- **HDF5 interoperability**: ITensors.jl-compatible storage uses the same ordering
- **Clear semantics**: flat buffers, `reshape`, and `flatten` all share one rule

### Function Documentation

Always document column-major layout in functions that handle tensor data:

```rust
/// Get the dense f64 data from a tensor in column-major order.
///
/// The data is returned in column-major layout, where the leftmost index
/// varies fastest in memory.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_data_f64(
    ptr: *const t4a_tensor,
    buf: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    // ...
}
```

Constructor functions follow the same contract: flat dense inputs are interpreted as
column-major buffers for the provided dimensions.

### Implementation Notes

- **No layout conversion inside the C API**: dense buffers are copied as-is
- **Storage compatibility**: internal Rust dense semantics and the C API now match
- **Boundary normalization**: callers from row-major ecosystems should normalize at the binding layer

## Memory Contiguity Requirements

**All data passed to C API functions must be contiguous in memory.**

### Why Contiguity Matters

- **Direct memory access**: the C API copies flat buffers directly
- **Performance**: non-contiguous views require materialization anyway
- **Safety**: contiguous buffers make FFI boundaries explicit and predictable

### C API Assumptions

The C API assumes that:
1. **Input buffers** (`data` in constructors) are contiguous flat buffers in column-major order
2. **Output buffers** (`buf` in accessors) are contiguous and writable
3. **Shape interpretation** is performed by the caller using column-major semantics

### Language Binding Responsibilities

Bindings should normalize dense arrays at the boundary:

- **Python / NumPy**: flatten with `order="F"` before calling constructors, and reshape with
  `order="F"` when reconstructing arrays from flat buffers
- **Julia**: native arrays are already column-major; ensure contiguity, then pass the flat data through

### Python Example

```python
flat = np.asarray(data, dtype=np.float64).ravel(order="F").copy()
ptr = lib.t4a_tensor_new_dense_f64(
    rank,
    index_ptrs,
    dims,
    ffi.cast("const double*", ffi.from_buffer(flat)),
    flat.size,
)

buf = np.empty(out_len[0], dtype=np.float64)
lib.t4a_tensor_get_data_f64(...)
arr = buf.reshape(dims, order="F")
```

### Julia Example

```julia
arr = Array(input)                  # materialize if needed
flat = vec(arr)                     # Julia already uses column-major linearization
# Pass `flat` and `size(arr)` directly to the C API entry point.
```

## Language Binding Normalization

Row-major ecosystems such as NumPy still need explicit normalization at the boundary. The rule is:

1. materialize or copy to a contiguous array if needed
2. flatten with column-major semantics
3. pass the flat buffer to the C API
4. reshape returned flat buffers with column-major semantics

This keeps the C API itself simple and makes layout handling explicit in the binding code instead
of distributing ad hoc conversions throughout the Rust implementation.

## Thread Safety

### Send + Sync

If the internal Rust type is `Send + Sync`, mark the opaque type accordingly:

```rust
unsafe impl Send for t4a_<TYPE> {}
unsafe impl Sync for t4a_<TYPE> {}
```

This enables safe use from multiple threads in the calling language.

### Documentation

Document thread safety guarantees in function documentation:

```rust
/// Get the dimension of an index.
///
/// This function is thread-safe and can be called from multiple threads
/// concurrently on different index objects.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_index
/// - `out_dim` must be a valid pointer to write the dimension
```

## Function Export

### Attributes

Always use:
- `#[unsafe(no_mangle)]` - Prevent name mangling (Rust 2024 edition syntax)
- `pub extern "C"` - C calling convention

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_<TYPE>_<operation>(...) -> ReturnType {
    // ...
}
```

### Documentation

Document all public functions with:
- Purpose description
- Arguments (with types and constraints)
- Return value
- Safety requirements
- Error conditions

```rust
/// Get the rank (number of indices) of a tensor.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `out_rank`: Output pointer for the rank
///
/// # Returns
/// Status code (T4A_SUCCESS or error code)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tensor
/// - `out_rank` must be a valid pointer to write the rank
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_get_rank(
    ptr: *const t4a_tensor,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    // ...
}
```

## Code Organization

### File Structure

Organize C API code by type. The structure may vary depending on the crate's scope:

**Example structure for `tensor4all-capi`:**
```
src/
├── lib.rs               # Main module, exports, status codes
├── types.rs             # Opaque type definitions and enums
├── macros.rs            # Common macros (impl_opaque_type_common!)
├── index.rs             # Index C API
├── tensor.rs            # Tensor C API
├── treetn.rs            # Tree tensor network (MPS/MPO) C API
├── simplett.rs          # Simple tensor train C API
├── quanticsgrids.rs     # Quantics grids C API
├── quanticstci.rs       # Quantics TCI C API
├── quanticstransform.rs # Quantics transformation operators C API
├── algorithm.rs         # Algorithm types and options
└── hdf5.rs              # HDF5 I/O (optional feature)
```

**General guidelines:**
- Keep related functionality together
- Separate types from functions
- Use macros for repetitive patterns
- Group by domain/type (e.g., one file per major type)

### Module Organization

1. **`types.rs`** (or similar): Define all opaque types and their implementations
2. **`macros.rs`** (if needed): Common macros for code generation
3. **`<type>.rs`**: C API functions for each type
4. **`lib.rs`**: Re-exports and status code definitions

### Section Comments

Use section comments to organize functions:

```rust
// ============================================================================
// Constructors
// ============================================================================

// ============================================================================
// Accessors
// ============================================================================

// ============================================================================
// Modifiers
// ============================================================================
```

### Tests

Add tests at the end of each module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_<operation>() {
        // Test implementation
    }
}
```

## Examples

### Complete Example: Opaque Type

```rust
// types.rs
#[repr(C)]
pub struct t4a_mytype {
    pub(crate) _private: *const c_void,
}

impl t4a_mytype {
    pub(crate) fn new(internal: InternalType) -> Self {
        Self {
            _private: Box::into_raw(Box::new(internal)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &InternalType {
        unsafe { &*(self._private as *const InternalType) }
    }
}

impl Clone for t4a_mytype {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_mytype {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalType);
            }
        }
    }
}

unsafe impl Send for t4a_mytype {}
unsafe impl Sync for t4a_mytype {}
```

### Complete Example: C API Functions

**Note**: This example uses `t4a_` prefix from `tensor4all-capi`. Replace with your crate's prefix.

```rust
// mytype.rs
use crate::types::t4a_mytype;
use crate::{StatusCode, T4A_SUCCESS, T4A_NULL_POINTER, T4A_INTERNAL_ERROR};
use std::panic::catch_unwind;

impl_opaque_type_common!(mytype);

#[unsafe(no_mangle)]
pub extern "C" fn t4a_mytype_new(value: i32) -> *mut t4a_mytype {
    if value < 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let internal = InternalType::new(value);
        Box::into_raw(Box::new(t4a_mytype::new(internal)))
    }));

    result.unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "C" fn t4a_mytype_get_value(
    ptr: *const t4a_mytype,
    out_value: *mut i32,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(|| {
        let obj = unsafe { &*ptr };
        unsafe { *out_value = obj.inner().value() };
        T4A_SUCCESS
    });

    result.unwrap_or(T4A_INTERNAL_ERROR)
}
```

## Summary Checklist

When adding a new opaque type to a C API crate:

- [ ] Choose a unique prefix for your crate (e.g., `t4a_` for tensor4all-capi)
- [ ] Define opaque type with `_private: *const c_void` in `types.rs` (or appropriate module)
- [ ] Implement `new()`, `inner()`, and `inner_mut()` methods
- [ ] Implement `Clone` and `Drop` traits
- [ ] Mark as `Send + Sync` if applicable
- [ ] Use `impl_opaque_type_common!` macro for lifecycle functions (or implement manually)
- [ ] Wrap all functions in `catch_unwind` for panic safety
- [ ] Define status codes with your crate's prefix
- [ ] Add `#[unsafe(no_mangle)]` and `extern "C"` to all exported functions
- [ ] Document all functions with safety requirements
- [ ] Document column-major layout for data access/creation functions (if applicable)
- [ ] Ensure memory contiguity requirements are met (language bindings)
- [ ] Normalize row-major caller data at the binding boundary if needed
- [ ] Add tests for all functions
- [ ] Follow naming conventions (use your crate's prefix consistently)

## Crate-Specific Notes

### tensor4all-capi

The `tensor4all-capi` crate provides C bindings for all tensor4all functionality:
- Uses `t4a_` prefix for all functions and types
- Uses `T4A_` prefix for status codes
- Provides: Index, Tensor, TreeTN (MPS/MPO), SimpleTT, QuanticsGrids, QuanticsTCI, QuanticsTransform, and HDF5
- See `crates/tensor4all-capi/` for implementation details

### Future C-API Crates

When creating new C-API crates:
1. Choose a unique prefix (e.g., `t4a_mci_` for matrix CI, `t4a_tci_` for tensor CI)
2. Follow all patterns in this document
3. Document the prefix choice in the crate's README
4. Ensure no naming conflicts with existing C-API crates

## Crate-Specific Type Reference (tensor4all-capi)

### Opaque Types

| Type | Wraps | Module | Description |
|------|-------|--------|-------------|
| `t4a_index` | `DynIndex` | `index.rs` | Tensor index with dimension, ID, and tags |
| `t4a_tensor` | `TensorDynLen` | `tensor.rs` | Dynamic-rank tensor (dense or diagonal) |
| `t4a_treetn` | `DefaultTreeTN<usize>` | `treetn.rs` | Tree tensor network (MPS, MPO, general TTN) |
| `t4a_simplett_f64` | `TensorTrain<f64>` | `simplett.rs` | Simple tensor train (f64) |
| `t4a_simplett_c64` | `TensorTrain<Complex64>` | `simplett.rs` | Simple tensor train (c64) |
| `t4a_qtci_f64` | `QuanticsTensorCI2<f64>` | `quanticstci.rs` | Quantics TCI result (f64) |
| `t4a_qgrid_disc` | `DiscretizedGrid` | `quanticsgrids.rs` | Discretized continuous grid |
| `t4a_qgrid_int` | `InherentDiscreteGrid` | `quanticsgrids.rs` | Integer discrete grid |
| `t4a_linop` | `LinearOperator<TensorDynLen, usize>` | `quanticstransform.rs` | Quantics linear operator (shift, flip, Fourier, etc.) |
| `t4a_treetci_graph` | `TreeTciGraph` | `treetci.rs` | Tree graph for TreeTCI |
| `t4a_treetci_f64` | `TreeTCI2<f64>` | `treetci.rs` | TreeTCI state (f64) |
| `t4a_treetci_c64` | `TreeTCI2<Complex64>` | `treetci.rs` | TreeTCI state (c64) |

### Enums

| Type | Values | Module | Description |
|------|--------|--------|-------------|
| `t4a_storage_kind` | DenseF64=0, DenseC64=1, DiagF64=2, DiagC64=3 | `types.rs` | Tensor storage kind |
| `t4a_unfolding_scheme` | Fused=0, Interleaved=1 | `types.rs` | Quantics unfolding scheme |
| `t4a_boundary_condition` | Periodic=0, Open=1 | `types.rs` | Boundary condition for quantics operators |

### Callback Types

| Type | Signature | Module | Description |
|------|-----------|--------|-------------|
| `QtciEvalCallbackF64` | `fn(coords: *const f64, n: size_t, result: *mut f64, user_data: *mut void) -> i32` | `quanticstci.rs` | QTCI continuous domain callback |
| `QtciEvalCallbackI64` | `fn(indices: *const i64, n: size_t, result: *mut f64, user_data: *mut void) -> i32` | `quanticstci.rs` | QTCI discrete domain callback |
| `TreeTciEvalCallback` | `fn(batch: *const size_t, n_sites: size_t, n_points: size_t, results: *mut f64, user_data: *mut void) -> i32` | `treetci.rs` | TreeTCI batch eval (f64) |
| `TreeTciEvalCallbackC64` | `fn(batch: *const size_t, n_sites: size_t, n_points: size_t, results: *mut f64, user_data: *mut void) -> i32` | `treetci.rs` | TreeTCI batch eval (c64, interleaved re/im) |

### Module Overview

| Module | Functions | Description |
|--------|-----------|-------------|
| `index.rs` | `t4a_index_*` | Index creation, tags, dimensions |
| `tensor.rs` | `t4a_tensor_*` | Tensor creation, data access, contraction |
| `treetn.rs` | `t4a_treetn_*` | Tree tensor networks: construction, orthogonalization, truncation, inner, norm, contract, linsolve |
| `simplett.rs` | `t4a_simplett_f64_*` | Simple TT operations |
| `quanticsgrids.rs` | `t4a_qgrid_disc_*`, `t4a_qgrid_int_*` | Quantics grid coordinate conversions |
| `quanticstci.rs` | `t4a_qtci_f64_*`, `t4a_quanticscrossinterpolate_*` | Quantics TCI interpolation |
| `quanticstransform.rs` | `t4a_qtransform_*`, `t4a_linop_*` | Quantics transformation operators |
| `treetci.rs` | `t4a_treetci_f64_*`, `t4a_treetci_c64_*`, `t4a_treetci_graph_*` | Tree-structured TCI |
| `hdf5.rs` | `t4a_hdf5_*` | HDF5 serialization (ITensors.jl compatible) |

## References

- `sparse-ir-capi`: Inspiration for design patterns
- Rust FFI best practices: https://doc.rust-lang.org/nomicon/ffi.html
- Opaque types pattern: https://rust-lang.github.io/rust-clippy/master/index.html#not_unsafe_ptr_arg_deref
