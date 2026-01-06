# C API Design Guidelines

This document describes the common design patterns and guidelines for C APIs in the tensor4all-rs project. These patterns apply to all crates that provide C-compatible FFI interfaces, including `tensor4all-capi` and any future C-API crates. These patterns ensure consistency, safety, and ease of use when creating language bindings (Julia, Python, C++, etc.).

**Scope**: This document provides general guidelines for C API design. Individual crates may have crate-specific naming conventions (e.g., `t4a_` prefix in `tensor4all-capi`), but the core patterns and safety requirements apply universally.

## Table of Contents

1. [Overview](#overview)
2. [Naming Conventions](#naming-conventions)
3. [Opaque Types](#opaque-types)
4. [Ownership Model](#ownership-model)
5. [Immutable vs Mutable Types](#immutable-vs-mutable-types)
6. [Lifecycle Management](#lifecycle-management)
7. [Error Handling](#error-handling)
8. [Panic Safety](#panic-safety)
9. [Buffer Management](#buffer-management)
10. [Data Layout (Row-Major)](#data-layout-row-major)
11. [Memory Contiguity Requirements](#memory-contiguity-requirements)
12. [Column-Major to Row-Major Conversion (Julia)](#column-major-to-row-major-conversion-julia)
13. [Thread Safety](#thread-safety)
14. [Function Export](#function-export)
15. [Code Organization](#code-organization)

## Overview

C APIs in tensor4all-rs follow patterns inspired by `sparse-ir-capi` and common FFI best practices:

- **Opaque pointers** to hide Rust implementation details
- **Explicit lifecycle functions** for memory management
- **Status codes** for error handling
- **Panic protection** to prevent Rust panics from crossing FFI boundaries
- **Consistent naming** with crate-specific prefixes
- **Row-major data layout** for multi-dimensional data (C-style, compatible with NumPy)
- **Contiguous memory requirements** - all data buffers must be contiguous
- **Layout conversion** handled by language bindings (Julia: column-major ↔ row-major)

**Note**: While this document uses examples from `tensor4all-capi` (with `t4a_` prefix), the patterns apply to all C-API crates in the tensor4all-rs project. Each crate should choose its own prefix to avoid naming conflicts.

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
// Caller owns the returned object
t4a_tensortrain* tt = t4a_tt_new_zeros(site_dims, num_sites);
// Must release when done
t4a_tt_release(tt);
```

**Rule 2: No ownership transfer**
```c
// Functions take const pointers - they don't take ownership
t4a_tensortrain* result = t4a_tt_scaled(tt, 2.0);
// tt is still valid and owned by caller
// result is a new object owned by caller
t4a_tt_release(tt);
t4a_tt_release(result);
```

**Rule 3: In-place operations modify the object**
```c
// In-place operation modifies the object in place
t4a_tt_scale_inplace(tt, 2.0);
// tt is modified, still owned by caller
t4a_tt_release(tt);
```

### Mutability Strategy

The C API provides both **immutable** and **in-place** operations:

#### Immutable Operations (Return New Objects)

These operations create new objects and leave the original unchanged:

```rust
/// Scale a tensor train by a factor, returning a new object
///
/// The original tensor train is unchanged.
#[no_mangle]
pub extern "C" fn t4a_tt_scaled(
    ptr: *const t4a_tensortrain,
    factor: libc::c_double,
) -> *mut t4a_tensortrain {
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
/// Scale a tensor train by a factor in place
///
/// Modifies the tensor train in place. More memory-efficient than scaled()
/// when you don't need the original.
#[no_mangle]
pub extern "C" fn t4a_tt_scale_inplace(
    ptr: *mut t4a_tensortrain,
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
  - `t4a_tt_scaled()` - returns new scaled object
  - `t4a_tt_added()` - returns new sum object
  - `t4a_tt_compressed()` - returns new compressed object

- **In-place operations**: Use `_inplace` suffix
  - `t4a_tt_scale_inplace()` - modifies object in place
  - `t4a_tt_add_inplace()` - modifies object in place
  - `t4a_tt_compress_inplace()` - modifies object in place

### Example: Complete Ownership Lifecycle

```c
// 1. Create object (caller owns it)
t4a_tensortrain* tt1 = t4a_tt_new_constant(site_dims, num_sites, 1.0);

// 2. Create another object
t4a_tensortrain* tt2 = t4a_tt_new_constant(site_dims, num_sites, 2.0);

// 3. Immutable operation (returns new object, caller owns it)
t4a_tensortrain* tt3 = t4a_tt_added(tt1, tt2);
// tt1 and tt2 are unchanged, still owned by caller

// 4. In-place operation (modifies tt3 in place)
t4a_tt_scale_inplace(tt3, 0.5);
// tt3 is modified, still owned by caller

// 5. Release all objects
t4a_tt_release(tt1);
t4a_tt_release(tt2);
t4a_tt_release(tt3);
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

Every opaque type must provide three standard lifecycle functions:

1. **`t4a_<TYPE>_release()`** - Release (drop) the object
2. **`t4a_<TYPE>_clone()`** - Create a clone
3. **`t4a_<TYPE>_is_assigned()`** - Check if pointer is valid

### Using the Macro

Use the `impl_opaque_type_common!` macro to generate these functions:

```rust
impl_opaque_type_common!(index);
// Generates: t4a_index_release, t4a_index_clone, t4a_index_is_assigned
```

### Manual Implementation

If you need custom behavior, implement manually following this pattern:

```rust
/// Release the object by dropping it
///
/// # Safety
/// The caller must ensure that the pointer is valid and not used after this call.
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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

## Data Layout (Row-Major)

**All tensor data in the C API uses row-major (C-style) memory layout.**

### Row-Major Order

Row-major order means that the rightmost index varies fastest in memory. For a tensor with dimensions `[d0, d1, d2, ...]`, element at position `[i0, i1, i2, ...]` is stored at:

```
offset = i0 * (d1 * d2 * ...) + i1 * (d2 * ...) + i2 * (...) + ...
```

**Example for a 2×3 tensor:**
```
Tensor dimensions: [2, 3]
Data layout: [a[0,0], a[0,1], a[0,2], a[1,0], a[1,1], a[1,2]]
```

### Why Row-Major?

- **C compatibility**: C arrays are row-major by default
- **NumPy compatibility**: NumPy arrays default to row-major (C-order)
- **Language bindings**: Most language bindings (Python, Julia, C++) expect row-major data
- **Performance**: Direct memory copy without layout conversion

### Function Documentation

Always document row-major layout in functions that handle tensor data:

```rust
/// Get the dense f64 data from a tensor in row-major order.
///
/// The data is returned in row-major (C-style) layout, where the rightmost
/// index varies fastest in memory.
///
/// # Arguments
/// - `ptr`: Tensor handle
/// - `buf`: Buffer to write data (if NULL, only out_len is written)
/// - `buf_len`: Length of the buffer
/// - `out_len`: Output: required buffer length
///
/// # Returns
/// - T4A_SUCCESS on success
/// - T4A_BUFFER_TOO_SMALL if buffer is too small (out_len is still written)
/// - T4A_INVALID_ARGUMENT if storage is not DenseF64
#[no_mangle]
pub extern "C" fn t4a_tensor_get_data_f64(
    ptr: *const t4a_tensor,
    buf: *mut libc::c_double,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    // ...
}
```

### Constructor Functions

When creating tensors from C data, the input data must be in row-major order:

```rust
/// Create a new dense f64 tensor from indices and data.
///
/// # Arguments
/// - `rank`: Number of indices
/// - `index_ptrs`: Array of t4a_index pointers (length = rank)
/// - `dims`: Array of dimensions (length = rank)
/// - `data`: Dense data in row-major order (length = product of dims)
/// - `data_len`: Length of data array
///
/// The data must be provided in row-major (C-style) layout.
#[no_mangle]
pub extern "C" fn t4a_tensor_new_dense_f64(
    rank: libc::size_t,
    index_ptrs: *const *const t4a_index,
    dims: *const libc::size_t,
    data: *const libc::c_double,
    data_len: libc::size_t,
) -> *mut t4a_tensor {
    // ...
}
```

### Implementation Notes

- **No layout conversion**: The C API does not perform layout conversion. Data is copied as-is, assuming row-major layout.
- **Storage compatibility**: The internal Rust storage may use a different layout, but the C API always presents data in row-major order.
- **Performance**: For languages that use column-major (Fortran-style) layout, callers must perform layout conversion if needed.

### Example: Converting from NumPy

```python
import numpy as np
from tensor4all import Tensor

# NumPy array (row-major by default)
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order='C')

# Create tensor - data is already row-major, direct copy
tensor = Tensor([i, j], arr)

# Get data back - also row-major
data = tensor.to_numpy()  # Returns row-major array
```

### Example: Converting from Julia

```julia
using Tensor4all

# Julia arrays are column-major by default
arr = [1 2 3; 4 5 6]  # 2×3 matrix

# Convert to row-major for C API
arr_rowmajor = permutedims(arr, (2, 1))  # Or use reshape with row-major layout

# Create tensor
tensor = Tensor([i, j], arr_rowmajor)
```

## Memory Contiguity Requirements

**All data passed to C API functions must be in contiguous memory.**

### Why Contiguity Matters

- **Direct memory access**: The C API uses `ptr::copy_nonoverlapping()` which requires contiguous memory
- **Performance**: Non-contiguous data requires element-by-element copying
- **Safety**: Non-contiguous buffers can cause undefined behavior or data corruption

### C API Assumptions

The C API assumes that:
1. **Input buffers** (`data` in constructors) are contiguous and in row-major order
2. **Output buffers** (`buf` in accessors) are contiguous and writable
3. **No layout conversion** is performed by the C API itself

### Language Binding Responsibilities

Each language binding must ensure contiguity before calling C API functions:

#### Python (NumPy)

Python bindings use `np.ascontiguousarray()` to ensure C-order (row-major) contiguous arrays:

```python
# Ensure contiguous C-order array
data = np.ascontiguousarray(data, dtype=np.float64)

# Now safe to pass to C API
ptr = lib.t4a_tensor_new_dense_f64(
    rank,
    index_ptrs,
    dims,
    ffi.cast("const double*", ffi.from_buffer(data)),
    data.size,
)
```

**Key points:**
- `np.ascontiguousarray()` creates a copy if the array is not contiguous
- NumPy arrays default to row-major (C-order), so layout conversion is usually not needed
- Non-contiguous arrays (e.g., transposed views) are automatically converted

#### Julia

Julia bindings use `_ensure_contiguous()` to ensure column-major contiguous arrays, then convert to row-major:

```julia
function _ensure_contiguous(A::AbstractArray{T,N}) where {T,N}
    if _is_column_major_contiguous(A)
        return A isa Array ? A : Array{T,N}(A)
    end
    # Materialize to contiguous Array
    return Array{T,N}(A)
end

function _column_to_row_major(arr::AbstractArray)
    arr = _ensure_contiguous(arr)  # Ensure contiguous first
    # Reverse dimensions to get row-major layout
    perm = reverse(1:ndims(arr))
    permuted = permutedims(arr, perm)
    permuted = _ensure_contiguous(permuted)  # Ensure contiguous after permutation
    return vec(permuted)
end
```

**Key points:**
- Julia arrays are column-major by default
- `_ensure_contiguous()` checks if the array is column-major contiguous
- If not contiguous, it materializes to a contiguous `Array`
- After layout conversion, contiguity is checked again

### Checking Contiguity

#### Python

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Check if C-contiguous (row-major)
assert arr.flags["C_CONTIGUOUS"]

# Check if F-contiguous (column-major)
assert arr.flags["F_CONTIGUOUS"]

# Ensure C-contiguous
arr = np.ascontiguousarray(arr, dtype=np.float64)
```

#### Julia

```julia
function _is_column_major_contiguous(A::AbstractArray)
    expected_strides = cumprod((1, size(A)...)[1:end-1])
    return strides(A) == expected_strides
end

# Check contiguity
if _is_column_major_contiguous(arr)
    # Safe to use
else
    # Need to materialize
    arr = Array(arr)
end
```

### Best Practices for Language Bindings

1. **Always check contiguity** before passing data to C API
2. **Document contiguity requirements** in binding documentation
3. **Provide helper functions** for users to ensure contiguity
4. **Handle non-contiguous arrays gracefully** by converting them automatically

## Column-Major to Row-Major Conversion (Julia)

Julia uses column-major (Fortran-style) memory layout by default, while the C API expects row-major (C-style) layout. Language bindings must handle this conversion.

### Conversion Strategy

The Julia binding uses a two-step process:

1. **Ensure contiguous column-major array**
2. **Convert to row-major** using dimension permutation

### Implementation Details

#### Column-Major to Row-Major

```julia
function _column_to_row_major(arr::AbstractArray)
    arr = _ensure_contiguous(arr)
    # Reverse dimensions to get row-major layout
    perm = reverse(1:ndims(arr))
    permuted = permutedims(arr, perm)
    permuted = _ensure_contiguous(permuted)
    return vec(permuted)
end
```

**How it works:**
- For a 2×3 array `[a[1,1] a[1,2] a[1,3]; a[2,1] a[2,2] a[2,3]]`:
  - Column-major: `[a[1,1], a[2,1], a[1,2], a[2,2], a[1,3], a[2,3]]`
  - After `permutedims(..., (2, 1))`: `[a[1,1], a[1,2], a[1,3], a[2,1], a[2,2], a[2,3]]` (row-major)

#### Row-Major to Column-Major

```julia
function _row_to_column_major(data::Vector{T}, dims::Tuple) where T
    # Data is row-major, so reverse dims for reshape
    arr = reshape(data, reverse(dims)...)
    # Reverse back to get column-major
    perm = reverse(1:length(dims))
    return permutedims(arr, perm)
end
```

**How it works:**
- Row-major data: `[a[1,1], a[1,2], a[1,3], a[2,1], a[2,2], a[2,3]]`
- Reshape with reversed dims: `[a[1,1] a[2,1]; a[1,2] a[2,2]; a[1,3] a[2,3]]`
- Permute back: `[a[1,1] a[1,2] a[1,3]; a[2,1] a[2,2] a[2,3]]` (column-major)

### Performance Considerations

- **Conversion overhead**: Layout conversion requires a full array copy
- **Memory usage**: Temporary arrays are created during conversion
- **Optimization**: For large arrays, consider in-place conversion if possible

### Example: Full Conversion Flow

```julia
using Tensor4all

# Create column-major Julia array
arr = [1 2 3; 4 5 6]  # 2×3 matrix, column-major

# Convert to row-major for C API
row_major_data = _column_to_row_major(arr)
# Result: [1, 2, 3, 4, 5, 6] (row-major flat vector)

# Create tensor
tensor = Tensor([i, j], arr)  # Conversion happens internally

# Get data back (converted to column-major)
result = data(tensor)  # Returns column-major Julia array
```

### Why Not Use Reshape Alone?

Simple `reshape()` is not sufficient because:
- `reshape()` does not change memory layout, only the view
- Column-major data reshaped to row-major dimensions still has column-major memory layout
- `permutedims()` is required to physically reorder elements

### Alternative Approaches (Not Used)

1. **Direct memory copy with stride calculation**: More complex, similar performance
2. **In-place conversion**: Possible but requires careful memory management
3. **Lazy conversion**: Would require tracking layout in the C API (not implemented)

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
- `#[no_mangle]` - Prevent name mangling
- `pub extern "C"` - C calling convention

```rust
#[no_mangle]
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
#[no_mangle]
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
├── lib.rs          # Main module, exports, status codes
├── types.rs        # Opaque type definitions
├── macros.rs       # Common macros (impl_opaque_type_common!)
├── index.rs        # Index C API functions
├── tensor.rs       # Tensor C API functions
└── ...
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

#[no_mangle]
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

#[no_mangle]
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
- [ ] Add `#[no_mangle]` and `extern "C"` to all exported functions
- [ ] Document all functions with safety requirements
- [ ] Document row-major layout for data access/creation functions (if applicable)
- [ ] Ensure memory contiguity requirements are met (language bindings)
- [ ] Handle column-major to row-major conversion if needed (Julia bindings)
- [ ] Add tests for all functions
- [ ] Follow naming conventions (use your crate's prefix consistently)

## Crate-Specific Notes

### tensor4all-capi

The `tensor4all-capi` crate provides C bindings for core tensor types:
- Uses `t4a_` prefix for all functions and types
- Uses `T4A_` prefix for status codes
- Provides Index and Tensor types
- See `crates/tensor4all-capi/` for implementation details

### Future C-API Crates

When creating new C-API crates:
1. Choose a unique prefix (e.g., `t4a_mci_` for matrix CI, `t4a_tci_` for tensor CI)
2. Follow all patterns in this document
3. Document the prefix choice in the crate's README
4. Ensure no naming conflicts with existing C-API crates

## References

- `sparse-ir-capi`: Inspiration for design patterns
- Rust FFI best practices: https://doc.rust-lang.org/nomicon/ffi.html
- Opaque types pattern: https://rust-lang.github.io/rust-clippy/master/index.html#not_unsafe_ptr_arg_deref

