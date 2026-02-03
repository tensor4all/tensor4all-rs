# HDF5 Runtime Library Loading via dlopen

**Issue:** https://github.com/tensor4all/tensor4all-rs/issues/209
**Date:** 2026-02-03
**Status:** Approved

## Overview

Replace build-time HDF5 linking with runtime library loading via dlopen/libloading. This enables:

1. **Cross-platform distribution** — Pre-built binaries without build-time HDF5 dependency
2. **Julia/Python integration** — Use HDF5 from language runtime environment (HDF5_jll, h5py)
3. **Flexible deployment** — Users specify which HDF5 library to use at runtime

## Design Decisions

- **Language-specific discovery** — Julia/Python bindings handle finding HDF5; Rust layer requires explicit `hdf5_init(path)`
- **Error on uninitialized** — Return `Hdf5Error::NotInitialized` if HDF5 operations called before initialization
- **Minimal code changes** — Copy hdf5-metno code, only replace sys layer with libloading

## Crate Structure

```
crates/tensor4all-hdf5-ffi/
├── Cargo.toml
└── src/
    ├── lib.rs              # hdf5-metno's lib.rs + init export
    ├── init.rs             # NEW: hdf5_init() only
    ├── sys/                # NEW: libloading-based (replaces hdf5-metno-sys)
    │   ├── mod.rs
    │   ├── types.rs        # hid_t, herr_t, hsize_t, etc.
    │   ├── constants.rs    # H5F_ACC_RDONLY, H5P_DEFAULT, etc.
    │   └── functions.rs    # Function table and wrappers
    ├── hl/                 # Copy from hdf5-metno/src/hl/
    │   ├── mod.rs
    │   ├── file.rs
    │   ├── group.rs
    │   ├── dataset.rs
    │   ├── attribute.rs
    │   ├── datatype.rs
    │   ├── dataspace.rs
    │   ├── plist.rs
    │   └── ...
    └── types/              # Copy from hdf5-metno-types
        ├── mod.rs
        ├── string.rs       # VarLenUnicode, FixedUnicode, VarLenAscii
        ├── complex.rs      # Complex64 support
        └── h5type.rs       # H5Type trait
```

## API

### Rust API (tensor4all-hdf5-ffi)

```rust
/// Initialize HDF5 by loading the library from the given path.
/// Must be called before any HDF5 operations.
///
/// Returns Ok(()) if already initialized with the same path.
/// Returns Err if already initialized with a different path.
pub fn hdf5_init(library_path: &str) -> Result<(), Hdf5Error>;

/// Check if HDF5 has been initialized.
pub fn hdf5_is_initialized() -> bool;

/// Get the path used for initialization (if any).
pub fn hdf5_library_path() -> Option<String>;
```

### Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum Hdf5Error {
    #[error("HDF5 not initialized - call hdf5_init() first")]
    NotInitialized,

    #[error("Failed to load HDF5 library from '{path}': {source}")]
    LibraryLoad { path: String, source: libloading::Error },

    #[error("HDF5 error: {0}")]
    Hdf5(String),

    #[error("Already initialized with different path: {0}")]
    AlreadyInitialized(String),
}
```

### C API (tensor4all-capi)

```c
// Initialize HDF5 library from the given path.
// Returns: 0=success, -1=null path, -2=load failed, -3=already initialized differently
int32_t t4a_hdf5_init(const char* library_path);

// Check if HDF5 has been initialized. Returns 1 if initialized, 0 otherwise.
int32_t t4a_hdf5_is_initialized(void);
```

## sys Module (libloading-based)

### Types (same as hdf5-metno-sys)

```rust
// sys/types.rs
pub type hid_t = i64;
pub type herr_t = i32;
pub type hsize_t = u64;
pub type hssize_t = i64;
pub type htri_t = i32;
```

### Function Loading

```rust
// sys/mod.rs
use libloading::Library;
use std::sync::OnceLock;

static LIB: OnceLock<LibState> = OnceLock::new();

struct LibState {
    _lib: Library,
    funcs: Functions,
    path: String,
}

pub fn load_library(path: &str) -> Result<(), Hdf5Error> {
    LIB.get_or_try_init(|| { /* load all symbols */ })?;
    Ok(())
}

fn funcs() -> &'static Functions {
    &LIB.get().expect("HDF5 not initialized").funcs
}
```

### Function Wrappers

```rust
// sys/functions.rs
pub struct Functions {
    H5Fcreate: unsafe extern "C" fn(...) -> hid_t,
    H5Fopen: unsafe extern "C" fn(...) -> hid_t,
    // ... ~20 functions total
}

// Wrappers with same signature as hdf5-metno-sys
pub unsafe fn H5Fcreate(...) -> hid_t {
    (funcs().H5Fcreate)(...)
}
```

### Required HDF5 C Functions (~20 total)

| Category | Functions |
|----------|-----------|
| File | `H5Fcreate`, `H5Fopen`, `H5Fclose` |
| Group | `H5Gcreate2`, `H5Gopen2`, `H5Gclose` |
| Dataset | `H5Dcreate2`, `H5Dopen2`, `H5Dclose`, `H5Dread`, `H5Dwrite`, `H5Dget_space` |
| Attribute | `H5Acreate2`, `H5Aopen`, `H5Aclose`, `H5Aread`, `H5Awrite` |
| Dataspace | `H5Screate_simple`, `H5Sclose` |
| Datatype | `H5Tcopy`, `H5Tclose`, `H5Tvlen_create` + predefined type IDs |
| Error | `H5Eget_msg` |

## Language Bindings

### Julia (Tensor4all.jl)

```julia
using HDF5_jll

function init_hdf5()
    if ccall((:t4a_hdf5_is_initialized, libtensor4all), Cint, ()) == 0
        libpath = HDF5_jll.libhdf5
        ret = ccall((:t4a_hdf5_init, libtensor4all), Cint, (Cstring,), libpath)
        ret == 0 || error("Failed to initialize HDF5: $ret")
    end
end

# Auto-initialize before HDF5 operations
function save_mps(filepath, name, tt)
    init_hdf5()
    # ...
end
```

### Python

```python
def _find_hdf5_library():
    """Find HDF5 library path from h5py or system."""
    try:
        import h5py
        return h5py.h5.get_config().lib_path
    except:
        pass
    # Fallback to common system paths...

def init_hdf5(path=None):
    if path is None:
        path = _find_hdf5_library()
    ret = _lib.t4a_hdf5_init(path.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to initialize HDF5: {ret}")
```

## tensor4all-hdf5 Changes

Only Cargo.toml change needed:

```toml
# Before
hdf5 = { package = "hdf5-metno", version = "0.12", features = ["complex"] }

# After
hdf5 = { package = "tensor4all-hdf5-ffi", path = "../tensor4all-hdf5-ffi" }
```

Source code unchanged — API is identical.

## Implementation Steps

1. **Analyze hdf5-metno**
   - Map module dependencies
   - List all hdf5-sys functions used

2. **Create tensor4all-hdf5-ffi crate**
   - Implement `sys/` module (libloading-based)
   - Copy `types/` from hdf5-metno-types
   - Copy `hl/` from hdf5-metno
   - Replace `hdf5_sys` → `crate::sys` imports

3. **Update tensor4all-hdf5**
   - Change Cargo.toml dependency
   - Verify existing tests pass

4. **Add C API to tensor4all-capi**
   - `t4a_hdf5_init()`, `t4a_hdf5_is_initialized()`

5. **Update language bindings**
   - Julia: Tensor4all.jl (separate repository)
   - Python: `python/tensor4all/`

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| hdf5-metno internal API changes | Copy minimal required modules only |
| HDF5 version incompatibility | Use only stable C API functions |
| Thread safety | Document HDF5's own constraints |

## License

hdf5-metno is MIT/Apache-2.0 dual licensed — compatible with tensor4all-rs (MIT).
