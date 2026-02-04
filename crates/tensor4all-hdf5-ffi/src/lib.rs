// Allow various clippy warnings for FFI code ported from hdf5-metno
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(dead_code)]
#![allow(unexpected_cfgs)]
#![allow(non_snake_case)]
#![allow(mismatched_lifetime_syntaxes)]

//! HDF5 bindings with runtime library loading (dlopen) for tensor4all-rs.
//!
//! This crate provides HDF5 functionality compatible with `hdf5-metno`, but loads
//! the HDF5 library at runtime using `libloading` instead of linking at build time.
//!
//! # Initialization
//!
//! Before using any HDF5 operations, you must initialize the library:
//!
//! ```ignore
//! use tensor4all_hdf5_ffi::hdf5_init;
//!
//! // Initialize with explicit library path
//! hdf5_init("/usr/lib/libhdf5.so")?;
//! ```

// Low-level FFI layer (replaces hdf5-sys)
pub mod sys;

// Type definitions (from hdf5-types)
pub mod types;

// Global constants (H5T_NATIVE_*, H5P_*, etc.)
pub mod globals;

// Error handling
mod error;
pub use error::{h5check, Error, Result};

// Synchronization
mod sync;
pub use sync::sync;

// Utility functions
mod util;
pub use util::{get_h5_str, string_from_cstr, to_cstring};

// Macros (must come before modules that use them)
#[macro_use]
mod macros;

// Core types
mod class;
mod dim;
mod handle;

pub use class::{from_id, ObjectClass};
pub use dim::{Dimension, Ix};
pub use handle::Handle;

// High-level API
pub mod hl;

// Re-export high-level types for convenience
pub use hl::{
    Attribute, AttributeBuilder, AttributeReader, AttributeWriter, Dataset, DatasetBuilder,
    DatasetReader, DatasetWriter, Dataspace, Datatype, File, FileBuilder, FixedAscii, FixedUnicode,
    Group, H5Type, H5TypeBuilder, OpenMode, VarLenAscii, VarLenUnicode,
};

// Initialization API
mod init;
pub use init::{ensure_hdf5_init, hdf5_init, hdf5_is_initialized, hdf5_library_path};

// C API
pub mod capi;
pub use capi::{
    hdf5_ffi_init, hdf5_ffi_is_initialized, hdf5_ffi_library_path, hdf5_ffi_status_message,
    hdf5_ffi_version, StatusCode, HDF5_FFI_ALREADY_INITIALIZED, HDF5_FFI_BUFFER_TOO_SMALL,
    HDF5_FFI_INTERNAL_ERROR, HDF5_FFI_INVALID_ARGUMENT, HDF5_FFI_LIBRARY_LOAD_ERROR,
    HDF5_FFI_NOT_INITIALIZED, HDF5_FFI_NULL_POINTER, HDF5_FFI_SUCCESS,
};
