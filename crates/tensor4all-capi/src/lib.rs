#![warn(missing_docs)]
//! C API for tensor4all Rust implementation
//!
//! This crate provides a C-compatible interface to the tensor4all library,
//! enabling usage from languages like Julia, Python, and C++.
//!
//! ## Milestone 1: Index types
//!
//! The initial implementation provides wrappers for:
//! - `DynIndex` (`Index<DynId, TagSet>`) - the default dynamic index type
//!
//! ## Milestone 2: Tensor types
//!
//! Extended to support:
//! - `TensorDynLen` with Dense/Diag storage
//! - Bidirectional conversion with ITensors.ITensor
//!
//! ## Design patterns
//!
//! Following the patterns from `sparse-ir-capi`:
//! - Opaque pointers with `_private: *const c_void`
//! - Explicit lifecycle functions: `*_release`, `*_clone`, `*_is_assigned`
//! - Status codes for error handling
//! - `catch_unwind` to prevent Rust panics from crossing FFI boundary

// C API requires unsafe operations with raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[macro_use]
mod macros;

mod algorithm;
#[cfg(feature = "hdf5")]
mod hdf5;
mod index;
mod simplett;
mod tensor;
mod tensorci;
mod treetn;
mod types;

pub use algorithm::*;
#[cfg(feature = "hdf5")]
pub use hdf5::*;
pub use index::*;
pub use simplett::*;
pub use tensor::*;
pub use tensorci::*;
pub use treetn::*;
pub use types::*;

/// Status code type for C API
pub type StatusCode = libc::c_int;

/// Operation completed successfully.
pub const T4A_SUCCESS: StatusCode = 0;
/// A null pointer was passed where a valid pointer was required.
pub const T4A_NULL_POINTER: StatusCode = -1;
/// An invalid argument was provided.
pub const T4A_INVALID_ARGUMENT: StatusCode = -2;
/// Too many tags would be added to a TagSet (exceeds maximum).
pub const T4A_TAG_OVERFLOW: StatusCode = -3;
/// A tag string exceeds the maximum allowed length.
pub const T4A_TAG_TOO_LONG: StatusCode = -4;
/// The provided output buffer is too small for the result.
pub const T4A_BUFFER_TOO_SMALL: StatusCode = -5;
/// An internal error occurred (e.g., a panic was caught).
pub const T4A_INTERNAL_ERROR: StatusCode = -6;
