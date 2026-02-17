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
mod quanticsgrids;
mod quanticstci;
mod quanticstransform;
mod simplett;
mod tensor;
mod tensorci;
mod treetn;
mod types;

pub use algorithm::*;
#[cfg(feature = "hdf5")]
pub use hdf5::*;
pub use index::*;
pub use quanticsgrids::*;
pub use quanticstci::*;
pub use quanticstransform::*;
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

// ============================================================================
// Thread-local error message storage
// ============================================================================

use std::cell::RefCell;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

/// Store an error message in thread-local storage.
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = msg.to_string();
    });
}

/// Store an error and return a status code.
pub(crate) fn err_status<E: std::fmt::Display>(err: E, code: StatusCode) -> StatusCode {
    set_last_error(&err.to_string());
    code
}

/// Store an error and return a null pointer.
pub(crate) fn err_null<T, E: std::fmt::Display>(err: E) -> *mut T {
    set_last_error(&err.to_string());
    std::ptr::null_mut()
}

/// Unwrap a `catch_unwind` result, storing any panic message.
pub(crate) fn unwrap_catch(result: std::thread::Result<StatusCode>) -> StatusCode {
    match result {
        Ok(code) => code,
        Err(panic) => {
            let msg = panic_message(&panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

/// Unwrap a `catch_unwind` pointer result, storing any panic message.
pub(crate) fn unwrap_catch_ptr<T>(result: std::thread::Result<*mut T>) -> *mut T {
    match result {
        Ok(ptr) => ptr,
        Err(panic) => {
            let msg = panic_message(&panic);
            set_last_error(&msg);
            std::ptr::null_mut()
        }
    }
}

fn panic_message(info: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = info.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = info.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    }
}

/// Retrieve the last error message.
///
/// # Arguments
/// * `buf` - Output buffer for the error message (UTF-8, null-terminated).
///           Pass null to query required length only.
/// * `buf_len` - Size of the buffer in bytes.
/// * `out_len` - Output: required buffer length including null terminator.
///
/// # Returns
/// * `T4A_SUCCESS` - Message written (or length query succeeded).
/// * `T4A_NULL_POINTER` - `out_len` is null.
/// * `T4A_BUFFER_TOO_SMALL` - Buffer too small; `out_len` has the required size.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_last_error_message(
    buf: *mut u8,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    LAST_ERROR.with(|cell| {
        let msg = cell.borrow();
        let required_len = msg.len() + 1; // +1 for null terminator

        unsafe { *out_len = required_len };

        if buf.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < required_len {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(msg.as_ptr(), buf, msg.len());
            *buf.add(msg.len()) = 0; // null terminator
        }

        T4A_SUCCESS
    })
}
