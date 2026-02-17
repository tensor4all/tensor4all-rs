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
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Store an error message in thread-local storage.
pub(crate) fn set_last_error(msg: &str) {
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
            let msg = panic_message(&*panic);
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
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            std::ptr::null_mut()
        }
    }
}

pub(crate) fn panic_message(info: &(dyn std::any::Any + Send)) -> String {
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
///   Pass null to query required length only.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: read the last error message as a Rust String.
    fn read_last_error() -> String {
        let mut out_len: libc::size_t = 0;
        // Query required length
        let status =
            t4a_last_error_message(std::ptr::null_mut(), 0, &mut out_len as *mut libc::size_t);
        assert_eq!(status, T4A_SUCCESS);
        if out_len <= 1 {
            return String::new();
        }
        let mut buf = vec![0u8; out_len];
        let status = t4a_last_error_message(
            buf.as_mut_ptr(),
            buf.len(),
            &mut out_len as *mut libc::size_t,
        );
        assert_eq!(status, T4A_SUCCESS);
        // Exclude null terminator
        String::from_utf8(buf[..out_len - 1].to_vec()).unwrap()
    }

    #[test]
    fn test_last_error_message_null_out_len() {
        let status = t4a_last_error_message(std::ptr::null_mut(), 0, std::ptr::null_mut());
        assert_eq!(status, T4A_NULL_POINTER);
    }

    #[test]
    fn test_last_error_message_roundtrip() {
        set_last_error("test error message");
        let msg = read_last_error();
        assert_eq!(msg, "test error message");
    }

    #[test]
    fn test_last_error_message_buffer_too_small() {
        set_last_error("hello");
        let mut out_len: libc::size_t = 0;
        let mut buf = [0u8; 2]; // too small for "hello\0"
        let status = t4a_last_error_message(
            buf.as_mut_ptr(),
            buf.len(),
            &mut out_len as *mut libc::size_t,
        );
        assert_eq!(status, T4A_BUFFER_TOO_SMALL);
        assert_eq!(out_len, 6); // "hello" + null
    }

    #[test]
    fn test_last_error_message_query_length_only() {
        set_last_error("abc");
        let mut out_len: libc::size_t = 0;
        let status =
            t4a_last_error_message(std::ptr::null_mut(), 0, &mut out_len as *mut libc::size_t);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(out_len, 4); // "abc" + null
    }

    #[test]
    fn test_err_status_stores_message() {
        let code = err_status("something failed", T4A_INVALID_ARGUMENT);
        assert_eq!(code, T4A_INVALID_ARGUMENT);
        assert_eq!(read_last_error(), "something failed");
    }

    #[test]
    fn test_err_null_stores_message() {
        let ptr: *mut u8 = err_null("null error");
        assert!(ptr.is_null());
        assert_eq!(read_last_error(), "null error");
    }

    #[test]
    fn test_unwrap_catch_ok() {
        let result: std::thread::Result<StatusCode> = Ok(T4A_SUCCESS);
        assert_eq!(unwrap_catch(result), T4A_SUCCESS);
    }

    #[test]
    fn test_unwrap_catch_panic() {
        let result: std::thread::Result<StatusCode> =
            std::panic::catch_unwind(|| panic!("test panic"));
        assert_eq!(unwrap_catch(result), T4A_INTERNAL_ERROR);
        assert_eq!(read_last_error(), "test panic");
    }

    #[test]
    fn test_unwrap_catch_ptr_ok() {
        let mut val = 42u8;
        let result: std::thread::Result<*mut u8> = Ok(&mut val as *mut u8);
        let ptr = unwrap_catch_ptr(result);
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_unwrap_catch_ptr_panic() {
        let result: std::thread::Result<*mut u8> = std::panic::catch_unwind(|| panic!("ptr panic"));
        let ptr = unwrap_catch_ptr(result);
        assert!(ptr.is_null());
        assert_eq!(read_last_error(), "ptr panic");
    }

    #[test]
    fn test_panic_message_str() {
        let result = std::panic::catch_unwind(|| panic!("str msg"));
        let msg = panic_message(&*result.unwrap_err());
        assert_eq!(msg, "str msg");
    }

    #[test]
    fn test_panic_message_string() {
        let result = std::panic::catch_unwind(|| panic!("{}", "string msg"));
        let msg = panic_message(&*result.unwrap_err());
        assert_eq!(msg, "string msg");
    }

    #[test]
    fn test_panic_message_unknown() {
        let result = std::panic::catch_unwind(|| std::panic::panic_any(42i32));
        let msg = panic_message(&*result.unwrap_err());
        assert_eq!(msg, "Unknown panic");
    }
}
