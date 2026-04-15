#![warn(missing_docs)]
//! C API for tensor4all Rust implementation
//!
//! This crate provides the minimal Julia-facing C-compatible interface to the
//! tensor4all library.
//!
//! The retained public surface is intentionally small:
//! - `Index`
//! - `Tensor`
//! - `TreeTN`
//! - canonical QTT layout descriptors and transform materialization entrypoints
//!
//! Convenience subsystems such as SimpleTT, TreeTCI, QuanticsTCI, quantics grid
//! objects, and HDF5 are not part of the redesigned C surface.

// C API requires unsafe operations with raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

mod index;
mod quanticstransform;
mod tensor;
mod treetn;
mod types;

pub use index::*;
pub use quanticstransform::*;
pub use tensor::*;
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
/// The requested API exists but is not implemented yet.
pub const T4A_NOT_IMPLEMENTED: StatusCode = -7;

/// Internal result type carrying both a status code and an error message.
pub(crate) type CapiResult<T> = Result<T, (StatusCode, String)>;

// ============================================================================
// Thread-local error message storage
// ============================================================================

use std::cell::RefCell;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Store an error message in thread-local storage.
///
/// When the `RUST_BACKTRACE` environment variable is set (to `1` or `full`),
/// a backtrace is captured and appended to the message.
pub(crate) fn set_last_error(msg: &str) {
    let bt = std::backtrace::Backtrace::capture();
    let full_msg = match bt.status() {
        std::backtrace::BacktraceStatus::Captured => {
            format!("{msg}\n\nRust backtrace:\n{bt}")
        }
        _ => msg.to_string(),
    };
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = full_msg;
    });
}

/// Store an error and return a status code.
pub(crate) fn err_status<E: std::fmt::Display>(err: E, code: StatusCode) -> StatusCode {
    set_last_error(&err.to_string());
    code
}

/// Store an error and return a null pointer.
#[allow(dead_code)]
pub(crate) fn err_null<T, E: std::fmt::Display>(err: E) -> *mut T {
    set_last_error(&err.to_string());
    std::ptr::null_mut()
}

/// Pair a status code with an error message for `run_catching`.
pub(crate) fn capi_error<E: std::fmt::Display>(code: StatusCode, err: E) -> (StatusCode, String) {
    (code, err.to_string())
}

/// Run a constructor-like closure and write the resulting handle to `out`.
pub(crate) fn run_catching<T, F>(out: *mut *mut T, f: F) -> StatusCode
where
    F: FnOnce() -> CapiResult<T>,
{
    if out.is_null() {
        return T4A_NULL_POINTER;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => {
            unsafe { *out = Box::into_raw(Box::new(value)) };
            T4A_SUCCESS
        }
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

/// Clone an opaque wrapper through the constructor-style `out` pattern.
pub(crate) fn clone_opaque<T: Clone>(src: *const T, out: *mut *mut T) -> StatusCode {
    if src.is_null() {
        return T4A_NULL_POINTER;
    }

    run_catching(out, || unsafe { Ok((&*src).clone()) })
}

/// Release an opaque wrapper allocated with `Box::into_raw`.
pub(crate) fn release_opaque<T>(obj: *mut T) {
    if obj.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(obj);
    }
}

/// Check whether an opaque wrapper pointer is valid.
pub(crate) fn is_assigned_opaque<T>(obj: *const T) -> i32 {
    if obj.is_null() {
        return 0;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*obj;
        1
    })) {
        Ok(v) => v,
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            0
        }
    }
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
#[allow(dead_code)]
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
mod tests;
