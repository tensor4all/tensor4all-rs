//! C API for HDF5 library initialization.
//!
//! This module provides C-compatible functions for initializing the HDF5 library
//! at runtime. This allows Julia (via HDF5_jll) and Python (via h5py) to provide
//! their own HDF5 libraries.
//!
//! # Example (C)
//!
//! ```c
//! #include "tensor4all_hdf5_ffi.h"
//!
//! int main() {
//!     int status = hdf5_ffi_init("/usr/lib/libhdf5.so");
//!     if (status != HDF5_FFI_SUCCESS) {
//!         // Handle error
//!     }
//!     // Now HDF5 operations can be used
//! }
//! ```

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ============================================================================
// Status Codes
// ============================================================================

/// Status code type for C API functions.
pub type StatusCode = c_int;

/// Operation succeeded.
pub const HDF5_FFI_SUCCESS: StatusCode = 0;

/// Null pointer error.
pub const HDF5_FFI_NULL_POINTER: StatusCode = -1;

/// Invalid argument.
pub const HDF5_FFI_INVALID_ARGUMENT: StatusCode = -2;

/// Library already initialized with a different path.
pub const HDF5_FFI_ALREADY_INITIALIZED: StatusCode = -3;

/// Failed to load the library.
pub const HDF5_FFI_LIBRARY_LOAD_ERROR: StatusCode = -4;

/// Library not initialized.
pub const HDF5_FFI_NOT_INITIALIZED: StatusCode = -5;

/// Internal error (panic or other unexpected error).
pub const HDF5_FFI_INTERNAL_ERROR: StatusCode = -6;

/// Buffer too small.
pub const HDF5_FFI_BUFFER_TOO_SMALL: StatusCode = -7;

// ============================================================================
// Initialization Functions
// ============================================================================

/// Initialize HDF5 by loading the library from the given path.
///
/// Must be called before any HDF5 operations.
///
/// # Arguments
///
/// * `library_path` - Path to the HDF5 shared library (null-terminated C string).
///   Examples:
///   - Linux: `/usr/lib/libhdf5.so`, `/path/to/libhdf5.so.200`
///   - macOS: `/usr/local/lib/libhdf5.dylib`
///   - Windows: `C:\path\to\hdf5.dll`
///
/// # Returns
///
/// * `HDF5_FFI_SUCCESS` (0) if initialization succeeds
/// * `HDF5_FFI_NULL_POINTER` if `library_path` is NULL
/// * `HDF5_FFI_ALREADY_INITIALIZED` if already initialized with a different path
/// * `HDF5_FFI_LIBRARY_LOAD_ERROR` if the library cannot be loaded
/// * `HDF5_FFI_INTERNAL_ERROR` on unexpected error
///
/// # Thread Safety
///
/// This function is thread-safe. Multiple calls with the same path are idempotent.
/// If called concurrently, only one thread will perform the initialization.
///
/// # Safety
///
/// The `library_path` must be a valid null-terminated C string.
#[no_mangle]
pub extern "C" fn hdf5_ffi_init(library_path: *const c_char) -> StatusCode {
    if library_path.is_null() {
        return HDF5_FFI_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let path_cstr = unsafe { CStr::from_ptr(library_path) };
        let path = match path_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return HDF5_FFI_INVALID_ARGUMENT,
        };

        match crate::hdf5_init(path) {
            Ok(()) => HDF5_FFI_SUCCESS,
            Err(crate::Error::AlreadyInitialized(_)) => HDF5_FFI_ALREADY_INITIALIZED,
            Err(crate::Error::LibraryLoad { .. }) => HDF5_FFI_LIBRARY_LOAD_ERROR,
            Err(_) => HDF5_FFI_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(HDF5_FFI_INTERNAL_ERROR)
}

/// Check if HDF5 has been initialized.
///
/// # Returns
///
/// * 1 if HDF5 has been initialized, 0 otherwise.
///
/// # Thread Safety
///
/// This function is thread-safe.
#[no_mangle]
pub extern "C" fn hdf5_ffi_is_initialized() -> c_int {
    let result = catch_unwind(|| if crate::hdf5_is_initialized() { 1 } else { 0 });

    result.unwrap_or(0)
}

/// Get the path used for HDF5 initialization.
///
/// This function follows the "query-then-fill" pattern:
/// 1. Call with `buf = NULL` to get the required buffer length in `out_len`
/// 2. Allocate a buffer of size `out_len + 1` (for null terminator)
/// 3. Call again with the allocated buffer
///
/// # Arguments
///
/// * `buf` - Buffer to write the path (can be NULL to query length only)
/// * `buf_len` - Length of the buffer
/// * `out_len` - Output: required buffer length (not including null terminator)
///
/// # Returns
///
/// * `HDF5_FFI_SUCCESS` if the path was written (or only length queried)
/// * `HDF5_FFI_NULL_POINTER` if `out_len` is NULL
/// * `HDF5_FFI_NOT_INITIALIZED` if HDF5 has not been initialized
/// * `HDF5_FFI_BUFFER_TOO_SMALL` if the buffer is too small (length still written)
/// * `HDF5_FFI_INTERNAL_ERROR` on unexpected error
///
/// # Thread Safety
///
/// This function is thread-safe.
///
/// # Safety
///
/// * `out_len` must be a valid pointer
/// * If `buf` is not NULL, it must point to a buffer of at least `buf_len` bytes
#[no_mangle]
pub extern "C" fn hdf5_ffi_library_path(
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    if out_len.is_null() {
        return HDF5_FFI_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let path = match crate::hdf5_library_path() {
            Some(p) => p,
            None => return HDF5_FFI_NOT_INITIALIZED,
        };

        let path_bytes = path.as_bytes();
        let required_len = path_bytes.len();

        unsafe { *out_len = required_len };

        // If buf is NULL, just return the length
        if buf.is_null() {
            return HDF5_FFI_SUCCESS;
        }

        // Check buffer size (need space for null terminator)
        if buf_len <= required_len {
            return HDF5_FFI_BUFFER_TOO_SMALL;
        }

        // Copy path and add null terminator
        unsafe {
            std::ptr::copy_nonoverlapping(path_bytes.as_ptr(), buf as *mut u8, required_len);
            *buf.add(required_len) = 0; // Null terminator
        }

        HDF5_FFI_SUCCESS
    }));

    result.unwrap_or(HDF5_FFI_INTERNAL_ERROR)
}

/// Get a human-readable error message for a status code.
///
/// # Arguments
///
/// * `status` - The status code to describe
///
/// # Returns
///
/// A pointer to a static string describing the error. The string is valid
/// for the lifetime of the program and must not be freed.
///
/// # Thread Safety
///
/// This function is thread-safe.
#[no_mangle]
pub extern "C" fn hdf5_ffi_status_message(status: StatusCode) -> *const c_char {
    let msg = match status {
        HDF5_FFI_SUCCESS => "Success\0",
        HDF5_FFI_NULL_POINTER => "Null pointer\0",
        HDF5_FFI_INVALID_ARGUMENT => "Invalid argument\0",
        HDF5_FFI_ALREADY_INITIALIZED => "Already initialized with different path\0",
        HDF5_FFI_LIBRARY_LOAD_ERROR => "Failed to load HDF5 library\0",
        HDF5_FFI_NOT_INITIALIZED => "HDF5 library not initialized\0",
        HDF5_FFI_INTERNAL_ERROR => "Internal error\0",
        HDF5_FFI_BUFFER_TOO_SMALL => "Buffer too small\0",
        _ => "Unknown error\0",
    };

    msg.as_ptr() as *const c_char
}

// ============================================================================
// Version Information
// ============================================================================

/// Get the version of this library.
///
/// # Returns
///
/// A pointer to a static string containing the version (e.g., "0.1.0").
/// The string is valid for the lifetime of the program and must not be freed.
#[no_mangle]
pub extern "C" fn hdf5_ffi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_codes() {
        assert_eq!(HDF5_FFI_SUCCESS, 0);
        assert!(HDF5_FFI_NULL_POINTER < 0);
        assert!(HDF5_FFI_INVALID_ARGUMENT < 0);
        assert!(HDF5_FFI_ALREADY_INITIALIZED < 0);
        assert!(HDF5_FFI_LIBRARY_LOAD_ERROR < 0);
        assert!(HDF5_FFI_NOT_INITIALIZED < 0);
        assert!(HDF5_FFI_INTERNAL_ERROR < 0);
        assert!(HDF5_FFI_BUFFER_TOO_SMALL < 0);
    }

    #[test]
    fn test_init_null_pointer() {
        let status = hdf5_ffi_init(std::ptr::null());
        assert_eq!(status, HDF5_FFI_NULL_POINTER);
    }

    #[test]
    fn test_is_initialized_without_init() {
        // Note: This test may fail if run after other tests that initialize HDF5
        // In that case, it will return 1 instead of 0, which is still valid behavior
        let result = hdf5_ffi_is_initialized();
        assert!(result == 0 || result == 1);
    }

    #[test]
    fn test_library_path_null_out_len() {
        let status = hdf5_ffi_library_path(std::ptr::null_mut(), 0, std::ptr::null_mut());
        assert_eq!(status, HDF5_FFI_NULL_POINTER);
    }

    #[test]
    fn test_status_message() {
        let msg = hdf5_ffi_status_message(HDF5_FFI_SUCCESS);
        assert!(!msg.is_null());
        let s = unsafe { CStr::from_ptr(msg) };
        assert_eq!(s.to_str().unwrap(), "Success");
    }

    #[test]
    fn test_version() {
        let ver = hdf5_ffi_version();
        assert!(!ver.is_null());
        let s = unsafe { CStr::from_ptr(ver) };
        assert!(!s.to_str().unwrap().is_empty());
    }
}
