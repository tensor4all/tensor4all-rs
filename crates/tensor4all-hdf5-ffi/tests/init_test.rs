//! Basic initialization tests for tensor4all-hdf5-ffi.

use tensor4all_hdf5_ffi::{hdf5_init, hdf5_is_initialized, hdf5_library_path};

/// Test that we can check initialization status before init.
#[test]
fn test_not_initialized() {
    // Note: This test may fail if run after other tests that initialize the library.
    // To properly test this, run it in isolation.
    if !hdf5_is_initialized() {
        assert!(hdf5_library_path().is_none());
    }
}

/// Test that initialization with invalid path fails (runtime-loading mode only).
/// This test only applies when link feature is NOT enabled (true runtime loading).
#[test]
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
fn test_init_invalid_path() {
    use tensor4all_hdf5_ffi::Error;

    let result = hdf5_init("/nonexistent/path/libhdf5.so");
    assert!(result.is_err());
    if let Err(Error::LibraryLoad { path, .. }) = result {
        assert!(path.contains("nonexistent"));
    }
}

/// Test that initialization succeeds in link mode (with runtime-loading feature also enabled).
/// When both features are enabled, link mode takes precedence.
#[test]
#[cfg(all(feature = "link", feature = "runtime-loading"))]
fn test_init_link_mode_with_runtime_loading() {
    // When both features are enabled, link mode takes precedence
    // so hdf5_init is a no-op and always succeeds
    let result = hdf5_init("");
    assert!(result.is_ok());
    assert!(hdf5_is_initialized());
}

/// Test that initialization succeeds in link mode (no-op).
#[test]
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
fn test_init_link_mode() {
    // In link mode, hdf5_init is a no-op and always succeeds
    let result = hdf5_init("");
    assert!(result.is_ok());
    assert!(hdf5_is_initialized());
}
