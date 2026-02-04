//! HDF5 library initialization.
//!
//! Two modes are supported:
//! - `link` feature (default): No initialization needed - library is linked at build time
//! - `runtime-loading` feature: Must call `hdf5_init()` before using HDF5 operations

use crate::error::Result;
use std::sync::Once;

static INIT: Once = Once::new();

/// Ensure HDF5 is initialized. Called internally before HDF5 operations.
///
/// In link mode, this calls H5open() to initialize the library.
/// In runtime-loading mode, this is a no-op (user must call hdf5_init first).
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
#[inline]
pub fn ensure_hdf5_init() {
    INIT.call_once(|| {
        crate::sync::sync(|| unsafe {
            hdf5_sys::h5::H5open();
        });
    });
}

/// Ensure HDF5 is initialized. Called internally before HDF5 operations.
#[cfg(feature = "runtime-loading")]
#[inline]
pub fn ensure_hdf5_init() {
    // In runtime-loading mode, user must call hdf5_init() first.
    // We just verify it's initialized.
    if !crate::sys::is_initialized() {
        panic!("HDF5 library not initialized. Call hdf5_init() first.");
    }
}

/// Initialize HDF5 by loading the library from the given path.
///
/// # Link mode (default)
///
/// With the `link` feature, this function is a no-op and always returns `Ok(())`.
/// HDF5 is already linked at build time.
///
/// # Runtime-loading mode
///
/// With the `runtime-loading` feature, this must be called before any HDF5 operations.
///
/// # Arguments
///
/// * `library_path` - Path to the HDF5 shared library (e.g., `/usr/lib/libhdf5.so`)
///
/// # Returns
///
/// * `Ok(())` if initialization succeeds or already initialized with the same path
/// * `Err(Error::AlreadyInitialized)` if already initialized with a different path
/// * `Err(Error::LibraryLoad)` if the library cannot be loaded
///
/// # Example
///
/// ```ignore
/// use tensor4all_hdf5_ffi::hdf5_init;
///
/// // With runtime-loading feature:
/// hdf5_init("/usr/lib/libhdf5.so")?;
///
/// // With link feature (default):
/// // This call is a no-op - HDF5 is already linked
/// hdf5_init("")?;
/// ```
#[cfg(feature = "runtime-loading")]
pub fn hdf5_init(library_path: &str) -> Result<()> {
    crate::sys::load_library(library_path)
}

/// Initialize HDF5 (no-op with link feature).
///
/// With the `link` feature, HDF5 is already linked at build time, so this
/// function does nothing and always returns `Ok(())`.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub fn hdf5_init(_library_path: &str) -> Result<()> {
    ensure_hdf5_init();
    Ok(())
}

/// Check if HDF5 has been initialized.
///
/// # Link mode (default)
///
/// Always returns `true` - library is always available.
///
/// # Runtime-loading mode
///
/// Returns `true` if [`hdf5_init`] has been successfully called, `false` otherwise.
#[cfg(feature = "runtime-loading")]
pub fn hdf5_is_initialized() -> bool {
    crate::sys::is_initialized()
}

/// Check if HDF5 has been initialized (always true with link feature).
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub fn hdf5_is_initialized() -> bool {
    true
}

/// Get the path used for HDF5 initialization.
///
/// # Link mode (default)
///
/// Always returns `None` - library path is determined at build time.
///
/// # Runtime-loading mode
///
/// Returns `Some(path)` if initialized, `None` otherwise.
#[cfg(feature = "runtime-loading")]
pub fn hdf5_library_path() -> Option<String> {
    crate::sys::library_path()
}

/// Get the path used for HDF5 initialization (None with link feature).
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub fn hdf5_library_path() -> Option<String> {
    None
}
