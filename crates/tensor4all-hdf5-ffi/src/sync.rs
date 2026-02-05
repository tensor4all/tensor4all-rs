//! Synchronization primitives for HDF5 operations.
//!
//! HDF5 is not thread-safe by default, so we need a global lock.
//!
//! This module follows the same pattern as hdf5-metno:
//! - LIBRARY_INIT ensures H5dont_atexit() and H5open() are called once
//! - sync() forces initialization before acquiring the lock

use std::sync::LazyLock;

// =============================================================================
// Link mode: use hdf5-sys LOCK
// Link mode takes precedence when both features are enabled.
// =============================================================================
#[cfg(feature = "link")]
mod link_impl {
    use super::*;

    /// Library initialization - called once before any HDF5 operations.
    /// This matches hdf5-metno's LIBRARY_INIT pattern.
    pub static LIBRARY_INIT: LazyLock<()> = LazyLock::new(|| {
        let _guard = hdf5_sys::LOCK.lock();
        unsafe {
            // Ensure HDF5 does not invalidate handles which might
            // still be live on other threads on program exit
            hdf5_sys::h5::H5dont_atexit();
            hdf5_sys::h5::H5open();
        }
    });

    /// Guards the execution of the provided closure with the hdf5-sys global mutex.
    /// Forces library initialization before acquiring the lock.
    pub fn sync<T, F>(func: F) -> T
    where
        F: FnOnce() -> T,
    {
        let _ = LazyLock::force(&LIBRARY_INIT);
        let _guard = hdf5_sys::LOCK.lock();
        func()
    }
}

// =============================================================================
// Runtime-loading mode: use our own lock
// Only compiled when link feature is not enabled.
// =============================================================================
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
mod runtime_impl {
    use super::*;
    use parking_lot::ReentrantMutex;

    /// Global reentrant mutex for HDF5 operations (runtime-loading mode).
    pub static LOCK: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));

    /// Library initialization for runtime-loading mode.
    /// User must call hdf5_init() first to load the library.
    pub static LIBRARY_INIT: LazyLock<()> = LazyLock::new(|| {
        let _guard = LOCK.lock();
        unsafe {
            crate::sys::H5dont_atexit();
            crate::sys::H5open();
        }
    });

    /// Guards the execution of the provided closure with a recursive static mutex.
    /// Forces library initialization before acquiring the lock.
    pub fn sync<T, F>(func: F) -> T
    where
        F: FnOnce() -> T,
    {
        // In runtime-loading mode, user must call hdf5_init() first
        if !crate::sys::is_initialized() {
            panic!("HDF5 library not initialized. Call hdf5_init() first.");
        }
        let _ = LazyLock::force(&LIBRARY_INIT);
        let _guard = LOCK.lock();
        func()
    }
}

// =============================================================================
// Public API
// Link mode takes precedence when both features are enabled.
// =============================================================================
#[cfg(feature = "link")]
pub use link_impl::{sync, LIBRARY_INIT};

#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
pub use runtime_impl::{sync, LIBRARY_INIT};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that sync is reentrant (can be called nested).
    /// In link mode, HDF5 is always available.
    #[test]
    #[cfg(feature = "link")]
    pub fn test_sync_reentrant() {
        let result = sync(|| sync(|| sync(|| 42)));
        assert_eq!(result, 42);
    }

    /// Test that sync is reentrant in runtime-loading mode.
    /// Requires HDF5_LIB environment variable to be set.
    #[test]
    #[cfg(all(feature = "runtime-loading", not(feature = "link")))]
    pub fn test_sync_reentrant() {
        // Skip if HDF5_LIB is not set
        let lib_path = match std::env::var("HDF5_LIB") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("Skipping test: HDF5_LIB environment variable not set");
                return;
            }
        };

        // Initialize HDF5
        crate::hdf5_init(&lib_path).expect("Failed to initialize HDF5");

        let result = sync(|| sync(|| sync(|| 42)));
        assert_eq!(result, 42);
    }
}
