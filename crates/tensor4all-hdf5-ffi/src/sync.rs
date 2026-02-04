//! Synchronization primitives for HDF5 operations.
//!
//! HDF5 is not thread-safe by default, so we need a global lock.
//!
//! In link mode, we use the lock from hdf5-sys to ensure consistency
//! with other code using hdf5-metno. In runtime-loading mode, we use
//! our own lock.

// =============================================================================
// Link mode: use hdf5-sys LOCK
// =============================================================================
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
mod link_impl {
    /// Guards the execution of the provided closure with the hdf5-sys global mutex.
    pub fn sync<T, F>(func: F) -> T
    where
        F: FnOnce() -> T,
    {
        let _guard = hdf5_sys::LOCK.lock();
        func()
    }
}

// =============================================================================
// Runtime-loading mode: use our own lock
// =============================================================================
#[cfg(feature = "runtime-loading")]
mod runtime_impl {
    use parking_lot::ReentrantMutex;
    use std::sync::LazyLock;

    /// Global reentrant mutex for HDF5 operations (runtime-loading mode).
    pub static LOCK: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));

    /// Guards the execution of the provided closure with a recursive static mutex.
    pub fn sync<T, F>(func: F) -> T
    where
        F: FnOnce() -> T,
    {
        let _guard = LOCK.lock();
        func()
    }
}

// =============================================================================
// Public API
// =============================================================================
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub use link_impl::sync;

#[cfg(feature = "runtime-loading")]
pub use runtime_impl::sync;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_sync_reentrant() {
        // Test that sync is reentrant (can be called nested)
        let result = sync(|| sync(|| sync(|| 42)));
        assert_eq!(result, 42);
    }
}
