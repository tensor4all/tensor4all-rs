//! Synchronization primitives for HDF5 operations.
//!
//! HDF5 is not thread-safe by default, so we need a global lock.

use std::sync::LazyLock;

use parking_lot::ReentrantMutex;

/// Global reentrant mutex for HDF5 operations.
pub static LOCK: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));

/// Guards the execution of the provided closure with a recursive static mutex.
pub fn sync<T, F>(func: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = LOCK.lock();
    func()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_reentrant_mutex() {
        let g1 = LOCK.try_lock();
        assert!(g1.is_some());
        let g2 = LOCK.lock();
        assert_eq!(*g2, ());
        let g3 = LOCK.try_lock();
        assert!(g3.is_some());
        let g4 = LOCK.lock();
        assert_eq!(*g4, ());
    }
}
