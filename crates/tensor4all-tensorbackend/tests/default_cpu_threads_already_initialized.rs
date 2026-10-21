//! `set_default_cpu_threads` is rejected once the default is fixed.
//!
//! This is its own test binary because the default is read once per process: the
//! first [`CpuBackend::new`] here is what fixes it.

use std::num::NonZeroUsize;

use tensor4all_tensorbackend::{set_default_cpu_threads, AlreadyInitialized, CpuBackend};

#[test]
fn set_after_initialization_is_rejected() {
    let backend = CpuBackend::new();
    assert!(backend.num_threads() >= 1, "a default backend has workers");
    assert_eq!(
        set_default_cpu_threads(NonZeroUsize::new(2).expect("two workers")),
        Err(AlreadyInitialized)
    );
}
