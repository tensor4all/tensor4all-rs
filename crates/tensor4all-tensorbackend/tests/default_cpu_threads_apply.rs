//! `set_default_cpu_threads` applies to the first [`CpuBackend::new`] of a process.
//!
//! This is its own test binary because the default is read once per process: inside
//! a binary shared with other tests, another thread's [`CpuBackend::new`] could
//! legitimately fix the value first.

use std::num::NonZeroUsize;

use tensor4all_tensorbackend::{set_default_cpu_threads, CpuBackend};

#[test]
fn default_cpu_threads_apply_to_new_backends() {
    set_default_cpu_threads(NonZeroUsize::new(2).expect("two workers"))
        .expect("the first set wins");
    let backend = CpuBackend::new();
    assert_eq!(backend.num_threads(), 2);
    assert!(backend.engine().pool().is_some());
}
