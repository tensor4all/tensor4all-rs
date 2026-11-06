//! `set_default_cpu_wait_policy` applies to later backends and is rejected the second time.
//!
//! This is its own test binary because the default is fixed once per process.

use tensor4all_tensorbackend::{
    default_cpu_wait_policy, set_default_cpu_wait_policy, CpuBackend, CpuWaitPolicy,
    WaitPolicyAlreadyInitialized,
};

#[test]
fn default_wait_policy_is_set_once() {
    assert_eq!(default_cpu_wait_policy(), CpuWaitPolicy::Active);
    set_default_cpu_wait_policy(CpuWaitPolicy::Passive).expect("the first set wins");
    assert_eq!(default_cpu_wait_policy(), CpuWaitPolicy::Passive);
    assert_eq!(
        set_default_cpu_wait_policy(CpuWaitPolicy::SpinPolls(10)),
        Err(WaitPolicyAlreadyInitialized)
    );
    assert_eq!(default_cpu_wait_policy(), CpuWaitPolicy::Passive);
    // A default backend and an explicit-count backend still build and keep their width.
    assert!(CpuBackend::new().num_threads() >= 1);
    assert_eq!(
        CpuBackend::with_threads(2).expect("backend").num_threads(),
        2
    );
}
