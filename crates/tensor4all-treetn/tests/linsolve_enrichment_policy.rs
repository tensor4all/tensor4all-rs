//! Tests that change the process-global default SVD truncation policy.
//!
//! They live in their own test target so that a changed policy never reaches
//! tests running concurrently in the `integration` binary. Tests added here
//! share the policy with each other, so each must restore it and must not
//! depend on another test's setting.
mod enrichment_step;

use enrichment_step::{bond_dim, distance_from_00, enrich_once};
use tensor4all_core::{
    default_svd_truncation_policy, set_default_svd_truncation_policy, SvdTruncationPolicy,
};
use tensor4all_treetn::ResidualEnrichment;

/// Restores the global SVD policy even when the test panics.
struct DefaultSvdPolicy(SvdTruncationPolicy);

impl Drop for DefaultSvdPolicy {
    fn drop(&mut self) {
        set_default_svd_truncation_policy(self.0).unwrap();
    }
}

#[test]
fn enrichment_cutoff_is_not_preempted_by_the_global_svd_policy() {
    let _restore = DefaultSvdPolicy(default_svd_truncation_policy());
    set_default_svd_truncation_policy(SvdTruncationPolicy::new(0.1)).unwrap();
    // 1e-3 is below the global threshold 0.1 but far above the enrichment
    // cutoff 1e-12 * ||b_local||.
    let (state, sites) = enrich_once(1e-3, ResidualEnrichment::default().with_max_added(2));
    assert_eq!(bond_dim(&state), 3);
    assert!(distance_from_00(&state, &sites) < 1e-12);
}
