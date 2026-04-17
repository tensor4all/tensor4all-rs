use super::*;
use tensor4all_core::SvdTruncationPolicy;

#[test]
fn test_default_options() {
    let opts = LinsolveOptions::default();
    assert_eq!(opts.nfullsweeps, 5);
    assert_eq!(opts.a0, 0.0);
    assert_eq!(opts.a1, 1.0);
    assert!(opts.convergence_tol.is_none());
}

#[test]
fn test_builder_pattern() {
    let policy = SvdTruncationPolicy::new(1e-9)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = LinsolveOptions::new(5)
        .with_max_rank(100)
        .with_svd_policy(policy)
        .with_krylov_tol(1e-8)
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6);

    assert_eq!(opts.nfullsweeps, 5);
    assert_eq!(opts.truncation.max_rank(), Some(100));
    assert_eq!(opts.truncation.svd_policy(), Some(policy));
    assert_eq!(opts.krylov_tol, 1e-8);
    assert_eq!(opts.a0, 1.0);
    assert_eq!(opts.a1, -1.0);
    assert_eq!(opts.convergence_tol, Some(1e-6));
}
