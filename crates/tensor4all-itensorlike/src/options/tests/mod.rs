use super::*;
use tensor4all_core::SvdTruncationPolicy;

#[test]
fn test_validate_svd_truncation_options_accepts_default_settings() {
    assert!(validate_svd_truncation_options(None, None).is_ok());
}

#[test]
fn test_validate_svd_truncation_options_rejects_negative_threshold() {
    let err =
        validate_svd_truncation_options(None, Some(SvdTruncationPolicy::new(-1.0))).unwrap_err();
    assert!(err.to_string().contains("svd_policy.threshold"));
}

#[test]
fn test_validate_svd_truncation_options_rejects_non_finite_threshold() {
    let err = validate_svd_truncation_options(None, Some(SvdTruncationPolicy::new(f64::NAN)))
        .unwrap_err();
    assert!(err.to_string().contains("svd_policy.threshold"));
}

#[test]
fn test_validate_svd_truncation_options_rejects_zero_max_rank() {
    let err =
        validate_svd_truncation_options(Some(0), Some(SvdTruncationPolicy::new(1e-8))).unwrap_err();
    assert!(err.to_string().contains("max_rank/maxdim"));
}

#[test]
fn test_truncate_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = TruncateOptions::svd()
        .with_svd_policy(policy)
        .with_max_rank(50)
        .with_site_range(0..5);

    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.max_rank(), Some(50));
    assert_eq!(opts.site_range(), Some(0..5));
}

#[test]
fn test_truncate_options_default() {
    let opts = TruncateOptions::default();
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_rank(), None);
    assert_eq!(opts.site_range(), None);
}

#[test]
fn test_contract_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-12);
    let opts = ContractOptions::zipup()
        .with_max_rank(100)
        .with_svd_policy(policy)
        .with_nhalfsweeps(6)
        .with_dense_reference_limit(256);

    assert_eq!(opts.method(), ContractMethod::Zipup);
    assert_eq!(opts.max_rank(), Some(100));
    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.nhalfsweeps(), 6);
    assert_eq!(opts.dense_reference_limit(), Some(256));
}

#[test]
fn test_contract_options_methods() {
    assert_eq!(ContractOptions::zipup().method(), ContractMethod::Zipup);
    assert_eq!(ContractOptions::fit().method(), ContractMethod::Fit);
    assert_eq!(ContractOptions::naive().method(), ContractMethod::Naive);
}

#[test]
fn test_contract_options_default() {
    let opts = ContractOptions::default();
    assert_eq!(opts.method(), ContractMethod::Zipup);
    assert_eq!(opts.nhalfsweeps(), 2);
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_rank(), None);
    assert_eq!(opts.dense_reference_limit(), None);
}

#[test]
fn test_contract_options_nsweeps() {
    let opts = ContractOptions::fit().with_nsweeps(5);
    assert_eq!(opts.nhalfsweeps(), 10);
}

#[test]
fn test_contract_options_odd_nhalfsweeps_allowed_in_builder() {
    let opts = ContractOptions::fit().with_nhalfsweeps(1);
    assert_eq!(opts.nhalfsweeps(), 1);
}

#[test]
fn test_contract_method_default() {
    let method: ContractMethod = Default::default();
    assert_eq!(method, ContractMethod::Zipup);
}

#[test]
fn test_linsolve_options_default() {
    let opts = LinsolveOptions::default();
    assert_eq!(opts.nhalfsweeps(), 10);
    assert_eq!(opts.coefficients(), (0.0, 1.0));
    assert_eq!(opts.krylov_tol(), 1e-10);
    assert_eq!(opts.krylov_maxiter(), 100);
    assert_eq!(opts.krylov_dim(), 30);
    assert_eq!(opts.convergence_tol(), None);
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_rank(), None);
}

#[test]
fn test_linsolve_options_new() {
    let opts = LinsolveOptions::new(5);
    assert_eq!(opts.nhalfsweeps(), 10);
}

#[test]
fn test_linsolve_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = LinsolveOptions::default()
        .with_nsweeps(10)
        .with_svd_policy(policy)
        .with_max_rank(100)
        .with_krylov_tol(1e-8)
        .with_krylov_maxiter(200)
        .with_krylov_dim(50)
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6);

    assert_eq!(opts.nhalfsweeps(), 20);
    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.max_rank(), Some(100));
    assert_eq!(opts.krylov_tol(), 1e-8);
    assert_eq!(opts.krylov_maxiter(), 200);
    assert_eq!(opts.krylov_dim(), 50);
    assert_eq!(opts.coefficients(), (1.0, -1.0));
    assert_eq!(opts.convergence_tol(), Some(1e-6));
}

#[test]
fn test_linsolve_options_nsweeps() {
    let opts = LinsolveOptions::default().with_nsweeps(3);
    assert_eq!(opts.nhalfsweeps(), 6);
}

#[test]
fn test_linsolve_options_odd_nhalfsweeps_allowed_in_builder() {
    let opts = LinsolveOptions::default().with_nhalfsweeps(3);
    assert_eq!(opts.nhalfsweeps(), 3);
}
