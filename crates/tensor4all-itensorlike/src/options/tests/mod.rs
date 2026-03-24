use super::*;

#[test]
fn test_truncate_options_builder() {
    let opts = TruncateOptions::svd().with_rtol(1e-10).with_max_rank(50);
    assert_eq!(opts.alg, TruncateAlg::SVD);
    assert_eq!(opts.rtol(), Some(1e-10));
    assert_eq!(opts.max_rank(), Some(50));
}

#[test]
fn test_truncate_options_algorithms() {
    assert_eq!(TruncateOptions::svd().alg, TruncateAlg::SVD);
    assert_eq!(TruncateOptions::lu().alg, TruncateAlg::LU);
    assert_eq!(TruncateOptions::ci().alg, TruncateAlg::CI);
}

#[test]
fn test_truncate_options_site_range() {
    let opts = TruncateOptions::svd().with_site_range(0..5);
    assert_eq!(opts.site_range, Some(0..5));
}

#[test]
fn test_truncate_options_default() {
    let opts = TruncateOptions::default();
    assert_eq!(opts.alg, TruncateAlg::SVD);
    assert_eq!(opts.rtol(), None);
    assert_eq!(opts.max_rank(), None);
    assert!(opts.site_range.is_none());
}

#[test]
fn test_truncate_options_has_truncation_params() {
    let mut opts = TruncateOptions::svd();
    assert!(opts.truncation_params().rtol.is_none());
    opts.truncation_params_mut().rtol = Some(1e-8);
    assert_eq!(opts.truncation_params().rtol, Some(1e-8));
}

#[test]
fn test_contract_options_builder() {
    let opts = ContractOptions::zipup()
        .with_max_rank(100)
        .with_rtol(1e-12)
        .with_nhalfsweeps(6); // 6 half-sweeps = 3 full sweeps
    assert_eq!(opts.method, ContractMethod::Zipup);
    assert_eq!(opts.max_rank(), Some(100));
    assert_eq!(opts.rtol(), Some(1e-12));
    assert_eq!(opts.nhalfsweeps, 6);
}

#[test]
fn test_contract_options_methods() {
    assert_eq!(ContractOptions::zipup().method, ContractMethod::Zipup);
    assert_eq!(ContractOptions::fit().method, ContractMethod::Fit);
    assert_eq!(ContractOptions::naive().method, ContractMethod::Naive);
}

#[test]
fn test_contract_options_default() {
    let opts = ContractOptions::default();
    assert_eq!(opts.method, ContractMethod::Zipup);
    assert_eq!(opts.nhalfsweeps, 2);
    assert_eq!(opts.rtol(), None);
    assert_eq!(opts.max_rank(), None);
}

#[test]
fn test_contract_options_has_truncation_params() {
    let mut opts = ContractOptions::zipup();
    assert!(opts.truncation_params().rtol.is_none());
    opts.truncation_params_mut().max_rank = Some(50);
    assert_eq!(opts.truncation_params().max_rank, Some(50));
}

#[test]
fn test_contract_options_nhalfsweeps_must_be_multiple_of_2() {
    // builder should not panic; validation happens at operation entry
    let opts = ContractOptions::fit().with_nhalfsweeps(1);
    assert_eq!(opts.nhalfsweeps(), 1);
}

#[test]
fn test_contract_method_default() {
    let method: ContractMethod = Default::default();
    assert_eq!(method, ContractMethod::Zipup);
}

#[test]
fn test_truncate_options_cutoff() {
    let opts = TruncateOptions::svd().with_cutoff(1e-10);
    assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
    assert_eq!(opts.truncation.cutoff, Some(1e-10));
}

#[test]
fn test_truncate_options_maxdim() {
    let opts = TruncateOptions::svd().with_maxdim(42);
    assert_eq!(opts.max_rank(), Some(42));
}

#[test]
fn test_contract_options_cutoff() {
    let opts = ContractOptions::zipup().with_cutoff(1e-8);
    assert!((opts.rtol().unwrap() - 1e-4).abs() < 1e-15);
    assert_eq!(opts.truncation.cutoff, Some(1e-8));
}

#[test]
fn test_contract_options_maxdim() {
    let opts = ContractOptions::fit().with_maxdim(30);
    assert_eq!(opts.max_rank(), Some(30));
}

#[test]
fn test_contract_options_nsweeps() {
    let opts = ContractOptions::fit().with_nsweeps(5);
    assert_eq!(opts.nhalfsweeps, 10);
}

#[test]
fn test_cutoff_rtol_last_wins_truncate() {
    // cutoff then rtol: rtol wins
    let opts = TruncateOptions::svd().with_cutoff(1e-10).with_rtol(1e-3);
    assert_eq!(opts.rtol(), Some(1e-3));
    assert_eq!(opts.truncation.cutoff, None);

    // rtol then cutoff: cutoff wins
    let opts = TruncateOptions::svd().with_rtol(1e-3).with_cutoff(1e-10);
    assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
    assert_eq!(opts.truncation.cutoff, Some(1e-10));
}

#[test]
fn test_linsolve_options_default() {
    let opts = LinsolveOptions::default();
    assert_eq!(opts.nhalfsweeps, 10);
    assert_eq!(opts.a0, 0.0);
    assert_eq!(opts.a1, 1.0);
    assert_eq!(opts.krylov_tol, 1e-10);
    assert_eq!(opts.krylov_maxiter, 100);
    assert_eq!(opts.krylov_dim, 30);
    assert!(opts.convergence_tol.is_none());
    assert_eq!(opts.rtol(), None);
    assert_eq!(opts.max_rank(), None);
}

#[test]
fn test_linsolve_options_new() {
    let opts = LinsolveOptions::new(5);
    assert_eq!(opts.nhalfsweeps(), 10); // 5 full sweeps = 10 half-sweeps
}

#[test]
fn test_linsolve_options_builder() {
    let opts = LinsolveOptions::default()
        .with_nsweeps(10)
        .with_cutoff(1e-10)
        .with_maxdim(100)
        .with_krylov_tol(1e-8)
        .with_krylov_maxiter(200)
        .with_krylov_dim(50)
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6);

    assert_eq!(opts.nhalfsweeps(), 20);
    assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
    assert_eq!(opts.truncation_params().cutoff, Some(1e-10));
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
    // builder should not panic; validation happens at operation entry
    let opts = LinsolveOptions::default().with_nhalfsweeps(3);
    assert_eq!(opts.nhalfsweeps(), 3);
}
