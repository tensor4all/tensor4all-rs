use super::*;

/// A wrapper struct to test HasTruncationParams trait default methods.
/// TruncationParams itself has inherent methods that shadow the trait defaults,
/// so we need a separate type to exercise the trait default implementations.
#[derive(Debug, Default)]
struct TestOptions {
    truncation: TruncationParams,
    _extra: i32,
}

impl HasTruncationParams for TestOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

#[test]
fn test_truncation_params_builder() {
    let params = TruncationParams::new().with_rtol(1e-10).with_max_rank(50);

    assert_eq!(params.rtol, Some(1e-10));
    assert_eq!(params.max_rank, Some(50));
}

#[test]
fn test_effective_values() {
    let params = TruncationParams::new();
    assert_eq!(params.effective_rtol(1e-12), 1e-12);
    assert_eq!(params.effective_max_rank(), usize::MAX);

    let params = params.with_rtol(1e-8).with_max_rank(100);
    assert_eq!(params.effective_rtol(1e-12), 1e-8);
    assert_eq!(params.effective_max_rank(), 100);
}

#[test]
fn test_decomposition_alg() {
    assert!(DecompositionAlg::SVD.is_svd_based());
    assert!(DecompositionAlg::RSVD.is_svd_based());
    assert!(!DecompositionAlg::QR.is_svd_based());
    assert!(!DecompositionAlg::LU.is_svd_based());
    assert!(!DecompositionAlg::CI.is_svd_based());

    assert!(DecompositionAlg::SVD.is_orthogonal());
    assert!(DecompositionAlg::RSVD.is_orthogonal());
    assert!(DecompositionAlg::QR.is_orthogonal());
    assert!(!DecompositionAlg::LU.is_orthogonal());
    assert!(!DecompositionAlg::CI.is_orthogonal());
}

#[test]
fn test_decomposition_alg_default() {
    let alg: DecompositionAlg = Default::default();
    assert_eq!(alg, DecompositionAlg::SVD);
}

#[test]
fn test_truncation_params_merge() {
    let params1 = TruncationParams::new().with_rtol(1e-10);
    let params2 = TruncationParams::new().with_max_rank(50);
    let merged = params1.merge(&params2);

    assert_eq!(merged.rtol, Some(1e-10));
    assert_eq!(merged.max_rank, Some(50));

    // Self values take precedence
    let params3 = TruncationParams::new().with_rtol(1e-8).with_max_rank(100);
    let merged2 = params3.merge(&params2);
    assert_eq!(merged2.rtol, Some(1e-8));
    assert_eq!(merged2.max_rank, Some(100));
}

#[test]
fn test_has_truncation_params_trait() {
    let mut params = TruncationParams::new();

    // Test trait methods
    assert_eq!(params.rtol(), None);
    assert_eq!(params.max_rank(), None);

    // Test mutable access
    params.truncation_params_mut().rtol = Some(1e-6);
    assert_eq!(params.truncation_params().rtol, Some(1e-6));
}

#[test]
fn test_with_cutoff() {
    let params = TruncationParams::new().with_cutoff(1e-10);
    assert_eq!(params.cutoff, Some(1e-10));
    assert!((params.rtol.unwrap() - 1e-5).abs() < 1e-15);
}

#[test]
fn test_with_maxdim() {
    let params = TruncationParams::new().with_maxdim(50);
    assert_eq!(params.max_rank, Some(50));
}

#[test]
fn test_cutoff_rtol_priority_last_wins() {
    // cutoff then rtol: rtol wins, cutoff cleared
    let params = TruncationParams::new().with_cutoff(1e-10).with_rtol(1e-3);
    assert_eq!(params.rtol, Some(1e-3));
    assert_eq!(params.cutoff, None);

    // rtol then cutoff: cutoff wins
    let params = TruncationParams::new().with_rtol(1e-3).with_cutoff(1e-10);
    assert_eq!(params.cutoff, Some(1e-10));
    assert!((params.rtol.unwrap() - 1e-5).abs() < 1e-15);
}

#[test]
fn test_has_truncation_params_cutoff() {
    let mut params = TruncationParams::new();
    params.set_cutoff(1e-8);
    assert_eq!(params.cutoff, Some(1e-8));
    assert!((params.rtol.unwrap() - 1e-4).abs() < 1e-15);

    params.set_maxdim(100);
    assert_eq!(params.max_rank, Some(100));
}

#[test]
fn test_trait_default_with_rtol() {
    // Exercise HasTruncationParams::with_rtol default implementation (lines 171-179)
    let opts = TestOptions::default().with_rtol(1e-6);
    assert_eq!(opts.truncation.rtol, Some(1e-6));
    assert_eq!(opts.truncation.cutoff, None);

    // Verify cutoff is cleared when rtol is set after cutoff
    let opts = TestOptions::default().with_cutoff(1e-10).with_rtol(1e-3);
    assert_eq!(opts.truncation.rtol, Some(1e-3));
    assert_eq!(opts.truncation.cutoff, None);
}

#[test]
fn test_trait_default_with_max_rank() {
    // Exercise HasTruncationParams::with_max_rank default implementation (lines 182-188)
    let opts = TestOptions::default().with_max_rank(42);
    assert_eq!(opts.truncation.max_rank, Some(42));
}

#[test]
fn test_trait_default_with_cutoff() {
    // Exercise HasTruncationParams::with_cutoff default implementation (lines 193-201)
    let opts = TestOptions::default().with_cutoff(1e-10);
    assert_eq!(opts.truncation.cutoff, Some(1e-10));
    assert!((opts.truncation.rtol.unwrap() - 1e-5).abs() < 1e-15);
}

#[test]
fn test_trait_default_with_maxdim() {
    // Exercise HasTruncationParams::with_maxdim default implementation (lines 204-210)
    let opts = TestOptions::default().with_maxdim(200);
    assert_eq!(opts.truncation.max_rank, Some(200));
}
