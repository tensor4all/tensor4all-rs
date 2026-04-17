use super::*;

#[test]
fn test_svd_truncation_policy_defaults() {
    let policy = SvdTruncationPolicy::new(1e-12);
    assert_eq!(policy.threshold, 1e-12);
    assert_eq!(policy.scale, ThresholdScale::Relative);
    assert_eq!(policy.measure, SingularValueMeasure::Value);
    assert_eq!(policy.rule, TruncationRule::PerValue);
}

#[test]
fn test_svd_truncation_policy_builders() {
    let policy = SvdTruncationPolicy::new(1e-8)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(policy.scale, ThresholdScale::Absolute);
    assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
    assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
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
