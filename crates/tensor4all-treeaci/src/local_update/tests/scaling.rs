//! Extreme-scale normalization regressions for #854.
// Test the wrapper separately from downstream factorizations and contractions.

fn audit_complex_normalization(scale: f64) {
    let value = num_complex::Complex64::new(scale, 0.5 * scale);
    let magnitude = value.norm();
    let mut matrix = tensor4all_tensorbackend::Matrix::from_col_major_vec(1, 1, vec![value]);
    super::super::LocalMatrixScale::new(magnitude).normalize(&mut matrix);
    let actual = matrix.as_col_major_slice()[0];
    let expected = num_complex::Complex64::new(scale / magnitude, 0.5 * scale / magnitude);
    eprintln!("SCALE_WRAPPER scale={scale:e} actual={actual:?} expected={expected:?}");
    assert!(
        actual.norm().is_finite() && (actual - expected).norm() < 1e-14,
        "finite complex normalization should preserve unit magnitude"
    );
}

#[test]
fn audit_complex_normalization_tiny() {
    audit_complex_normalization(1e-200);
}

#[test]
fn audit_complex_normalization_huge() {
    audit_complex_normalization(1e200);
}

#[test]
fn audit_complex_normalization_ordinary_control() {
    audit_complex_normalization(1.0);
}

#[test]
fn audit_real_normalization_extreme_control() {
    for scale in [1e-200, 1e200] {
        let mut matrix = tensor4all_tensorbackend::Matrix::from_col_major_vec(1, 1, vec![scale]);
        super::super::LocalMatrixScale::new(scale).normalize(&mut matrix);
        assert_eq!(matrix.as_col_major_slice(), &[1.0]);
    }
}
