use num_complex::Complex64;

pub(crate) fn assert_scalar_close(actual: f64, expected: f64, max_sample: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol * max_sample.max(1.0),
        "got {}, expected {}, tol {}, max_sample {}",
        actual,
        expected,
        tol,
        max_sample
    );
}

pub(crate) fn assert_complex_slice_close(
    actual: &[Complex64],
    expected: &[Complex64],
    max_sample: f64,
    tol: f64,
) {
    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff <= tol * max_sample.max(1.0),
        "maxabs diff {} exceeds tol {} * max_sample {}",
        max_diff,
        tol,
        max_sample
    );
}

#[cfg(test)]
mod tests {
    use super::{assert_complex_slice_close, assert_scalar_close};
    use num_complex::Complex64;

    #[test]
    fn scalar_close_accepts_small_error_relative_to_max_sample() {
        assert_scalar_close(1.0 + 1e-12, 1.0, 2.0, 1e-10);
    }

    #[test]
    #[should_panic]
    fn scalar_close_rejects_large_error_relative_to_max_sample() {
        assert_scalar_close(1.1, 1.0, 1.0, 1e-3);
    }

    #[test]
    fn complex_slice_close_accepts_small_maxabs_error_relative_to_max_sample() {
        let actual = [Complex64::new(1.0, 2.0 + 1e-12), Complex64::new(-3.0, 4.0)];
        let expected = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        assert_complex_slice_close(&actual, &expected, 5.0, 1e-10);
    }

    #[test]
    #[should_panic]
    fn complex_slice_close_rejects_large_maxabs_error_relative_to_max_sample() {
        let actual = [Complex64::new(1.0, 2.5)];
        let expected = [Complex64::new(1.0, 2.0)];
        assert_complex_slice_close(&actual, &expected, 1.0, 1e-3);
    }
}
