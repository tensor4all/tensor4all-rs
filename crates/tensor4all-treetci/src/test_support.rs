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
mod tests;
