use super::*;

#[test]
fn source_function_peaks_at_one() {
    let config = DEFAULT_PARTIAL_FOURIER2D_CONFIG;
    assert!((source_function(0.0, 0.0, &config) - 1.0).abs() < 1e-12);
}

#[test]
fn partial_fourier_reference_is_real_and_even() {
    let config = DEFAULT_PARTIAL_FOURIER2D_CONFIG;
    let left = partial_fourier_reference(-1.25, 0.25, &config);
    let right = partial_fourier_reference(1.25, 0.25, &config);
    assert!((left.re - right.re).abs() < 1e-12);
    assert!(left.im.abs() < 1e-12);
    assert!(right.im.abs() < 1e-12);
    assert!((partial_fourier_reference(0.0, 0.0, &config).re - (2.0 * PI).sqrt()).abs() < 1e-12);
}
