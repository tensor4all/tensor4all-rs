use super::*;

#[test]
fn gaussian_target_peaks_at_one() {
    assert!((gaussian_target(0.0) - 1.0).abs() < 1e-12);
}

#[test]
fn gaussian_fourier_reference_is_real_and_even() {
    let left = gaussian_fourier_reference(-1.25);
    let right = gaussian_fourier_reference(1.25);

    assert!((left.re - right.re).abs() < 1e-12);
    assert!(left.im.abs() < 1e-12);
    assert!(right.im.abs() < 1e-12);
    assert!((gaussian_fourier_reference(0.0).re - (2.0 * PI).sqrt()).abs() < 1e-12);
}

#[test]
fn global_index_to_quantics_sites_uses_msb_first_order() {
    assert_eq!(global_index_to_quantics_sites(0, 3), vec![0, 0, 0]);
    assert_eq!(global_index_to_quantics_sites(4, 3), vec![1, 0, 0]);
    assert_eq!(global_index_to_quantics_sites(7, 3), vec![1, 1, 1]);
}
