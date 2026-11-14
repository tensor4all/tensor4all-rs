use super::*;

#[test]
fn estimate_counts_factor_columns_from_the_far_end() {
    // Three sites, joint bonds 6 and 4, binary outputs.
    // cols[1] = min(4, 2 * 1) = 2; cols[0] = min(6, 2 * 2) = 4.
    let estimate = estimate_sweep(&[6, 4], &[2, 2, 2]);
    assert_eq!(estimate.factor_elements, 6 * 4 + 4 * 2);
    // The largest QR input and its Q: 2 * (6 * 2 * 2) = 48, against
    // 2 * (4 * 2 * 1) = 16.
    assert_eq!(estimate.peak_elements, estimate.factor_elements + 48);
}

#[test]
fn estimate_saturates_instead_of_overflowing() {
    let estimate = estimate_sweep(&[usize::MAX, usize::MAX], &[usize::MAX; 3]);
    assert!(estimate.peak_elements >= estimate.factor_elements);
    assert!(estimate.factor_elements > u128::from(u64::MAX));
}

#[test]
fn gamma_grows_with_the_count_and_saturates() {
    let rounding = Rounding {
        unit: f64::EPSILON / 2.0,
    };
    let small = rounding.gamma(10);
    let ju = 10.0 * f64::EPSILON / 2.0;
    assert!((small - ju / (1.0 - ju)).abs() <= f64::EPSILON * small);
    assert!(small > ju);
    assert!(rounding.gamma(20) > small);
    assert_eq!(rounding.gamma(usize::MAX), f64::INFINITY);
    assert!(rounding.norm_upper(1.0, 4) > 1.0);
}
