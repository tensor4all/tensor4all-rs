use super::*;

#[test]
fn exponent_is_exact_for_normal_and_subnormal_values() {
    assert_eq!(exponent(1.0), 0);
    assert_eq!(exponent(1.999_999_999_999_999_8), 0);
    assert_eq!(exponent(2.0), 1);
    assert_eq!(exponent(0.75), -1);
    assert_eq!(exponent(f64::MAX), 1023);
    assert_eq!(exponent(f64::MIN_POSITIVE), -1022);
    assert_eq!(exponent(f64::MIN_POSITIVE - f64::from_bits(1)), -1023);
    assert_eq!(exponent(f64::from_bits(1)), -1074);
    assert_eq!(exponent(f64::from_bits(3)), -1073);
    assert_eq!(exponent(-4.0), 2);
}

#[test]
fn ldexp_steps_through_ranges_a_single_factor_cannot_hold() {
    assert_eq!(ldexp(1.0, 0), 1.0);
    assert_eq!(ldexp(f64::from_bits(1), 2034), ldexp(1.0, 960));
    assert_eq!(ldexp(1.0, -1074), f64::from_bits(1));
    assert_eq!(ldexp(ldexp(1.0, 1023), -2097), f64::from_bits(1));
    assert_eq!(ldexp(1.0, 1024), f64::INFINITY);
    assert_eq!(ldexp(1.0, -1075), 0.0);
}

#[test]
fn scaled_product_does_not_overflow_before_the_exponent_applies() {
    // f64::MAX * sqrt(2) overflows, but f64::MAX * sqrt(2) * 2^-1000 is
    // finite: the product is formed from the mantissa of the coefficient.
    let value = scaled_product(f64::MAX, std::f64::consts::SQRT_2, -1000);
    let expected = ldexp(f64::MAX * 0.5 * std::f64::consts::SQRT_2, -999);
    assert!(value.is_finite());
    assert_eq!(value, expected);
    // A result above f64::MAX is still infinite.
    assert_eq!(scaled_product(f64::MAX, 2.0, 0), f64::INFINITY);
    // And a product far below the subnormal range is zero.
    assert_eq!(scaled_product(1e-300, 1.0, -2000), 0.0);
}

#[test]
fn smallest_exponent_reads_an_overflowing_complex_modulus() {
    use num_complex::Complex64;
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(3);
    // Every nonzero modulus is sqrt(2) * f64::MAX, above f64::MAX.
    let all_huge = IdxTensor::from_dense(
        vec![i.clone()],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(f64::MAX, f64::MAX),
            Complex64::new(-f64::MAX, f64::MAX),
        ],
    )
    .unwrap();
    let (exp, smallest) = smallest_exponent(&all_huge).unwrap().unwrap();
    assert_eq!(exp, 1024);
    assert_eq!(smallest, f64::INFINITY);
    assert_eq!(largest_exponent(&all_huge).unwrap(), 1024);

    let mixed = IdxTensor::from_dense(
        vec![i.clone()],
        vec![
            Complex64::new(f64::from_bits(1), 0.0),
            Complex64::new(f64::MAX, f64::MAX),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(
        smallest_exponent(&mixed).unwrap(),
        Some((-1074, f64::from_bits(1)))
    );

    let zero = IdxTensor::from_dense(vec![i.clone()], vec![0.0_f64; 3]).unwrap();
    assert_eq!(smallest_exponent(&zero).unwrap(), None);

    // An infinite entry that is also the smallest nonzero one is an error
    // (otherwise `largest_exponent` reports it).
    let infinite = IdxTensor::from_dense(vec![i], vec![0.0, f64::INFINITY, 0.0]).unwrap();
    assert!(smallest_exponent(&infinite).is_err());
}
