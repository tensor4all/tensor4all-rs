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
