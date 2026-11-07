use super::*;

#[test]
fn combined_scalar_product_quotient_avoids_intermediate_range_loss() {
    for (x, y, z, expected) in [
        (1e300, 1e300, 1e300, 1e300),
        (1e-300, 1e-300, 1e-300, 1e-300),
        (1e300, 1e-300, 1e300, 1e-300),
        (0.5, f64::from_bits(1), f64::from_bits(1), 0.5),
    ] {
        let actual = scaled_mul_div(&x.into(), &y.into(), &z.into());
        assert!((actual.real() / expected - 1.0).abs() < 1e-14);
        assert_eq!(actual.imag(), 0.0);
    }
    let actual = scaled_mul_div(
        &AnyScalar::new_complex(1e-200, 1e-200),
        &AnyScalar::new_complex(1e200, -1e200),
        &1.0.into(),
    );
    assert!((actual.real() - 2.0).abs() < 1e-14);
    assert_eq!(actual.imag(), 0.0);
}

#[test]
fn combined_scalar_product_quotient_preserves_gradients() {
    let x = AnyScalar::new_real(0.75).enable_grad().unwrap();
    let y = AnyScalar::new_real(0.5).enable_grad().unwrap();
    let z = AnyScalar::new_real(0.25).enable_grad().unwrap();
    let result = scaled_mul_div(&x, &y, &z);
    assert_eq!(result.real(), 1.5);
    assert!(result.tracks_grad());
    result.backward().unwrap();
    for (value, expected) in [(x, 2.0), (y, 3.0), (z, -6.0)] {
        assert!((value.grad().unwrap().unwrap().real() - expected).abs() < 1e-12);
    }
    let x = AnyScalar::new_real(0.5).enable_grad().unwrap();
    let tiny = AnyScalar::new_real(f64::from_bits(1));
    let result = scaled_mul_div(&x, &tiny, &tiny);
    assert_eq!(result.real(), 0.5);
    assert!(result.tracks_grad());
    result.backward().unwrap();
    assert_eq!(x.grad().unwrap().unwrap().real(), 1.0);
}
