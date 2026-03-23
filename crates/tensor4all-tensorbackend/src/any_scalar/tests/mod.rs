use super::*;

#[test]
fn scalar_from_value_supports_all_supported_element_types() {
    let f32_scalar = Scalar::from_value(1.25_f32);
    let f64_scalar = Scalar::from_value(-2.5_f64);
    let c32_scalar = Scalar::from_value(Complex32::new(3.0, -0.5));
    let c64_scalar = Scalar::from_value(Complex64::new(-1.0, 2.0));

    assert_eq!(f32_scalar.real(), 1.25);
    assert_eq!(f64_scalar.real(), -2.5);
    assert_eq!(c32_scalar.real(), 3.0);
    assert_eq!(c32_scalar.imag(), -0.5);
    assert_eq!(Complex64::from(c64_scalar), Complex64::new(-1.0, 2.0));
}

#[test]
fn any_scalar_sum_from_real_storage_stays_real() {
    let dense = Storage::from_dense_f64_col_major(vec![1.0, -2.5], &[2]).unwrap();
    let diag = Storage::from_diag_f64_col_major(vec![3.0, 4.5], 2).unwrap();

    let dense_sum = AnyScalar::sum_from_storage(&dense);
    let diag_sum = AnyScalar::sum_from_storage(&diag);

    assert!(dense_sum.is_real());
    assert_eq!(dense_sum.real(), -1.5);
    assert!(diag_sum.is_real());
    assert_eq!(diag_sum.real(), 7.5);
}

#[test]
fn any_scalar_sum_from_complex_storage_stays_complex() {
    let dense = Storage::from_dense_c64_col_major(
        vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
        &[2],
    )
    .unwrap();

    let sum = AnyScalar::sum_from_storage(&dense);
    let sum_c64: Complex64 = sum.into();
    assert_eq!(sum_c64, Complex64::new(0.5, 1.0));
}

#[test]
fn scalar_arithmetic_uses_runtime_bridge() {
    let sum = AnyScalar::from_real(1.5) + AnyScalar::from_real(2.0);
    let diff = AnyScalar::from_complex(3.0, -1.0) - AnyScalar::from_real(1.0);
    let prod = AnyScalar::from_real(2.0) * AnyScalar::from_complex(0.0, 1.0);

    assert_eq!(sum.as_f64(), Some(3.5));
    assert_eq!(Complex64::from(diff), Complex64::new(2.0, -1.0));
    assert_eq!(Complex64::from(prod), Complex64::new(0.0, 2.0));
}
