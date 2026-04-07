use super::*;

fn assert_scalar_close(actual: &Scalar, expected: &Scalar) {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(a), Some(e)) => assert!((a - e).abs() < 1e-6),
        _ => {
            let a = actual
                .as_c64()
                .unwrap_or_else(|| Complex64::new(actual.real(), actual.imag()));
            let e = expected
                .as_c64()
                .unwrap_or_else(|| Complex64::new(expected.real(), expected.imag()));
            assert!((a.re - e.re).abs() < 1e-6);
            assert!((a.im - e.im).abs() < 1e-6);
        }
    }
}

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
    let dense = Storage::from_dense_col_major(vec![1.0, -2.5], &[2]).unwrap();
    let diag = Storage::from_diag_col_major(vec![3.0, 4.5], 2).unwrap();

    let dense_sum = AnyScalar::sum_from_storage(&dense);
    let diag_sum = AnyScalar::sum_from_storage(&diag);

    assert!(dense_sum.is_real());
    assert_eq!(dense_sum.real(), -1.5);
    assert!(diag_sum.is_real());
    assert_eq!(diag_sum.real(), 7.5);
}

#[test]
fn any_scalar_sum_from_complex_storage_stays_complex() {
    let dense = Storage::from_dense_col_major(
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

#[test]
fn promote_scalar_native_covers_all_scalar_type_pairs() {
    let cases = vec![
        (
            Scalar::from_value(1.25_f32),
            ScalarType::F32,
            Scalar::from_value(1.25_f32),
        ),
        (
            Scalar::from_value(1.25_f32),
            ScalarType::F64,
            Scalar::from_value(1.25_f64),
        ),
        (
            Scalar::from_value(1.25_f32),
            ScalarType::C32,
            Scalar::from_value(Complex32::new(1.25, 0.0)),
        ),
        (
            Scalar::from_value(1.25_f32),
            ScalarType::C64,
            Scalar::from_value(Complex64::new(1.25, 0.0)),
        ),
        (
            Scalar::from_value(-2.5_f64),
            ScalarType::F32,
            Scalar::from_value(-2.5_f32),
        ),
        (
            Scalar::from_value(-2.5_f64),
            ScalarType::F64,
            Scalar::from_value(-2.5_f64),
        ),
        (
            Scalar::from_value(-2.5_f64),
            ScalarType::C32,
            Scalar::from_value(Complex32::new(-2.5, 0.0)),
        ),
        (
            Scalar::from_value(-2.5_f64),
            ScalarType::C64,
            Scalar::from_value(Complex64::new(-2.5, 0.0)),
        ),
        (
            Scalar::from_value(Complex32::new(3.0, -0.5)),
            ScalarType::F32,
            Scalar::from_value(3.0_f32),
        ),
        (
            Scalar::from_value(Complex32::new(3.0, -0.5)),
            ScalarType::F64,
            Scalar::from_value(3.0_f64),
        ),
        (
            Scalar::from_value(Complex32::new(3.0, -0.5)),
            ScalarType::C32,
            Scalar::from_value(Complex32::new(3.0, -0.5)),
        ),
        (
            Scalar::from_value(Complex32::new(3.0, -0.5)),
            ScalarType::C64,
            Scalar::from_value(Complex64::new(3.0, -0.5)),
        ),
        (
            Scalar::from_value(Complex64::new(-1.0, 2.0)),
            ScalarType::F32,
            Scalar::from_value(-1.0_f32),
        ),
        (
            Scalar::from_value(Complex64::new(-1.0, 2.0)),
            ScalarType::F64,
            Scalar::from_value(-1.0_f64),
        ),
        (
            Scalar::from_value(Complex64::new(-1.0, 2.0)),
            ScalarType::C32,
            Scalar::from_value(Complex32::new(-1.0, 2.0)),
        ),
        (
            Scalar::from_value(Complex64::new(-1.0, 2.0)),
            ScalarType::C64,
            Scalar::from_value(Complex64::new(-1.0, 2.0)),
        ),
    ];

    for (source, target, expected) in cases {
        let promoted =
            Scalar::from_native(promote_scalar_native(source.as_native(), target).unwrap())
                .expect("promoted scalar");
        assert_eq!(promoted.native.scalar_type(), expected.native.scalar_type());
        assert_scalar_close(&promoted, &expected);
    }
}

#[test]
fn scalar_utility_methods_cover_real_and_complex_cases() {
    let zero = Scalar::zero();
    let real = Scalar::from_real(-4.0);
    let complex = Scalar::from_complex(3.0, -4.0);

    assert!(zero.is_zero());
    assert_eq!(zero.as_f64(), Some(0.0));
    assert!(real.is_real());
    assert!(!real.is_complex());
    assert_eq!(real.abs(), 4.0);
    assert_eq!(real.as_f64(), Some(-4.0));
    assert_eq!(real.as_c64(), None);

    assert!(complex.is_complex());
    assert!(!complex.is_real());
    assert_eq!(complex.abs(), 5.0);
    assert_eq!(complex.as_f64(), None);
    assert_eq!(complex.as_c64(), Some(Complex64::new(3.0, -4.0)));
    assert_eq!(complex.conj(), Scalar::from_complex(3.0, 4.0));
    assert_eq!(complex.real_part(), Scalar::from_real(3.0));
    assert_eq!(complex.imag_part(), Scalar::from_real(-4.0));

    let recomposed = Scalar::compose_complex(Scalar::from_real(1.5), Scalar::from_real(-2.0))
        .expect("compose complex");
    assert_eq!(recomposed, Scalar::from_complex(1.5, -2.0));
    assert!(
        Scalar::compose_complex(Scalar::from_complex(1.0, 1.0), Scalar::from_real(0.0)).is_err()
    );

    assert_eq!(Scalar::from_real(9.0).sqrt(), Scalar::from_real(3.0));
    assert_eq!(
        Scalar::from_real(-4.0).sqrt(),
        Scalar::from_complex(0.0, 2.0)
    );
    assert_scalar_close(
        &Scalar::from_complex(3.0, -4.0).sqrt().powi(2),
        &Scalar::from_complex(3.0, -4.0),
    );
    assert_eq!(Scalar::from_real(-2.0).powi(3), Scalar::from_real(-8.0));
    assert_scalar_close(
        &Scalar::from_real(-4.0).powf(0.5),
        &Scalar::from_complex(0.0, 2.0),
    );
}

#[test]
fn scalar_trait_helpers_cover_ordering_and_conversions() {
    assert_eq!(Scalar::default(), Scalar::zero());
    assert_eq!(Scalar::one(), Scalar::from_real(1.0));
    assert!(Scalar::from_value(1.0_f32) < Scalar::from_value(2.0_f32));
    assert!(Scalar::from_value(1.0_f32) < Scalar::from_value(2.0_f64));
    assert!(Scalar::from_complex(1.0, 1.0)
        .partial_cmp(&Scalar::from_complex(1.0, 0.0))
        .is_none());

    assert_eq!(f64::try_from(Scalar::from_real(2.5)).unwrap(), 2.5);
    assert_eq!(f64::try_from(Scalar::from_value(1.25_f32)).unwrap(), 1.25);
    assert!(f64::try_from(Scalar::from_complex(1.0, 0.5)).is_err());

    let debug = format!("{:?}", Scalar::from_real(1.0));
    assert!(debug.contains("Scalar"));
    assert!(debug.contains("scalar_type"));
}
