use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_core::AnyScalar;

#[test]
fn test_is_complex() {
    assert!(!AnyScalar::new_real(1.0).is_complex());
    assert!(AnyScalar::new_complex(1.0, 0.0).is_complex());
}

#[test]
fn test_real() {
    assert_eq!(AnyScalar::new_real(3.5).real(), 3.5);
    assert_eq!(AnyScalar::new_complex(2.0, 3.0).real(), 2.0);
}

#[test]
fn test_abs() {
    assert_eq!(AnyScalar::new_real(-5.0).abs(), 5.0);
    assert_eq!(AnyScalar::new_complex(3.0, 4.0).abs(), 5.0);
}

#[test]
fn test_sqrt() {
    // Positive real
    let result = AnyScalar::new_real(4.0).sqrt();
    assert!(!result.is_complex());
    assert!((result.real() - 2.0).abs() < 1e-10);

    // Negative real -> complex
    let result = AnyScalar::new_real(-4.0).sqrt();
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 2.0).abs() < 1e-10);

    // Complex
    let result = AnyScalar::new_complex(-1.0, 0.0).sqrt();
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 1.0).abs() < 1e-10);
}

#[test]
fn test_powf() {
    // Positive real
    let result = AnyScalar::new_real(4.0).powf(0.5);
    assert!(!result.is_complex());
    assert!((result.real() - 2.0).abs() < 1e-10);

    // Negative real -> complex
    let result = AnyScalar::new_real(-4.0).powf(0.5);
    assert!(result.is_complex());

    // Complex
    let result = AnyScalar::new_complex(1.0, 1.0).powf(2.0);
    assert!(result.is_complex());
}

#[test]
fn test_powi() {
    // Real
    let result = AnyScalar::new_real(2.0).powi(3);
    assert!(!result.is_complex());
    assert!((result.real() - 8.0).abs() < 1e-10);

    let result = AnyScalar::new_real(2.0).powi(-2);
    assert!(!result.is_complex());
    assert!((result.real() - 0.25).abs() < 1e-10);

    // Complex
    let result = AnyScalar::new_complex(1.0, 1.0).powi(2);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 2.0).abs() < 1e-10);
}

#[test]
fn test_add() {
    // F64 + F64
    let result = AnyScalar::new_real(1.0) + AnyScalar::new_real(2.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    // F64 + C64
    let result = AnyScalar::new_real(1.0) + AnyScalar::new_complex(2.0, 3.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 3.0);

    // C64 + F64
    let result = AnyScalar::new_complex(1.0, 2.0) + AnyScalar::new_real(3.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 4.0);
    assert_eq!(z.im, 2.0);

    // C64 + C64
    let result = AnyScalar::new_complex(1.0, 2.0) + AnyScalar::new_complex(3.0, 4.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 4.0);
    assert_eq!(z.im, 6.0);
}

#[test]
fn test_sub() {
    let result = AnyScalar::new_real(5.0) - AnyScalar::new_real(2.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(5.0, 6.0) - AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 6.0);
}

#[test]
fn test_mul() {
    let result = AnyScalar::new_real(3.0) * AnyScalar::new_real(4.0);
    assert!(!result.is_complex());
    assert!((result.real() - 12.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(1.0, 2.0) * AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 2.0);
    assert_eq!(z.im, 4.0);
}

#[test]
fn test_div() {
    let result = AnyScalar::new_real(12.0) / AnyScalar::new_real(4.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(6.0, 8.0) / AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 4.0);
}

#[test]
fn test_neg() {
    let result = -AnyScalar::new_real(5.0);
    assert!(!result.is_complex());
    assert!((result.real() - (-5.0)).abs() < 1e-10);

    let result = -AnyScalar::new_complex(1.0, 2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, -1.0);
    assert_eq!(z.im, -2.0);
}

#[test]
fn test_from_f64() {
    let s: AnyScalar = 3.5.into();
    assert!(!s.is_complex());
    assert_eq!(s.real(), 3.5);
}

#[test]
fn test_new_real() {
    let s = AnyScalar::new_real(3.5);
    assert!(!s.is_complex());
    assert_eq!(s.real(), 3.5);
}

#[test]
fn test_new_complex() {
    let s = AnyScalar::new_complex(1.0, 2.0);
    assert!(s.is_complex());
    assert_eq!(s.real(), 1.0);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, 2.0);
}

#[test]
fn test_from_complex64() {
    let z = Complex64::new(1.0, 2.0);
    let s: AnyScalar = z.into();
    assert!(s.is_complex());
    let c: Complex64 = s.into();
    assert_eq!(c.re, 1.0);
    assert_eq!(c.im, 2.0);
}

#[test]
fn test_try_from_anyscalar_to_f64() {
    // Success case
    let s = AnyScalar::new_real(3.5);
    let x: Result<f64, _> = s.try_into();
    assert_eq!(x.unwrap(), 3.5);

    // Error case (complex)
    let s = AnyScalar::new_complex(1.0, 2.0);
    let x: Result<f64, _> = s.try_into();
    assert!(x.is_err());
}

#[test]
fn test_from_anyscalar_to_complex64() {
    // Real -> Complex
    let s = AnyScalar::new_real(3.5);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 3.5);
    assert_eq!(z.im, 0.0);

    // Complex -> Complex
    let s = AnyScalar::new_complex(1.0, 2.0);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, 2.0);
}

#[test]
fn test_default() {
    let s = AnyScalar::default();
    assert!(s.is_zero());
    assert!(!s.is_complex());
}

#[test]
fn test_zero() {
    let z = AnyScalar::zero();
    assert!(z.is_zero());
    assert!(!AnyScalar::new_real(0.0).is_complex());
    assert!(AnyScalar::new_real(0.0).is_zero());
    assert!(!AnyScalar::new_real(1.0).is_zero());
    assert!(AnyScalar::new_complex(0.0, 0.0).is_zero());
    assert!(!AnyScalar::new_complex(1.0, 0.0).is_zero());
}

#[test]
fn test_one() {
    let one = AnyScalar::one();
    assert!(!one.is_complex());
    assert!((one.real() - 1.0).abs() < 1e-10);
}

#[test]
fn test_partial_ord() {
    // Real numbers can be compared
    assert!(AnyScalar::new_real(1.0) < AnyScalar::new_real(2.0));
    assert!(AnyScalar::new_real(2.0) > AnyScalar::new_real(1.0));
    assert!(AnyScalar::new_real(1.0) <= AnyScalar::new_real(1.0));

    // Complex numbers cannot be compared
    let c1 = AnyScalar::new_complex(1.0, 0.0);
    let c2 = AnyScalar::new_complex(2.0, 0.0);
    assert!(c1.partial_cmp(&c2).is_none());

    // Mixed types cannot be compared
    assert!(AnyScalar::new_real(1.0)
        .partial_cmp(&AnyScalar::new_complex(1.0, 0.0))
        .is_none());
}

#[test]
fn test_display() {
    #[allow(clippy::approx_constant)]
    let s1 = AnyScalar::new_real(3.14);
    assert_eq!(format!("{}", s1), "3.14");

    let s2 = AnyScalar::new_complex(1.0, 2.0);
    let display_str = format!("{}", s2);
    assert!(display_str.contains("1") && display_str.contains("2"));
}
