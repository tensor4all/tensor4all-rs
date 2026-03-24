use super::*;

#[test]
fn test_global_default() {
    static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

    assert!((TEST_DEFAULT.get() - 1e-12).abs() < 1e-20);

    TEST_DEFAULT.set(1e-10).unwrap();
    assert!((TEST_DEFAULT.get() - 1e-10).abs() < 1e-20);
}

#[test]
fn test_invalid_values() {
    static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

    assert!(TEST_DEFAULT.set(f64::NAN).is_err());
    assert!(TEST_DEFAULT.set(f64::INFINITY).is_err());
    assert!(TEST_DEFAULT.set(-1.0).is_err());
}

#[test]
fn test_set_unchecked() {
    static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

    TEST_DEFAULT.set_unchecked(1e-8);
    assert!((TEST_DEFAULT.get() - 1e-8).abs() < 1e-20);
}

#[test]
fn test_error_display() {
    let err = InvalidRtolError(-1.0);
    let msg = format!("{}", err);
    assert!(msg.contains("-1"));
    assert!(msg.contains("rtol"));
}

#[test]
fn test_runtime_construction() {
    // Create GlobalDefault at runtime (not as static) to exercise
    // the constructor through runtime code paths.
    let default = GlobalDefault::new(std::f64::consts::PI);
    assert!((default.get() - std::f64::consts::PI).abs() < 1e-15);

    default.set(2.71).unwrap();
    assert!((default.get() - 2.71).abs() < 1e-15);
}

#[test]
fn test_new_with_zero() {
    let default = GlobalDefault::new(0.0);
    assert_eq!(default.get(), 0.0);

    // Setting to zero should be valid
    default.set(0.0).unwrap();
    assert_eq!(default.get(), 0.0);
}

#[test]
fn test_negative_infinity_error() {
    let default = GlobalDefault::new(1.0);
    assert!(default.set(f64::NEG_INFINITY).is_err());
}

#[test]
fn test_error_debug_and_clone() {
    let err = InvalidRtolError(f64::NAN);
    let err2 = err;
    let debug_str = format!("{:?}", err2);
    assert!(debug_str.contains("InvalidRtolError"));
}
