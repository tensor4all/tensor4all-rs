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
