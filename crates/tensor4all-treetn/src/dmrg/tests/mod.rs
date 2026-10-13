use super::*;

#[test]
fn checked_real_scalar_rejects_significant_imaginary_part() {
    let error =
        checked_real_scalar(&AnyScalar::new_complex(1.25, 0.5), 1.0e-3, "test scalar").unwrap_err();

    assert!(matches!(
        error,
        DmrgError::NonRealScalar {
            context: "test scalar",
            real: 1.25,
            imag: 0.5,
            tolerance: 1.0e-3,
        }
    ));
}
