use super::*;

#[test]
fn difference_tensor_dimensions_reject_product_overflow() {
    let error = checked_difference_tensor_dims(usize::MAX, 2, 1, 1).unwrap_err();
    assert!(error.to_string().contains("left bond product"));

    let error = checked_difference_tensor_dims(1, 1, usize::MAX, 2).unwrap_err();
    assert!(error.to_string().contains("right bond product"));
}

#[test]
fn difference_tensor_backing_reservation_reports_failure() {
    let error =
        try_vec_with_capacity::<Complex64>("difference-kernel tensor", usize::MAX).unwrap_err();
    assert!(
        error.to_string().contains("allocation failed"),
        "unexpected error: {error}"
    );
}
