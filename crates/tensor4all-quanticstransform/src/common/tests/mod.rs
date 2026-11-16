use super::*;
#[cfg(feature = "simplett-transforms")]
use tensor4all_core::DenseBudgetError;

#[cfg(feature = "simplett-transforms")]
#[test]
fn test_boundary_condition_default() {
    assert_eq!(BoundaryCondition::default(), BoundaryCondition::Periodic);
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn test_carry_direction_default() {
    assert_eq!(CarryDirection::default(), CarryDirection::LeftToRight);
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn test_identity_mpo() {
    let mpo = identity_mpo(4).unwrap();
    assert_eq!(mpo.len(), 4);

    // Check that it's an identity operator
    for i in 0..4 {
        let t = mpo.site_tensor(i);
        assert_eq!(t.left_dim(), 1);
        assert_eq!(t.site_dim(), 4);
        assert_eq!(t.right_dim(), 1);
    }
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn asymmetric_conversion_skips_empty_site_payload() {
    let core = tensor3_zeros(1, 0, 1).unwrap();
    let train = SimpleTensorTrain::new(vec![core]).unwrap();
    let input = 1 << 30;
    let error = tensortrain_to_linear_operator_asymmetric(&train, &[input], &[0]).unwrap_err();
    assert!(error.to_string().contains("invalid dimension 0"));
}

#[test]
fn checked_allocation_len_rejects_complex_byte_limit_before_allocation() {
    let elements = isize::MAX as usize / std::mem::size_of::<Complex64>() + 1;
    let error = checked_allocation_len::<Complex64>(&[elements], "complex tensor").unwrap_err();
    assert!(error.to_string().contains("byte length"));
}

#[test]
fn checked_allocation_len_covers_product_and_zero_size_branches() {
    let error = checked_allocation_len::<u8>(&[usize::MAX, 2], "overflow").unwrap_err();
    assert!(error.to_string().contains("element count"));

    assert_eq!(
        checked_allocation_len::<u8>(&[0, usize::MAX], "zero").unwrap(),
        0
    );
    assert_eq!(checked_allocation_len::<u8>(&[], "empty").unwrap(), 1);
    assert_eq!(
        checked_allocation_len::<()>(&[usize::MAX], "zst").unwrap(),
        usize::MAX
    );
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn identity_mpo_rejects_oversized_site_list() {
    let error = identity_mpo(usize::MAX).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn embed_rejects_total_over_budget() {
    // `identity_mpo(2)` sites are 1x4x1, so two embedded 2-variable sites are
    // 2 * 16 = 32 elements.
    let mpo = identity_mpo(2).unwrap();
    let error = embed_single_var_mpo(&mpo, 2, 0, DenseElementBudget::new(31)).unwrap_err();
    assert!(
        matches!(
            error,
            QuanticsTransformError::DenseBudget(DenseBudgetError::Exceeded {
                elements: 32,
                max_elements: 31,
                ..
            })
        ),
        "unexpected error: {error}"
    );
}

#[cfg(feature = "simplett-transforms")]
#[test]
fn embed_accepts_total_at_budget() {
    let mpo = identity_mpo(2).unwrap();
    let embedded = embed_single_var_mpo(&mpo, 2, 0, DenseElementBudget::new(32)).unwrap();
    assert_eq!(embedded.len(), 2);
}

#[test]
fn try_vec_with_capacity_reports_capacity_overflow_context() {
    let error = try_vec_with_capacity::<u8>("test site list", usize::MAX).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("test site list"));
    assert!(
        message.contains("allocation failed"),
        "unexpected error: {message}"
    );
}
