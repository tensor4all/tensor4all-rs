use super::*;
#[test]
fn size_arithmetic_saturates_or_reports_overflow_before_allocation() {
    assert_eq!(capped_mul(usize::MAX, 2, 7), 7);
    assert_eq!(capped_mul(2, 3, 7), 6);
    assert_eq!(capped_mul(0, usize::MAX, 7), 0);
    assert!(matches!(
        checked_product([usize::MAX, 2], "test"),
        Err(TreeRsiError::SizeOverflow { .. })
    ));
    assert_eq!(checked_product([], "test").unwrap(), 1);
    assert!(enforce_limit("test", 7, 7).is_ok());
    assert!(matches!(
        enforce_limit("test", 8, 7),
        Err(TreeRsiError::ResourceLimit { .. })
    ));
}
