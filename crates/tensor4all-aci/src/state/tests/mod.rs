use super::{checked_frame_add, checked_frame_mul};

#[test]
fn frame_shape_helpers_reject_overflow() {
    assert!(checked_frame_mul(usize::MAX, 2, "product").is_err());
    assert!(checked_frame_add(usize::MAX, 1, "sum").is_err());
}
