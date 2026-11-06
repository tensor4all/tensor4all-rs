use super::*;

#[test]
fn matrix2_zeros_fills_a_column_major_zero_matrix() {
    let matrix: Matrix2<f64> = matrix2_zeros(2, 3).unwrap();

    assert_eq!(matrix.dims(), &[2, 3]);
    assert_eq!(matrix.to_col_major_vec(), vec![0.0; 6]);
}

#[test]
fn matrix2_zeros_maps_unrepresentable_shapes_to_invalid_operation() {
    // `usize::MAX * 2` overflows the element count before any allocation.
    let overflow = matrix2_zeros::<f64>(usize::MAX, 2).unwrap_err();
    assert!(matches!(
        overflow,
        MPOError::InvalidOperation { ref message } if message.contains("overflows usize")
    ));

    // `usize::MAX` elements fit in `usize` but cannot be reserved.
    let unallocatable = matrix2_zeros::<f64>(usize::MAX, 1).unwrap_err();
    assert!(matches!(
        unallocatable,
        MPOError::InvalidOperation { ref message } if message.contains("cannot allocate")
    ));
}
