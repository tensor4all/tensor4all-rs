use super::*;
use tensor4all_core::ColMajorArray;

#[test]
fn two_dimensional_column_helpers_report_shape_errors() {
    let one_dimensional = ColMajorArray::new(vec![1usize, 2, 3], vec![3]).unwrap();
    let err = ncols_2d(&one_dimensional).unwrap_err();
    assert!(err.to_string().contains("expected 2D ColMajorArray"));
    assert!(err.to_string().contains("[3]"));

    let matrix = ColMajorArray::new(vec![1usize, 2, 3, 4], vec![2, 2]).unwrap();
    let err = column_2d(&matrix, 2).unwrap_err();
    assert!(err.to_string().contains("column 2 is out of range"));
    assert!(err.to_string().contains("[2, 2]"));
}
