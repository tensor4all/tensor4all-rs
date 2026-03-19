
use super::*;

#[test]
fn test_matrixaca_new() {
    let aca = MatrixACA::<f64>::new(5, 5);
    assert_eq!(aca.nrows(), 5);
    assert_eq!(aca.ncols(), 5);
    assert_eq!(aca.rank(), 0);
    assert!(aca.is_empty());
}

#[test]
fn test_matrixaca_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let aca = MatrixACA::from_matrix_with_pivot(&m, (1, 1)).unwrap();
    assert_eq!(aca.nrows(), 3);
    assert_eq!(aca.ncols(), 3);
    assert_eq!(aca.rank(), 1);

    // Check that evaluation at pivot is correct
    let val = aca.evaluate(1, 1);
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn test_matrixaca_add_pivot() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    assert!(aca.add_pivot(&m, (1, 1)).is_ok());
    assert_eq!(aca.rank(), 2);
}

#[test]
fn test_matrixaca_zero_pivot_returns_error() {
    let m = from_vec2d(vec![vec![0.0, 1.0], vec![1.0, 1.0]]);
    // Pivot at (0,0) which is 0.0 should fail
    let result = MatrixACA::from_matrix_with_pivot(&m, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_matrixaca_near_zero_pivot_returns_error() {
    let m = from_vec2d(vec![vec![1e-200, 1.0], vec![1.0, 1.0]]);
    let result = MatrixACA::from_matrix_with_pivot(&m, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_matrixaca_add_pivot_row_zero_diagonal() {
    // Matrix where second pivot would cause division by zero
    let m = from_vec2d(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    // Adding pivot at (1,1) where value is 0 should fail during add_pivot_row
    let result = aca.add_pivot(&m, (1, 1));
    assert!(result.is_err());
}
