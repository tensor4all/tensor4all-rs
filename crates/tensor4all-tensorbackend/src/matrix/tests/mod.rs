use super::*;

#[test]
fn test_matrix_basic() {
    let mut m = Matrix::<f64>::zeros(3, 3);
    m[[0, 0]] = 1.0;
    m[[1, 1]] = 2.0;
    m[[2, 2]] = 3.0;

    assert_eq!(m[[0, 0]], 1.0);
    assert_eq!(m[[1, 1]], 2.0);
    assert_eq!(m[[2, 2]], 3.0);
}

#[test]
fn test_matrix_column_major_storage() {
    let m = Matrix::from_col_major_vec(2, 3, vec![1, 4, 2, 5, 3, 6]);

    assert_eq!(m[[0, 0]], 1);
    assert_eq!(m[[0, 1]], 2);
    assert_eq!(m[[0, 2]], 3);
    assert_eq!(m[[1, 0]], 4);
    assert_eq!(m[[1, 1]], 5);
    assert_eq!(m[[1, 2]], 6);
    assert_eq!(m.as_col_major_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_matrix_transpose() {
    let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let mt = transpose(&m);

    assert_eq!(mt.nrows(), 3);
    assert_eq!(mt.ncols(), 2);
    assert_eq!(mt[[0, 0]], 1.0);
    assert_eq!(mt[[0, 1]], 4.0);
    assert_eq!(mt[[2, 0]], 3.0);
}

#[test]
fn test_submatrix_argmax() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let (r, c, _) = submatrix_argmax(&m, 0..3, 0..3);
    assert_eq!((r, c), (2, 2));
}

#[test]
fn test_mat_mul() {
    let a = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = mat_mul(&a, &b).unwrap();

    assert_eq!(c[[0, 0]], 19.0);
    assert_eq!(c[[0, 1]], 22.0);
    assert_eq!(c[[1, 0]], 43.0);
    assert_eq!(c[[1, 1]], 50.0);
}

#[test]
fn test_mat_mul_rectangular_preserves_column_major_layout() {
    let a = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b = from_vec2d(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
    let c = mat_mul(&a, &b).unwrap();

    assert_eq!(c.nrows(), 2);
    assert_eq!(c.ncols(), 2);
    assert_eq!(c[[0, 0]], 58.0);
    assert_eq!(c[[0, 1]], 64.0);
    assert_eq!(c[[1, 0]], 139.0);
    assert_eq!(c[[1, 1]], 154.0);
    assert_eq!(c.as_col_major_slice(), &[58.0, 139.0, 64.0, 154.0]);
}

#[test]
fn mat_mul_reports_dimension_mismatch() {
    let a = Matrix::from_col_major_vec(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let err = mat_mul(&a, &b).unwrap_err();

    assert!(err.to_string().contains("matrix dimensions"));
}
