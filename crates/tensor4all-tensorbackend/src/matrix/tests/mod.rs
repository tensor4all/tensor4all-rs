use super::*;
use tenferro::TypedTensor;

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
fn matrix_into_col_major_vec_consumes_storage() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let data = m.into_col_major_vec();

    assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_to_typed_tensor_preserves_column_major_layout() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let tensor = m.to_typed_tensor();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_into_typed_tensor_consumes_column_major_layout() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let tensor = m.into_typed_tensor();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_from_typed_tensor_consumes_column_major_layout() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]);

    let m = Matrix::try_from_typed_tensor(tensor).unwrap();

    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(m[[0, 1]], 2.0);
}

#[test]
fn matrix_from_typed_tensor_rejects_non_matrix_rank() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 1, 1], vec![1.0, 2.0]);

    let err = Matrix::try_from_typed_tensor(tensor).unwrap_err();

    assert!(err.to_string().contains("rank-2 tensor"));
}

#[test]
fn try_from_vec2d_rejects_longer_rows() {
    let err = try_from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0, 5.0]]).unwrap_err();

    assert!(matches!(
        err,
        MatrixShapeError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 3,
        }
    ));
    assert_eq!(err.to_string(), "row 1 has length 3, expected 2");
}

#[test]
fn try_from_vec2d_rejects_shorter_rows() {
    let err = try_from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0]]).unwrap_err();

    assert!(matches!(
        err,
        MatrixShapeError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 1,
        }
    ));
    assert_eq!(err.to_string(), "row 1 has length 1, expected 2");
}

#[test]
#[should_panic(expected = "row 1 has length 3, expected 2")]
fn from_vec2d_panics_with_shape_error_for_ragged_rows() {
    let _ = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0, 5.0]]);
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
fn mat_mul_owned_matches_borrowed_multiplication() {
    let a = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

    let c = mat_mul_owned(a, b).unwrap();

    assert_eq!(c.as_col_major_slice(), &[19.0, 43.0, 22.0, 50.0]);
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
fn batched_mat_mul_same_shape_preserves_column_major_batches() {
    let a0 = vec![1.0, 3.0, 2.0, 4.0];
    let a1 = vec![2.0, 0.0, 0.0, 3.0];
    let b0 = vec![5.0, 7.0, 6.0, 8.0];
    let b1 = vec![1.0, 4.0, 2.0, 5.0];
    let mut a = a0;
    a.extend(a1);
    let mut b = b0;
    b.extend(b1);

    let out = batched_mat_mul_same_shape(2, 2, 2, 2, &a, &b).unwrap();

    assert_eq!(out.len(), 8);
    assert_eq!(&out[0..4], &[19.0, 43.0, 22.0, 50.0]);
    assert_eq!(&out[4..8], &[2.0, 12.0, 4.0, 15.0]);
}

#[test]
fn batched_mat_mul_same_shape_owned_matches_borrowed() {
    let a = vec![1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 3.0];
    let b = vec![5.0, 7.0, 6.0, 8.0, 1.0, 4.0, 2.0, 5.0];

    let borrowed = batched_mat_mul_same_shape(2, 2, 2, 2, &a, &b).unwrap();
    let owned = batched_mat_mul_same_shape_owned(2, 2, 2, 2, a, b).unwrap();

    assert_eq!(owned, borrowed);
}

#[test]
fn mat_mul_reports_dimension_mismatch() {
    let a = Matrix::from_col_major_vec(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let err = mat_mul(&a, &b).unwrap_err();

    assert!(err.to_string().contains("matrix dimensions"));
}
