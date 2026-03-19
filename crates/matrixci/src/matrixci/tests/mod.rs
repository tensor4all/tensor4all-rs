
use super::*;

#[test]
fn test_matrixci_new() {
    let ci = MatrixCI::<f64>::new(5, 5);
    assert_eq!(ci.nrows(), 5);
    assert_eq!(ci.ncols(), 5);
    assert_eq!(ci.rank(), 0);
    assert!(ci.is_empty());
}

#[test]
fn test_matrixci_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let ci = MatrixCI::from_matrix_with_pivot(&m, (2, 2));
    assert_eq!(ci.nrows(), 3);
    assert_eq!(ci.ncols(), 3);
    assert_eq!(ci.rank(), 1);
    assert_eq!(ci.row_indices(), &[2]);
    assert_eq!(ci.col_indices(), &[2]);
}

#[test]
fn test_matrixci_add_pivot() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    assert!(ci.add_pivot(&m, (1, 1)).is_ok());
    assert_eq!(ci.rank(), 2);
}

#[test]
fn test_crossinterpolate() {
    // Create a rank-1 matrix
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let ci = crossinterpolate(&m, None);

    // For a rank-1 matrix, should need only 1 pivot
    assert!(ci.rank() >= 1);

    // Check approximation quality
    let approx = ci.to_matrix();
    for i in 0..3 {
        for j in 0..3 {
            let diff = (m[[i, j]] - approx[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Approximation error too large at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_from_parts() {
    let pivot_cols = from_vec2d(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    let pivot_rows = from_vec2d(vec![
        vec![10.0, 20.0, 30.0, 40.0],
        vec![50.0, 60.0, 70.0, 80.0],
    ]);
    let ci = MatrixCI::from_parts(vec![0, 2], vec![1, 3], pivot_cols, pivot_rows);
    assert_eq!(ci.nrows(), 3);
    assert_eq!(ci.ncols(), 4);
    assert_eq!(ci.rank(), 2);
    assert!(!ci.is_empty());
    assert_eq!(ci.iset(), &[0, 2]);
    assert_eq!(ci.jset(), &[1, 3]);
}

#[test]
fn test_left_matrix_empty() {
    let ci = MatrixCI::<f64>::new(3, 4);
    let left = ci.left_matrix();
    assert_eq!(nrows(&left), 3);
    assert_eq!(ncols(&left), 0);
}

#[test]
fn test_right_matrix_empty() {
    let ci = MatrixCI::<f64>::new(3, 4);
    let right = ci.right_matrix();
    assert_eq!(nrows(&right), 0);
    assert_eq!(ncols(&right), 4);
}

#[test]
fn test_left_right_matrix_non_empty() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let ci = MatrixCI::from_matrix_with_pivot(&m, (2, 2));
    let left = ci.left_matrix();
    let right = ci.right_matrix();

    // For rank-1: left = pivot_cols * P^{-1}, right = P^{-1} * pivot_rows
    // Reconstruct: approx = left * right (should equal pivot_cols * P^{-1} * pivot_rows)
    let _approx = mat_mul(&left, &ci.pivot_rows);
    // This is L * A[I,:] which is the CI approximation
    // For a single pivot at (2,2), P = A[2,2] = 9.0
    // left = A[:,2] / 9.0 = [3/9, 6/9, 9/9] = [1/3, 2/3, 1]
    assert!((left[[0, 0]] - 3.0 / 9.0).abs() < 1e-10);
    assert!((left[[1, 0]] - 6.0 / 9.0).abs() < 1e-10);
    assert!((left[[2, 0]] - 1.0).abs() < 1e-10);

    // right = A[2,:] / 9.0 = [7/9, 8/9, 1]
    assert!((right[[0, 0]] - 7.0 / 9.0).abs() < 1e-10);
    assert!((right[[0, 1]] - 8.0 / 9.0).abs() < 1e-10);
    assert!((right[[0, 2]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_first_pivot_value_empty() {
    let ci = MatrixCI::<f64>::new(3, 3);
    // Empty CI returns T::one()
    assert!((ci.first_pivot_value() - 1.0).abs() < 1e-10);
}

#[test]
fn test_first_pivot_value_non_empty() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let ci = MatrixCI::from_matrix_with_pivot(&m, (1, 0));
    // first_pivot_value = pivot_cols[row_indices[0], 0] = A[1, 0] = 3.0
    assert!((ci.first_pivot_value() - 3.0).abs() < 1e-10);
}

#[test]
fn test_evaluate_empty() {
    let ci = MatrixCI::<f64>::new(3, 3);
    // Empty CI evaluates to zero
    assert!((ci.evaluate(0, 0)).abs() < 1e-10);
    assert!((ci.evaluate(1, 2)).abs() < 1e-10);
}

#[test]
fn test_evaluate_non_empty() {
    // Rank-1 matrix: outer product of [1,2,3] and [1,2,3]
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let ci = MatrixCI::from_matrix_with_pivot(&m, (2, 2));
    // CI should approximate rank-1 matrix exactly with one pivot
    for i in 0..3 {
        for j in 0..3 {
            let diff = (ci.evaluate(i, j) - m[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "evaluate({},{}) = {} vs {}",
                i,
                j,
                ci.evaluate(i, j),
                m[[i, j]]
            );
        }
    }
}

#[test]
fn test_submatrix_empty() {
    let ci = MatrixCI::<f64>::new(4, 5);
    let sub = ci.submatrix(&[0, 2], &[1, 3]);
    assert_eq!(nrows(&sub), 2);
    assert_eq!(ncols(&sub), 2);
    // All zeros for empty CI
    for i in 0..2 {
        for j in 0..2 {
            assert!((sub[[i, j]]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_submatrix_non_empty() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let ci = MatrixCI::from_matrix_with_pivot(&m, (2, 2));
    let sub = ci.submatrix(&[0, 1], &[0, 2]);
    // For rank-1 matrix, should be exact
    assert!((sub[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((sub[[0, 1]] - 3.0).abs() < 1e-10);
    assert!((sub[[1, 0]] - 2.0).abs() < 1e-10);
    assert!((sub[[1, 1]] - 6.0).abs() < 1e-10);
}

#[test]
fn test_add_pivot_row_dimension_mismatch() {
    let m3x3 = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let m2x3 = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m3x3, (0, 0));
    let result = ci.add_pivot_row(&m2x3, 1);
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_row_index_oob() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let result = ci.add_pivot_row(&m, 5);
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_row_duplicate() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let result = ci.add_pivot_row(&m, 0);
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_col_dimension_mismatch() {
    let m3x3 = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let m3x2 = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m3x3, (0, 0));
    let result = ci.add_pivot_col(&m3x2, 1);
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_col_index_oob() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let result = ci.add_pivot_col(&m, 5);
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_col_duplicate() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let result = ci.add_pivot_col(&m, 0);
    assert!(result.is_err());
}

#[test]
fn test_add_best_pivot() {
    let m = from_vec2d(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (1, 1));
    // add_best_pivot should find the next pivot with maximum error
    let result = ci.add_best_pivot(&m);
    assert!(result.is_ok());
    let (i, j) = result.unwrap();
    assert_eq!(ci.rank(), 2);
    // The best pivot should be at the position with largest remaining error
    // After pivot at (1,1), the remaining error max is at (2,2)=3 or (0,0)=1
    // So (2,2) should be picked
    assert_eq!(i, 2);
    assert_eq!(j, 2);
}

#[test]
fn test_crossinterpolate_higher_rank() {
    // Create a rank-2 matrix (not rank-1)
    let m = from_vec2d(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.0, 0.0],
        vec![0.0, 0.0, 0.0, 4.0],
    ]);

    let ci = crossinterpolate(
        &m,
        Some(CrossInterpolateOptions {
            tolerance: 1e-12,
            max_iter: 100,
        }),
    );

    // Diagonal matrix has rank 4, CI should pick up to 4 pivots
    assert!(ci.rank() >= 2);

    // Check approximation quality
    let approx = ci.to_matrix();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - approx[[i, j]]).abs();
            assert!(
                diff < 1e-8,
                "Approximation error too large at ({}, {}): diff = {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_pivot_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let _ = ci.add_pivot(&m, (1, 1));

    let p = ci.pivot_matrix();
    // P = A[I, J] = A[{0,1}, {0,1}] = [[1,2],[4,5]]
    assert!((p[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((p[[0, 1]] - 2.0).abs() < 1e-10);
    assert!((p[[1, 0]] - 4.0).abs() < 1e-10);
    assert!((p[[1, 1]] - 5.0).abs() < 1e-10);
}

#[test]
fn test_to_matrix() {
    // For a rank-1 matrix, to_matrix should reconstruct exactly
    let m = from_vec2d(vec![vec![2.0, 4.0], vec![3.0, 6.0]]);

    let ci = MatrixCI::from_matrix_with_pivot(&m, (1, 1));
    let approx = ci.to_matrix();
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (m[[i, j]] - approx[[i, j]]).abs() < 1e-10,
                "Mismatch at ({}, {})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_add_pivot_errors_on_dim_mismatch() {
    let m3x3 = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let m2x2 = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m3x3, (0, 0));
    let result = ci.add_pivot(&m2x2, (1, 1));
    assert!(result.is_err());
}

#[test]
fn test_add_pivot_errors_on_oob() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    let result = ci.add_pivot(&m, (5, 0));
    assert!(result.is_err());
    let result2 = ci.add_pivot(&m, (0, 5));
    assert!(result2.is_err());
}

#[test]
fn test_add_pivot_errors_on_duplicate() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
    // Duplicate row
    let result = ci.add_pivot(&m, (0, 1));
    assert!(result.is_err());
    // Duplicate col
    let result2 = ci.add_pivot(&m, (1, 0));
    assert!(result2.is_err());
}
