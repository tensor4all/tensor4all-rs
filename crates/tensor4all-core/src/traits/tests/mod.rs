use super::*;
use crate::error::MatrixCIError;
use tensor4all_tensorbackend::from_vec2d;

struct ExactCi {
    matrix: Matrix<f64>,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
}

impl ExactCi {
    fn new(matrix: Matrix<f64>, row_indices: Vec<usize>, col_indices: Vec<usize>) -> Self {
        Self {
            matrix,
            row_indices,
            col_indices,
        }
    }
}

impl AbstractMatrixCI<f64> for ExactCi {
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn rank(&self) -> usize {
        self.row_indices.len()
    }

    fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    fn evaluate(&self, i: usize, j: usize) -> f64 {
        self.matrix[[i, j]]
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<f64> {
        tensor4all_tensorbackend::submatrix(&self.matrix, rows, cols)
    }
}

fn sample_matrix() -> Matrix<f64> {
    from_vec2d(vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ])
}

#[test]
fn default_helpers_return_expected_views_and_available_indices() {
    let matrix = sample_matrix();
    let ci = ExactCi::new(matrix.clone(), vec![1], vec![2]);

    assert!(!ci.is_empty());
    assert_eq!(ci.row(2), vec![7.0, 8.0, 10.0]);
    assert_eq!(ci.col(1), vec![2.0, 5.0, 8.0]);
    assert_eq!(ci.available_rows(), vec![0, 2]);
    assert_eq!(ci.available_cols(), vec![0, 1]);

    let full = ci.to_matrix();
    assert_eq!(full.nrows(), matrix.nrows());
    assert_eq!(full.ncols(), matrix.ncols());
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            assert_eq!(full[[i, j]], matrix[[i, j]]);
        }
    }

    let local_error = ci.local_error(&matrix, &[0, 2], &[0, 1]);
    for i in 0..local_error.nrows() {
        for j in 0..local_error.ncols() {
            assert_eq!(local_error[[i, j]], 0.0);
        }
    }
}

#[test]
fn find_new_pivot_covers_success_and_error_paths() {
    let ci = ExactCi::new(sample_matrix(), vec![1], vec![2]);
    let mut perturbed = sample_matrix();
    perturbed[[2, 1]] += 10.0;

    let ((row, col), error) = ci.find_new_pivot(&perturbed).unwrap();

    assert_eq!((row, col), (2, 1));
    assert_eq!(error, 10.0);

    let ((row, col), error) = ci.find_new_pivot_in(&perturbed, &[0, 2], &[0, 1]).unwrap();

    assert_eq!((row, col), (2, 1));
    assert_eq!(error, 10.0);

    assert!(matches!(
        ci.find_new_pivot_in(&perturbed, &[], &[0]),
        Err(MatrixCIError::EmptyIndexSet { dimension }) if dimension == "rows"
    ));
    assert!(matches!(
        ci.find_new_pivot_in(&perturbed, &[0], &[]),
        Err(MatrixCIError::EmptyIndexSet { dimension }) if dimension == "cols"
    ));

    let full_rank = ExactCi::new(sample_matrix(), vec![0, 1, 2], vec![0, 1, 2]);
    assert!(matches!(
        full_rank.find_new_pivot_in(&perturbed, &[0], &[0]),
        Err(MatrixCIError::FullRank)
    ));
}
