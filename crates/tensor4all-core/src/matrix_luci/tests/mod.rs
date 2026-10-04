use super::*;
use crate::rrlu;
use num_complex::{Complex32, Complex64};
use std::time::{Duration, Instant};
use tensor4all_tensorbackend::{from_vec2d, mat_mul};

#[test]
fn test_matrixluci_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.nrows(), 3);
    assert_eq!(luci.ncols(), 3);
    assert_eq!(luci.rank(), 3);
}

#[test]
fn matrix_luci_factors_from_matrix_owned_matches_borrowed() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ]);

    let borrowed = matrix_luci_factors_from_matrix(&m, None).unwrap();
    let owned = matrix_luci_factors_from_matrix_owned(m, None).unwrap();

    assert_eq!(owned.rank, borrowed.rank);
    assert_eq!(owned.row_indices, borrowed.row_indices);
    assert_eq!(owned.col_indices, borrowed.col_indices);
    assert_eq!(
        owned.left.as_col_major_slice(),
        borrowed.left.as_col_major_slice()
    );
    assert_eq!(
        owned.right.as_col_major_slice(),
        borrowed.right.as_col_major_slice()
    );
}

#[test]
fn matrix_luci_owned_rectangular_roundtrip_preserves_column_major_storage() {
    let data = vec![1.0_f64, 3.0, 2.0, 5.0, 4.0, 6.0];
    let matrix = Matrix::try_from_col_major_vec(2, 3, data.clone()).unwrap();
    assert_eq!(matrix.into_col_major_vec(), data);

    let borrowed_matrix = Matrix::try_from_col_major_vec(2, 3, data.clone()).unwrap();
    let borrowed = matrix_luci_factors_from_matrix(&borrowed_matrix, None).unwrap();
    let owned = matrix_luci_factors_from_matrix_owned(
        Matrix::try_from_col_major_vec(2, 3, data).unwrap(),
        None,
    )
    .unwrap();

    assert_eq!(owned.rank, borrowed.rank);
    assert_eq!(owned.left.nrows(), 2);
    assert_eq!(owned.right.ncols(), 3);
    let reconstructed = mat_mul(&owned.left, &owned.right).unwrap();
    for col in 0..3 {
        for row in 0..2 {
            let expected = [1.0_f64, 3.0, 2.0, 5.0, 4.0, 6.0][row + 2 * col];
            assert!((reconstructed[[row, col]] - expected).abs() < 1.0e-10);
        }
    }
}

#[test]
fn matrix_luci_owned_matches_borrowed_for_all_scalar_kinds() {
    fn assert_matches<T>(data: Vec<T>)
    where
        T: crate::Scalar + crate::MatrixLuciScalar + std::fmt::Debug + PartialEq,
    {
        let borrowed_matrix = Matrix::try_from_col_major_vec(2, 3, data.clone()).unwrap();
        let owned_matrix = Matrix::try_from_col_major_vec(2, 3, data).unwrap();
        let borrowed = matrix_luci_factors_from_matrix(&borrowed_matrix, None).unwrap();
        let owned = matrix_luci_factors_from_matrix_owned(owned_matrix, None).unwrap();

        assert_eq!(owned.rank, borrowed.rank);
        assert_eq!(owned.row_indices, borrowed.row_indices);
        assert_eq!(owned.col_indices, borrowed.col_indices);
        assert_eq!(
            owned.left.as_col_major_slice(),
            borrowed.left.as_col_major_slice()
        );
        assert_eq!(
            owned.right.as_col_major_slice(),
            borrowed.right.as_col_major_slice()
        );
    }

    assert_matches(vec![1.0_f32, 3.0, 2.0, 5.0, 4.0, 6.0]);
    assert_matches(vec![1.0_f64, 3.0, 2.0, 5.0, 4.0, 6.0]);
    assert_matches(vec![
        Complex32::new(1.0, 0.25),
        Complex32::new(3.0, -0.5),
        Complex32::new(2.0, 0.75),
        Complex32::new(5.0, -1.0),
        Complex32::new(4.0, 1.25),
        Complex32::new(6.0, -1.5),
    ]);
    assert_matches(vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(3.0, -0.5),
        Complex64::new(2.0, 0.75),
        Complex64::new(5.0, -1.0),
        Complex64::new(4.0, 1.25),
        Complex64::new(6.0, -1.5),
    ]);
}

#[test]
fn test_matrixluci_reconstruct() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    let approx = luci.to_matrix();

    for i in 0..2 {
        for j in 0..2 {
            let diff = (m[[i, j]] - approx[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank2_iplusj_left_orthogonal() {
    // Pi matrix for f(i,j) = i + j on 4x4 grid
    let m = from_vec2d(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let opts = RrLUOptions {
        left_orthogonal: true,
        ..Default::default()
    };
    let luci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();
    assert_eq!(luci.rank(), 2);

    // Check left() * right() = Pi
    let left = luci.left();
    let right = luci.right();
    let reconstructed = mat_mul(&left, &right).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): expected {} got {} (diff {})",
                i,
                j,
                m[[i, j]],
                reconstructed[[i, j]],
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank2_iplusj_right_orthogonal() {
    let m = from_vec2d(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let opts = RrLUOptions {
        left_orthogonal: false,
        ..Default::default()
    };
    let luci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();
    assert_eq!(luci.rank(), 2);

    let reconstructed = mat_mul(&luci.left(), &luci.right()).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): expected {} got {} (diff {})",
                i,
                j,
                m[[i, j]],
                reconstructed[[i, j]],
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank_deficient() {
    // Rank-1 matrix
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.rank(), 1);
}

#[derive(Clone, Copy, Default)]
struct MatrixLuciHilbertTiming {
    selection: Duration,
    gather: Duration,
    left_factor: Duration,
    right_factor: Duration,
    rank: usize,
    last_error: f64,
    checksum: f64,
}

impl MatrixLuciHilbertTiming {
    fn total(self) -> Duration {
        self.selection + self.gather + self.left_factor + self.right_factor
    }
}

fn hilbert_matrix(size: usize) -> Matrix<f64> {
    let mut data = vec![0.0; size * size];
    for col in 0..size {
        for row in 0..size {
            data[row + size * col] = 1.0 / ((row + col + 1) as f64);
        }
    }
    Matrix::from_col_major_vec(size, size, data)
}

fn timed_hilbert_matrix_luci_once(size: usize, left_orthogonal: bool) -> MatrixLuciHilbertTiming {
    let matrix = hilbert_matrix(size);
    let options = RrLUOptions {
        max_bond_dim: usize::MAX,
        rel_tol: 0.0,
        abs_tol: 1.0e-10,
        left_orthogonal,
    };

    let mut timing = MatrixLuciHilbertTiming::default();

    let start = Instant::now();
    let lu = rrlu(&matrix, Some(options)).unwrap();
    timing.selection = start.elapsed();
    timing.rank = lu.npivots();
    timing.last_error = lu.pivot_errors().last().copied().unwrap_or(0.0);

    let start = Instant::now();
    let left = if left_orthogonal {
        rrlu_cols_times_pivot_solve(
            &lu,
            &tensor4all_tensorbackend::default_cpu_execution_context(),
        )
        .unwrap()
    } else {
        rrlu_colmatrix(
            &lu,
            &tensor4all_tensorbackend::default_cpu_execution_context(),
        )
        .unwrap()
    };
    timing.left_factor = start.elapsed();

    let start = Instant::now();
    let right = if left_orthogonal {
        rrlu_rowmatrix(
            &lu,
            &tensor4all_tensorbackend::default_cpu_execution_context(),
        )
        .unwrap()
    } else {
        rrlu_pivot_solve_times_rows(
            &lu,
            &tensor4all_tensorbackend::default_cpu_execution_context(),
        )
        .unwrap()
    };
    timing.right_factor = start.elapsed();

    let left_checksum = if left.nrows() > 0 && left.ncols() > 0 {
        left[[0, 0]].abs()
    } else {
        0.0
    };
    let right_checksum = if right.nrows() > 0 && right.ncols() > 0 {
        right[[0, 0]].abs()
    } else {
        0.0
    };
    timing.checksum = left_checksum + right_checksum;
    timing
}

fn timing_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn timing_median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

#[test]
#[ignore]
fn matrix_luci_hilbert_timing() {
    let repeats = std::env::var("T4A_MATRIX_LUCI_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let sizes = std::env::var("T4A_MATRIX_LUCI_SIZES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![4, 8, 16, 32, 64]);

    println!(
        "impl,matrix,size,repeats,left_orthogonal,selection_ms,gather_ms,left_factor_ms,right_factor_ms,total_ms,rank,last_error,checksum"
    );
    for size in sizes {
        for left_orthogonal in [true, false] {
            let runs = (0..repeats)
                .map(|_| timed_hilbert_matrix_luci_once(size, left_orthogonal))
                .collect::<Vec<_>>();
            let first = runs[0];
            println!(
                "rust,hilbert,{size},{repeats},{left_orthogonal},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6e},{:.6e}",
                timing_median(runs.iter().map(|run| timing_ms(run.selection)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.gather)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.left_factor)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.right_factor)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.total())).collect()),
                first.rank,
                first.last_error,
                first.checksum,
            );
        }
    }
}

/// Runs `f` on another thread while this thread holds the process-global
/// backend lock; completing within two seconds proves `f` never consulted the
/// global context.
fn run_while_default_context_is_busy<R: Send + 'static>(
    f: impl FnOnce() -> R + Send + 'static,
) -> R {
    tensor4all_tensorbackend::with_default_backend(|_| {
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        let result = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("factorization blocked on the process-global context");
        handle.join().expect("worker thread panicked");
        result
    })
}

fn assert_owned_in_matches_global_factors<T>(matrix: Matrix<T>, left_orthogonal: bool)
where
    T: Scalar + crate::MatrixLuciScalar + std::fmt::Debug,
{
    let options = RrLUOptions {
        left_orthogonal,
        ..RrLUOptions::default()
    };
    let expected =
        matrix_luci_factors_from_matrix_owned(matrix.clone(), Some(options.clone())).unwrap();
    let input = matrix.clone();
    let actual = run_while_default_context_is_busy(move || {
        let context = tensor4all_tensorbackend::CpuExecutionContext::from_backend(
            tenferro_cpu::CpuBackend::new(),
        );
        matrix_luci_factors_from_matrix_owned_in(input, Some(options), &context).unwrap()
    });
    assert_eq!(actual.rank, expected.rank);
    assert_eq!(actual.row_indices, expected.row_indices);
    assert_eq!(actual.col_indices, expected.col_indices);
    for (x, y) in [
        (&actual.left, &expected.left),
        (&actual.right, &expected.right),
    ] {
        for (a, b) in x.as_col_major_slice().iter().zip(y.as_col_major_slice()) {
            assert!((*a - *b).abs_sq() < 1.0e-24, "{a:?} != {b:?}");
        }
    }
    let rebuilt = mat_mul(&actual.left, &actual.right).unwrap();
    for (a, b) in rebuilt
        .as_col_major_slice()
        .iter()
        .zip(matrix.as_col_major_slice())
    {
        assert!((*a - *b).abs_sq() < 1.0e-20, "{a:?} != {b:?}");
    }
}

#[test]
fn owned_in_factors_match_global_factors_in_both_gauges_f64() {
    let matrix = from_vec2d(vec![
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
        vec![2.0, 1.0, 0.5],
    ]);
    assert_owned_in_matches_global_factors(matrix.clone(), true);
    assert_owned_in_matches_global_factors(matrix, false);
}

#[test]
fn owned_in_factors_match_global_factors_in_both_gauges_c64() {
    let z = |re: f64, im: f64| Complex64::new(re, im);
    let matrix = from_vec2d(vec![
        vec![z(1.0, 1.0), z(0.0, 2.0), z(3.0, -1.0)],
        vec![z(2.0, 0.0), z(-1.0, 1.0), z(0.5, 0.5)],
    ]);
    assert_owned_in_matches_global_factors(matrix.clone(), true);
    assert_owned_in_matches_global_factors(matrix, false);
}

#[test]
fn row_interpolation_context_matches_full_factors_without_right_factor() {
    let context = tensor4all_tensorbackend::default_cpu_execution_context();
    let a = tensor4all_tensorbackend::Matrix::from_col_major_vec(
        3,
        2,
        vec![1.0_f64, 2.0, 4.0, 3.0, 6.0, 12.0],
    );
    let row = super::matrix_luci_row_interpolation_owned_in(a.clone(), None, &context).unwrap();
    assert_eq!(row.rows, vec![2]);
    assert_eq!(row.interpolation.as_col_major_slice(), &[0.25, 0.5, 1.0]);
    let full = super::matrix_luci_factors_from_matrix_owned_in(
        a,
        Some(crate::RrLUOptions {
            left_orthogonal: true,
            ..Default::default()
        }),
        &context,
    )
    .unwrap();
    assert_eq!(row.rows, full.row_indices);
    assert_eq!(
        row.interpolation.as_col_major_slice(),
        full.left.as_col_major_slice()
    );
    let zero = super::matrix_luci_row_interpolation_owned_in(
        tensor4all_tensorbackend::Matrix::<f64>::zeros(3, 2),
        None,
        &context,
    )
    .unwrap();
    assert!(zero.rows.is_empty());
    assert_eq!(zero.interpolation.ncols(), 0);
}

#[test]
fn owned_context_facades_reject_invalid_controls_and_nonfinite_inputs() {
    fn check<T: Scalar + crate::MatrixLuciScalar>() {
        let context = tensor4all_tensorbackend::default_cpu_execution_context();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            for absolute in [false, true] {
                let mut options = RrLUOptions::default();
                if absolute {
                    options.abs_tol = value;
                } else {
                    options.rel_tol = value;
                }
                let a = Matrix::<T>::zeros(0, 2);
                assert!(matches!(
                    matrix_luci_row_interpolation_owned_in(
                        a.clone(),
                        Some(options.clone()),
                        &context
                    ),
                    Err(MatrixCIError::InvalidArgument { .. })
                ));
                assert!(matches!(
                    matrix_luci_factors_from_matrix_owned_in(a, Some(options), &context),
                    Err(MatrixCIError::InvalidArgument { .. })
                ));
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let a = Matrix::from_col_major_vec(1, 2, vec![T::one(), T::from_f64(value)]);
            assert!(matrix_luci_row_interpolation_owned_in(a.clone(), None, &context).is_err());
            assert!(matrix_luci_factors_from_matrix_owned_in(a, None, &context).is_err());
        }
        // A rank cap of zero is valid and still reports the uneliminated residual.
        let a = Matrix::from_col_major_vec(1, 2, vec![T::one(), T::from_f64(2.0)]);
        let options = RrLUOptions {
            max_bond_dim: 0,
            ..Default::default()
        };
        let id = matrix_luci_row_interpolation_owned_in(a, Some(options), &context).unwrap();
        assert!(id.rows.is_empty());
        assert_eq!(id.pivot_magnitudes, vec![2.0]);
    }
    check::<f64>();
    check::<f32>();
    check::<num_complex::Complex64>();
    check::<num_complex::Complex32>();
}

#[test]
fn row_only_rank_cap_diagnostic_measures_the_remaining_schur_complement() {
    fn check<T: Scalar + crate::MatrixLuciScalar>() {
        let context = tensor4all_tensorbackend::default_cpu_execution_context();
        for tail in [0.0, 1.0] {
            for cap in 0..=3 {
                let a = Matrix::from_col_major_vec(
                    3,
                    3,
                    [4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, tail]
                        .map(T::from_f64)
                        .to_vec(),
                );
                let id = matrix_luci_row_interpolation_owned_in(
                    a,
                    Some(RrLUOptions {
                        max_bond_dim: cap,
                        ..Default::default()
                    }),
                    &context,
                )
                .unwrap();
                let rank = cap.min(if tail == 0.0 { 2 } else { 3 });
                assert_eq!(id.rows.len(), rank);
                assert_eq!(id.pivot_magnitudes, [4.0, 2.0, tail, 0.0][..=rank]);
            }
        }
    }
    check::<f64>();
    check::<f32>();
    check::<num_complex::Complex64>();
    check::<num_complex::Complex32>();
}
