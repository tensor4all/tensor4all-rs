use num_complex::{Complex32, Complex64};
use tensor4all_core::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix, rrlu, MatrixLuciScalar,
    RrLUOptions,
};
use tensor4all_tensorbackend::{mat_mul, Matrix};

fn check_small_pivots<T: MatrixLuciScalar>(scale: T, tolerance: f64) {
    for (nrows, ncols) in [(2, 2), (2, 3), (3, 2)] {
        let data: Vec<T> = (0..ncols)
            .flat_map(|col| {
                (0..nrows).map(move |row| scale * T::from_f64((1 + row + 2 * col) as f64))
            })
            .collect();
        let matrix = Matrix::from_col_major_vec(nrows, ncols, data.clone());
        for left_orthogonal in [false, true] {
            for rel_tol in [0.0, tolerance] {
                let options = RrLUOptions {
                    rel_tol,
                    abs_tol: 0.0,
                    left_orthogonal,
                    ..Default::default()
                };
                let lu = rrlu(&matrix, Some(options.clone())).unwrap();
                let dense =
                    matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap();
                let lazy = matrix_luci_factors_from_blocks(
                    nrows,
                    ncols,
                    |rows, cols, out| {
                        for (j, &col) in cols.iter().enumerate() {
                            for (i, &row) in rows.iter().enumerate() {
                                out[i + rows.len() * j] = data[row + nrows * col];
                            }
                        }
                    },
                    options,
                )
                .unwrap();
                for (rank, left, right) in [
                    (lu.npivots(), lu.left(true), lu.right(true)),
                    (dense.rank, dense.left, dense.right),
                    (lazy.rank, lazy.left, lazy.right),
                ] {
                    assert_eq!(rank, 2);
                    let actual = mat_mul(&left, &right).unwrap();
                    let error = actual
                        .as_col_major_slice()
                        .iter()
                        .zip(&data)
                        .map(|(&a, &b)| (a - b).abs_val() / scale.abs_val())
                        .fold(0.0_f64, f64::max);
                    assert!(error < tolerance, "relative reconstruction error: {error}");
                }
            }
        }
    }
}

#[test]
fn small_nonzero_pivots_reconstruct_in_every_scalar_kind() {
    for scale in [1.0, 1e-20] {
        check_small_pivots(scale, 1e-12);
        check_small_pivots(Complex64::new(scale, scale / 2.0), 1e-12);
    }
    for scale in [1.0_f32, 1e-18] {
        check_small_pivots(scale, 1e-4);
        check_small_pivots(Complex32::new(scale, scale / 2.0), 1e-4);
    }
}

#[test]
fn small_pivots_still_obey_explicit_tolerances_and_stop_at_zero() {
    for left_orthogonal in [false, true] {
        for (data, rel_tol, abs_tol, rank, expected) in [
            (
                vec![1e-20, 0.0, 0.0, 1e-23],
                1e-2,
                0.0,
                1,
                vec![1e-20, 0.0, 0.0, 0.0],
            ),
            (
                vec![1e-20, 0.0, 0.0, 1e-23],
                0.0,
                1e-22,
                1,
                vec![1e-20, 0.0, 0.0, 0.0],
            ),
            (vec![0.0; 4], 0.0, 0.0, 0, vec![0.0; 4]),
        ] {
            let options = RrLUOptions {
                rel_tol,
                abs_tol,
                left_orthogonal,
                ..Default::default()
            };
            let matrix = Matrix::from_col_major_vec(2, 2, data.clone());
            let lu = rrlu(&matrix, Some(options.clone())).unwrap();
            let dense = matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap();
            let lazy = matrix_luci_factors_from_blocks(
                2,
                2,
                |rows, cols, out| {
                    for (j, &col) in cols.iter().enumerate() {
                        for (i, &row) in rows.iter().enumerate() {
                            out[i + rows.len() * j] = data[row + 2 * col];
                        }
                    }
                },
                options,
            )
            .unwrap();
            for (actual_rank, left, right) in [
                (lu.npivots(), lu.left(true), lu.right(true)),
                (dense.rank, dense.left, dense.right),
                (lazy.rank, lazy.left, lazy.right),
            ] {
                assert_eq!(actual_rank, rank);
                let actual = mat_mul(&left, &right).unwrap();
                assert_eq!(actual.as_col_major_slice(), &expected);
            }
        }
    }
}
