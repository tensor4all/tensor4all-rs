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

#[test]
fn extreme_nonzero_pivots_reconstruct_in_every_scalar_kind() {
    for scale in [2f64.powi(-600), 2f64.powi(600)] {
        check_small_pivots(scale, 1e-12);
        check_small_pivots(Complex64::new(scale, scale / 2.0), 1e-12);
    }
    for scale in [2f32.powi(-90), 2f32.powi(90)] {
        check_small_pivots(scale, 1e-4);
        check_small_pivots(Complex32::new(scale, scale / 2.0), 1e-4);
    }
}

fn check_scale_invariance<T: MatrixLuciScalar>(phase: T, factors: &[f64], tolerance: f64) {
    // Rank three with exact binary coefficients, unique pivots, and a zero tail.
    let data = vec![
        16.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 2.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    ];
    let base: Vec<T> = data.into_iter().map(|x| phase * T::from_f64(x)).collect();
    for left_orthogonal in [false, true] {
        for rel_tol in [0.0, tolerance] {
            let options = RrLUOptions {
                left_orthogonal,
                rel_tol,
                ..Default::default()
            };
            let reference = rrlu(
                &Matrix::from_col_major_vec(5, 4, base.clone()),
                Some(options.clone()),
            )
            .unwrap();
            assert_eq!(reference.npivots(), 3);
            for &factor in factors {
                let scaled: Vec<T> = base.iter().map(|&v| v * T::from_f64(factor)).collect();
                let matrix = Matrix::from_col_major_vec(5, 4, scaled.clone());
                let lu = rrlu(&matrix, Some(options.clone())).unwrap();
                let dense =
                    matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap();
                let lazy = matrix_luci_factors_from_blocks(
                    5,
                    4,
                    |rows, cols, out| {
                        for (j, &col) in cols.iter().enumerate() {
                            for (i, &row) in rows.iter().enumerate() {
                                out[i + rows.len() * j] = scaled[row + 5 * col];
                            }
                        }
                    },
                    options.clone(),
                )
                .unwrap();
                for (rank, rows, cols, errors, left, right) in [
                    (
                        lu.npivots(),
                        lu.row_indices(),
                        lu.col_indices(),
                        lu.pivot_errors(),
                        lu.left(true),
                        lu.right(true),
                    ),
                    (
                        dense.rank,
                        dense.row_indices,
                        dense.col_indices,
                        dense.pivot_errors,
                        dense.left,
                        dense.right,
                    ),
                    (
                        lazy.rank,
                        lazy.row_indices,
                        lazy.col_indices,
                        lazy.pivot_errors,
                        lazy.left,
                        lazy.right,
                    ),
                ] {
                    assert_eq!(rank, 3, "scale {factor:e}");
                    assert_eq!(rows, reference.row_indices());
                    assert_eq!(cols, reference.col_indices());
                    for (&actual, expected) in errors.iter().zip(reference.pivot_errors()) {
                        assert!(
                            (actual / factor - expected).abs()
                                <= tolerance * 16.0 * phase.abs_val(),
                            "scaled pivot error {actual:e} vs {expected:e}"
                        );
                    }
                    let actual = mat_mul(&left, &right).unwrap();
                    let error = actual
                        .as_col_major_slice()
                        .iter()
                        .zip(&scaled)
                        .map(|(&a, &b)| (a - b).abs_val() / factor)
                        .fold(0.0_f64, f64::max);
                    assert!(
                        error <= tolerance * 16.0 * phase.abs_val(),
                        "reconstruction error {error:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn rank_pivots_and_errors_are_scale_invariant() {
    let factors = [
        2f64.powi(-600),
        2f64.powi(-70),
        1.0,
        2f64.powi(70),
        2f64.powi(600),
    ];
    check_scale_invariance(1.0_f64, &factors, 1e-12);
    check_scale_invariance(Complex64::new(1.0, 0.5), &factors, 1e-12);
    let factors = [2f64.powi(-90), 1.0, 2f64.powi(90)];
    check_scale_invariance(1.0_f32, &factors, 1e-4);
    check_scale_invariance(Complex32::new(1.0, 0.5), &factors, 1e-4);
}

#[test]
fn non_finite_inputs_are_errors_even_outside_the_selected_pivot() {
    use tensor4all_core::MatrixCIError;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..4 {
            let mut data = vec![4.0, 0.0, 0.0, 2.0];
            data[position] = invalid;
            let matrix = Matrix::from_col_major_vec(2, 2, data);
            for options in [
                None,
                Some(RrLUOptions {
                    max_bond_dim: 0,
                    ..Default::default()
                }),
            ] {
                assert!(matches!(
                    rrlu(&matrix, options),
                    Err(MatrixCIError::NaNEncountered { .. })
                ));
            }
        }
    }
}

#[test]
fn unrepresentable_reciprocals_stop_in_the_component_type() {
    let matrix = Matrix::from_col_major_vec(1, 1, vec![f32::from_bits(1)]);
    assert_eq!(rrlu(&matrix, None).unwrap().npivots(), 0);
    let matrix = Matrix::from_col_major_vec(1, 1, vec![f64::from_bits(1)]);
    assert_eq!(rrlu(&matrix, None).unwrap().npivots(), 0);
    // Subnormal does not itself mean undividable: these reciprocals fit.
    for value in [f64::MIN_POSITIVE / 2.0, f64::MIN_POSITIVE] {
        let lu = rrlu(&Matrix::from_col_major_vec(1, 1, vec![value]), None).unwrap();
        assert_eq!(lu.npivots(), 1);
        assert_eq!(lu.pivot_errors(), vec![value, 0.0]);
    }
}

#[test]
fn triangular_conversion_handles_extreme_complex_pivots() {
    use tensor4all_core::matrixlu::{cols_to_l_matrix, rows_to_u_matrix};
    for scale in [2f64.powi(-600), 2f64.powi(600)] {
        let pivot = Complex64::new(scale, scale / 2.0);
        let p = Matrix::from_col_major_vec(1, 1, vec![pivot]);
        let mut col = Matrix::from_col_major_vec(2, 1, vec![pivot, pivot * 2.0]);
        cols_to_l_matrix(&mut col, &p, true).unwrap();
        let mut row = Matrix::from_col_major_vec(1, 2, vec![pivot, pivot * 2.0]);
        rows_to_u_matrix(&mut row, &p, false).unwrap();
        for actual in [col, row] {
            for (&value, expected) in actual.as_col_major_slice().iter().zip([1.0, 2.0]) {
                assert!((value - Complex64::new(expected, 0.0)).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn lazy_non_finite_residual_has_the_public_numerical_error_kind() {
    use tensor4all_core::MatrixCIError;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = matrix_luci_factors_from_blocks(
            1,
            1,
            |_, _, out| out[0] = invalid,
            RrLUOptions::default(),
        );
        assert!(matches!(result, Err(MatrixCIError::NaNEncountered { .. })));
    }
}

#[test]
fn absolute_only_truncation_does_not_acquire_a_relative_epsilon_floor() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1e-17]);
    let lu = rrlu(
        &matrix,
        Some(RrLUOptions {
            rel_tol: 0.0,
            abs_tol: 1e-20,
            ..Default::default()
        }),
    )
    .unwrap();
    assert_eq!(lu.npivots(), 2);
    assert_eq!(lu.pivot_errors(), vec![1.0, 1e-17, 0.0]);
}

#[test]
fn triangular_conversion_rejects_non_finite_and_undividable_pivots() {
    use tensor4all_core::{
        matrixlu::{cols_to_l_matrix, rows_to_u_matrix},
        MatrixCIError,
    };
    for pivot in [f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        let p = Matrix::from_col_major_vec(1, 1, vec![pivot]);
        let mut c = Matrix::from_col_major_vec(1, 1, vec![1.0]);
        let mut r = c.clone();
        for result in [
            cols_to_l_matrix(&mut c, &p, true),
            rows_to_u_matrix(&mut r, &p, false),
        ] {
            if pivot.is_finite() {
                assert!(matches!(result, Err(MatrixCIError::SingularMatrix)));
            } else {
                assert!(matches!(result, Err(MatrixCIError::NaNEncountered { .. })));
            }
        }
    }
}

#[test]
fn elimination_overflow_is_reported_as_a_numerical_error() {
    use tensor4all_core::MatrixCIError;
    let matrix = Matrix::from_col_major_vec(2, 2, vec![f64::MAX, f64::MAX, -f64::MAX, f64::MAX]);
    assert!(matches!(
        rrlu(&matrix, None),
        Err(MatrixCIError::NaNEncountered { .. })
    ));
}
