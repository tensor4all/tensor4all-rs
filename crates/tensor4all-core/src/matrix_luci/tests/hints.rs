use super::*;

fn options(left_orthogonal: bool) -> RrLUOptions {
    RrLUOptions {
        rel_tol: 0.0,
        abs_tol: 0.02,
        left_orthogonal,
        ..Default::default()
    }
}

fn verify<T: Scalar + crate::MatrixLuciScalar>(phase: T, rounding: f64) {
    for left_orthogonal in [false, true] {
        for (rows, cols) in [(3, 4), (4, 3), (3, 3)] {
            let mut a = Matrix::zeros(rows, cols);
            a[[0, 0]] = T::from_f64(1.0) * phase;
            a[[1, 1]] = T::from_f64(1.5) * phase;
            a[[2, 2]] = T::from_f64(0.01) * phase;
            let opts = options(left_orthogonal);
            let fresh =
                matrix_luci_factors_from_matrix_owned(a.clone(), Some(opts.clone())).unwrap();
            assert_eq!(fresh.row_indices, vec![1, 0]);
            // A weak hint is ignored, a strong hint wins, and hints on used
            // axes are skipped. Inverse maps must follow both permutations.
            let hinted = matrix_luci_factors_from_matrix_owned_with_hints(
                a.clone(),
                opts,
                &[(2, 2), (0, 0), (0, 1)],
            )
            .unwrap();
            assert_eq!(hinted.row_indices, vec![0, 1]);
            assert_eq!(hinted.col_indices, vec![0, 1]);
            let rebuilt = mat_mul(&hinted.left, &hinted.right).unwrap();
            let residual = a
                .as_col_major_slice()
                .iter()
                .zip(rebuilt.as_col_major_slice())
                .map(|(&a, &b)| (a - b).abs_val())
                .fold(0.0_f64, f64::max);
            let expected = 0.01 * phase.abs_val();
            assert!(
                (residual - expected).abs() < rounding,
                "residual={residual:e}"
            );
            assert!((hinted.pivot_errors.last().unwrap() - expected).abs() < rounding);
            assert_eq!(hinted.pivot_errors.len(), 3);
        }
    }
}

#[test]
fn hinted_crosses_preserve_residuals_for_all_scalars_orientations_and_shapes() {
    verify(1.0_f32, 1e-6);
    verify(1.0_f64, 1e-12);
    verify(Complex32::new(1.0, 0.25), 1e-6);
    verify(Complex64::new(1.0, 0.25), 1e-12);
}

#[test]
fn hints_never_hide_a_large_residual_or_impose_a_rank_floor() {
    let a = Matrix::from_col_major_vec(2, 2, vec![0.01_f64, 0.0, 0.0, 1.0]);
    for left in [false, true] {
        for cap in [0, 1, 2] {
            let f = matrix_luci_factors_from_matrix_owned_with_hints(
                a.clone(),
                RrLUOptions {
                    max_bond_dim: cap,
                    ..options(left)
                },
                &[(0, 0)],
            )
            .unwrap();
            assert_eq!(f.rank, cap.min(1));
            assert_eq!(f.row_indices, if cap == 0 { vec![] } else { vec![1] });
            assert_eq!(
                *f.pivot_errors.last().unwrap(),
                if cap == 0 { 1.0 } else { 0.01 }
            );
        }
        let zero = matrix_luci_factors_from_matrix_owned_with_hints(
            Matrix::<f64>::zeros(2, 2),
            options(left),
            &[(0, 0), (1, 1)],
        )
        .unwrap();
        assert_eq!(zero.rank, 0);
        assert_eq!(zero.pivot_errors, vec![0.0]);
    }
    // Half the maximum is strong enough, but it must also meet abs_tol.
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.5]);
    let f = matrix_luci_factors_from_matrix_owned_with_hints(
        a,
        RrLUOptions {
            abs_tol: 1.2,
            ..options(true)
        },
        &[(0, 0)],
    )
    .unwrap();
    assert_eq!(f.row_indices, vec![1]);
    assert_eq!(f.pivot_errors, vec![1.5, 1.0]);
}

#[test]
fn hinted_selection_still_enforces_relative_tolerance_and_reports_capped_residual() {
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.5]);
    for cap in [1, 2] {
        let f = matrix_luci_factors_from_matrix_owned_with_hints(
            a.clone(),
            RrLUOptions {
                max_bond_dim: cap,
                rel_tol: 0.8,
                abs_tol: 0.0,
                ..options(true)
            },
            &[(0, 0)],
        )
        .unwrap();
        assert_eq!(f.row_indices, vec![1]);
        assert_eq!(f.pivot_errors, vec![1.5, 1.0]);
    }
    let f = matrix_luci_factors_from_matrix_owned_with_hints(
        a,
        RrLUOptions {
            max_bond_dim: 1,
            ..options(true)
        },
        &[(0, 0)],
    )
    .unwrap();
    assert_eq!(f.row_indices, vec![0]);
    assert_eq!(f.pivot_errors, vec![1.0, 1.5]);
}

#[test]
fn empty_hints_match_the_existing_owned_route_and_duplicates_are_harmless() {
    let a = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![3.0, 2.0, 1.0]]);
    for left in [false, true] {
        let fresh = matrix_luci_factors_from_matrix_owned(a.clone(), Some(options(left))).unwrap();
        let hinted =
            matrix_luci_factors_from_matrix_owned_with_hints(a.clone(), options(left), &[])
                .unwrap();
        assert_eq!(hinted.row_indices, fresh.row_indices);
        assert_eq!(hinted.col_indices, fresh.col_indices);
        assert_eq!(hinted.pivot_errors, fresh.pivot_errors);
        assert_eq!(
            hinted.left.as_col_major_slice(),
            fresh.left.as_col_major_slice()
        );
        assert_eq!(
            hinted.right.as_col_major_slice(),
            fresh.right.as_col_major_slice()
        );
        let single =
            matrix_luci_factors_from_matrix_owned_with_hints(a.clone(), options(left), &[(0, 2)])
                .unwrap();
        let duplicate = matrix_luci_factors_from_matrix_owned_with_hints(
            a.clone(),
            options(left),
            &[(0, 2), (0, 2)],
        )
        .unwrap();
        assert_eq!(single.row_indices, duplicate.row_indices);
        assert_eq!(single.col_indices, duplicate.col_indices);
    }
}

#[test]
fn hints_validate_coordinates_lengths_options_and_nonfinite_values_before_shortcuts() {
    for (rows, cols, hints) in [
        (0, 2, vec![(0, 0)]),
        (2, 0, vec![(0, 0)]),
        (2, 2, vec![(2, 0)]),
        (2, 2, vec![(0, 2)]),
        (2, 2, vec![(0, 0); 3]),
    ] {
        let result = matrix_luci_factors_from_matrix_owned_with_hints(
            Matrix::<f64>::zeros(rows, cols),
            RrLUOptions {
                max_bond_dim: 0,
                ..options(true)
            },
            &hints,
        );
        assert!(matches!(result, Err(MatrixCIError::InvalidArgument { .. })));
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = matrix_luci_factors_from_matrix_owned_with_hints(
            Matrix::from_col_major_vec(1, 1, vec![bad]),
            options(true),
            &[(0, 0)],
        );
        assert!(result.is_err());
        for (abs_tol, rel_tol) in [(bad, 0.0), (0.0, bad), (-1.0, 0.0), (0.0, -1.0)] {
            let result = matrix_luci_factors_from_matrix_owned_with_hints(
                Matrix::from_col_major_vec(1, 1, vec![1.0_f64]),
                RrLUOptions {
                    abs_tol,
                    rel_tol,
                    ..options(true)
                },
                &[],
            );
            assert!(matches!(result, Err(MatrixCIError::InvalidArgument { .. })));
        }
    }
    let empty = matrix_luci_factors_from_matrix_owned_with_hints(
        Matrix::<f64>::zeros(0, 2),
        options(true),
        &[],
    )
    .unwrap();
    assert_eq!(empty.rank, 0);
    assert_eq!(empty.pivot_errors, vec![0.0]);
}

#[test]
fn session_hint_factors_match_the_global_route() {
    let context = tensor4all_tensorbackend::default_cpu_execution_context();
    let a = from_vec2d(vec![vec![1.0_f64, 0.0], vec![0.0, 1.5]]);
    for left in [false, true] {
        let fresh =
            matrix_luci_factors_from_matrix_owned_with_hints(a.clone(), options(left), &[(0, 0)])
                .unwrap();
        let session = context
            .with_concrete_session(|session| {
                matrix_luci_factors_from_matrix_owned_with_hints_on_session(
                    a.clone(),
                    options(left),
                    &[(0, 0)],
                    session,
                )
            })
            .unwrap()
            .unwrap();
        assert_eq!(session.row_indices, fresh.row_indices);
        assert_eq!(session.col_indices, fresh.col_indices);
        assert_eq!(session.pivot_errors, fresh.pivot_errors);
        assert_eq!(
            session.left.as_col_major_slice(),
            fresh.left.as_col_major_slice()
        );
        assert_eq!(
            session.right.as_col_major_slice(),
            fresh.right.as_col_major_slice()
        );
    }
}

#[test]
fn hints_follow_nontrivial_inverse_permutations_and_reject_elimination_overflow() {
    for left in [false, true] {
        let a = from_vec2d(vec![
            vec![3.0_f64, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let f = matrix_luci_factors_from_matrix_owned_with_hints(
            a.clone(),
            options(left),
            &[(2, 2), (1, 1), (0, 0)],
        )
        .unwrap();
        assert_eq!(f.row_indices, vec![1, 0, 2]);
        assert_eq!(f.col_indices, vec![1, 0, 2]);
        assert_eq!(
            mat_mul(&f.left, &f.right).unwrap().as_col_major_slice(),
            a.as_col_major_slice()
        );
        let result = matrix_luci_factors_from_matrix_owned_with_hints(
            from_vec2d(vec![vec![1e308_f64, -1e308], vec![1e308, 1e308]]),
            options(left),
            &[(0, 0)],
        );
        assert!(matches!(result, Err(MatrixCIError::NaNEncountered { .. })));
    }
}

#[test]
fn half_maximum_hint_boundary_is_inclusive() {
    for (hint, selected) in [(1.0_f64, 0), (1.0 - f64::EPSILON, 1)] {
        let f = matrix_luci_factors_from_matrix_owned_with_hints(
            from_vec2d(vec![vec![hint, 0.0], vec![0.0, 2.0]]),
            RrLUOptions {
                max_bond_dim: 1,
                ..options(true)
            },
            &[(0, 0)],
        )
        .unwrap();
        assert_eq!(f.row_indices, vec![selected]);
        assert_eq!(f.col_indices, vec![selected]);
        assert_eq!(
            *f.pivot_errors.last().unwrap(),
            if selected == 0 { 2.0 } else { hint }
        );
    }
}
