use super::*;
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::from_vec2d;

fn assert_selected_cross<T: Scalar + MatrixLuciScalar>(
    from_parts: impl Fn(f64, f64) -> T,
    complex: bool,
    rounding: f64,
) {
    let u = [
        from_parts(1.0, if complex { 1.0 } else { 0.0 }),
        from_parts(2.0, if complex { -1.0 } else { 0.0 }),
    ];
    let v = [
        from_parts(1.0, if complex { -1.0 } else { 0.0 }),
        from_parts(2.0, if complex { 1.0 } else { 0.0 }),
        from_parts(3.0, if complex { -2.0 } else { 0.0 }),
    ];
    let delta = from_parts(0.125, if complex { 0.125 } else { 0.0 });
    let mut values = (0..3)
        .flat_map(|j| u.iter().map(move |&x| x * v[j]))
        .collect::<Vec<_>>();
    values[5] = values[5] + delta;
    let a = Matrix::from_col_major_vec(2, 3, values);
    for left_orthogonal in [false, true] {
        let cross = matrix_luci_factors_from_pivots(&a, &[0], &[0], left_orthogonal).unwrap();
        assert_eq!(cross.rank, 1);
        assert_eq!(cross.row_indices, vec![0]);
        assert_eq!(cross.col_indices, vec![0]);
        let reconstructed = mat_mul(&cross.left, &cross.right).unwrap();
        for j in 0..3 {
            for (i, &x) in u.iter().enumerate() {
                assert!(Scalar::abs_val(reconstructed[[i, j]] - x * v[j]) < rounding);
            }
        }
        assert_eq!(cross.pivot_errors.len(), 1);
        assert!((cross.pivot_errors[0] - Scalar::abs_val(delta)).abs() < rounding);
        let identity = if left_orthogonal {
            cross.left[[0, 0]]
        } else {
            cross.right[[0, 0]]
        };
        assert_eq!(
            Scalar::abs_val(identity - <T as Scalar>::from_f64(1.0)),
            0.0
        );
    }
}

#[test]
fn fixed_cross_measures_full_residual_for_all_scalars_and_orientations() {
    assert_selected_cross::<f32>(|re, _| re as f32, false, 1e-5);
    assert_selected_cross::<f64>(|re, _| re, false, 1e-12);
    assert_selected_cross::<Complex32>(|re, im| Complex32::new(re as f32, im as f32), true, 1e-5);
    assert_selected_cross::<Complex64>(Complex64::new, true, 1e-12);
}

#[test]
fn fixed_cross_preserves_prescribed_order_without_conjugating() {
    let a = from_vec2d(vec![
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        vec![Complex64::new(3.0, -2.0), Complex64::new(5.0, 1.0)],
    ]);
    for left in [false, true] {
        let cross = matrix_luci_factors_from_pivots(&a, &[1, 0], &[0, 1], left).unwrap();
        assert_eq!(cross.row_indices, vec![1, 0]);
        assert_eq!(cross.col_indices, vec![0, 1]);
        assert!(cross.pivot_errors[0] < 1e-12);
        for (pivot, &row) in cross.row_indices.iter().enumerate() {
            for (bond, &col) in cross.col_indices.iter().enumerate() {
                let value = if left {
                    cross.left[[row, bond]]
                } else {
                    cross.right[[pivot, col]]
                };
                assert_eq!(
                    value,
                    Complex64::new(if pivot == bond { 1.0 } else { 0.0 }, 0.0)
                );
            }
        }
        let reconstructed = mat_mul(&cross.left, &cross.right).unwrap();
        for (&x, &y) in a
            .as_col_major_slice()
            .iter()
            .zip(reconstructed.as_col_major_slice())
        {
            assert!((x - y).norm() < 1e-12);
        }
    }
}

#[test]
fn fixed_cross_validates_selection_before_reconstruction() {
    let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    for (rows, cols) in [(vec![], vec![]), (vec![0, 1], vec![0])] {
        assert!(matches!(
            matrix_luci_factors_from_pivots(&a, &rows, &cols, true),
            Err(MatrixCIError::InvalidArgument { .. })
        ));
    }
    assert!(matches!(
        matrix_luci_factors_from_pivots(&a, &[2], &[0], true),
        Err(MatrixCIError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        matrix_luci_factors_from_pivots(&a, &[0], &[2], true),
        Err(MatrixCIError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        matrix_luci_factors_from_pivots(&a, &[0, 0], &[0, 1], true),
        Err(MatrixCIError::DuplicatePivotRow { .. })
    ));
    assert!(matches!(
        matrix_luci_factors_from_pivots(&a, &[0, 1], &[1, 1], true),
        Err(MatrixCIError::DuplicatePivotCol { .. })
    ));
}

#[test]
fn fixed_cross_rejects_zero_and_dependent_pivot_blocks() {
    for a in [
        from_vec2d(vec![vec![0.0_f64, 0.0], vec![0.0, 0.0]]),
        from_vec2d(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
    ] {
        assert!(matches!(
            matrix_luci_factors_from_pivots(&a, &[0, 1], &[0, 1], true),
            Err(MatrixCIError::SingularMatrix)
        ));
    }
}

#[test]
fn fixed_cross_rejects_nonfinite_values_outside_pivots() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, invalid]]);
        assert!(matches!(
            matrix_luci_factors_from_pivots(&a, &[0], &[0], true),
            Err(MatrixCIError::NaNEncountered { .. })
        ));
    }
}

#[test]
fn fixed_cross_singularity_check_is_scale_independent() {
    for scale in [1e-200_f64, 1.0, 1e200] {
        let a = from_vec2d(vec![
            vec![scale, 2.0 * scale],
            vec![3.0 * scale, 5.0 * scale],
        ]);
        let cross = matrix_luci_factors_from_pivots(&a, &[1, 0], &[0, 1], true).unwrap();
        assert!(cross.pivot_errors[0] / scale < 1e-12);
    }
}

fn assert_one_axis<T: Scalar + MatrixLuciScalar>(
    from_parts: impl Fn(f64, f64) -> T,
    rounding: f64,
) {
    let a = Matrix::from_col_major_vec(
        2,
        3,
        [
            (1.0, 1.0),
            (2.0, -1.0),
            (3.0, 0.0),
            (4.0, 2.0),
            (5.0, -1.0),
            (7.0, 1.0),
        ]
        .into_iter()
        .map(|(r, i)| from_parts(r, i))
        .collect(),
    );
    for left in [false, true] {
        for rows_fixed in [false, true] {
            let (rows, cols) = if rows_fixed {
                (&[1, 0][..], &[][..])
            } else {
                (&[][..], &[2, 0][..])
            };
            let cross = matrix_luci_factors_from_pivots(&a, rows, cols, left).unwrap();
            assert_eq!(cross.rank, 2);
            if rows_fixed {
                assert_eq!(cross.row_indices, rows);
            } else {
                assert_eq!(cross.col_indices, cols);
            }
            let reconstructed = mat_mul(&cross.left, &cross.right).unwrap();
            let residual = a
                .as_col_major_slice()
                .iter()
                .zip(reconstructed.as_col_major_slice())
                .map(|(&a, &b)| Scalar::abs_val(a - b))
                .fold(0.0_f64, f64::max);
            assert!(residual < rounding, "residual={residual}");
            assert_eq!(cross.pivot_errors, vec![residual]);
        }
    }
}

#[test]
fn prescribed_axis_completion_preserves_order_and_complex_values() {
    assert_one_axis::<f32>(|r, _| r as f32, 2e-5);
    assert_one_axis::<f64>(|r, _| r, 1e-12);
    assert_one_axis::<Complex32>(|r, i| Complex32::new(r as f32, i as f32), 2e-5);
    assert_one_axis::<Complex64>(Complex64::new, 1e-12);
}

#[test]
fn prescribed_axis_rejects_dependence_and_measures_unsampled_residual() {
    let deficient = from_vec2d(vec![vec![1.0_f64, 2.0], vec![2.0, 4.0]]);
    for (rows, cols) in [(&[0, 1][..], &[][..]), (&[][..], &[0, 1][..])] {
        assert!(matches!(
            matrix_luci_factors_from_pivots(&deficient, rows, cols, true),
            Err(MatrixCIError::SingularMatrix)
        ));
    }
    let a = from_vec2d(vec![vec![1.0_f64, 2.0, 0.0], vec![2.0, 4.0, 0.125]]);
    for left in [false, true] {
        let cross = matrix_luci_factors_from_pivots(&a, &[], &[0], left).unwrap();
        assert_eq!(cross.col_indices, vec![0]);
        assert_eq!(cross.row_indices, vec![1]);
        assert!((cross.pivot_errors[0] - 0.0625).abs() < 1e-12);
    }
}

#[test]
fn prescribed_axis_completion_is_independent_of_overall_scale() {
    for scale in [1e-200, 1.0, 1e200] {
        let a = from_vec2d(vec![
            vec![scale, 2.0 * scale],
            vec![3.0 * scale, 4.0 * scale],
        ]);
        for (rows, cols) in [(&[1, 0][..], &[][..]), (&[][..], &[1, 0][..])] {
            let cross = matrix_luci_factors_from_pivots(&a, rows, cols, true).unwrap();
            assert_eq!(cross.rank, 2);
            assert!(cross.pivot_errors[0] / scale < 1e-12);
        }
    }
}
