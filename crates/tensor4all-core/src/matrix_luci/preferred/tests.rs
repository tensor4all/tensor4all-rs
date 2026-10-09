use super::*;
use num_complex::{Complex32, Complex64};

fn verify<T: Scalar + MatrixLuciScalar>(
    parts: impl Fn(f64, f64) -> T,
    complex: bool,
    rounding: f64,
) {
    let phase = parts(1.0, if complex { 0.5 } else { 0.0 });
    let a = Matrix::from_col_major_vec(
        3,
        3,
        [1.0, 2.0, 3.0, 2.0, 4.01, 6.0, 3.0, 6.0, 9.0]
            .into_iter()
            .map(|x| T::from_f64(x) * phase)
            .collect(),
    );
    let threshold = 0.02 * Scalar::abs_val(phase);
    for left in [false, true] {
        let bad = matrix_luci_factors_from_pivots(&a, &[1], &[1], left).unwrap();
        assert!(bad.pivot_errors[0] > threshold);
        let restored =
            matrix_luci_factors_with_preferred_pivots(&a, &[2], &[2], &[1], &[1], left, threshold)
                .unwrap();
        assert_eq!(restored.rank, 1);
        assert_eq!(restored.row_indices, vec![2]);
        assert_eq!(restored.col_indices, vec![1]);
        assert!((restored.pivot_errors[0] - 0.015 * Scalar::abs_val(phase)).abs() < rounding);
        assert!(restored.pivot_errors[0] <= threshold);
        let partial = matrix_luci_factors_with_preferred_pivots(
            &a,
            &[2],
            &[2],
            &[0, 1],
            &[1],
            left,
            threshold,
        )
        .unwrap();
        assert_eq!(partial.rank, 1);
        assert_eq!(partial.row_indices, vec![0]);
        assert_eq!(partial.col_indices, vec![1]);
        assert!(partial.pivot_errors[0] <= threshold);
        let empty =
            matrix_luci_factors_with_preferred_pivots(&a, &[2], &[2], &[], &[], left, threshold)
                .unwrap();
        assert_eq!(empty.row_indices, vec![2]);
        assert_eq!(empty.col_indices, vec![2]);
    }
    let re_im = |re, im| parts(re, if complex { im } else { 0.0 });
    let u = Matrix::from_col_major_vec(
        4,
        2,
        vec![
            re_im(1.0, 1.0),
            re_im(2.0, -1.0),
            re_im(3.0, 2.0),
            re_im(1.0, -3.0),
            re_im(2.0, 0.0),
            re_im(1.0, 0.5),
            re_im(-1.0, 0.0),
            re_im(2.0, 1.0),
        ],
    );
    let v = Matrix::from_col_major_vec(
        2,
        5,
        vec![
            re_im(1.0, 0.0),
            re_im(2.0, 1.0),
            re_im(2.0, -1.0),
            re_im(0.5, 0.0),
            re_im(1.0, 1.0),
            re_im(3.0, -2.0),
            re_im(4.0, 0.0),
            re_im(1.0, 1.0),
            re_im(-1.0, 0.2),
            re_im(2.0, 0.0),
        ],
    );
    let a = mat_mul(&u, &v).unwrap();
    for left in [false, true] {
        let restored = matrix_luci_factors_with_preferred_pivots(
            &a,
            &[0, 1],
            &[0, 1],
            &[2],
            &[3],
            left,
            rounding,
        )
        .unwrap();
        assert_eq!(restored.rank, 2);
        assert!(restored.row_indices.contains(&2));
        assert!(restored.col_indices.contains(&3));
        let result = mat_mul(&restored.left, &restored.right).unwrap();
        let error = a
            .as_col_major_slice()
            .iter()
            .zip(result.as_col_major_slice())
            .map(|(&x, &y)| Scalar::abs_val(x - y))
            .fold(0.0_f64, f64::max);
        assert!(error < rounding);
    }
}

#[test]
fn screen_restoration_checks_actual_residual_all_scalars_and_axes() {
    verify::<f64>(|re, _| re, false, 1e-10);
    verify::<f32>(|re, _| re as f32, false, 1e-3);
    verify::<Complex64>(Complex64::new, true, 1e-10);
    verify::<Complex32>(|re, im| Complex32::new(re as f32, im as f32), true, 1e-3);
}

#[test]
fn preferences_and_tolerance_are_checked_before_access() {
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
    for threshold in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &[], &[], true, threshold),
            Err(MatrixCIError::InvalidArgument { .. })
        ));
    }
    for (rows, cols) in [(vec![2], vec![]), (vec![], vec![2])] {
        assert!(matches!(
            matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &rows, &cols, true, 1.0),
            Err(MatrixCIError::IndexOutOfBounds { .. })
        ));
    }
    assert!(matches!(
        matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &[0, 0], &[], true, 1.0),
        Err(MatrixCIError::DuplicatePivotRow { .. })
    ));
    assert!(matches!(
        matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &[], &[0, 0], true, 1.0),
        Err(MatrixCIError::DuplicatePivotCol { .. })
    ));
    assert!(matches!(
        matrix_luci_factors_with_preferred_pivots(&a, &[], &[], &[], &[], true, 1.0),
        Err(MatrixCIError::InvalidArgument { .. })
    ));
    for (rows, cols) in [(vec![0], vec![]), (vec![], vec![0]), (vec![0, 1], vec![0])] {
        assert!(matches!(
            matrix_luci_factors_with_preferred_pivots(&a, &rows, &cols, &[], &[], true, 1.0),
            Err(MatrixCIError::InvalidArgument { .. })
        ));
    }
    let nonfinite = Matrix::from_col_major_vec(1, 1, vec![f64::NAN]);
    assert!(matches!(
        matrix_luci_factors_with_preferred_pivots(&nonfinite, &[0], &[0], &[], &[], true, 1.0),
        Err(MatrixCIError::NaNEncountered { .. })
    ));
}

#[test]
fn an_inadmissible_seed_is_returned_with_its_measured_error() {
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
    let factors =
        matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &[1], &[1], true, 0.5).unwrap();
    assert_eq!(factors.row_indices, vec![0]);
    assert_eq!(factors.col_indices, vec![0]);
    assert_eq!(factors.pivot_errors, vec![1.0]);
}

#[test]
fn singular_restorations_and_zero_coefficients_leave_the_seed_intact() {
    for tiny in [0.0, 1e-17] {
        let a = Matrix::from_col_major_vec(
            3,
            3,
            vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, tiny, 1.0],
        );
        let factors = matrix_luci_factors_with_preferred_pivots(
            &a,
            &[0, 1],
            &[0, 1],
            &[],
            &[0, 2],
            true,
            1e-12,
        )
        .unwrap();
        assert_eq!(factors.col_indices, vec![0, 1]);
        assert_eq!(factors.rank, 2);
        assert!(factors.pivot_errors[0] <= 1e-12);
    }
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
    let protected = matrix_luci_factors_with_preferred_pivots(
        &a,
        &[0, 1],
        &[0, 1],
        &[0, 1],
        &[0, 1],
        true,
        0.0,
    )
    .unwrap();
    assert_eq!(protected.pivot_errors, vec![0.0]);
    assert_eq!(protected.rank, 2);
}

#[test]
fn nonfinite_residual_predictions_are_rejected_without_changing_a_valid_seed() {
    let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 1e308, 1e-300, 0.0]);
    let seed =
        matrix_luci_factors_with_preferred_pivots(&a, &[0], &[0], &[], &[1], true, 2e8).unwrap();
    assert_eq!(seed.col_indices, vec![0]);
    assert!(seed.pivot_errors[0] <= 2e8);
    assert!((seed.pivot_errors[0] / 1e8 - 1.0).abs() < 1e-12);
}

#[test]
fn admissible_exchanges_prefer_the_largest_determinant_ratio_and_protect_preferences() {
    let a = Matrix::from_col_major_vec(
        3,
        4,
        vec![
            1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 0.0, 2.0,
        ],
    );
    for left in [false, true] {
        let largest =
            matrix_luci_factors_with_preferred_pivots(&a, &[0, 1], &[0, 1], &[], &[2], left, 1e-12)
                .unwrap();
        assert_eq!(largest.col_indices, vec![2, 1]);
        assert!(largest.pivot_errors[0] <= 1e-12);
        // Column 3 cannot replace unprotected column 1. Rejecting it must
        // leave the solved coefficients valid for the later column 2 trial;
        // protecting column 0 then overrides the largest-ratio position.
        let protected = matrix_luci_factors_with_preferred_pivots(
            &a,
            &[0, 1],
            &[0, 1],
            &[],
            &[0, 3, 2],
            left,
            1e-12,
        )
        .unwrap();
        assert_eq!(protected.col_indices, vec![0, 2]);
        assert_eq!(protected.rank, 2);
        assert!(protected.pivot_errors[0] <= 1e-12);
    }
}
