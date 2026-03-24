use super::*;
use crate::defaults::tensordynlen::TensorDynLen;
use crate::defaults::DynIndex;
/// Helper to create a 1D tensor (vector) with given data and shared index.
fn make_vector_with_index(data: Vec<f64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

fn scale_vector_f64(x: &TensorDynLen, diag: &[f64]) -> Result<TensorDynLen> {
    let x_data = x.to_vec_f64()?;
    let result_data: Vec<f64> = x_data
        .iter()
        .zip(diag.iter())
        .map(|(&xi, &di)| xi * di)
        .collect();
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

fn apply_matrix2_f64(x: &TensorDynLen, a_data: &[f64; 4]) -> Result<TensorDynLen> {
    let x_data = x.to_vec_f64()?;
    let result_data = vec![
        a_data[0] * x_data[0] + a_data[1] * x_data[1],
        a_data[2] * x_data[0] + a_data[3] * x_data[1],
    ];
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

#[test]
fn test_givens_rotation_real() {
    let a = AnyScalar::new_real(3.0);
    let b = AnyScalar::new_real(4.0);
    let (c, s) = compute_givens_rotation(&a, &b);

    // c = 3/5 = 0.6, s = 4/5 = 0.8
    assert!(!c.is_complex());
    assert!(!s.is_complex());
    assert!((c.real() - 0.6).abs() < 1e-10);
    assert!((s.real() - 0.8).abs() < 1e-10);
}

#[test]
fn test_apply_givens_rotation_real() {
    let c = AnyScalar::new_real(0.6);
    let s = AnyScalar::new_real(0.8);
    let x = AnyScalar::new_real(3.0);
    let y = AnyScalar::new_real(4.0);

    let (new_x, new_y) = apply_givens_rotation(&c, &s, &x, &y);

    // new_x = 0.6*3 + 0.8*4 = 1.8 + 3.2 = 5.0
    // new_y = -0.8*3 + 0.6*4 = -2.4 + 2.4 = 0.0
    assert!(!new_x.is_complex());
    assert!(!new_y.is_complex());
    assert!((new_x.real() - 5.0).abs() < 1e-10);
    assert!(new_y.real().abs() < 1e-10);
}

#[test]
fn test_givens_rotation_complex() {
    let a = AnyScalar::new_complex(3.0, 4.0);
    let b = AnyScalar::new_complex(1.0, -2.0);
    let (c, s) = compute_givens_rotation(&a, &b);

    assert!(c.is_complex());
    assert!(s.is_complex());
    assert_eq!(c.is_real(), s.is_real());

    // c*a + s*b should recover sqrt(|a|^2 + |b|^2) on the real axis.
    let rotated = c.clone() * a + s.clone() * b;
    assert!(rotated.is_complex());
    assert!(rotated.real().is_finite());
    assert!(rotated.imag().is_finite());
}

#[test]
fn test_apply_givens_rotation_complex() {
    let c = AnyScalar::new_complex(0.6, 0.1);
    let s = AnyScalar::new_complex(0.8, -0.2);
    let x = AnyScalar::new_complex(3.0, 1.0);
    let y = AnyScalar::new_complex(4.0, -2.0);

    let (new_x, new_y) = apply_givens_rotation(&c, &s, &x, &y);

    assert!(new_x.is_complex());
    assert!(new_y.is_complex());
    assert!(new_x.real().is_finite() && new_x.imag().is_finite());
    assert!(new_y.real().is_finite() && new_y.imag().is_finite());
}

#[test]
fn test_gmres_identity_operator() {
    // Solve A x = b where A = I (identity)
    // Solution: x = b
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    // Identity operator: A x = x
    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge for identity");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm() < 1e-10, "Solution should equal b");
}

#[test]
fn test_gmres_diagonal_matrix() {
    // Solve A x = b where A = diag(2, 3, 4)
    // b = [2, 6, 12] → x = [1, 2, 3]
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    // Diagonal scaling operator
    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm() < 1e-8,
        "Solution error too large: {}",
        diff.norm()
    );
}

#[test]
fn test_gmres_nonsymmetric_matrix() {
    // Solve A x = b where A is a 2x2 non-symmetric matrix
    // A = [[2, 1], [0, 3]]
    // A x = [2*1 + 1*2, 0*1 + 3*2] = [4, 6]
    // So b = [4, 6] → x = [1, 2]
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0], &idx);

    // Matrix A = [[2, 1], [0, 3]]
    let a_data = [2.0, 1.0, 0.0, 3.0];

    let apply_a = move |x: &TensorDynLen| apply_matrix2_f64(x, &a_data);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm() < 1e-8,
        "Solution error too large: {}",
        diff.norm()
    );
}

#[test]
fn test_gmres_with_good_initial_guess() {
    // If initial guess is already the solution, should converge immediately
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx); // Already the solution for A=diag(2,2,2)

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> {
        // A = 2*I
        x.scale(AnyScalar::new_real(2.0))
    };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 0, "Should converge with 0 iterations");
}

/// Helper to create a 1D complex tensor (vector) with given data and shared index.
fn make_vector_c64_with_index(data: Vec<num_complex::Complex64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

#[test]
fn test_gmres_identity_operator_c64() {
    // Solve A x = b where A = I (identity), b is complex
    // Solution: x = b
    use num_complex::Complex64;

    let idx = DynIndex::new_dyn(4);
    let b = make_vector_c64_with_index(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.5, 3.5),
        ],
        &idx,
    );
    let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);

    // Identity operator: A x = x
    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let err = diff.norm();

    assert!(result.converged, "GMRES should converge for identity (c64)");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );
    assert!(err < 1e-8, "Solution should equal b, error: {}", err);
}

#[test]
fn test_gmres_diagonal_c64() {
    // Solve A x = b where A = diag(2+i, 3-i, 1+2i, 4)
    use num_complex::Complex64;

    let idx = DynIndex::new_dyn(4);
    let diag = [
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(4.0, 0.0),
    ];
    let x_true = [
        Complex64::new(1.0, -1.0),
        Complex64::new(0.5, 2.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    let b_data: Vec<Complex64> = diag.iter().zip(x_true.iter()).map(|(d, x)| d * x).collect();

    let b = make_vector_c64_with_index(b_data, &idx);
    let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);
    let expected = make_vector_c64_with_index(x_true.to_vec(), &idx);

    let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
        let x_data = x.to_vec_c64()?;
        let result_data: Vec<Complex64> = x_data
            .iter()
            .zip(diag.iter())
            .map(|(&xi, &di)| di * xi)
            .collect();
        Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
    };

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    let err = diff.norm();

    assert!(result.converged, "GMRES should converge for diagonal (c64)");
    assert!(err < 1e-8, "Solution error too large: {}", err);
}

#[test]
fn test_gmres_zero_rhs() {
    // Solve A x = 0 → x = 0 (for any invertible A)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions::default();

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    // With zero RHS, should return initial guess immediately
    assert!(result.converged);
}

// ==========================================================================
// Tests for restart_gmres_with_truncation
// ==========================================================================

#[test]
fn test_restart_gmres_identity_operator() {
    // Solve A x = b where A = I (identity)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    // No-op truncation
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(
        result.converged,
        "Restart GMRES should converge for identity"
    );
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm() < 1e-10, "Solution should equal b");
}

#[test]
fn test_restart_gmres_diagonal_matrix() {
    // Solve A x = b where A = diag(2, 3, 4)
    // b = [2, 6, 12] → x = [1, 2, 3]
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-10,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: None,
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Restart GMRES should converge");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm() < 1e-8,
        "Solution error too large: {}",
        diff.norm()
    );
}

#[test]
fn test_restart_gmres_with_initial_guess() {
    // Solve A x = b with a good initial guess
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.9, 1.9, 2.9], &idx); // Close to [1, 2, 3]
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, Some(&x0), &options, truncate).unwrap();

    assert!(result.converged, "Should converge with good initial guess");

    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm() < 1e-8,
        "Solution error too large: {}",
        diff.norm()
    );
}

#[test]
fn test_restart_gmres_outer_iterations_tracked() {
    // Verify that outer_iterations is tracked correctly
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Use small inner_max_iter to encourage multiple outer iterations
    let options = RestartGmresOptions {
        max_outer_iters: 20,
        rtol: 1e-10,
        inner_max_iter: 2,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Should converge");
    // Verify iteration counts are reasonable
    assert!(
        result.iterations >= 1,
        "Should have at least 1 inner iteration"
    );
    // outer_iterations can be 0 if converged at first check
    assert!(
        result.outer_iterations <= options.max_outer_iters,
        "outer_iterations should not exceed max_outer_iters"
    );
}

#[test]
fn test_restart_gmres_zero_rhs() {
    // Solve A x = 0 → x = 0
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Should converge for zero RHS");
    assert_eq!(result.iterations, 0, "Should converge immediately");
    assert_eq!(result.outer_iterations, 0, "Should have 0 outer iterations");
}

#[test]
fn test_restart_gmres_options_builder() {
    let options = RestartGmresOptions::new()
        .with_max_outer_iters(30)
        .with_rtol(1e-12)
        .with_inner_max_iter(15)
        .with_inner_max_restarts(1)
        .with_min_reduction(0.95)
        .with_inner_rtol(0.01)
        .with_verbose(true);

    assert_eq!(options.max_outer_iters, 30);
    assert!((options.rtol - 1e-12).abs() < 1e-15);
    assert_eq!(options.inner_max_iter, 15);
    assert_eq!(options.inner_max_restarts, 1);
    assert_eq!(options.min_reduction, Some(0.95));
    assert_eq!(options.inner_rtol, Some(0.01));
    assert!(options.verbose);
}

#[test]
fn test_gmres_with_truncation_check_true_residual_safe() {
    // When check_true_residual is enabled and convergence is reported,
    // the residual_norm should reflect the checked residual (not the
    // potentially inaccurate Hessenberg estimate).
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation: convergence should work normally with check enabled
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: true,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();

    assert!(
        result.converged,
        "Should converge with true residual check and no-op truncation"
    );

    // Verify the reported residual is actually the checked residual
    let ax = apply_a(&result.solution).unwrap();
    let r = ax
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let true_rel_res = r.norm() / b.norm();
    assert!(
        true_rel_res < 1e-8,
        "True residual should be small: {}",
        true_rel_res
    );
}

#[test]
#[allow(clippy::needless_borrows_for_generic_args)]
fn test_gmres_with_truncation_check_true_residual_consistency() {
    // Test that when check_true_residual is enabled, the reported residual_norm
    // is consistent with the actual checked (truncated) residual.
    let idx = DynIndex::new_dyn(4);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0, 4.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0, 0.0], &idx);

    // A = diag(1, 2, 3, 4)
    let diag = [1.0, 2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation: the solver should converge normally
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Without check: converged with Hessenberg residual
    let options_no_check = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: false,
    };
    let result_no_check =
        gmres_with_truncation(&apply_a, &b, &x0, &options_no_check, &truncate).unwrap();

    // With check: converged with verified residual
    let options_check = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: true,
    };
    let result_check = gmres_with_truncation(&apply_a, &b, &x0, &options_check, &truncate).unwrap();

    // Both should converge for this simple problem
    assert!(result_no_check.converged, "No-check should converge");
    assert!(result_check.converged, "With-check should converge");

    // With check, the reported residual should be the true residual
    let ax = apply_a(&result_check.solution).unwrap();
    let r = ax
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let true_rel_res = r.norm() / b.norm();

    // The reported residual should match the true residual closely
    assert!(
        (result_check.residual_norm - true_rel_res).abs() < 1e-8,
        "Reported residual ({:.6e}) should match true residual ({:.6e})",
        result_check.residual_norm,
        true_rel_res
    );
}

// ==========================================================================
// Tests for uncovered code paths
// ==========================================================================

#[test]
fn test_gmres_verbose_output() {
    // Exercise verbose code paths in gmres (lines 151, 158-161, 242)
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);

    let a_data = [2.0, 1.0, 0.0, 3.0];
    let apply_a = move |x: &TensorDynLen| apply_matrix2_f64(x, &a_data);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 2,
        verbose: true,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    assert!(result.converged);
}

#[test]
fn test_gmres_no_convergence_exhausts_restarts() {
    // Force gmres to exhaust all restarts without converging.
    // Use max_iter=1 and max_restarts=1 with a harder problem so it can't converge.
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // Very few iterations and restarts, tight tolerance
    let options = GmresOptions {
        max_iter: 1,
        rtol: 1e-15,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    // With max_iter=1 and max_restarts=1, it may or may not converge,
    // but the end-of-restart and final-residual code paths should be hit.
    // The important thing is it runs without error.
    assert!(result.residual_norm >= 0.0);
}

#[test]
fn test_gmres_lucky_breakdown() {
    // For a 1D system, the Krylov subspace is exact after 1 iteration,
    // causing h_{j+1,j} = 0 (lucky breakdown).
    // A = [5], b = [10], x0 = [0] -> solution x = [2]
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![10.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { scale_vector_f64(x, &[5.0]) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    assert!(result.converged, "Should converge via lucky breakdown");

    let expected = make_vector_with_index(vec![2.0], &idx);
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(diff.norm() < 1e-10, "Solution should be [2.0]");
}

#[test]
fn test_gmres_with_truncation_zero_rhs() {
    // Exercise the zero-b early return in gmres_with_truncation (lines 345-350)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_gmres_with_truncation_already_converged() {
    // Initial guess already satisfies the equation, triggering early return (lines 375-380)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx); // Exact solution for diag(2,3,4)

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_gmres_with_truncation_verbose() {
    // Exercise verbose output paths in gmres_with_truncation
    // (lines 361, 368-371, 446-457, 463, 467, 503, 517, 522-525)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 2,
        verbose: true,
        check_true_residual: true,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
}

#[test]
fn test_restart_gmres_stagnation_detection() {
    // Test stagnation detection in restart_gmres_with_truncation.
    // Use a system where the exact solution has irrational-like components
    // that can't be represented after rounding, causing stagnation.
    let idx = DynIndex::new_dyn(3);
    // A = diag(3, 7, 11), b = [1, 1, 1] -> x = [1/3, 1/7, 1/11]
    // These solutions can't be exactly represented after rounding to 1 decimal place.
    let b = make_vector_with_index(vec![1.0, 1.0, 1.0], &idx);

    let diag = [3.0, 7.0, 11.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // Truncation that rounds to 1 decimal place - the exact solution [1/3, 1/7, 1/11]
    // gets rounded to [0.3, 0.1, 0.1], preventing convergence below the rounding error.
    let truncate = |x: &mut TensorDynLen| -> Result<()> {
        let data = x.to_vec_f64()?;
        let new_data: Vec<f64> = data.iter().map(|&v| (v * 10.0).round() / 10.0).collect();
        *x = TensorDynLen::from_dense(x.indices.clone(), new_data).unwrap();
        Ok(())
    };

    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-12,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: Some(0.999),
        inner_rtol: Some(0.1),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    // Stagnation should be detected and converged should be false
    assert!(!result.converged, "Should detect stagnation");
}

#[test]
fn test_restart_gmres_exhausts_outer_iters() {
    // Test the path where restart_gmres exhausts all outer iterations without converging.
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Very few iterations and extremely tight tolerance to prevent convergence
    let options = RestartGmresOptions {
        max_outer_iters: 1,
        rtol: 1e-100, // impossibly tight
        inner_max_iter: 1,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.5),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(!result.converged);
    assert_eq!(result.outer_iterations, options.max_outer_iters);
}

#[test]
fn test_restart_gmres_verbose() {
    // Exercise verbose output paths in restart_gmres_with_truncation
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-10,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: true,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(result.converged);
}

#[test]
fn test_restart_gmres_zero_rhs_with_x0() {
    // Cover the zero-b-with-x0 path in restart_gmres_with_truncation (line 779)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 5,
        rtol: 1e-10,
        inner_max_iter: 3,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: None,
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, Some(&x0), &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    // Solution should be x0 when b is zero
    assert!(result.solution.distance(&x0) < 1e-12);
}

#[test]
fn test_restart_gmres_stagnation_verbose() {
    // Cover the stagnation verbose output path (lines 847-851)
    let n = 10;
    let idx = DynIndex::new_dyn(n);
    let b_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let b = make_vector_with_index(b_data, &idx);

    let diag: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powf(i as f64 * 3.0 / (n as f64 - 1.0)))
        .collect();
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |x: &mut TensorDynLen| -> Result<()> {
        let data = x.to_vec_f64()?;
        let new_data: Vec<f64> = data.iter().map(|&v| (v * 100.0).round() / 100.0).collect();
        *x = TensorDynLen::from_dense(x.indices.clone(), new_data).unwrap();
        Ok(())
    };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-12,
        inner_max_iter: 3,
        inner_max_restarts: 0,
        min_reduction: Some(0.999),
        inner_rtol: Some(0.1),
        verbose: true, // Cover verbose stagnation output
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(!result.converged);
}
