use super::*;
use crate::{contract, TensorElement};

fn check_hpd_solution<S: TensorElement>(
    matrix: Vec<S>,
    rhs: Vec<S>,
    expected: Vec<S>,
    init: Vec<S>,
) {
    let index = DynIndex::new_dyn(2);
    let input = index.prime();
    let matrix = IdxTensor::from_dense(vec![index.clone(), input.clone()], matrix).unwrap();
    let rhs = IdxTensor::from_dense(vec![index.clone()], rhs).unwrap();
    let expected = IdxTensor::from_dense(vec![index.clone()], expected).unwrap();
    let init = IdxTensor::from_dense(vec![index.clone()], init).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    let result = cg(apply, &rhs, &init, &CgOptions::default()).unwrap();
    assert!(result.converged);
    assert!(result.iterations <= 2);
    assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);
    let residual = rhs
        .sub(&apply(&result.solution).unwrap())
        .unwrap()
        .norm()
        .unwrap();
    assert!(residual < 1e-10);
    assert!((residual - result.residual_norm).abs() < 1e-14);
}

#[test]
fn cg_real_spd_known_solution() {
    check_hpd_solution(
        vec![4.0, 1.0, 1.0, 3.0],
        vec![6.0, 7.0],
        vec![1.0, 2.0],
        vec![0.0, 0.0],
    );
}

#[test]
fn cg_complex_hpd_known_solution_from_nonzero_guess() {
    let c = Complex64::new;
    // Column-major Hermitian matrix [[4, 1+i], [1-i, 3]].
    check_hpd_solution(
        vec![c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)],
        vec![c(7.0, 5.0), c(8.0, -3.0)],
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        vec![c(0.5, 0.0), c(0.0, -1.0)],
    );
}

#[test]
fn cg_single_precision_real_and_complex_hpd() {
    check_hpd_solution(
        vec![2.0_f32, 0.0, 0.0, 2.0],
        vec![2.0_f32, 4.0],
        vec![1.0_f32, 2.0],
        vec![0.0_f32, 0.0],
    );
    let c = num_complex::Complex32::new;
    check_hpd_solution(
        vec![c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)],
        vec![c(2.0, 2.0), c(4.0, -2.0)],
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        vec![c(0.0, 0.0), c(0.0, 0.0)],
    );
}

#[test]
fn cg_handles_extreme_f64_scales_without_false_convergence_or_breakdown() {
    let index = DynIndex::new_dyn(1);
    let initial = IdxTensor::zeros::<f64>(vec![index.clone()]).unwrap();
    for (factor, rhs_value, expected) in [
        (1.0, f64::from_bits(1), f64::from_bits(1)),
        (1.0, 1e-200, 1e-200),
        (1.0, 1e200, 1e200),
        (1.0, 1e308, 1e308),
        (1e20, 1e150, 1e130),
        (1e-200, 1e-200, 1.0),
        (1e308, 1.0, 1e-308),
    ] {
        let rhs = make_vector_with_index(vec![rhs_value], &index);
        let apply =
            |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.scale(AnyScalar::new_real(factor))?) };
        let result = cg(apply, &rhs, &initial, &CgOptions::default()).unwrap();
        assert!(result.converged);
        let actual = result.solution.to_vec::<f64>().unwrap()[0];
        assert!(
            (actual / expected - 1.0).abs() < 1e-12,
            "factor={factor:e}, rhs={rhs_value:e}, x={actual:e}"
        );
        assert!(((factor * actual - rhs_value) / rhs_value).abs() <= 1e-10);
        assert!(result.residual_norm / rhs_value <= 1e-10);
    }
}

#[test]
fn cg_handles_extreme_single_precision_real_and_complex_scales() {
    use num_complex::Complex32;
    let options = CgOptions {
        rtol: 1e-5,
        hermitian_tol: 1e-5,
        ..CgOptions::default()
    };
    let index = DynIndex::new_dyn(1);
    for scale in [f32::from_bits(1), 1e-30_f32, 1e20] {
        let real = IdxTensor::from_dense(vec![index.clone()], vec![scale]).unwrap();
        let complex = IdxTensor::from_dense(
            vec![index.clone()],
            vec![Complex32::new(3.0 * scale, 4.0 * scale)],
        )
        .unwrap();
        for rhs in [real, complex] {
            let initial = rhs.scale(AnyScalar::new_real(0.0)).unwrap();
            let identity = |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.clone()) };
            let result = cg(identity, &rhs, &initial, &options).unwrap();
            assert!(result.converged);
            let error =
                result.solution.sub(&rhs).unwrap().maxabs().unwrap() / rhs.maxabs().unwrap();
            assert!(error <= 1e-5, "relative solution error={error:e}");
            assert!(result.residual_norm / rhs.norm().unwrap() <= options.rtol);
        }
    }
}

fn check_scaled_hpd<S: TensorElement>(
    matrix: Vec<S>,
    solution: Vec<S>,
    scales: &[f64],
    options: CgOptions,
) {
    let index = DynIndex::new_dyn(2);
    let input = index.prime();
    let matrix = IdxTensor::from_dense(vec![index.clone(), input.clone()], matrix).unwrap();
    let solution = IdxTensor::from_dense(vec![index.clone()], solution).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    for &scale in scales {
        let expected = solution.scale(AnyScalar::new_real(scale)).unwrap();
        let rhs = apply(&expected).unwrap();
        let initial = expected.scale(AnyScalar::new_real(0.0)).unwrap();
        let result = cg(apply, &rhs, &initial, &options).unwrap();
        assert!(result.converged);
        let error =
            result.solution.sub(&expected).unwrap().maxabs().unwrap() / expected.maxabs().unwrap();
        assert!(
            error <= options.rtol,
            "scale={scale:e}, solution error={error:e}"
        );
        let residual = rhs
            .sub(&apply(&result.solution).unwrap())
            .unwrap()
            .norm()
            .unwrap();
        assert!(residual / rhs.norm().unwrap() <= options.rtol);
        assert_eq!(result.residual_norm, residual);
    }
}

#[test]
fn cg_scaled_nondiagonal_hpd_recurrences_cover_all_scalar_types() {
    let real = vec![4.0, 1.0, 1.0, 3.0];
    check_scaled_hpd(
        real.clone(),
        vec![1.0, 2.0],
        &[1e-200, 1.0, 1e150],
        CgOptions::default(),
    );
    let c = Complex64::new;
    let complex = vec![c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    check_scaled_hpd(
        complex.clone(),
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        &[1e-200, 1.0, 1e150],
        CgOptions::default(),
    );
    let options = CgOptions {
        rtol: 1e-5,
        hermitian_tol: 1e-5,
        ..CgOptions::default()
    };
    check_scaled_hpd(
        real.iter().map(|&x| x as f32).collect(),
        vec![1.0_f32, 2.0],
        &[1e-30, 1.0, 1e20],
        options.clone(),
    );
    let c = num_complex::Complex32::new;
    check_scaled_hpd(
        complex
            .iter()
            .map(|x| c(x.re as f32, x.im as f32))
            .collect(),
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        &[1e-30, 1.0, 1e20],
        options,
    );
}

#[test]
fn cg_accepts_hpd_curvature_with_cancellation_roundoff() {
    let index = DynIndex::new_dyn(2);
    let input = index.prime();
    let c = Complex64::new;
    // Hermitian with positive leading entry and determinant about 1.3e-7.
    let matrix = IdxTensor::from_dense(
        vec![index.clone(), input.clone()],
        vec![c(1.3, 0.0), c(0.7, -0.9), c(0.7, 0.9), c(1.0000001, 0.0)],
    )
    .unwrap();
    let rhs = make_vector_c64(vec![c(-0.7 / 1.3, -0.9 / 1.3), c(1.0, 0.0)], &index);
    let initial = rhs.scale(AnyScalar::new_real(0.0)).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    let options = CgOptions {
        rtol: 1e-6,
        ..CgOptions::default()
    };
    let result = cg(apply, &rhs, &initial, &options).unwrap();
    assert!(result.converged);
    let actual_residual = rhs
        .sub(&apply(&result.solution).unwrap())
        .unwrap()
        .norm()
        .unwrap();
    assert!(actual_residual <= options.rtol * rhs.norm().unwrap());
    assert_eq!(result.residual_norm, actual_residual);
}

fn diagonal(x: &PlainVector) -> Result<PlainVector> {
    Ok(PlainVector {
        data: vec![2.0 * x.data[0], 3.0 * x.data[1]],
    })
}

#[test]
fn cg_uses_only_vector_space_operations() {
    let rhs = PlainVector {
        data: vec![2.0, 6.0],
    };
    let init = PlainVector {
        data: vec![0.0, 0.0],
    };
    let result = cg(diagonal, &rhs, &init, &CgOptions::default()).unwrap();
    assert!(result.converged);
    assert!((result.solution.data[0] - 1.0).abs() < 1e-12);
    assert!((result.solution.data[1] - 2.0).abs() < 1e-12);
}

#[test]
fn cg_zero_rhs_and_exact_initial_guesses() {
    for data in [vec![0.0, 0.0], vec![1.0, -2.0]] {
        let initial = PlainVector { data };
        let rhs = diagonal(&initial).unwrap();
        let result = cg(diagonal, &rhs, &initial, &CgOptions::default()).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.residual_norm, 0.0);
        assert_eq!(result.solution.data, initial.data);
    }
}

#[test]
fn cg_zero_rhs_with_nonzero_initial_guess() {
    let rhs = PlainVector {
        data: vec![0.0, 0.0],
    };
    let initial = PlainVector {
        data: vec![1.0, -2.0],
    };
    let options = CgOptions {
        atol: 1e-12,
        ..CgOptions::default()
    };
    let result = cg(diagonal, &rhs, &initial, &options).unwrap();
    assert!(result.converged);
    assert!(result.iterations > 0);
    assert!(result.solution.maxabs().unwrap() < 1e-12);
    assert!(diagonal(&result.solution).unwrap().norm().unwrap() <= options.atol);
}

#[test]
fn cg_iteration_budget_returns_true_residual() {
    let rhs = PlainVector {
        data: vec![2.0, 6.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for budget in [0, 1] {
        let options = CgOptions {
            max_iter: budget,
            ..CgOptions::default()
        };
        let result = cg(diagonal, &rhs, &initial, &options).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, budget);
        let true_residual = rhs
            .sub(&diagonal(&result.solution).unwrap())
            .unwrap()
            .norm()
            .unwrap();
        assert!((result.residual_norm - true_residual).abs() < 1e-14);
    }
}

#[test]
fn cg_absolute_and_relative_tolerances() {
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for scale in [1e-8, 1.0, 1e8] {
        let rhs = PlainVector {
            data: vec![2.0 * scale, 6.0 * scale],
        };
        let result = cg(diagonal, &rhs, &initial, &CgOptions::default()).unwrap();
        assert!(result.converged);
        assert!(result.residual_norm <= 1e-10 * rhs.norm().unwrap());
        assert!((result.solution.data[0] / scale - 1.0).abs() < 1e-12);
        assert!((result.solution.data[1] / scale - 2.0).abs() < 1e-12);
    }
    let rhs = PlainVector {
        data: vec![0.001, 0.0],
    };
    let options = CgOptions {
        rtol: 0.0,
        atol: 0.001,
        ..CgOptions::default()
    };
    let result = cg(diagonal, &rhs, &initial, &options).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, options.atol);
}

#[test]
fn cg_rejects_invalid_options_before_applying_operator() {
    let vector = PlainVector {
        data: vec![1.0, 2.0],
    };
    let apply =
        |_: &PlainVector| -> Result<PlainVector> { panic!("invalid options reached operator") };
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for options in [
            CgOptions {
                rtol: invalid,
                ..CgOptions::default()
            },
            CgOptions {
                atol: invalid,
                ..CgOptions::default()
            },
            CgOptions {
                hermitian_tol: invalid,
                ..CgOptions::default()
            },
        ] {
            assert!(matches!(
                cg(apply, &vector, &vector, &options),
                Err(KrylovError::InvalidOptions { solver: "cg", .. })
            ));
        }
    }
}

#[test]
fn cg_rejects_nonpositive_curvature() {
    let rhs = PlainVector {
        data: vec![1.0, 2.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for factor in [0.0, -1.0] {
        let apply =
            |x: &PlainVector| -> Result<PlainVector> { Ok(x.scale(AnyScalar::new_real(factor))?) };
        assert!(matches!(
            cg(apply, &rhs, &initial, &CgOptions::default()),
            Err(KrylovError::NonPositiveCurvature { iteration: 1, .. })
        ));
    }
}

#[test]
fn cg_rejects_observable_nonhermitian_curvature() {
    let index = DynIndex::new_dyn(1);
    let rhs = make_vector_c64(vec![Complex64::new(1.0, 0.0)], &index);
    let initial = make_vector_c64(vec![Complex64::new(0.0, 0.0)], &index);
    let apply =
        |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.scale(AnyScalar::new_complex(1.0, 1.0))?) };
    assert!(matches!(
        cg(apply, &rhs, &initial, &CgOptions::default()),
        Err(KrylovError::NonHermitian { solver: "cg", .. })
    ));
}

#[test]
fn cg_preserves_operator_error_source() {
    let vector = PlainVector {
        data: vec![1.0, 2.0],
    };
    let apply = |_: &PlainVector| -> Result<PlainVector> {
        Err(std::io::Error::other("CG callback failed").into())
    };
    let error = cg(apply, &vector, &vector, &CgOptions::default()).unwrap_err();
    match error {
        KrylovError::Operation {
            solver: "cg",
            source,
        } => assert!(source.downcast_ref::<std::io::Error>().is_some()),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn cg_zero_rhs_still_checks_index_compatibility() {
    let index = DynIndex::new_dyn(2);
    let rhs = make_vector_with_index(vec![0.0, 0.0], &index);
    let identity = |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.clone()) };
    for other in [DynIndex::new_dyn(2), index.prime()] {
        let initial = make_vector_with_index(vec![0.0, 0.0], &other);
        assert!(matches!(
            cg(identity, &rhs, &initial, &CgOptions::default()),
            Err(KrylovError::Operation { .. })
        ));
    }
}

#[test]
fn cg_rechecks_residual_and_restarts_after_roundoff() {
    // After three CG steps, recursive residual < 5e-17 but true residual > 5e-17.
    let rhs = PlainVector {
        data: vec![1.0, 1.0, 1.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0, 0.0],
    };
    for budget in [3, 4] {
        let calls = AtomicUsize::new(0);
        let apply = |x: &PlainVector| -> Result<PlainVector> {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(PlainVector {
                data: vec![x.data[0], 2.0 * x.data[1], 3.0 * x.data[2]],
            })
        };
        let options = CgOptions {
            max_iter: budget,
            rtol: 0.0,
            atol: 5e-17,
            ..CgOptions::default()
        };
        let result = cg(apply, &rhs, &initial, &options).unwrap();
        assert_eq!(result.iterations, budget);
        assert_eq!(result.converged, budget == 4);
        let true_residual = rhs
            .sub(&apply(&result.solution).unwrap())
            .unwrap()
            .norm()
            .unwrap();
        assert_eq!(result.residual_norm, true_residual);
        // Initialization, each step, checks after step 3 (and 4), and test oracle.
        assert_eq!(
            calls.load(Ordering::Relaxed),
            if budget == 3 { 6 } else { 8 }
        );
        if result.converged {
            assert!(true_residual <= options.atol);
            assert_eq!(result.solution.data, vec![1.0, 0.5, 1.0 / 3.0]);
        }
    }
}

#[test]
fn cg_rejects_overflowing_threshold_before_operator_call() {
    let rhs = PlainVector {
        data: vec![2.0, 0.0],
    };
    let options = CgOptions {
        rtol: f64::MAX,
        ..CgOptions::default()
    };
    let apply = |_: &PlainVector| -> Result<PlainVector> {
        panic!("overflowing threshold reached operator")
    };
    assert!(matches!(
        cg(apply, &rhs, &rhs, &options),
        Err(KrylovError::InvalidOptions { solver: "cg", .. })
    ));
}

#[test]
fn cg_reports_nonfinite_directional_arithmetic() {
    let rhs = PlainVector {
        data: vec![1.0, 1.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let calls = AtomicUsize::new(0);
    let apply = |x: &PlainVector| -> Result<PlainVector> {
        if calls.fetch_add(1, Ordering::Relaxed) == 0 {
            Ok(x.clone())
        } else {
            Ok(PlainVector {
                data: vec![f64::MAX, f64::MAX],
            })
        }
    };
    assert!(matches!(
        cg(apply, &rhs, &initial, &CgOptions::default()),
        Err(KrylovError::NumericalBreakdown { iteration: 1, .. })
    ));
}

#[test]
fn cg_reports_operator_errors_during_steps_and_residual_verification() {
    let rhs = PlainVector {
        data: vec![1.0, 1.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for fail_at in [1, 2] {
        let calls = AtomicUsize::new(0);
        let apply = |x: &PlainVector| -> Result<PlainVector> {
            if calls.fetch_add(1, Ordering::Relaxed) == fail_at {
                Err(std::io::Error::other("CG operator failed").into())
            } else {
                Ok(x.clone())
            }
        };
        assert!(matches!(
            cg(apply, &rhs, &initial, &CgOptions::default()),
            Err(KrylovError::Operation { solver: "cg", .. })
        ));
    }
}

#[test]
fn cg_zero_tolerances_allow_exact_convergence() {
    let rhs = PlainVector {
        data: vec![1.0, 2.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let options = CgOptions {
        rtol: 0.0,
        atol: 0.0,
        ..CgOptions::default()
    };
    let result = cg(|x: &PlainVector| Ok(x.clone()), &rhs, &initial, &options).unwrap();
    assert!(result.converged);
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.solution.data, rhs.data);
}

#[test]
fn cg_rejects_nonfinite_inputs_and_operator_outputs() {
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for value in [f64::NAN, f64::INFINITY] {
        let rhs = PlainVector {
            data: vec![value, 1.0],
        };
        assert!(matches!(
            cg(diagonal, &rhs, &initial, &CgOptions::default()),
            Err(KrylovError::NumericalBreakdown { .. })
        ));
        let rhs = PlainVector {
            data: vec![1.0, 1.0],
        };
        let apply = |_: &PlainVector| -> Result<PlainVector> {
            Ok(PlainVector {
                data: vec![value, 0.0],
            })
        };
        assert!(matches!(
            cg(apply, &rhs, &initial, &CgOptions::default()),
            Err(KrylovError::NumericalBreakdown { .. })
        ));
    }
}
