use super::*;
use crate::{contract, TensorElement};
use std::sync::Arc;

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
    let calls = AtomicUsize::new(0);
    let counted = |x: &PlainVector| {
        calls.fetch_add(1, Ordering::Relaxed);
        diagonal(x)
    };
    let result = cg(counted, &rhs, &initial, &CgOptions::default()).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(result.solution.data, rhs.data);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn cg_wraps_krylov_errors_from_the_operator_callback() {
    let rhs = PlainVector {
        data: vec![1.0, 2.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for fail_at in [0, 1] {
        let calls = AtomicUsize::new(0);
        let apply = |x: &PlainVector| -> Result<PlainVector> {
            if calls.fetch_add(1, Ordering::Relaxed) == fail_at {
                return Err(KrylovError::NonPositiveCurvature {
                    solver: "cg",
                    iteration: 7,
                    curvature: -1.0,
                }
                .into());
            }
            diagonal(x)
        };
        match cg(apply, &rhs, &initial, &CgOptions::default()).unwrap_err() {
            KrylovError::Operation {
                solver: "cg",
                source,
            } => assert!(matches!(
                source.downcast_ref::<KrylovError>(),
                Some(KrylovError::NonPositiveCurvature { iteration: 7, .. })
            )),
            other => panic!("callback error was not wrapped: {other:?}"),
        }
    }
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

/// Vector-space operation that a [`FaultyVector`] can be told to reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailingOp {
    Validate,
    Scale,
    Axpby,
    InnerProduct,
}

/// Way in which a [`FaultyVector`] misbehaves.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fault {
    /// Reject the `at`-th call of `op` with a backend error.
    FailAt { op: FailingOp, at: usize },
    /// Return `value` from every self-inner-product instead of `||x||^2`.
    SelfProduct(f64),
    /// Drop the update from the `from`-th `axpby` call onwards, the way a
    /// backend that truncates its result to zero would.
    DropUpdatesFrom { from: usize },
}

/// Per-operation call counters, shared through every clone of a
/// [`FaultyVector`].
#[derive(Debug, Default)]
struct CallCounts {
    validate: AtomicUsize,
    scale: AtomicUsize,
    axpby: AtomicUsize,
    inner_product: AtomicUsize,
}

impl CallCounts {
    fn op(&self, op: FailingOp) -> &AtomicUsize {
        match op {
            FailingOp::Validate => &self.validate,
            FailingOp::Scale => &self.scale,
            FailingOp::Axpby => &self.axpby,
            FailingOp::InnerProduct => &self.inner_product,
        }
    }
}

/// Vector-space double that misbehaves in one configured way.
///
/// The operator callback clones its input, so one shared counter follows the
/// residual through every recurrence stage. Walking the rejection point over
/// the call count of a clean run therefore fails each stage in turn without
/// hard-coding the solver's call order.
#[derive(Debug, Clone)]
struct FaultyVector {
    data: Vec<f64>,
    fault: Fault,
    calls: Arc<CallCounts>,
}

impl FaultyVector {
    fn new(data: Vec<f64>, fault: Fault) -> Self {
        Self {
            data,
            fault,
            calls: Arc::new(CallCounts::default()),
        }
    }

    /// Same fault and counters, different payload: what the operator callback
    /// returns so the next stage is still armed.
    fn derived(&self, data: Vec<f64>) -> Self {
        Self {
            data,
            ..self.clone()
        }
    }

    /// Number of calls made so far to `op`.
    fn count(&self, op: FailingOp) -> usize {
        self.calls.op(op).load(Ordering::Relaxed)
    }

    /// Count a call of `op`, rejecting it when it is the configured failure.
    fn check(&self, op: FailingOp) -> std::result::Result<usize, TensorVectorSpaceError> {
        let call = self.calls.op(op).fetch_add(1, Ordering::Relaxed);
        match self.fault {
            Fault::FailAt { op: failed, at } if failed == op && at == call => Err(
                TensorVectorSpaceError::from(anyhow::anyhow!("injected {op:?} failure")),
            ),
            _ => Ok(call),
        }
    }
}

impl TensorIndex for FaultyVector {
    type Index = DynIndex;
    type Error = TensorVectorSpaceError;

    fn external_indices(&self) -> Vec<Self::Index> {
        Vec::new()
    }

    fn replaceind(
        &self,
        _old_index: &Self::Index,
        _new_index: &Self::Index,
    ) -> std::result::Result<Self, Self::Error> {
        Ok(self.clone())
    }

    fn replace_indices(
        &self,
        _old_indices: &[Self::Index],
        _new_indices: &[Self::Index],
    ) -> std::result::Result<Self, Self::Error> {
        Ok(self.clone())
    }
}

impl TensorVectorSpace for FaultyVector {
    fn validate(&self) -> std::result::Result<(), Self::Error> {
        self.check(FailingOp::Validate).map(|_| ())
    }

    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.data.iter().map(|x| x * x).sum())
    }

    fn axpby(
        &self,
        a: AnyScalar,
        other: &Self,
        b: AnyScalar,
    ) -> std::result::Result<Self, Self::Error> {
        let call = self.check(FailingOp::Axpby)?;
        if let Fault::DropUpdatesFrom { from } = self.fault {
            if call >= from {
                return Ok(self.derived(vec![0.0; self.data.len()]));
            }
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| a.real() * x + b.real() * y)
            .collect();
        Ok(self.derived(data))
    }

    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error> {
        self.check(FailingOp::Scale)?;
        Ok(self.derived(self.data.iter().map(|&x| scalar.real() * x).collect()))
    }

    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error> {
        self.check(FailingOp::InnerProduct)?;
        if let (Fault::SelfProduct(value), true) = (self.fault, std::ptr::eq(self, other)) {
            return Ok(AnyScalar::new_real(value));
        }
        Ok(AnyScalar::new_real(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&x, &y)| x * y)
                .sum(),
        ))
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.data.iter().map(|x| x.abs()).fold(0.0, f64::max))
    }
}

/// A solve whose recurrence stages exercise a distinct scaling branch.
struct Scenario {
    name: &'static str,
    rhs: Vec<f64>,
    diagonal: Vec<f64>,
    options: CgOptions,
}

impl Scenario {
    /// Solve `diag(diagonal) x = rhs` with misbehaving vectors and return the
    /// result with the per-operation call counters, so a caller can walk every
    /// rejection point of this scenario.
    fn run(
        &self,
        fault: Fault,
    ) -> (
        std::result::Result<CgResult<FaultyVector>, KrylovError>,
        Arc<CallCounts>,
    ) {
        let rhs = FaultyVector::new(self.rhs.clone(), fault);
        let calls = Arc::clone(&rhs.calls);
        let initial = rhs.derived(vec![0.0; self.rhs.len()]);
        let apply = |x: &FaultyVector| -> Result<FaultyVector> {
            Ok(x.derived(
                x.data
                    .iter()
                    .zip(self.diagonal.iter())
                    .map(|(&value, &scale)| scale * value)
                    .collect(),
            ))
        };
        (cg(apply, &rhs, &initial, &self.options), calls)
    }
}

/// Scenarios that reach every recurrence stage: a two-step solve whose
/// residual norm is at least one, a sub-unit residual that makes the
/// power-of-two rescaling non-trivial, the zero-right-hand-side shortcut, and
/// the true-residual restart after an optimistic recursive residual.
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "two-step",
            rhs: vec![2.0, 6.0],
            diagonal: vec![2.0, 3.0],
            options: CgOptions::default(),
        },
        Scenario {
            name: "sub-unit residual",
            rhs: vec![0.2, 0.6],
            diagonal: vec![2.0, 3.0],
            options: CgOptions::default(),
        },
        Scenario {
            name: "zero right-hand side",
            rhs: vec![0.0, 0.0],
            diagonal: vec![2.0, 3.0],
            options: CgOptions::default(),
        },
        Scenario {
            name: "true-residual restart",
            rhs: vec![1.0, 1.0, 1.0],
            diagonal: vec![1.0, 2.0, 3.0],
            options: CgOptions {
                max_iter: 4,
                rtol: 0.0,
                atol: 5e-17,
                ..CgOptions::default()
            },
        },
    ]
}

#[test]
fn cg_forwards_every_vector_space_failure_as_an_operation_error() {
    for scenario in scenarios() {
        for op in [
            FailingOp::Validate,
            FailingOp::Scale,
            FailingOp::Axpby,
            FailingOp::InnerProduct,
        ] {
            let clean_fault = Fault::FailAt { op, at: usize::MAX };
            let (clean, calls) = scenario.run(clean_fault);
            let clean = clean.unwrap_or_else(|error| {
                panic!("clean run failed for {} / {op:?}: {error}", scenario.name)
            });
            if scenario.rhs == [0.0, 0.0] {
                assert!(clean.converged);
                assert_eq!(clean.iterations, 0);
            } else {
                assert!(clean.iterations > 0, "{} stopped early", scenario.name);
            }
            let total = calls.op(op).load(Ordering::Relaxed);
            // The zero right-hand side returns after its initial residual, so
            // it reaches only the stages before the iteration begins.
            if total == 0 {
                assert_eq!(scenario.name, "zero right-hand side");
                continue;
            }
            for at in 0..total {
                let (result, _) = scenario.run(Fault::FailAt { op, at });
                match result {
                    Err(KrylovError::Operation {
                        solver: "cg",
                        source,
                    }) => assert!(
                        format!("{source:#}").contains("injected"),
                        "{} / {op:?} call {at} lost the backend error: {source:#}",
                        scenario.name
                    ),
                    other => panic!(
                        "{} / {op:?} call {at} did not surface as an operation failure: {other:?} \
                         (clean run made {total} calls)",
                        scenario.name
                    ),
                }
            }
        }
    }
}

#[test]
fn cg_rejects_degenerate_residual_inner_products() {
    // A backend whose self-inner-product is non-positive or nonfinite cannot
    // define a CG recurrence; report it instead of iterating on garbage.
    let scenario = Scenario {
        name: "two-step",
        rhs: vec![2.0, 6.0],
        diagonal: vec![2.0, 3.0],
        options: CgOptions::default(),
    };
    for value in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
        let (result, _) = scenario.run(Fault::SelfProduct(value));
        match result {
            Err(KrylovError::NumericalBreakdown {
                solver: "cg",
                iteration: 0,
                quantity: "residual inner product",
            }) => {}
            other => panic!("self-product {value} was not rejected: {other:?}"),
        }
    }
}

#[test]
fn cg_reports_a_vanishing_search_direction() {
    // A backend that drops the update leaves the conjugacy recurrence with a
    // zero direction. CG must report it at the next step instead of dividing
    // by a zero curvature or looping on a stale residual.
    let rhs = FaultyVector::new(vec![2.0, 6.0], Fault::DropUpdatesFrom { from: 3 });
    let initial = rhs.derived(vec![0.0, 0.0]);
    let apply = |x: &FaultyVector| -> Result<FaultyVector> {
        Ok(x.derived(vec![2.0 * x.data[0], 3.0 * x.data[1]]))
    };
    match cg(apply, &rhs, &initial, &CgOptions::default()) {
        Err(KrylovError::NumericalBreakdown {
            solver: "cg",
            iteration: 2,
            quantity: "search direction norm",
        }) => {}
        other => panic!("vanishing direction was not reported: {other:?}"),
    }
    assert_eq!(
        rhs.count(FailingOp::Axpby),
        4,
        "the vanished direction must be reported before another update"
    );
}

#[test]
fn cg_reports_an_unrepresentable_physical_residual() {
    // A rotational (non-normal) operator keeps the quadratic form positive, so
    // the Hermitian and curvature checks accept it, but CG diverges on it. At
    // unit scale the divergence is merely non-convergence; scaling the right-
    // hand side to the top of the f64 range pushes the physical residual past
    // f64::MAX, which the solver must report rather than return as a norm.
    let index = DynIndex::new_dyn(2);
    let input = index.prime();
    let omega = 1e6;
    let matrix = IdxTensor::from_dense(
        vec![index.clone(), input.clone()],
        vec![1.0, omega, -omega, 1.0],
    )
    .unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    let solve = |magnitude: f64| {
        let rhs = IdxTensor::from_dense(vec![index.clone()], vec![magnitude, magnitude]).unwrap();
        let initial = IdxTensor::zeros::<f64>(vec![index.clone()]).unwrap();
        cg(apply, &rhs, &initial, &CgOptions::default())
    };
    let moderate = solve(1.0).expect("a representable residual must not error");
    assert!(!moderate.converged);
    assert!(moderate.residual_norm.is_finite());
    assert!(moderate.residual_norm > 1.0, "CG diverges on this operator");
    match solve(1e305) {
        Err(KrylovError::NumericalBreakdown {
            solver: "cg",
            iteration: 1,
            quantity: "physical residual norm",
        }) => {}
        other => panic!("unrepresentable residual was not reported: {other:?}"),
    }
}

#[test]
fn cg_reports_the_true_residual_when_the_iteration_budget_runs_out() {
    // On a system spanning twelve decades the recursive residual drifts from
    // the true one. Stopping before either meets the tolerance must still
    // report `||b - A x||`, not the recurrence's estimate.
    let diagonal: Vec<f64> = (0..4)
        .map(|site| 1e12_f64.powf(-(site as f64) / 3.0))
        .collect();
    let spectrum = diagonal.clone();
    let apply = move |x: &PlainVector| -> Result<PlainVector> {
        Ok(PlainVector {
            data: x
                .data
                .iter()
                .zip(spectrum.iter())
                .map(|(&value, &scale)| scale * value)
                .collect(),
        })
    };
    let rhs = PlainVector {
        data: vec![1.0; diagonal.len()],
    };
    let initial = PlainVector {
        data: vec![0.0; diagonal.len()],
    };
    let options = CgOptions {
        max_iter: 5,
        rtol: 1e-14,
        ..CgOptions::default()
    };
    let result = cg(&apply, &rhs, &initial, &options).unwrap();
    assert!(!result.converged);
    assert_eq!(result.iterations, options.max_iter);
    let true_residual = rhs
        .sub(&apply(&result.solution).unwrap())
        .unwrap()
        .norm()
        .unwrap();
    let drift = (result.residual_norm - true_residual).abs() / true_residual;
    assert!(
        drift <= 1e-13,
        "reported {:e} but b - A x has norm {true_residual:e} (relative drift {drift:e})",
        result.residual_norm
    );
}

#[test]
fn cg_static_options_validation() {
    CgOptions {
        max_iter: 0,
        rtol: 0.0,
        atol: 0.0,
        hermitian_tol: 0.0,
    }
    .validate()
    .unwrap();
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
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
                options.validate(),
                Err(KrylovError::InvalidOptions { solver: "cg", .. })
            ));
        }
    }
}
