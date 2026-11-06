use super::*;
use crate::{contract, TensorElement};

fn check_tiny_shadow<S: TensorElement + From<f64>>(epsilon: f64) {
    let i = DynIndex::new_dyn(3);
    let input = i.prime();
    let a = IdxTensor::from_dense(
        vec![i.clone(), input.clone()],
        [1.0, epsilon, 1.0, 1.0, 1.5, -1.0, 0.0, 1.0, 1.5]
            .map(S::from)
            .to_vec(),
    )
    .unwrap();
    let b = IdxTensor::from_dense(vec![i.clone()], [1.0, 0.0, 0.0].map(S::from).to_vec()).unwrap();
    let expected = IdxTensor::from_dense(
        vec![i.clone()],
        [13.0 / 17.0, 4.0 / 17.0, -6.0 / 17.0].map(S::from).to_vec(),
    )
    .unwrap();
    let initial = b.scale(0.0.into()).unwrap();
    let apply =
        |x: &IdxTensor| -> Result<IdxTensor> { Ok(contract(&[&a, &x.replaceind(&i, &input)?])?) };
    let result = bicgstab(apply, &b, &initial, &BiCgStabOptions::default()).unwrap();
    assert!(result.converged);
    assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);
    let residual = b
        .sub(&apply(&result.solution).unwrap())
        .unwrap()
        .norm()
        .unwrap();
    assert_eq!(result.residual_norm, residual);
    assert!(residual <= 1e-10);
}

#[test]
fn bicgstab_tiny_shadow_products_remain_in_range() {
    for epsilon in [1e-200, 1e-320] {
        check_tiny_shadow::<f64>(epsilon);
    }
}

#[test]
fn bicgstab_complex_tiny_shadow_products_remain_in_range() {
    for epsilon in [1e-200, 1e-320] {
        check_tiny_shadow::<Complex64>(epsilon);
    }
}

#[test]
fn bicgstab_does_not_recover_from_callback_breakdown_errors() {
    let rhs = PlainVector { data: vec![1.0] };
    let initial = PlainVector { data: vec![0.0] };
    for solver in ["bicgstab", "cg"] {
        for fail_at in [0, 1, 2] {
            let calls = AtomicUsize::new(0);
            let apply = |x: &PlainVector| -> Result<PlainVector> {
                if calls.fetch_add(1, Ordering::Relaxed) == fail_at {
                    return Err(KrylovError::NumericalBreakdown {
                        solver,
                        iteration: 99,
                        quantity: "callback inner solve",
                    }
                    .into());
                }
                Ok(x.clone())
            };
            let error = bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()).unwrap_err();
            match error {
                KrylovError::Operation {
                    solver: "bicgstab",
                    source,
                } => assert!(matches!(
                source.downcast_ref::<KrylovError>(),
                Some(KrylovError::NumericalBreakdown { solver: inner, iteration: 99, quantity: "callback inner solve" }) if *inner == solver)),
                other => panic!("callback error lost its operation wrapper: {other:?}"),
            }
            assert_eq!(calls.load(Ordering::Relaxed), fail_at + 1);
        }
    }
}

fn check_solution<S: TensorElement>(matrix: Vec<S>, rhs: Vec<S>, expected: Vec<S>) {
    let index = DynIndex::new_dyn(2);
    let input = index.prime();
    let matrix = IdxTensor::from_dense(vec![index.clone(), input.clone()], matrix).unwrap();
    let rhs = IdxTensor::from_dense(vec![index.clone()], rhs).unwrap();
    let expected = IdxTensor::from_dense(vec![index.clone()], expected).unwrap();
    let init = expected.scale(0.25.into()).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    let result = bicgstab(apply, &rhs, &init, &BiCgStabOptions::default()).unwrap();
    assert!(result.converged);
    assert!(result.iterations <= 2);
    assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);
    let residual = rhs
        .sub(&apply(&result.solution).unwrap())
        .unwrap()
        .norm()
        .unwrap();
    assert!(residual <= 1e-10 * rhs.norm().unwrap());
    assert_eq!(result.residual_norm, residual);
}

#[test]
fn bicgstab_real_nonsymmetric_known_solution() {
    // Column-major [[4, 1], [2, 3]], x = [1, 2].
    check_solution(vec![4.0, 2.0, 1.0, 3.0], vec![6.0, 8.0], vec![1.0, 2.0]);
}

#[test]
fn bicgstab_complex_nonhermitian_known_solution() {
    let c = Complex64::new;
    check_solution(
        vec![c(4.0, 0.0), c(2.0, -1.0), c(1.0, 1.0), c(3.0, 2.0)],
        vec![c(7.0, 5.0), c(11.0, 2.0)],
        vec![c(1.0, 1.0), c(2.0, -1.0)],
    );
}

#[test]
fn bicgstab_identity_converges_before_stabilization() {
    let rhs = PlainVector {
        data: vec![1.0, -2.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let result = bicgstab(
        |x: &PlainVector| Ok(x.clone()),
        &rhs,
        &initial,
        &BiCgStabOptions {
            max_iter: 1,
            ..BiCgStabOptions::default()
        },
    )
    .unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.solution.data, rhs.data);
    assert_eq!(result.residual_norm, 0.0);
}

#[test]
fn bicgstab_full_step_convergence_and_budget() {
    let apply = |x: &PlainVector| {
        Ok(PlainVector {
            data: vec![x.data[0] + x.data[1], 2.0 * x.data[1]],
        })
    };
    let rhs = PlainVector {
        data: vec![0.0, 1.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for budget in [0, 1] {
        let result = bicgstab(
            apply,
            &rhs,
            &initial,
            &BiCgStabOptions {
                max_iter: budget,
                ..BiCgStabOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.iterations, budget);
        assert_eq!(result.converged, budget == 1);
        assert_eq!(result.residual_norm, if budget == 0 { 1.0 } else { 0.0 });
        if result.converged {
            assert_eq!(result.solution.data, vec![-0.5, 0.5]);
        }
    }
}

#[test]
fn bicgstab_zero_rhs_and_exact_nonzero_guesses() {
    let apply = |x: &PlainVector| {
        Ok(PlainVector {
            data: vec![x.data[0] + x.data[1], 2.0 * x.data[1]],
        })
    };
    let initial = PlainVector {
        data: vec![1.0, -2.0],
    };
    let exact_rhs = apply(&initial).unwrap();
    let result = bicgstab(apply, &exact_rhs, &initial, &BiCgStabOptions::default()).unwrap();
    assert_eq!(result.iterations, 0);
    assert!(result.converged);
    assert_eq!(result.solution.data, initial.data);
    let zero = PlainVector {
        data: vec![0.0, 0.0],
    };
    let calls = AtomicUsize::new(0);
    let counted = |x: &PlainVector| {
        calls.fetch_add(1, Ordering::Relaxed);
        apply(x)
    };
    let result = bicgstab(counted, &zero, &initial, &BiCgStabOptions::default()).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(result.solution.data, zero.data);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn bicgstab_breakdown_survives_failed_recovery_check() {
    // Cycle one breaks down at omega after three operator calls; the fourth
    // call is the recovery's true-residual check.
    let rhs = PlainVector {
        data: vec![1.0, 0.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let calls = AtomicUsize::new(0);
    let nonfinite_check = |x: &PlainVector| -> Result<PlainVector> {
        if calls.fetch_add(1, Ordering::Relaxed) >= 3 {
            return Ok(PlainVector {
                data: vec![f64::NAN, 0.0],
            });
        }
        Ok(PlainVector {
            data: vec![x.data[0] + x.data[1], x.data[0]],
        })
    };
    assert!(matches!(
        bicgstab(nonfinite_check, &rhs, &initial, &BiCgStabOptions::default()),
        Err(KrylovError::NumericalBreakdown {
            solver: "bicgstab",
            iteration: 1,
            quantity: "omega",
        })
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 4);

    let calls = AtomicUsize::new(0);
    let failing_check = |x: &PlainVector| -> Result<PlainVector> {
        if calls.fetch_add(1, Ordering::Relaxed) >= 3 {
            return Err(std::io::Error::other("recovery operator failed").into());
        }
        Ok(PlainVector {
            data: vec![x.data[0] + x.data[1], x.data[0]],
        })
    };
    match bicgstab(failing_check, &rhs, &initial, &BiCgStabOptions::default()).unwrap_err() {
        KrylovError::Operation {
            solver: "bicgstab",
            source,
        } => assert!(source.downcast_ref::<std::io::Error>().is_some()),
        other => panic!("recovery callback failure was replaced: {other:?}"),
    }
}

#[test]
fn bicgstab_exhaustion_reports_true_residual() {
    let rhs = PlainVector {
        data: vec![6.0, 8.0],
    };
    let initial = PlainVector {
        data: vec![0.5, -0.5],
    };
    let apply = |x: &PlainVector| {
        Ok(PlainVector {
            data: vec![
                4.0 * x.data[0] + x.data[1],
                2.0 * x.data[0] + 3.0 * x.data[1],
            ],
        })
    };
    for budget in [0, 1] {
        let result = bicgstab(
            apply,
            &rhs,
            &initial,
            &BiCgStabOptions {
                max_iter: budget,
                ..BiCgStabOptions::default()
            },
        )
        .unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, budget);
        let norm = rhs
            .sub(&apply(&result.solution).unwrap())
            .unwrap()
            .norm()
            .unwrap();
        assert_eq!(result.residual_norm, norm);
    }
}

#[test]
fn bicgstab_absolute_threshold_accepts_equality_and_zero_tolerances() {
    let rhs = PlainVector {
        data: vec![0.001, 0.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let identity = |x: &PlainVector| Ok(x.clone());
    for atol in [0.0, 0.001] {
        let result = bicgstab(
            identity,
            &rhs,
            &initial,
            &BiCgStabOptions {
                rtol: 0.0,
                atol,
                ..BiCgStabOptions::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, if atol == 0.0 { 1 } else { 0 });
        assert_eq!(result.residual_norm, atol);
    }
}

#[test]
fn bicgstab_invalid_options_precede_operator() {
    let vector = PlainVector {
        data: vec![2.0, 0.0],
    };
    let apply =
        |_: &PlainVector| -> Result<PlainVector> { panic!("invalid options reached operator") };
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for options in [
            BiCgStabOptions {
                rtol: invalid,
                ..BiCgStabOptions::default()
            },
            BiCgStabOptions {
                atol: invalid,
                ..BiCgStabOptions::default()
            },
        ] {
            assert!(matches!(
                bicgstab(apply, &vector, &vector, &options),
                Err(KrylovError::InvalidOptions {
                    solver: "bicgstab",
                    ..
                })
            ));
        }
    }
    let options = BiCgStabOptions {
        rtol: f64::MAX,
        ..BiCgStabOptions::default()
    };
    assert!(matches!(
        bicgstab(apply, &vector, &vector, &options),
        Err(KrylovError::InvalidOptions {
            solver: "bicgstab",
            ..
        })
    ));
}

#[test]
fn bicgstab_reports_alpha_stabilization_and_omega_breakdowns() {
    let rhs = PlainVector {
        data: vec![1.0, 0.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for (matrix, quantity) in [
        ([0.0, 0.0, 0.0, 0.0], "alpha denominator"),
        ([0.0, -1.0, 1.0, 0.0], "alpha denominator"),
        ([1.0, 0.0, 1.0, 0.0], "stabilization denominator"),
        ([1.0, 1.0, 1.0, 0.0], "omega"),
    ] {
        let apply = |x: &PlainVector| {
            Ok(PlainVector {
                data: vec![
                    matrix[0] * x.data[0] + matrix[1] * x.data[1],
                    matrix[2] * x.data[0] + matrix[3] * x.data[1],
                ],
            })
        };
        assert!(
            matches!(bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()),
            Err(KrylovError::NumericalBreakdown { solver: "bicgstab", iteration: 1, quantity: actual })
            if actual == quantity)
        );
    }
}

#[test]
fn bicgstab_reports_shadow_residual_breakdown_on_nonsingular_system() {
    // det(A) = -2; after cycle one r = [0, 0, 1], orthogonal to r_shadow.
    let apply = |x: &PlainVector| {
        Ok(PlainVector {
            data: vec![
                -x.data[0] - x.data[1] - x.data[2],
                -x.data[0] - x.data[1],
                x.data[0] - x.data[1] - x.data[2],
            ],
        })
    };
    let rhs = PlainVector {
        data: vec![1.0, 0.0, 0.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0, 0.0],
    };
    assert!(matches!(
        bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()),
        Err(KrylovError::NumericalBreakdown {
            quantity: "shadow residual inner product",
            ..
        })
    ));
}

#[test]
fn bicgstab_preserves_operator_error_sources_at_each_step() {
    let rhs = PlainVector {
        data: vec![6.0, 8.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for fail_at in [0, 1, 2, 3] {
        let calls = AtomicUsize::new(0);
        let apply = |x: &PlainVector| -> Result<PlainVector> {
            if calls.fetch_add(1, Ordering::Relaxed) == fail_at {
                return Err(std::io::Error::other("BiCGSTAB operator failed").into());
            }
            Ok(PlainVector {
                data: vec![
                    4.0 * x.data[0] + x.data[1],
                    2.0 * x.data[0] + 3.0 * x.data[1],
                ],
            })
        };
        let options = BiCgStabOptions {
            max_iter: 1,
            ..BiCgStabOptions::default()
        };
        match bicgstab(apply, &rhs, &initial, &options).unwrap_err() {
            KrylovError::Operation {
                solver: "bicgstab",
                source,
            } => assert!(source.downcast_ref::<std::io::Error>().is_some()),
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[test]
fn bicgstab_zero_rhs_checks_full_index_identity() {
    let index = DynIndex::new_dyn(2);
    let rhs = make_vector_with_index(vec![0.0, 0.0], &index);
    for other in [DynIndex::new_dyn(2), index.prime()] {
        let initial = make_vector_with_index(vec![0.0, 0.0], &other);
        assert!(matches!(
            bicgstab(
                |x: &IdxTensor| Ok(x.clone()),
                &rhs,
                &initial,
                &BiCgStabOptions::default()
            ),
            Err(KrylovError::Operation { .. })
        ));
    }
}

#[test]
fn bicgstab_rejects_nonfinite_inputs_and_operator_outputs() {
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    for value in [f64::NAN, f64::INFINITY] {
        let rhs = PlainVector {
            data: vec![value, 1.0],
        };
        assert!(matches!(
            bicgstab(
                |x: &PlainVector| Ok(x.clone()),
                &rhs,
                &initial,
                &BiCgStabOptions::default()
            ),
            Err(KrylovError::NumericalBreakdown { .. })
        ));
        let rhs = PlainVector {
            data: vec![1.0, 1.0],
        };
        let apply = |_: &PlainVector| {
            Ok(PlainVector {
                data: vec![value, 0.0],
            })
        };
        assert!(matches!(
            bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()),
            Err(KrylovError::NumericalBreakdown { .. })
        ));
    }
}

#[test]
fn bicgstab_checks_true_residual_before_nonfinite_recurrence_error() {
    let rhs = PlainVector {
        data: vec![1.0, 0.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };
    let inputs = Mutex::new(Vec::new());
    let apply = |x: &PlainVector| -> Result<PlainVector> {
        inputs.lock().unwrap().push(x.data.clone());
        Ok(PlainVector {
            data: vec![1.7e308 * x.data[0], 1.7e308 * x.data[0]],
        })
    };
    assert!(matches!(
        bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()),
        Err(KrylovError::NumericalBreakdown { iteration: 1, .. })
    ));
    let inputs = inputs.into_inner().unwrap();
    assert_eq!(inputs.len(), 3);
    assert_eq!(inputs[2], initial.data);
}

#[test]
fn bicgstab_stops_on_unrepresentable_physical_residual() {
    let index = DynIndex::new_dyn(2);
    let rhs = make_vector_with_index(vec![1e308, 0.0], &index);
    let initial = make_vector_with_index(vec![0.0, 0.0], &index);
    let input = index.prime();
    let matrix = IdxTensor::from_dense(
        vec![index.clone(), input.clone()],
        vec![1.0_f64, 2.0, 0.0, 1.0],
    )
    .unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?)
    };
    let options = BiCgStabOptions {
        max_iter: 2,
        ..BiCgStabOptions::default()
    };
    assert!(matches!(
        bicgstab(apply, &rhs, &initial, &options),
        Err(KrylovError::NumericalBreakdown { iteration: 1, .. })
    ));
}

#[test]
fn bicgstab_extreme_f64_identity_and_operator_scales() {
    let index = DynIndex::new_dyn(1);
    let initial = IdxTensor::zeros::<f64>(vec![index.clone()]).unwrap();
    for (factor, value, expected) in [
        (1.0, f64::from_bits(1), f64::from_bits(1)),
        (1.0, 1e-200, 1e-200),
        (1.0, 1e200, 1e200),
        (1.0, 1e308, 1e308),
        (1.0, f64::MAX, f64::MAX),
        (1e20, 1e150, 1e130),
        (1e-200, 1e-200, 1.0),
        (1e308, 1.0, 1e-308),
    ] {
        let rhs = make_vector_with_index(vec![value], &index);
        let apply = |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.scale(factor.into())?) };
        let result = bicgstab(apply, &rhs, &initial, &BiCgStabOptions::default()).unwrap();
        assert!(result.converged);
        let actual = result.solution.to_vec::<f64>().unwrap()[0];
        assert!(
            (actual / expected - 1.0).abs() < 1e-12,
            "factor={factor:e} rhs={value:e} x={actual:e}"
        );
        assert!(((factor * actual - value) / value).abs() <= 1e-10);
        assert!(result.residual_norm / value <= 1e-10);
    }
}

fn check_scaled<S: TensorElement>(
    matrix: Vec<S>,
    solution: Vec<S>,
    scales: &[(f64, f64)],
    rtol: f64,
) {
    let index = DynIndex::new_dyn(solution.len());
    let input = index.prime();
    let matrix = IdxTensor::from_dense(vec![index.clone(), input.clone()], matrix).unwrap();
    let solution = IdxTensor::from_dense(vec![index.clone()], solution).unwrap();
    for &(scale, operator_scale) in scales {
        let apply = |x: &IdxTensor| -> Result<IdxTensor> {
            Ok(contract(&[&matrix, &x.replaceind(&index, &input)?])?
                .scale(operator_scale.into())?)
        };
        let expected = solution.scale(scale.into()).unwrap();
        let rhs = apply(&expected).unwrap();
        let initial = expected.scale(0.0.into()).unwrap();
        let options = BiCgStabOptions {
            rtol,
            ..BiCgStabOptions::default()
        };
        let result = bicgstab(apply, &rhs, &initial, &options).unwrap();
        assert!(result.converged);
        let error =
            result.solution.sub(&expected).unwrap().maxabs().unwrap() / expected.maxabs().unwrap();
        assert!(
            error <= rtol,
            "scale={scale:e}, relative solution error={error:e}"
        );
        let residual = rhs
            .sub(&apply(&result.solution).unwrap())
            .unwrap()
            .norm()
            .unwrap();
        assert!(residual / rhs.norm().unwrap() <= rtol);
        assert_eq!(result.residual_norm, residual);
    }
}

#[test]
fn bicgstab_scaled_nonsymmetric_recurrence_all_scalar_types() {
    let real = vec![4.0, 2.0, 1.0, 3.0];
    check_scaled(
        real.clone(),
        vec![1.0, 2.0],
        &[
            (1e-200, 1.0),
            (1.0, 1.0),
            (1e150, 1.0),
            (1.0, 1e-200),
            (1.0, 1e200),
        ],
        1e-10,
    );
    check_scaled(
        real.iter().map(|&x| x as f32).collect(),
        vec![1.0_f32, 2.0],
        &[
            (1e-30, 1.0),
            (1.0, 1.0),
            (1e20, 1.0),
            (1.0, 1e-20),
            (1.0, 1e20),
        ],
        1e-5,
    );
    let c = Complex64::new;
    let complex = vec![c(4.0, 0.0), c(2.0, -1.0), c(1.0, 1.0), c(3.0, 2.0)];
    check_scaled(
        complex.clone(),
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        &[
            (1e-200, 1.0),
            (1.0, 1.0),
            (1e150, 1.0),
            (1.0, 1e-200),
            (1.0, 1e200),
        ],
        1e-10,
    );
    let c = num_complex::Complex32::new;
    check_scaled(
        complex
            .iter()
            .map(|z| c(z.re as f32, z.im as f32))
            .collect(),
        vec![c(1.0, 1.0), c(2.0, -1.0)],
        &[
            (1e-30, 1.0),
            (1.0, 1.0),
            (1e20, 1.0),
            (1.0, 1e-20),
            (1.0, 1e20),
        ],
        1e-5,
    );
}

#[test]
fn bicgstab_subnormal_single_precision_real_and_complex_identity() {
    let index = DynIndex::new_dyn(1);
    let c = num_complex::Complex32::new;
    for scale in [f32::from_bits(1), 1e-30_f32, 1e20] {
        for rhs in [
            IdxTensor::from_dense(vec![index.clone()], vec![scale]).unwrap(),
            IdxTensor::from_dense(vec![index.clone()], vec![c(3.0 * scale, 4.0 * scale)]).unwrap(),
        ] {
            let initial = rhs.scale(0.0.into()).unwrap();
            let options = BiCgStabOptions {
                rtol: 1e-5,
                ..BiCgStabOptions::default()
            };
            let result = bicgstab(|x: &IdxTensor| Ok(x.clone()), &rhs, &initial, &options).unwrap();
            assert!(result.converged);
            assert!(
                result.solution.sub(&rhs).unwrap().maxabs().unwrap() / rhs.maxabs().unwrap()
                    <= options.rtol
            );
            assert!(result.residual_norm / rhs.norm().unwrap() <= options.rtol);
        }
    }
}

#[test]
fn bicgstab_extreme_complex_double_identity_and_blocks() {
    use crate::block_tensor::BlockTensor;
    let index = DynIndex::new_dyn(1);
    let c = Complex64::new;
    for scale in [f64::from_bits(1), 1e-200, 1e200] {
        let first =
            IdxTensor::from_dense(vec![index.clone()], vec![c(3.0 * scale, 4.0 * scale)]).unwrap();
        let second = IdxTensor::from_dense(vec![index.clone()], vec![c(-scale, scale)]).unwrap();
        let rhs = BlockTensor::new(vec![first, second], (2, 1)).unwrap();
        let initial = rhs.scale(0.0.into()).unwrap();
        let result = bicgstab(
            |x: &BlockTensor<IdxTensor>| Ok(x.clone()),
            &rhs,
            &initial,
            &BiCgStabOptions::default(),
        )
        .unwrap();
        assert!(result.converged);
        assert!(
            result.solution.sub(&rhs).unwrap().maxabs().unwrap() / rhs.maxabs().unwrap() < 1e-12
        );
        assert_eq!(result.residual_norm, 0.0);
    }
}

#[test]
fn bicgstab_replaces_optimistic_residual_at_both_convergence_points() {
    let rhs = PlainVector {
        data: vec![1.0, 1.0, 1.0],
    };
    let initial = PlainVector {
        data: vec![0.0, 0.0, 0.0],
    };
    // The three-cycle estimate passes, but the true residual is about 1e-16.
    // 5e-17 triggers at the BiCG step; 1e-18 triggers after stabilization.
    for (atol, three_calls, four_calls) in [(5e-17, 7, 9), (1e-18, 8, 10)] {
        for budget in [3, 4] {
            let calls = AtomicUsize::new(0);
            let apply = |x: &PlainVector| -> Result<PlainVector> {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(PlainVector {
                    data: vec![x.data[0], 2.0 * x.data[1], 3.0 * x.data[2]],
                })
            };
            let options = BiCgStabOptions {
                max_iter: budget,
                rtol: 0.0,
                atol,
            };
            let result = bicgstab(apply, &rhs, &initial, &options).unwrap();
            assert_eq!(result.iterations, budget);
            assert_eq!(result.converged, budget == 4);
            assert_eq!(
                calls.load(Ordering::Relaxed),
                if budget == 3 { three_calls } else { four_calls }
            );
            let true_norm = rhs
                .sub(&apply(&result.solution).unwrap())
                .unwrap()
                .norm()
                .unwrap();
            assert_eq!(result.residual_norm, true_norm);
            assert_eq!(true_norm <= atol, result.converged);
            if result.converged {
                assert!(
                    result
                        .solution
                        .sub(&PlainVector {
                            data: vec![1.0, 0.5, 1.0 / 3.0]
                        })
                        .unwrap()
                        .maxabs()
                        .unwrap()
                        < 1e-15
                );
            }
        }
    }
}

#[test]
fn bicgstab_longer_nonsymmetric_recurrence_all_scalar_types() {
    let real = vec![
        4.0, 2.0, 0.0, 1.0, 1.0, 5.0, -1.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 1.0, 6.0,
    ];
    check_scaled(
        real.clone(),
        vec![1.0, -2.0, 3.0, -4.0],
        &[(1.0, 1.0)],
        1e-10,
    );
    check_scaled(
        real.iter().map(|&x| x as f32).collect(),
        vec![1.0_f32, -2.0, 3.0, -4.0],
        &[(1.0, 1.0)],
        1e-5,
    );
    let c = Complex64::new;
    let mut complex: Vec<_> = real.iter().map(|&x| c(x, 0.0)).collect();
    complex[0] = c(4.0, 1.0);
    complex[3] = c(1.0, -1.0);
    complex[5] = c(5.0, -1.0);
    complex[9] = c(1.0, 1.0);
    complex[10] = c(3.0, 2.0);
    check_scaled(
        complex.clone(),
        vec![c(1.0, 1.0), c(-2.0, 1.0), c(3.0, -2.0), c(-4.0, -0.5)],
        &[(1.0, 1.0)],
        1e-10,
    );
    let c = num_complex::Complex32::new;
    check_scaled(
        complex
            .iter()
            .map(|z| c(z.re as f32, z.im as f32))
            .collect(),
        vec![c(1.0, 1.0), c(-2.0, 1.0), c(3.0, -2.0), c(-4.0, -0.5)],
        &[(1.0, 1.0)],
        1e-5,
    );
}
