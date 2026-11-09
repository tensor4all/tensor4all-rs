use super::*;

#[test]
fn local_gmres_options_match_krylovkit_restart_convention() {
    let options = LinsolveOptions::default()
        .with_gmres_restart_dim(30)
        .with_gmres_max_restarts(10)
        .with_gmres_tol(1.0e-8);

    let gmres_options = local_gmres_options(&options).unwrap();

    assert_eq!(gmres_options.max_iter, 30);
    assert_eq!(gmres_options.max_restarts, 10);
    assert_eq!(gmres_options.rtol, 1.0e-8);
}

#[test]
fn local_gmres_options_does_not_convert_maxiter_to_total_step_limit() {
    let options = LinsolveOptions::default()
        .with_gmres_restart_dim(30)
        .with_gmres_max_restarts(100);

    let gmres_options = local_gmres_options(&options).unwrap();

    assert_eq!(gmres_options.max_iter, 30);
    assert_eq!(gmres_options.max_restarts, 100);
}

#[test]
fn local_gmres_options_reject_zero_iteration_parameters() {
    assert!(local_gmres_options(&LinsolveOptions::default().with_gmres_restart_dim(0)).is_err());
    assert!(local_gmres_options(&LinsolveOptions::default().with_gmres_max_restarts(0)).is_err());
}

fn choices() -> Vec<LocalSolver> {
    vec![
        LocalSolver::Gmres,
        LocalSolver::Cg(CgOptions::default()),
        LocalSolver::BiCgStab(BiCgStabOptions::default()),
    ]
}

#[test]
fn local_solver_preserves_callback_error_sources() {
    #[derive(Debug, thiserror::Error)]
    #[error("callback test failure")]
    struct CallbackFailure;
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(2);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![1.0, -2.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i], vec![0.0; 2]).unwrap();
    for solver in choices() {
        let options = LinsolveOptions::new(1).with_local_solver(solver);
        let error = solve_local_system(
            &options,
            |_: &IdxTensor| Err(CallbackFailure.into()),
            &b,
            &init,
            false,
        )
        .err()
        .unwrap();
        assert!(
            error.chain().any(|e| e.is::<CallbackFailure>()),
            "{error:?}"
        );
    }
}

#[test]
fn local_solver_reports_native_residual_units_on_budget_exhaustion() {
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(3);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![100.0, -200.0, 300.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i.clone()], vec![0.0; 3]).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        let values = x.to_vec::<f64>()?;
        Ok(IdxTensor::from_dense(
            vec![i.clone()],
            values
                .iter()
                .zip([1.0, 2.0, 4.0])
                .map(|(v, d)| v * d)
                .collect::<Vec<_>>(),
        )?)
    };
    let mut absolute = LinsolveOptions::new(1).with_gmres_absolute_tolerance();
    absolute.gmres_restart_dim = 1;
    absolute.gmres_max_restarts = 1;
    let options = [
        LinsolveOptions::new(1)
            .with_gmres_restart_dim(1)
            .with_gmres_max_restarts(1),
        absolute,
        LinsolveOptions::new(1).with_local_solver(LocalSolver::Cg(CgOptions {
            max_iter: 1,
            rtol: 0.0,
            ..Default::default()
        })),
        LinsolveOptions::new(1).with_local_solver(LocalSolver::BiCgStab(BiCgStabOptions {
            max_iter: 1,
            rtol: 0.0,
            ..Default::default()
        })),
    ];
    for (index, options) in options.into_iter().enumerate() {
        let result = solve_local_system(&options, apply, &b, &init, false).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        let norm = apply(&result.solution)
            .unwrap()
            .sub(&b)
            .unwrap()
            .norm()
            .unwrap();
        let expected = if index == 0 {
            norm / b.norm().unwrap()
        } else {
            norm
        };
        assert!(
            (result.residual_norm - expected).abs() < 1e-10,
            "{index}: {} vs {expected}",
            result.residual_norm
        );
        assert_eq!(result.residual_is_relative, index == 0);
    }
}

#[test]
fn local_solver_skips_zero_affine_operator_term() {
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(2);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![2.0, 4.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i], vec![0.0; 2]).unwrap();
    for solver in choices() {
        let options = LinsolveOptions::new(1)
            .with_coefficients(2.0, 0.0)
            .with_local_solver(solver);
        let result = solve_local_system(
            &options,
            |_: &IdxTensor| anyhow::bail!("zero affine term must not call A"),
            &b,
            &init,
            false,
        )
        .unwrap();
        assert!(result.converged);
        assert!(
            result
                .solution
                .sub(&b.scale(0.5.into()).unwrap())
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
    }
}

#[test]
fn local_solver_handles_zero_and_tiny_rhs() {
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(2);
    let init = IdxTensor::from_dense(vec![i.clone()], vec![0.0; 2]).unwrap();
    for scale in [0.0, 1.0, 1e-20] {
        let b = IdxTensor::from_dense(vec![i.clone()], vec![scale, -2.0 * scale]).unwrap();
        for solver in choices() {
            let options = LinsolveOptions::new(1)
                .with_coefficients(0.5, 0.5)
                .with_local_solver(solver);
            let result =
                solve_local_system(&options, |x: &IdxTensor| Ok(x.clone()), &b, &init, false)
                    .unwrap();
            assert!(result.converged);
            let error = result.solution.sub(&b).unwrap().maxabs().unwrap();
            assert!(error <= 1e-12 * scale, "scale {scale}: {error}");
        }
    }
}

#[test]
fn local_solver_rejects_zero_affine_coefficients_for_every_solver() {
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(2);
    let init = IdxTensor::from_dense(vec![i.clone()], vec![0.0; 2]).unwrap();
    for values in [vec![0.0, 0.0], vec![1.0, -2.0]] {
        let b = IdxTensor::from_dense(vec![i.clone()], values).unwrap();
        for solver in choices() {
            let options = LinsolveOptions::new(1)
                .with_coefficients(0.0, 0.0)
                .with_local_solver(solver);
            let error = solve_local_system(
                &options,
                |_: &IdxTensor| anyhow::bail!("zero coefficients must not call A"),
                &b,
                &init,
                false,
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("both zero"), "{error}");
        }
    }
}

#[test]
fn local_solver_rejects_invalid_active_gmres_options() {
    for options in [
        LinsolveOptions::default().with_gmres_tol(f64::NAN),
        LinsolveOptions::default().with_gmres_tol(0.0),
        LinsolveOptions::default().with_gmres_restart_dim(0),
        LinsolveOptions::default().with_gmres_restart_dim(usize::MAX),
        LinsolveOptions::default().with_gmres_max_restarts(0),
    ] {
        assert!(validate_local_options(&options).is_err());
    }
}

#[test]
fn local_solver_reports_unallocatable_gmres_restart_dim_as_invalid_options() {
    use tensor4all_core::krylov::KrylovError;
    use tensor4all_core::{DynIndex, IdxTensor};
    // `usize::MAX - 1` passes the overflow check above, but its Krylov
    // workspace cannot be allocated; the solve fails instead of panicking.
    let options = LinsolveOptions::default().with_gmres_restart_dim(usize::MAX - 1);
    validate_local_options(&options).unwrap();
    let i = DynIndex::new_dyn(2);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![1.0, -2.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i], vec![0.0; 2]).unwrap();
    let error = solve_local_system(&options, |x: &IdxTensor| Ok(x.clone()), &b, &init, false)
        .err()
        .unwrap();
    assert!(
        error.chain().any(|e| matches!(
            e.downcast_ref::<KrylovError>(),
            Some(KrylovError::InvalidOptions {
                solver: "gmres",
                ..
            })
        )),
        "{error:?}"
    );
}

#[test]
fn local_solver_rejects_changed_gmres_controls_for_other_solvers() {
    for solver in [
        LocalSolver::Cg(CgOptions::default()),
        LocalSolver::BiCgStab(BiCgStabOptions::default()),
    ] {
        let base = LinsolveOptions::default().with_local_solver(solver);
        base.validate_local_solver().unwrap();
        for (options, name) in [
            (base.clone().with_gmres_tol(f64::NAN), "gmres_tol"),
            (base.clone().with_gmres_tol(1e-8), "gmres_tol"),
            (
                base.clone().with_gmres_absolute_tolerance(),
                "gmres_tolerance_mode",
            ),
            (
                base.clone().with_gmres_max_restarts(0),
                "gmres_max_restarts",
            ),
            (base.clone().with_gmres_restart_dim(5), "gmres_restart_dim"),
        ] {
            let error = options.validate_local_solver().unwrap_err();
            assert!(
                matches!(error, TreeTNOperationError::InvalidConfiguration { .. }),
                "{error:?}"
            );
            let message = format!("{name} configures LocalSolver::Gmres only");
            assert!(error.to_string().contains(&message), "{error}");
        }
    }
}

#[test]
fn local_solver_applies_the_operator_coefficient_when_a0_is_zero() {
    use tensor4all_core::{DynIndex, IdxTensor};
    let i = DynIndex::new_dyn(2);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![6.0, -4.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i.clone()], vec![0.0; 2]).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        let values = x.to_vec::<f64>()?;
        Ok(IdxTensor::from_dense(
            vec![i.clone()],
            vec![3.0 * values[0], 2.0 * values[1]],
        )?)
    };
    for a1 in [1.0, 2.0] {
        let expected = IdxTensor::from_dense(vec![i.clone()], vec![2.0 / a1, -2.0 / a1]).unwrap();
        for solver in choices() {
            let options = LinsolveOptions::new(1)
                .with_coefficients(0.0, a1)
                .with_local_solver(solver);
            let result = solve_local_system(&options, apply, &b, &init, false).unwrap();
            assert!(result.converged);
            let error = result.solution.sub(&expected).unwrap().maxabs().unwrap();
            assert!(error < 1e-10, "a1 = {a1}: {error}");
        }
    }
}

#[test]
fn exactly_one_accepts_real_and_complex_spellings_only() {
    assert!(is_exactly_one(&AnyScalar::new_real(1.0)));
    assert!(is_exactly_one(&AnyScalar::new_complex(1.0, 0.0)));
    assert!(!is_exactly_one(&AnyScalar::new_complex(1.0, 1e-300)));
    assert!(!is_exactly_one(&AnyScalar::new_real(-1.0)));
    assert!(!is_exactly_one(&AnyScalar::new_real(1.0 + f64::EPSILON)));
}

#[test]
fn local_solver_rejects_non_finite_affine_coefficients_for_every_solver() {
    for solver in choices() {
        for (a0, a1, name) in [
            (
                AnyScalar::new_real(f64::NAN),
                AnyScalar::new_real(1.0),
                "a0",
            ),
            (
                AnyScalar::new_real(f64::INFINITY),
                AnyScalar::new_real(1.0),
                "a0",
            ),
            (
                AnyScalar::new_complex(1.0, f64::NAN),
                AnyScalar::new_real(1.0),
                "a0",
            ),
            (
                AnyScalar::new_real(1.0),
                AnyScalar::new_real(f64::NEG_INFINITY),
                "a1",
            ),
            (
                AnyScalar::new_real(1.0),
                AnyScalar::new_complex(f64::NAN, 0.0),
                "a1",
            ),
        ] {
            let options = LinsolveOptions::new(1)
                .with_coefficients(a0, a1)
                .with_local_solver(solver.clone());
            let error = options.validate_local_solver().unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("{name} must be finite")),
                "{error}"
            );
        }
    }
}

#[test]
fn local_solver_keeps_tracked_unit_and_zero_coefficients_in_the_graph() {
    use tensor4all_core::{DynIndex, IdxTensor};
    // (a0 + 2 a1) x = 1 gives x = 1 / (a0 + 2 a1), dx/da0 = -x^2 and
    // dx/da1 = -2 x^2: (x, dx/da0, dx/da1) = (1/2, -1/4, -1/2) at
    // (a0, a1) = (0, 1) and (1, -1, -2) at (1, 0).
    let i = DynIndex::new_dyn(1);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![1.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i], vec![0.0]).unwrap();
    let apply = |x: &IdxTensor| -> Result<IdxTensor> { Ok(x.scale(AnyScalar::new_real(2.0))?) };
    for (solver, (a0, a1, x, da0, da1)) in choices().into_iter().flat_map(|solver| {
        [(0.0, 1.0, 0.5, -0.25, -0.5), (1.0, 0.0, 1.0, -1.0, -2.0)]
            .map(|case| (solver.clone(), case))
    }) {
        let a0 = AnyScalar::new_real(a0).enable_grad().unwrap();
        let a1 = AnyScalar::new_real(a1).enable_grad().unwrap();
        let mut options = LinsolveOptions::new(1).with_local_solver(solver.clone());
        options.a0 = a0.clone();
        options.a1 = a1.clone();
        let result = solve_local_system(&options, apply, &b, &init, false).unwrap();
        let total = result.solution.sum().unwrap();
        assert!((total.real() - x).abs() < 1e-12, "{solver:?}: {total:?}");
        total.backward().unwrap();
        for (name, value, expected) in [("a0", a0, da0), ("a1", a1, da1)] {
            let grad = value.grad().unwrap();
            let grad = grad.unwrap_or_else(|| panic!("{solver:?}: {name} has no gradient"));
            assert!(
                (grad.real() - expected).abs() < 1e-12,
                "{solver:?}: d/d{name} = {grad:?}"
            );
        }
    }
}

#[test]
fn local_solver_gradient_is_that_of_the_returned_iterate() {
    use tensor4all_core::{DynIndex, IdxTensor};
    // (a0 I + a1 A) x = b with A = diag(2, 3), b = (1, 2), a0 = 1 and a
    // tracked a1 = 0. Every solver returns x = alpha b after one step, with
    // a zero residual. The exact solution has d sum(x)/da1 = -sum(A b) = -8;
    // the returned iterate has dalpha/da1 = -b'Ab / b'b = -14/5, so
    // d sum(x)/da1 = -8.4. The gradient is that of the iterate.
    let i = DynIndex::new_dyn(2);
    let b = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let init = IdxTensor::from_dense(vec![i.clone()], vec![0.0; 2]).unwrap();
    let e2 = IdxTensor::from_dense(vec![i], vec![0.0, 1.0]).unwrap();
    // A = diag(2, 3) = 2 I + e2 e2^T, built from graph-preserving operations.
    let apply = |x: &IdxTensor| -> Result<IdxTensor> {
        let projected = e2.scale(e2.inner_product(x)?)?;
        Ok(x.axpby(
            AnyScalar::new_real(2.0),
            &projected,
            AnyScalar::new_real(1.0),
        )?)
    };
    for solver in choices() {
        let a1 = AnyScalar::new_real(0.0).enable_grad().unwrap();
        let mut options = LinsolveOptions::new(1).with_local_solver(solver.clone());
        options.a0 = AnyScalar::new_real(1.0);
        options.a1 = a1.clone();
        let result = solve_local_system(&options, apply, &b, &init, false).unwrap();
        let total = result.solution.sum().unwrap();
        assert_eq!(result.iterations, 1, "{solver:?}");
        assert!((total.real() - 3.0).abs() < 1e-12, "{solver:?}: {total:?}");
        total.backward().unwrap();
        let grad = a1.grad().unwrap();
        let grad = grad.unwrap_or_else(|| panic!("{solver:?}: a1 has no gradient"));
        assert!(
            (grad.real() + 8.4).abs() < 1e-12,
            "{solver:?}: d/da1 = {grad:?}"
        );
    }
}
