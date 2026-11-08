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
