use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{
    krylov::{BiCgStabOptions, CgOptions, KrylovError},
    AnyScalar, DynIndex, IdxTensor, IndexLike,
};
use tensor4all_treetn::{
    square_linsolve, IndexMapping, LinsolveOptions, LocalSolver, SquareLinsolveResult, TreeTN,
    TreeTNOperationError,
};

type State = TreeTN<IdxTensor, usize>;

fn scalar(z: Complex64, complex: bool) -> AnyScalar {
    if complex {
        z.into()
    } else {
        AnyScalar::new_real(z.re)
    }
}

fn tensor(indices: Vec<DynIndex>, values: Vec<Complex64>, complex: bool) -> IdxTensor {
    if complex {
        IdxTensor::from_dense(indices, values).unwrap()
    } else {
        IdxTensor::from_dense(indices, values.iter().map(|z| z.re).collect::<Vec<_>>()).unwrap()
    }
}

fn solvers() -> Vec<LocalSolver> {
    vec![
        LocalSolver::Gmres,
        LocalSolver::Cg(CgOptions {
            rtol: 1e-12,
            ..Default::default()
        }),
        LocalSolver::BiCgStab(BiCgStabOptions {
            rtol: 1e-12,
            ..Default::default()
        }),
    ]
}

struct Problem {
    operator: State,
    rhs: State,
    init: State,
    expected: IdxTensor,
    input: HashMap<usize, IndexMapping<DynIndex>>,
    output: HashMap<usize, IndexMapping<DynIndex>>,
    options: LinsolveOptions,
}

impl Problem {
    fn new(a: [Complex64; 4], complex: bool, a0: Complex64, a1: Complex64) -> Self {
        let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
        let site = DynIndex::new_dyn(2);
        let s_in = site.sim();
        let s_out = s_in.prime();
        let x = [z(1.0, 0.5), z(-2.0, 0.25)];
        let b = (0..2)
            .map(|row| a0 * x[row] + a1 * (a[row] * x[0] + a[row + 2] * x[1]))
            .collect();
        let state = |t| State::from_tensors(vec![t], vec![0]).unwrap();
        let operator = state(tensor(
            vec![s_out.clone(), s_in.clone()],
            a.to_vec(),
            complex,
        ));
        let rhs = state(tensor(vec![site.clone()], b, complex));
        let init = state(tensor(vec![site.clone()], vec![z(0.0, 0.0); 2], complex));
        let expected = tensor(vec![site.clone()], x.to_vec(), complex);
        let input = HashMap::from([(
            0,
            IndexMapping {
                true_index: site.clone(),
                internal_index: s_in,
            },
        )]);
        let output = HashMap::from([(
            0,
            IndexMapping {
                true_index: site,
                internal_index: s_out,
            },
        )]);
        let options = LinsolveOptions::new(2)
            .with_nsite(1)
            .with_gmres_tol(1e-12)
            .with_coefficients(scalar(a0, complex), scalar(a1, complex))
            .with_convergence_tol(1e-9);
        Self {
            operator,
            rhs,
            init,
            expected,
            input,
            output,
            options,
        }
    }

    fn hpd(complex: bool) -> Self {
        let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
        Self::new(
            [z(3.0, 0.0), z(0.25, 0.4), z(0.25, -0.4), z(2.0, 0.0)],
            complex,
            z(0.5, 0.0),
            z(0.25, 0.0),
        )
    }

    /// Options for `solver`; the 1e-12 GMRES tolerance is kept only for GMRES,
    /// because other local solvers reject changed `gmres_*` controls.
    fn with_solver(&self, solver: LocalSolver) -> LinsolveOptions {
        let gmres_tol = if matches!(solver, LocalSolver::Gmres) {
            self.options.gmres_tol
        } else {
            LinsolveOptions::default().gmres_tol
        };
        self.options
            .clone()
            .with_gmres_tol(gmres_tol)
            .with_local_solver(solver)
    }

    fn solve(
        &self,
        options: LinsolveOptions,
    ) -> Result<SquareLinsolveResult<IdxTensor, usize>, TreeTNOperationError> {
        square_linsolve(
            &self.operator,
            &self.rhs,
            self.init.clone(),
            &0,
            options,
            Some(self.input.clone()),
            Some(self.output.clone()),
        )
    }

    fn assert_solution(&self, result: &SquareLinsolveResult<IdxTensor, usize>) {
        assert!(result.converged, "residual {:?}", result.residual);
        assert!(result.residual.unwrap() < 1e-9);
        assert!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .sub(&self.expected)
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-9
        );
    }
}

fn has_krylov_error(
    error: &TreeTNOperationError,
    predicate: impl Fn(&KrylovError) -> bool,
) -> bool {
    error
        .source
        .chain()
        .filter_map(|e| e.downcast_ref::<KrylovError>())
        .any(predicate)
}

#[test]
fn local_solver_selection_solves_real_and_complex_hpd_affine_systems() {
    for complex in [false, true] {
        let p = Problem::hpd(complex);
        for solver in solvers() {
            p.assert_solution(&p.solve(p.with_solver(solver)).unwrap());
        }
        let default = p.solve(p.options.clone()).unwrap();
        let explicit = p.solve(p.with_solver(LocalSolver::Gmres)).unwrap();
        assert!(
            default
                .solution
                .contract_to_tensor()
                .unwrap()
                .sub(&explicit.solution.contract_to_tensor().unwrap())
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
        assert!(matches!(
            LinsolveOptions::default().local_solver,
            LocalSolver::Gmres
        ));
    }
}

#[test]
fn local_solver_selection_handles_general_real_and_complex_operators() {
    for complex in [false, true] {
        let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
        let p = Problem::new(
            [z(2.0, 0.5), z(0.1, -0.1), z(0.6, 0.2), z(3.0, -0.3)],
            complex,
            z(0.1, -0.2),
            z(0.7, 0.15),
        );
        for solver in [
            LocalSolver::Gmres,
            LocalSolver::BiCgStab(BiCgStabOptions::default()),
        ] {
            p.assert_solution(&p.solve(p.with_solver(solver)).unwrap());
        }
    }
}

#[test]
fn local_solver_selection_uses_effective_affine_operator_for_cg() {
    let z = |re| Complex64::new(re, 0.0);
    let p = Problem::new([z(-3.0), z(0.0), z(0.0), z(-2.0)], false, z(5.0), z(1.0));
    p.assert_solution(
        &p.solve(p.with_solver(LocalSolver::Cg(CgOptions::default())))
            .unwrap(),
    );
    let p = Problem::new([z(3.0), z(0.0), z(0.0), z(2.0)], false, z(-5.0), z(1.0));
    let error = p
        .solve(p.with_solver(LocalSolver::Cg(CgOptions::default())))
        .unwrap_err();
    assert!(
        has_krylov_error(&error, |e| matches!(
            e,
            KrylovError::NonPositiveCurvature { solver: "cg", .. }
        )),
        "{error:?}"
    );
    let p = Problem::new(
        [
            Complex64::new(3.0, 1.0),
            z(0.0),
            z(0.0),
            Complex64::new(2.0, 1.0),
        ],
        true,
        z(0.0),
        z(1.0),
    );
    let error = p
        .solve(p.with_solver(LocalSolver::Cg(CgOptions::default())))
        .unwrap_err();
    assert!(
        has_krylov_error(&error, |e| matches!(
            e,
            KrylovError::NonHermitian { solver: "cg", .. }
        )),
        "{error:?}"
    );
}

#[test]
fn local_solver_selection_cg_fails_on_an_indefinite_operator() {
    // A = diag(3, -2) is Hermitian but indefinite. With x = (1, -2), b = (3, 4)
    // and the first CG direction b has curvature 3 * 9 - 2 * 16 = -5, so the
    // solve must stop with a typed error rather than return an iterate.
    let z = |re| Complex64::new(re, 0.0);
    let p = Problem::new([z(3.0), z(0.0), z(0.0), z(-2.0)], false, z(0.0), z(1.0));
    let error = p
        .solve(p.with_solver(LocalSolver::Cg(CgOptions::default())))
        .unwrap_err();
    assert!(
        has_krylov_error(&error, |e| matches!(
            e,
            KrylovError::NonPositiveCurvature { solver: "cg", .. }
        )),
        "{error:?}"
    );
    // GMRES solves the same indefinite system.
    p.assert_solution(&p.solve(p.with_solver(LocalSolver::Gmres)).unwrap());
}

#[test]
fn local_solver_selection_cg_does_not_converge_on_a_real_non_symmetric_operator() {
    // A = [[2, 1], [0, 2]] is real, non-symmetric, and has a positive symmetric
    // part. For real scalars CG's Hermitian-consistency check has nothing to
    // measure; with this right-hand side, b = A (1, -2) = (0, -4), every visited
    // curvature is positive and CG exhausts its budget. The sweep reports that
    // through `converged = false` and its true residual instead of returning a
    // plausible solution as converged.
    let z = |re| Complex64::new(re, 0.0);
    let p = Problem::new([z(2.0), z(0.0), z(1.0), z(2.0)], false, z(0.0), z(1.0));
    let result = p
        .solve(p.with_solver(LocalSolver::Cg(CgOptions::default())))
        .unwrap();
    assert!(!result.converged);
    assert!(result.residual.unwrap() > 1e-3, "{:?}", result.residual);
    // BiCGSTAB and GMRES solve the same operator.
    p.assert_solution(
        &p.solve(p.with_solver(LocalSolver::BiCgStab(BiCgStabOptions::default())))
            .unwrap(),
    );
    p.assert_solution(&p.solve(p.with_solver(LocalSolver::Gmres)).unwrap());
}

#[test]
fn local_solver_selection_rejects_invalid_convergence_tolerance() {
    let p = Problem::hpd(false);
    for tol in [f64::NAN, -1.0] {
        let error = p
            .solve(p.options.clone().with_convergence_tol(tol))
            .unwrap_err();
        assert!(
            error
                .source
                .chain()
                .any(|e| e.to_string().contains("convergence_tol must be finite")),
            "{error:?}"
        );
    }
}

#[test]
fn local_solver_selection_propagates_bicgstab_breakdown_without_fallback() {
    let z = |re| Complex64::new(re, 0.0);
    let p = Problem::new([z(0.0), z(-1.0), z(1.0), z(0.0)], false, z(0.0), z(1.0));
    let error = p
        .solve(p.with_solver(LocalSolver::BiCgStab(BiCgStabOptions::default())))
        .unwrap_err();
    assert!(
        has_krylov_error(&error, |e| matches!(
            e,
            KrylovError::NumericalBreakdown {
                solver: "bicgstab",
                ..
            }
        )),
        "{error:?}"
    );
    p.assert_solution(&p.solve(p.options.clone()).unwrap());
}

#[test]
fn local_solver_selection_rejects_changed_inactive_gmres_controls() {
    use tensor4all_treetn::GmresToleranceMode;
    let p = Problem::hpd(false);
    for solver in [
        LocalSolver::Cg(CgOptions::default()),
        LocalSolver::BiCgStab(BiCgStabOptions::default()),
    ] {
        let base = p.with_solver(solver);
        p.assert_solution(&p.solve(base.clone()).unwrap());
        for (options, name) in [
            (base.clone().with_gmres_tol(f64::NAN), "gmres_tol"),
            (base.clone().with_gmres_tol(1e-12), "gmres_tol"),
            (base.clone().with_gmres_restart_dim(0), "gmres_restart_dim"),
            (
                base.clone().with_gmres_max_restarts(7),
                "gmres_max_restarts",
            ),
            (
                base.clone()
                    .with_gmres_tolerance_mode(GmresToleranceMode::Absolute),
                "gmres_tolerance_mode",
            ),
        ] {
            let error = p.solve(options).unwrap_err();
            let message = format!("{name} configures LocalSolver::Gmres only");
            assert!(
                error
                    .source
                    .chain()
                    .any(|e| e.to_string().contains(&message)),
                "{message}: {error:?}"
            );
        }
    }
}

#[test]
fn local_solver_selection_validates_selected_options_before_shortcuts() {
    let p = Problem::hpd(false);
    for sweeps in [0, 1] {
        for identity_only in [false, true] {
            for solver in [
                LocalSolver::Cg(CgOptions {
                    rtol: f64::NAN,
                    ..Default::default()
                }),
                LocalSolver::BiCgStab(BiCgStabOptions {
                    atol: -1.0,
                    ..Default::default()
                }),
            ] {
                let mut options = p.with_solver(solver).with_nfullsweeps(sweeps);
                if identity_only {
                    options = options.with_coefficients(1.0, 0.0);
                }
                let error = p.solve(options).unwrap_err();
                assert!(
                    has_krylov_error(&error, |e| matches!(e, KrylovError::InvalidOptions { .. })),
                    "{error:?}"
                );
            }
        }
    }
}

#[test]
fn local_solver_selection_honors_native_zero_iteration_budgets() {
    let p = Problem::hpd(false);
    for solver in [
        LocalSolver::Cg(CgOptions {
            max_iter: 0,
            ..Default::default()
        }),
        LocalSolver::BiCgStab(BiCgStabOptions {
            max_iter: 0,
            ..Default::default()
        }),
    ] {
        let result = p.solve(p.with_solver(solver).with_nfullsweeps(1)).unwrap();
        assert!(!result.converged);
        assert_eq!(result.sweeps, 1);
        assert!(result.residual.unwrap() > 1e-3);
        assert_eq!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
    }
}

#[test]
fn local_solver_selection_checks_exact_guess_with_zero_native_budget() {
    let mut p = Problem::hpd(true);
    p.init = State::from_tensors(vec![p.expected.clone()], vec![0]).unwrap();
    for solver in [
        LocalSolver::Cg(CgOptions {
            max_iter: 0,
            ..Default::default()
        }),
        LocalSolver::BiCgStab(BiCgStabOptions {
            max_iter: 0,
            ..Default::default()
        }),
    ] {
        p.assert_solution(&p.solve(p.with_solver(solver)).unwrap());
    }
}

#[test]
fn local_solver_selection_keeps_local_and_sweep_tolerances_separate() {
    let p = Problem::hpd(false);
    for solver in [
        LocalSolver::Cg(CgOptions {
            rtol: 0.0,
            atol: 100.0,
            ..Default::default()
        }),
        LocalSolver::BiCgStab(BiCgStabOptions {
            rtol: 0.0,
            atol: 100.0,
            ..Default::default()
        }),
    ] {
        let result = p.solve(p.with_solver(solver).with_nfullsweeps(1)).unwrap();
        assert!(!result.converged);
        assert!(result.residual.unwrap() > 1e-3);
        assert_eq!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
    }
}

#[test]
fn local_solver_selection_reports_unallocatable_gmres_restart_dim() {
    let p = Problem::hpd(false);
    let error = p
        .solve(p.options.clone().with_gmres_restart_dim(usize::MAX - 1))
        .unwrap_err();
    assert!(
        has_krylov_error(&error, |e| matches!(
            e,
            KrylovError::InvalidOptions {
                solver: "gmres",
                ..
            }
        )),
        "{error:?}"
    );
}
