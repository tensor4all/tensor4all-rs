//! Dense references are restricted to these small ALS regression systems.
use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{
    AnyScalar, Canonical, DynIndex, FactorizeOptions, IdxTensor, IndexLike, SvdTruncationPolicy,
};
use tensor4all_treetn::{
    factorize_tensor_to_treetn_with, square_linsolve, IndexMapping, LinsolveOptions,
    SquareLinsolveResult, TreeTN, TreeTNOperationError, TreeTopology,
};

type State = TreeTN<IdxTensor, usize>;

fn tensor(indices: Vec<DynIndex>, values: Vec<Complex64>, complex: bool) -> IdxTensor {
    if complex {
        IdxTensor::from_dense(indices, values).unwrap()
    } else {
        IdxTensor::from_dense(indices, values.iter().map(|v| v.re).collect::<Vec<_>>()).unwrap()
    }
}

struct Problem {
    operator: State,
    rhs: State,
    init: State,
    expected: IdxTensor,
    input: HashMap<usize, IndexMapping<DynIndex>>,
    output: HashMap<usize, IndexMapping<DynIndex>>,
    edges: Vec<(usize, usize)>,
    a0: Complex64,
    a1: Complex64,
}

impl Problem {
    fn new(edges: &[(usize, usize)], complex: bool) -> Self {
        let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
        Self::with_local_operator(
            edges,
            complex,
            [z(3.0, 0.1), z(0.25, -0.2), z(-0.15, 0.3), z(2.0, -0.2)],
            z(0.3, 0.2),
            z(0.7, -0.1),
        )
    }

    fn with_local_operator(
        edges: &[(usize, usize)],
        complex: bool,
        local: [Complex64; 4],
        a0: Complex64,
        a1: Complex64,
    ) -> Self {
        let n = edges
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .max()
            .map_or(1, |v| v + 1);
        let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
        let topology = TreeTopology::new(
            sites
                .iter()
                .enumerate()
                .map(|(i, s)| (i, vec![s.clone()]))
                .collect(),
            edges.to_vec(),
        );
        let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
        let values: Vec<_> = (0..1 << n)
            .map(|i| z((i + 1) as f64 / 3.0, (i as f64).cos()))
            .collect();
        let mut b = vec![z(0.0, 0.0); values.len()];
        for (row, b_row) in b.iter_mut().enumerate() {
            *b_row = a0 * values[row];
            for (col, value) in values.iter().enumerate() {
                let element: Complex64 = (0..n)
                    .map(|k| local[((row >> k) & 1) + 2 * ((col >> k) & 1)])
                    .product();
                *b_row += a1 * element * value;
            }
        }
        let expected = tensor(sites.clone(), values, complex);
        let rhs_dense = tensor(sites.clone(), b, complex);
        let init_dense = tensor(
            sites,
            (0..1 << n)
                .map(|i| z((i as f64 + 0.4).sin(), (i as f64 + 0.7).cos()))
                .collect(),
            complex,
        );
        let decompose = |dense: &IdxTensor| {
            factorize_tensor_to_treetn_with(
                dense,
                &topology,
                FactorizeOptions::qr().with_canonical(Canonical::Left),
                &0,
            )
            .unwrap()
        };
        let rhs = decompose(&rhs_dense);
        let init = decompose(&init_dense);
        let mut operator = State::new();
        let bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(1)).collect();
        let mut input = HashMap::new();
        let mut output = HashMap::new();
        for node in 0..n {
            let site = topology.nodes[&node][0].clone();
            let s_in = site.sim();
            let s_out = s_in.prime();
            let mut indices = vec![s_out.clone(), s_in.clone()];
            for (i, &(u, v)) in edges.iter().enumerate() {
                if u == node || v == node {
                    indices.push(bonds[i].clone());
                }
            }
            operator
                .add_tensor(node, tensor(indices, local.to_vec(), complex))
                .unwrap();
            input.insert(
                node,
                IndexMapping {
                    true_index: site.clone(),
                    internal_index: s_in,
                },
            );
            output.insert(
                node,
                IndexMapping {
                    true_index: site,
                    internal_index: s_out,
                },
            );
        }
        for (i, &(u, v)) in edges.iter().enumerate() {
            operator
                .connect(
                    operator.node_index(&u).unwrap(),
                    &bonds[i],
                    operator.node_index(&v).unwrap(),
                    &bonds[i],
                )
                .unwrap();
        }
        Self {
            operator,
            rhs,
            init,
            expected,
            input,
            output,
            edges: edges.to_vec(),
            a0,
            a1,
        }
    }

    fn options(&self, sweeps: usize) -> LinsolveOptions {
        LinsolveOptions::new(sweeps)
            .with_nsite(1)
            .with_coefficients(
                if self.a0.im == 0.0 {
                    AnyScalar::new_real(self.a0.re)
                } else {
                    self.a0.into()
                },
                if self.a1.im == 0.0 {
                    AnyScalar::new_real(self.a1.re)
                } else {
                    self.a1.into()
                },
            )
            .with_gmres_tol(1e-12)
            .with_gmres_restart_dim(20)
            .with_convergence_tol(1e-9)
    }

    /// `options(sweeps)` with `solver`; GMRES keeps the tighter `gmres_*`
    /// settings, other solvers get the defaults because they reject changed
    /// GMRES controls.
    fn options_with_solver(
        &self,
        sweeps: usize,
        solver: tensor4all_treetn::LocalSolver,
    ) -> LinsolveOptions {
        let options = self.options(sweeps);
        let options = if matches!(solver, tensor4all_treetn::LocalSolver::Gmres) {
            options
        } else {
            let defaults = LinsolveOptions::default();
            options
                .with_gmres_tol(defaults.gmres_tol)
                .with_gmres_restart_dim(defaults.gmres_restart_dim)
        };
        options.with_local_solver(solver)
    }

    fn solve(
        &self,
        init: State,
        center: usize,
        options: LinsolveOptions,
    ) -> SquareLinsolveResult<IdxTensor, usize> {
        square_linsolve(
            &self.operator,
            &self.rhs,
            init,
            &center,
            options,
            Some(self.input.clone()),
            Some(self.output.clone()),
        )
        .unwrap()
    }

    fn assert_solution(&self, result: &SquareLinsolveResult<IdxTensor, usize>) {
        let actual = result.solution.contract_to_tensor().unwrap();
        let error = actual.sub(&self.expected).unwrap().maxabs().unwrap();
        assert!(error < 1e-8, "dense solution error {error}");
        assert!(result.converged, "residual {:?}", result.residual);
        assert!(result.residual.unwrap() < 1e-9);
        for &(u, v) in &self.edges {
            let before = self
                .init
                .bond_index(self.init.edge_between(&u, &v).unwrap())
                .unwrap()
                .dim();
            let after = result
                .solution
                .bond_index(result.solution.edge_between(&u, &v).unwrap())
                .unwrap()
                .dim();
            assert_eq!(after, before);
        }
    }
}

#[test]
fn one_site_als_real_and_complex_chains_and_branching_trees() {
    for complex in [false, true] {
        for edges in [vec![(0, 1), (1, 2)], vec![(0, 1), (0, 2), (0, 3)]] {
            let p = Problem::new(&edges, complex);
            for center in [0, edges.last().unwrap().1] {
                let result = p.solve(p.init.clone(), center, p.options(4));
                p.assert_solution(&result);
            }
        }
    }
}

#[test]
fn one_site_als_single_node_and_zero_initial_guess() {
    for complex in [false, true] {
        let p = Problem::new(&[], complex);
        let mut init = p.init.clone();
        init.scale_mut(AnyScalar::new_real(0.0)).unwrap();
        p.assert_solution(&p.solve(init, 0, p.options(2)));
    }
}

#[test]
fn one_site_als_zero_initial_guess_chain() {
    let p = Problem::new(&[(0, 1), (1, 2)], true);
    let mut init = p.init.clone();
    init.scale_mut(AnyScalar::new_real(0.0)).unwrap();
    p.assert_solution(&p.solve(init, 0, p.options(4)));
}

#[test]
fn one_site_als_exact_guess_and_repeated_sweeps_keep_environments_fresh() {
    let p = Problem::new(&[(0, 1), (0, 2), (0, 3)], true);
    let first = p.solve(p.init.clone(), 0, p.options(2));
    p.assert_solution(&first);
    let result = p.solve(first.solution, 0, p.options(3));
    p.assert_solution(&result);
    assert_eq!(result.sweeps, 1);
    let mut options = p.options(3);
    options.convergence_tol = None;
    let result = p.solve(p.init.clone(), 0, options);
    assert_eq!(result.sweeps, 3);
    assert!(!result.converged);
    assert!(
        result
            .solution
            .contract_to_tensor()
            .unwrap()
            .sub(&p.expected)
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-8
    );
    assert!(result.residual.unwrap() < 1e-9);
}

#[test]
fn one_site_als_zero_sweeps_and_skipped_residual() {
    let p = Problem::new(&[(0, 1)], false);
    let before = p.init.contract_to_tensor().unwrap();
    let result = p.solve(p.init.clone(), 0, p.options(0));
    assert_eq!(result.sweeps, 0);
    assert!(!result.converged);
    assert!(result.residual.unwrap() > 1e-3);
    assert!(
        result
            .solution
            .contract_to_tensor()
            .unwrap()
            .sub(&before)
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-12
    );
    let mut options = p.options(1).with_residual_check(false);
    options.convergence_tol = None;
    let result = p.solve(p.init.clone(), 0, options);
    assert!(result.residual.is_none());
    assert!(!result.converged);
    assert!(
        result
            .solution
            .contract_to_tensor()
            .unwrap()
            .sub(&p.expected)
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-8
    );
}

#[test]
fn one_site_als_zero_rhs_from_nonzero_guess() {
    for complex in [false, true] {
        let mut p = Problem::new(&[(0, 1), (1, 2)], complex);
        p.rhs.scale_mut(AnyScalar::new_real(0.0)).unwrap();
        let result = p.solve(p.init.clone(), 0, p.options(2));
        assert!(result.converged, "zero RHS residual {:?}", result.residual);
        assert!(result.residual.unwrap() < 1e-10);
        assert!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-10
        );
    }
}

#[test]
fn one_site_als_does_not_grow_an_insufficient_initial_rank() {
    let mut p = Problem::new(&[(0, 1)], false);
    let dense = p.init.contract_to_tensor().unwrap();
    let sites: Vec<_> = (0..2)
        .map(|i| (i, p.init.site_space(&i).unwrap().iter().cloned().collect()))
        .collect();
    p.init = factorize_tensor_to_treetn_with(
        &dense,
        &TreeTopology::new(sites.into_iter().collect(), p.edges.clone()),
        FactorizeOptions::svd().with_max_bond_dim(1),
        &0,
    )
    .unwrap();
    let result = p.solve(p.init.clone(), 0, p.options(3));
    assert!(!result.converged);
    assert_eq!(result.sweeps, 3);
    assert!(result.residual.unwrap() > 1e-4);
    assert_eq!(
        result
            .solution
            .bond_index(result.solution.edge_between(&0, &1).unwrap())
            .unwrap()
            .dim(),
        1
    );
}

#[test]
fn one_site_als_validates_mode_and_truncation_before_identity_shortcut() {
    let p = Problem::new(&[(0, 1)], false);
    assert_eq!(LinsolveOptions::default().nsite, 2);
    for options in [
        p.options(1).with_nsite(0),
        p.options(1).with_nsite(3),
        p.options(1).with_max_bond_dim(2),
        p.options(1).with_svd_policy(SvdTruncationPolicy::new(0.0)),
    ] {
        for identity_only in [false, true] {
            let options = if identity_only {
                options.clone().with_coefficients(1.0, 0.0)
            } else {
                options.clone()
            };
            let err = square_linsolve(
                &p.operator,
                &p.rhs,
                p.init.clone(),
                &0,
                options,
                Some(p.input.clone()),
                Some(p.output.clone()),
            )
            .unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("nsite") || message.contains("one-site"),
                "{message}"
            );
        }
    }
}

#[test]
fn one_site_als_identity_term_only_keeps_initial_bond_spaces() {
    let mut p = Problem::new(&[(0, 1)], false);
    let sites = (0..2)
        .map(|i| (i, p.init.site_space(&i).unwrap().iter().cloned().collect()))
        .collect();
    p.init = factorize_tensor_to_treetn_with(
        &p.init.contract_to_tensor().unwrap(),
        &TreeTopology::new(sites, p.edges.clone()),
        FactorizeOptions::svd().with_max_bond_dim(1),
        &0,
    )
    .unwrap();
    for zero_operator in [false, true] {
        if zero_operator {
            p.operator.scale_mut(AnyScalar::new_real(0.0)).unwrap();
        }
        let options = p
            .options(3)
            .with_coefficients(2.0, if zero_operator { 1.0 } else { 0.0 });
        let result = p.solve(p.init.clone(), 0, options);
        assert_eq!(
            result
                .solution
                .bond_index(result.solution.edge_between(&0, &1).unwrap())
                .unwrap()
                .dim(),
            1
        );
        assert!(!result.converged);
        assert!(result.residual.unwrap() > 1e-4);
    }
    let err = square_linsolve(
        &p.operator,
        &p.rhs,
        p.init.clone(),
        &0,
        p.options(1).with_coefficients(0.0, 1.0),
        Some(p.input.clone()),
        Some(p.output.clone()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("both zero"));
}

#[test]
fn one_site_als_absolute_local_tolerance_and_required_residual() {
    let p = Problem::new(&[(0, 1), (1, 2)], true);
    let result = p.solve(
        p.init.clone(),
        0,
        p.options(3)
            .with_gmres_absolute_tolerance()
            .with_residual_check(false),
    );
    p.assert_solution(&result);
    assert!(result.residual.is_some());
}

#[test]
fn one_site_als_rejects_empty_network_and_missing_center() {
    let empty = State::new();
    let error = square_linsolve(
        &empty,
        &empty,
        empty.clone(),
        &0,
        LinsolveOptions::new(1).with_nsite(1),
        None,
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("nonempty"));
    let p = Problem::new(&[], false);
    assert!(square_linsolve(
        &p.operator,
        &p.rhs,
        p.init.clone(),
        &1,
        p.options(1),
        Some(p.input.clone()),
        Some(p.output.clone())
    )
    .is_err());
}

#[test]
fn one_site_als_rank_two_chain_requires_alternating_local_solves() {
    let mut p = Problem::new(&[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], false);
    let sites = (0..6)
        .map(|i| (i, p.init.site_space(&i).unwrap().iter().cloned().collect()))
        .collect();
    p.init = factorize_tensor_to_treetn_with(
        &p.init.contract_to_tensor().unwrap(),
        &TreeTopology::new(sites, p.edges.clone()),
        FactorizeOptions::svd().with_max_bond_dim(2),
        &0,
    )
    .unwrap();
    let result = p.solve(
        p.init.clone(),
        0,
        p.options(30)
            .with_convergence_tol(1e-12)
            .with_gmres_tol(1e-14),
    );
    p.assert_solution(&result);
    assert!(
        result.sweeps > 1,
        "regression must exercise alternating solves"
    );
}

#[test]
fn small_nonzero_operator_is_not_treated_as_zero() {
    let mut p = Problem::new(&[(0, 1)], false);
    let scale = 1.0e-20;
    p.operator.scale_mut(AnyScalar::new_real(scale)).unwrap();
    for nsite in [1, 2] {
        let options = p
            .options(20)
            .with_nsite(nsite)
            .with_coefficients(p.a0.re, p.a1.re / scale);
        p.assert_solution(&p.solve(p.init.clone(), 0, options));
    }
}

fn local_solver_choices() -> Vec<tensor4all_treetn::LocalSolver> {
    use tensor4all_core::krylov::{BiCgStabOptions, CgOptions};
    use tensor4all_treetn::LocalSolver;
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

#[test]
fn local_solver_selection_supports_one_and_two_site_tree_sweeps() {
    for complex in [false, true] {
        for edges in [vec![(0, 1), (1, 2)], vec![(0, 1), (0, 2), (0, 3)]] {
            let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
            let p = Problem::with_local_operator(
                &edges,
                complex,
                [z(3.0, 0.0), z(0.25, 0.4), z(0.25, -0.4), z(2.0, 0.0)],
                z(0.3, 0.0),
                z(0.7, 0.0),
            );
            for nsite in [1, 2] {
                for solver in local_solver_choices() {
                    let options = p.options_with_solver(4, solver).with_nsite(nsite);
                    p.assert_solution(&p.solve(p.init.clone(), edges.last().unwrap().1, options));
                }
            }
        }
    }
}

#[test]
fn local_solver_selection_bicgstab_handles_general_tree_operators() {
    use tensor4all_core::krylov::BiCgStabOptions;
    use tensor4all_treetn::LocalSolver;
    for complex in [false, true] {
        let p = Problem::new(&[(0, 1), (0, 2), (0, 3)], complex);
        for nsite in [1, 2] {
            let options = p
                .options_with_solver(
                    4,
                    LocalSolver::BiCgStab(BiCgStabOptions {
                        rtol: 1e-12,
                        ..Default::default()
                    }),
                )
                .with_nsite(nsite);
            p.assert_solution(&p.solve(p.init.clone(), 0, options));
        }
    }
}

#[test]
fn local_solver_selection_respects_zero_budget_for_identity_only_sweeps() {
    use tensor4all_core::krylov::{BiCgStabOptions, CgOptions};
    use tensor4all_treetn::LocalSolver;
    let p = Problem::new(&[(0, 1)], false);
    let before = p.init.contract_to_tensor().unwrap();
    let closed_form = p
        .rhs
        .contract_to_tensor()
        .unwrap()
        .scale(AnyScalar::new_real(0.5))
        .unwrap();
    let zero_budget_solvers = || {
        [
            LocalSolver::Gmres,
            LocalSolver::Cg(CgOptions {
                max_iter: 0,
                ..Default::default()
            }),
            LocalSolver::BiCgStab(BiCgStabOptions {
                max_iter: 0,
                ..Default::default()
            }),
        ]
    };
    // One-site ALS sweeps an identity-only system, so native budgets apply.
    for solver in zero_budget_solvers().into_iter().skip(1) {
        let options = p
            .options_with_solver(1, solver)
            .with_nsite(1)
            .with_coefficients(2.0, 0.0);
        let result = p.solve(p.init.clone(), 0, options);
        assert!(!result.converged);
        assert_eq!(result.sweeps, 1);
        assert!(result.residual.unwrap() > 1e-3);
        assert!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .sub(&before)
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
    }
    // Two-site MALS returns the same closed form whichever solver is selected.
    for solver in zero_budget_solvers() {
        let options = p
            .options_with_solver(1, solver)
            .with_nsite(2)
            .with_coefficients(2.0, 0.0);
        let result = p.solve(p.init.clone(), 0, options);
        assert!(
            result
                .solution
                .contract_to_tensor()
                .unwrap()
                .sub(&closed_form)
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
    }
}

#[test]
fn local_solver_selection_zero_and_tiny_rhs_across_sweep_modes() {
    for complex in [false, true] {
        for scale in [0.0, 1e-20] {
            let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
            let mut p = Problem::with_local_operator(
                &[(0, 1)],
                complex,
                [z(3.0, 0.0), z(0.25, 0.2), z(0.25, -0.2), z(2.0, 0.0)],
                z(0.3, 0.0),
                z(0.7, 0.0),
            );
            p.rhs.scale_mut(AnyScalar::new_real(scale)).unwrap();
            let expected = p.expected.scale(AnyScalar::new_real(scale)).unwrap();
            if scale != 0.0 {
                p.init.scale_mut(AnyScalar::new_real(0.0)).unwrap();
            }
            for nsite in [1, 2] {
                for solver in local_solver_choices() {
                    let options = p.options_with_solver(2, solver).with_nsite(nsite);
                    let result = p.solve(p.init.clone(), 0, options);
                    assert!(
                        result.converged,
                        "{complex} {scale} {nsite}: {:?}",
                        result.residual
                    );
                    let error = result
                        .solution
                        .contract_to_tensor()
                        .unwrap()
                        .sub(&expected)
                        .unwrap()
                        .maxabs()
                        .unwrap();
                    assert!(error <= 1e-8 * scale, "{complex} {scale} {nsite}: {error}");
                }
            }
        }
    }
}

#[test]
fn a_single_index_mapping_is_rejected_before_the_identity_shortcut() {
    // Two-site MALS with a1 = 0 takes the identity-only closed form, which does
    // not use the mappings when no residual is requested; the mismatch must
    // still be reported.
    let p = Problem::new(&[(0, 1)], false);
    let options = p
        .options(1)
        .with_nsite(2)
        .with_coefficients(2.0, 0.0)
        .with_residual_check(false);
    assert!(options.convergence_tol.is_some());
    let options = LinsolveOptions {
        convergence_tol: None,
        ..options
    };
    for (input, output) in [
        (Some(p.input.clone()), None),
        (None, Some(p.output.clone())),
    ] {
        let error = square_linsolve(
            &p.operator,
            &p.rhs,
            p.init.clone(),
            &0,
            options.clone(),
            input,
            output,
        )
        .unwrap_err();
        assert!(
            matches!(error, TreeTNOperationError::InvalidConfiguration { .. }),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("must both be Some or both be None"),
            "{error}"
        );
    }
}

#[test]
fn non_finite_coefficients_are_rejected_before_the_identity_shortcut() {
    // Two-site MALS with a1 = 0 and no residual takes the closed form
    // rhs / a0 without calling a local solver, so the coefficients must be
    // checked by option validation.
    let p = Problem::new(&[(0, 1)], false);
    for a0 in [f64::NAN, f64::INFINITY] {
        let options = LinsolveOptions {
            convergence_tol: None,
            ..p.options(1)
                .with_nsite(2)
                .with_coefficients(a0, 0.0)
                .with_residual_check(false)
        };
        let error = square_linsolve(
            &p.operator,
            &p.rhs,
            p.init.clone(),
            &0,
            options,
            Some(p.input.clone()),
            Some(p.output.clone()),
        )
        .unwrap_err();
        assert!(
            matches!(error, TreeTNOperationError::InvalidConfiguration { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("a0 must be finite"), "{error}");
    }
}

#[test]
fn identity_shortcut_divides_by_a_tiny_or_huge_a0_without_overflow() {
    // Two-site MALS with a1 = 0 and no residual returns rhs / a0 in closed
    // form. For these coefficients 1 / a0, or |a0|^2 for complex a0, leaves
    // the f64 range although rhs / a0 is of order 2^70 or 2^-18.
    // Split the exponent: 2^-1070 itself is subnormal but 1 / 2^1070 overflows.
    let pow2 = |k: i32| 2.0_f64.powi(k / 2) * 2.0_f64.powi(k - k / 2);
    // (complex problem, a0, rhs scale, rhs scale / a0)
    let cases = [
        (
            false,
            Complex64::new(3.0 * pow2(-1070), 0.0),
            pow2(-1000),
            Complex64::new(pow2(70) / 3.0, 0.0),
        ),
        (
            true,
            Complex64::new(3.0, 4.0) * pow2(-1070),
            pow2(-1000),
            Complex64::new(3.0, -4.0) * (pow2(70) / 25.0),
        ),
        (
            true,
            Complex64::new(3.0, 4.0) * pow2(1018),
            pow2(1000),
            Complex64::new(3.0, -4.0) * (pow2(-18) / 25.0),
        ),
    ];
    for (complex, a0, rhs_scale, quotient) in cases {
        let mut p = Problem::new(&[(0, 1)], complex);
        let expected = p
            .rhs
            .contract_to_tensor()
            .unwrap()
            .scale(AnyScalar::new_complex(quotient.re, quotient.im))
            .unwrap();
        p.rhs.scale_mut(AnyScalar::new_real(rhs_scale)).unwrap();
        let options = LinsolveOptions {
            convergence_tol: None,
            ..p.options(1)
                .with_nsite(2)
                .with_coefficients(AnyScalar::new_complex(a0.re, a0.im), 0.0)
                .with_residual_check(false)
        };
        let result = p.solve(p.init.clone(), 0, options);
        assert_eq!(result.sweeps, 0);
        assert!(result.residual.is_none());
        let error = result
            .solution
            .contract_to_tensor()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap();
        let scale = expected.maxabs().unwrap();
        assert!(
            error.is_finite() && error <= 1e-14 * scale,
            "{a0:?}: {error} vs {scale}"
        );
    }
}
