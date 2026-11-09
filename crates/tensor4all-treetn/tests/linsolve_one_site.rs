//! Dense references are restricted to these small ALS regression systems.
use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{
    AnyScalar, Canonical, DynIndex, FactorizeOptions, IdxTensor, IndexLike, SvdTruncationPolicy,
};
use tensor4all_treetn::{
    factorize_tensor_to_treetn_with, square_linsolve, IndexMapping, LinsolveOptions,
    ResidualEnrichment, SquareLinsolveResult, TreeTN, TreeTopology,
};

use crate::enrichment_step::{bond_dim, distance_from_00, enrich_once};

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
        let local = [z(3.0, 0.1), z(0.25, -0.2), z(-0.15, 0.3), z(2.0, -0.2)];
        let a0 = z(0.3, 0.2);
        let a1 = z(0.7, -0.1);
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

    /// The initial guess truncated by SVD to bonds of at most `max_bond_dim`.
    fn rank_limited_init(&self, max_bond_dim: usize) -> State {
        self.decompose(&self.init.contract_to_tensor().unwrap(), max_bond_dim)
    }

    /// `dense` decomposed on the problem's tree with bonds of at most `max_bond_dim`.
    fn decompose(&self, dense: &IdxTensor, max_bond_dim: usize) -> State {
        let sites = self
            .init
            .node_names()
            .into_iter()
            .map(|i| {
                (
                    i,
                    self.init.site_space(&i).unwrap().iter().cloned().collect(),
                )
            })
            .collect();
        factorize_tensor_to_treetn_with(
            dense,
            &TreeTopology::new(sites, self.edges.clone()),
            FactorizeOptions::svd().with_max_bond_dim(max_bond_dim),
            &0,
        )
        .unwrap()
    }

    fn bond_dims(&self, state: &State) -> Vec<usize> {
        self.edges
            .iter()
            .map(|(u, v)| {
                state
                    .bond_index(state.edge_between(u, v).unwrap())
                    .unwrap()
                    .dim()
            })
            .collect()
    }

    fn assert_dense_solution(&self, result: &SquareLinsolveResult<IdxTensor, usize>) {
        let actual = result.solution.contract_to_tensor().unwrap();
        let error = actual.sub(&self.expected).unwrap().maxabs().unwrap();
        assert!(error < 1e-8, "dense solution error {error}");
        assert!(result.converged, "residual {:?}", result.residual);
        assert!(result.residual.unwrap() < 1e-9);
    }

    fn assert_solution(&self, result: &SquareLinsolveResult<IdxTensor, usize>) {
        self.assert_dense_solution(result);
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

#[test]
fn enrichment_grows_an_insufficient_initial_rank() {
    for complex in [false, true] {
        for edges in [vec![(0, 1), (1, 2), (2, 3)], vec![(0, 1), (0, 2), (0, 3)]] {
            let p = Problem::new(&edges, complex);
            let init = p.rank_limited_init(1);
            assert!(p.bond_dims(&init).iter().all(|&dim| dim == 1));
            let without = p.solve(init.clone(), 0, p.options(4));
            assert!(!without.converged);
            let options = p.options(6).with_enrichment(ResidualEnrichment::default());
            let result = p.solve(init, 0, options);
            p.assert_dense_solution(&result);
            // Every bond grows, and never beyond the dimension of its smaller side.
            for (dim, &(u, v)) in p.bond_dims(&result.solution).into_iter().zip(&edges) {
                let side = |node: usize, away: usize| {
                    let mut seen = vec![node];
                    let mut stack = vec![node];
                    while let Some(n) = stack.pop() {
                        for &(a, b) in &edges {
                            for (from, to) in [(a, b), (b, a)] {
                                if from == n && to != away && !seen.contains(&to) {
                                    seen.push(to);
                                    stack.push(to);
                                }
                            }
                        }
                    }
                    1usize << seen.len()
                };
                let bound = side(u, v).min(side(v, u));
                assert!(
                    (2..=bound).contains(&dim),
                    "edge ({u}, {v}): {dim} > {bound}"
                );
            }
        }
    }
}

#[test]
fn enrichment_respects_per_step_and_total_bond_limits() {
    // The complex solution needs rank 4 across the middle bond.
    let p = Problem::new(&[(0, 1), (1, 2), (2, 3)], true);
    let init = p.rank_limited_init(1);
    let mut options = p
        .options(1)
        .with_enrichment(ResidualEnrichment::default().with_max_added(1));
    options.convergence_tol = None;
    // One full sweep visits every edge twice, adding at most one direction
    // per visit.
    let dims = p.bond_dims(&p.solve(init.clone(), 0, options.clone()).solution);
    assert!(dims.iter().all(|&dim| (2..=3).contains(&dim)), "{dims:?}");

    let capped = options.with_nfullsweeps(4).with_enrichment(
        ResidualEnrichment::default()
            .with_max_added(4)
            .with_max_bond_dim(2),
    );
    let result = p.solve(init, 0, capped);
    assert_eq!(p.bond_dims(&result.solution), vec![2, 2, 2]);
    assert!(result.residual.unwrap() > 1e-4);
}

#[test]
fn enrichment_skips_zero_rhs_and_single_node_problems() {
    let mut p = Problem::new(&[(0, 1), (1, 2)], true);
    p.rhs.scale_mut(AnyScalar::new_real(0.0)).unwrap();
    let init = p.rank_limited_init(1);
    let options = p.options(2).with_enrichment(ResidualEnrichment::default());
    let result = p.solve(init, 0, options);
    assert!(result.converged);
    assert_eq!(p.bond_dims(&result.solution), vec![1, 1]);
    assert!(
        result
            .solution
            .contract_to_tensor()
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-10
    );

    let p = Problem::new(&[], false);
    let options = p.options(2).with_enrichment(ResidualEnrichment::default());
    p.assert_solution(&p.solve(p.init.clone(), 0, options));
}

#[test]
fn enrichment_adds_nothing_once_the_residual_is_negligible() {
    // The real solution has rank 2 across every bond, below the middle
    // bond's bound of 4, so only the residual cutoff stops growth.
    let p = Problem::new(&[(0, 1), (1, 2), (2, 3)], false);
    let exact = p.decompose(&p.expected, 2);
    assert_eq!(p.bond_dims(&exact), vec![2, 2, 2]);
    let options = p
        .options(2)
        .with_enrichment(ResidualEnrichment::default().with_cutoff(1e-8));
    let result = p.solve(exact, 0, options);
    p.assert_dense_solution(&result);
    assert_eq!(p.bond_dims(&result.solution), vec![2, 2, 2]);
}

#[test]
fn enrichment_rejects_two_site_mode_and_invalid_settings() {
    let p = Problem::new(&[(0, 1)], false);
    for (options, expected) in [
        (
            p.options(1)
                .with_nsite(2)
                .with_enrichment(ResidualEnrichment::default()),
            "residual enrichment applies only to one-site ALS",
        ),
        (
            p.options(1)
                .with_enrichment(ResidualEnrichment::default().with_max_added(0)),
            "max_added must be at least 1",
        ),
        (
            p.options(1)
                .with_enrichment(ResidualEnrichment::default().with_cutoff(f64::NAN)),
            "cutoff must be finite and nonnegative",
        ),
        (
            p.options(1)
                .with_enrichment(ResidualEnrichment::default().with_cutoff(-1.0)),
            "cutoff must be finite and nonnegative",
        ),
        (
            p.options(1)
                .with_enrichment(ResidualEnrichment::default().with_max_bond_dim(0)),
            "max_bond_dim must be at least 1",
        ),
    ] {
        for identity_only in [false, true] {
            let options = if identity_only {
                options.clone().with_coefficients(1.0, 0.0)
            } else {
                options.clone()
            };
            let message = square_linsolve(
                &p.operator,
                &p.rhs,
                p.init.clone(),
                &0,
                options,
                Some(p.input.clone()),
                Some(p.output.clone()),
            )
            .unwrap_err()
            .to_string();
            assert!(message.contains(expected), "{message}");
        }
    }
}

#[test]
fn enrichment_leaves_the_state_unchanged_before_the_next_solve() {
    for small in [0.5, 1e-3] {
        let (state, sites) = enrich_once(small, ResidualEnrichment::default().with_max_added(2));
        assert_eq!(bond_dim(&state), 3);
        let distance = distance_from_00(&state, &sites);
        assert!(
            distance < 1e-12,
            "enrichment changed the state by {distance}"
        );
    }
}

#[test]
fn enrichment_cutoff_zero_keeps_tiny_residual_directions() {
    // A singular value of 1e-13 exceeds a zero cutoff, although the default
    // SVD policy (relative 1e-12) alone would drop it.
    let enrichment = ResidualEnrichment::default()
        .with_max_added(2)
        .with_cutoff(0.0);
    let (state, sites) = enrich_once(1e-13, enrichment);
    assert_eq!(bond_dim(&state), 3);
    assert!(distance_from_00(&state, &sites) < 1e-12);
}

#[test]
fn enrichment_with_an_overflowing_cutoff_adds_nothing() {
    // `cutoff * ||b_local||` overflows to infinity; a finite cutoff that large
    // keeps no direction rather than failing.
    let enrichment = ResidualEnrichment::default()
        .with_max_added(2)
        .with_cutoff(f64::MAX);
    let (state, sites) = enrich_once(0.5, enrichment);
    assert_eq!(bond_dim(&state), 1);
    assert!(distance_from_00(&state, &sites) < 1e-12);
}
