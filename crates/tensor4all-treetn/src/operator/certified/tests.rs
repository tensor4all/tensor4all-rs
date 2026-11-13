use super::*;
use crate::operator::IndexMapping;
use crate::random::{random_treetn, LinkSpace};
use crate::site_index_network::SiteIndexNetwork;
use crate::{apply_linear_operator, ApplyOptions};
use num_complex::Complex64;
use rand::SeedableRng;
use std::collections::HashMap;
use tensor4all_core::{
    Canonical, DynIndex, FactorizeAlg, IdxTensor, TensorContractionLike, TensorFactorizationLike,
};

type Network = SiteIndexNetwork<usize, DynIndex>;

/// The chain `0 - 1 - ... - (n - 1)` with one binary site per node.
fn chain(n: usize) -> Network {
    let mut network = Network::new();
    for node in 0..n {
        network
            .add_node(node, HashSet::from([DynIndex::new_dyn(2)]))
            .unwrap();
    }
    for node in 1..n {
        network.add_edge(&(node - 1), &node).unwrap();
    }
    network
}

fn site(network: &Network, node: usize) -> DynIndex {
    network
        .site_space(&node)
        .and_then(|space| space.iter().next().cloned())
        .unwrap()
}

/// A random chain state whose bond `k` carries the weights `decay^i`, so its
/// Schmidt spectra decay and moderate budgets truncate.
fn random_state(
    sites: &Network,
    complex: bool,
    bond: usize,
    seed: u64,
) -> TreeTN<IdxTensor, usize> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let state = if complex {
        random_treetn::<Complex64, _, _>(&mut rng, sites, LinkSpace::uniform(bond))
    } else {
        random_treetn::<f64, _, _>(&mut rng, sites, LinkSpace::uniform(bond))
    }
    .unwrap();
    let decay = 0.2_f64;
    let mut nodes: Vec<usize> = state.node_names();
    nodes.sort();
    let mut tensors = Vec::with_capacity(nodes.len());
    for &node in &nodes {
        let mut tensor = state
            .tensor(state.node_index(&node).unwrap())
            .unwrap()
            .clone();
        // Weight the bond towards the next node, if the state has one there.
        if let Some(edge) = state.edge_between(&node, &(node + 1)) {
            let link = state.bond_index(edge).unwrap().clone();
            let primed = link.sim();
            let dim = link.dim();
            let mut weights = vec![0.0; dim * dim];
            for i in 0..dim {
                weights[i + dim * i] = decay.powi(i as i32);
            }
            let diagonal =
                IdxTensor::from_dense(vec![link.clone(), primed.clone()], weights).unwrap();
            tensor = IdxTensor::contract(&[&tensor, &diagonal])
                .unwrap()
                .replaceind(&primed, &link)
                .unwrap();
        }
        tensors.push(tensor);
    }
    TreeTN::from_tensors(tensors, nodes).unwrap()
}

/// A random operator on `nodes` with MPO `edges` and bond `bond`, mapping each
/// node's state site index to itself.
fn random_operator(
    state: &Network,
    nodes: &[usize],
    edges: &[(usize, usize)],
    complex: bool,
    bond: usize,
    seed: u64,
) -> LinearOperator<IdxTensor, usize> {
    let mut mpo_sites = Network::new();
    let mut inputs = HashMap::new();
    let mut outputs = HashMap::new();
    for &node in nodes {
        let true_index = site(state, node);
        let (input, output) = (true_index.sim(), true_index.sim());
        mpo_sites
            .add_node(node, HashSet::from([input.clone(), output.clone()]))
            .unwrap();
        inputs.insert(
            node,
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: input,
            },
        );
        outputs.insert(
            node,
            IndexMapping {
                true_index,
                internal_index: output,
            },
        );
    }
    for &(a, b) in edges {
        mpo_sites.add_edge(&a, &b).unwrap();
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mpo = if complex {
        random_treetn::<Complex64, _, _>(&mut rng, &mpo_sites, LinkSpace::uniform(bond))
    } else {
        random_treetn::<f64, _, _>(&mut rng, &mpo_sites, LinkSpace::uniform(bond))
    }
    .unwrap();
    LinearOperator::new(mpo, inputs, outputs)
}

fn chain_edges(nodes: &[usize]) -> Vec<(usize, usize)> {
    nodes.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

/// Smallest rank (at least one) whose discarded Schmidt weight at the cut
/// after node `cut` is at most `threshold`.
fn rank_for(exact: &IdxTensor, sites: &Network, cut: usize, threshold: f64) -> usize {
    let left: Vec<DynIndex> = (0..=cut).map(|node| site(sites, node)).collect();
    let spectrum = exact
        .factorize_full_rank(&left, FactorizeAlg::SVD, Canonical::Left)
        .unwrap()
        .singular_values
        .unwrap();
    let mut tail = 0.0_f64;
    let mut rank = 0;
    for (position, value) in spectrum.iter().enumerate().rev() {
        tail += value * value;
        if tail.sqrt() > threshold {
            rank = position + 1;
            break;
        }
    }
    rank.max(1)
}

fn bond_between(network: &TreeTN<IdxTensor, usize>, a: usize, b: usize) -> usize {
    network
        .edge_between(&a, &b)
        .and_then(|edge| network.bond_index(edge))
        .map(|index| index.dim())
        .unwrap()
}

fn dense_error(network: &TreeTN<IdxTensor, usize>, exact: &IdxTensor) -> f64 {
    network
        .to_dense()
        .unwrap()
        .sub(exact)
        .unwrap()
        .norm()
        .unwrap()
}

/// Apply `operator` to `state` with budgets relative to `||A x||` and check,
/// against the exact application, that
/// - the bound is at most the budget,
/// - the true error is the bound up to roundoff (the step errors are
///   orthogonal, so the identity is exact in exact arithmetic),
/// - every kept bond lies between the rank any approximation within the budget
///   needs and the rank at the smallest per-step share.
fn check_against_exact(
    sites: &Network,
    state: &TreeTN<IdxTensor, usize>,
    operator: &LinearOperator<IdxTensor, usize>,
) {
    let exact = apply_linear_operator(operator, state, ApplyOptions::naive())
        .unwrap()
        .to_dense()
        .unwrap();
    let scale = exact.norm().unwrap();
    let cuts = sites.node_count() - 1;
    let mut truncations = 0;
    for relative in [1e-1, 1e-3, 1e-6] {
        let budget = relative * scale;
        let certified =
            apply_linear_operator_certified(operator, state, CertifiedApplyOptions::new(budget))
                .unwrap();
        let bound = certified.bound();
        assert!(bound <= budget, "bound {bound} above budget {budget}");
        assert_eq!(certified.step_bounds().len(), cuts);
        assert!(certified.estimated_bytes() >= certified.factor_bytes());

        let error = dense_error(certified.network(), &exact);
        assert!(
            (bound - error).abs() <= 1e-8 * bound + 1e-12 * scale,
            "true error {error} against bound {bound} (budget {budget})"
        );

        let share = budget / (cuts as f64).sqrt();
        for cut in 0..cuts {
            let kept = bond_between(certified.network(), cut, cut + 1);
            let necessary = rank_for(&exact, sites, cut, budget);
            let per_share = rank_for(&exact, sites, cut, 0.99 * share);
            assert!(
                necessary <= kept && kept <= per_share,
                "cut {cut}: kept {kept}, necessary {necessary}, per-share {per_share}"
            );
            if kept < rank_for(&exact, sites, cut, 0.0) {
                truncations += 1;
            }
        }
    }
    // The decaying states make the budgets truncate on several cuts.
    assert!(truncations >= 2, "only {truncations} truncated cuts");
}

#[test]
fn certified_application_matches_the_exact_one_within_its_bound() {
    let sites = chain(6);
    let nodes: Vec<usize> = (0..6).collect();
    for complex in [false, true] {
        let state = random_state(&sites, complex, 4, 11);
        let operator = random_operator(&sites, &nodes, &chain_edges(&nodes), complex, 3, 12);
        check_against_exact(&sites, &state, &operator);
    }
}

#[test]
fn certified_application_of_a_partial_operator_fills_the_gaps() {
    let sites = chain(6);
    // Support {1, 2} and {4}, joined across the gap at 3.
    for complex in [false, true] {
        let state = random_state(&sites, complex, 3, 21);
        let operator = random_operator(&sites, &[1, 2, 4], &[(1, 2), (2, 4)], complex, 3, 22);
        check_against_exact(&sites, &state, &operator);
    }
}

#[test]
fn a_real_state_under_a_complex_operator_is_certified() {
    let sites = chain(5);
    let nodes: Vec<usize> = (0..5).collect();
    let state = random_state(&sites, false, 3, 31);
    let operator = random_operator(&sites, &nodes, &chain_edges(&nodes), true, 2, 32);
    check_against_exact(&sites, &state, &operator);
}

#[test]
fn a_single_site_is_applied_exactly_with_a_zero_bound() {
    let sites = chain(1);
    let state = random_state(&sites, false, 1, 41);
    let operator = random_operator(&sites, &[0], &[], false, 1, 42);
    let certified =
        apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(0.0))
            .unwrap();
    assert_eq!(certified.bound(), 0.0);
    assert!(certified.step_bounds().is_empty());
    let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive())
        .unwrap()
        .to_dense()
        .unwrap();
    assert!(dense_error(certified.network(), &exact) <= 1e-14 * exact.norm().unwrap());
}

#[test]
fn an_unreachable_budget_is_reported_not_relaxed() {
    let sites = chain(5);
    let nodes: Vec<usize> = (0..5).collect();
    let state = random_state(&sites, false, 4, 51);
    let operator = random_operator(&sites, &nodes, &chain_edges(&nodes), false, 3, 52);
    let error = apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(0.0))
        .unwrap_err();
    let CertifiedApplyError::BudgetUnreachable {
        budget,
        step_bound,
        share,
        ..
    } = error
    else {
        panic!("expected BudgetUnreachable, got {error}");
    };
    assert_eq!(budget, 0.0);
    assert_eq!(share, 0.0);
    assert!(step_bound > 0.0);
}

#[test]
fn the_memory_gate_refuses_an_estimate_above_the_limit() {
    let sites = chain(5);
    let nodes: Vec<usize> = (0..5).collect();
    let state = random_state(&sites, false, 4, 61);
    let operator = random_operator(&sites, &nodes, &chain_edges(&nodes), false, 3, 62);
    let passing =
        apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(1.0))
            .unwrap();
    let estimate = passing.estimated_bytes();
    assert!(estimate > passing.factor_bytes());
    // A limit one byte below the estimate is refused with that estimate.
    let error = apply_linear_operator_certified(
        &operator,
        &state,
        CertifiedApplyOptions::new(1.0).with_memory_limit(estimate - 1),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CertifiedApplyError::MemoryLimit { estimated_bytes, limit_bytes }
            if estimated_bytes == estimate && limit_bytes == estimate - 1
    ));
    // The limit itself passes.
    apply_linear_operator_certified(
        &operator,
        &state,
        CertifiedApplyOptions::new(1.0).with_memory_limit(estimate),
    )
    .unwrap();
}

#[test]
fn the_estimate_counts_complex_entries_twice() {
    let sites = chain(4);
    let nodes: Vec<usize> = (0..4).collect();
    let estimate = |complex: bool| {
        let state = random_state(&sites, complex, 3, 71);
        let operator = random_operator(&sites, &nodes, &chain_edges(&nodes), complex, 2, 72);
        apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(1e3))
            .unwrap()
            .estimated_bytes()
    };
    assert_eq!(estimate(true), 2 * estimate(false));
}

#[test]
fn a_branched_state_is_unsupported() {
    // A star: 1 is joined to 0, 2 and 3.
    let mut sites = Network::new();
    for node in 0..4 {
        sites
            .add_node(node, HashSet::from([DynIndex::new_dyn(2)]))
            .unwrap();
    }
    for leaf in [0, 2, 3] {
        sites.add_edge(&1, &leaf).unwrap();
    }
    let state = random_state(&sites, false, 2, 81);
    let operator = random_operator(&sites, &[1], &[], false, 1, 82);
    let error = apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(1.0))
        .unwrap_err();
    assert!(matches!(
        error,
        CertifiedApplyError::UnsupportedTopology { .. }
    ));
}

#[test]
fn a_full_operator_with_other_edges_is_unsupported() {
    let sites = chain(4);
    let state = random_state(&sites, false, 2, 91);
    // A star MPO around node 1 on the chain's nodes.
    let operator = random_operator(
        &sites,
        &[0, 1, 2, 3],
        &[(1, 0), (1, 2), (1, 3)],
        false,
        2,
        92,
    );
    let error = apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(1.0))
        .unwrap_err();
    assert!(matches!(
        error,
        CertifiedApplyError::UnsupportedTopology { .. }
    ));
}

#[test]
fn invalid_budgets_and_foreign_nodes_are_rejected() {
    let sites = chain(3);
    let state = random_state(&sites, false, 2, 101);
    let operator = random_operator(&sites, &[0, 1], &[(0, 1)], false, 2, 102);
    for budget in [-1.0, f64::NAN, f64::INFINITY] {
        let error =
            apply_linear_operator_certified(&operator, &state, CertifiedApplyOptions::new(budget))
                .unwrap_err();
        assert!(matches!(error, CertifiedApplyError::InvalidOptions { .. }));
    }
    let other = chain(5);
    let foreign = random_operator(&other, &[4], &[], false, 1, 103);
    let error = apply_linear_operator_certified(&foreign, &state, CertifiedApplyOptions::new(1.0))
        .unwrap_err();
    assert!(matches!(error, CertifiedApplyError::Operation { .. }));
}

#[test]
fn the_binding_form_reports_binding_errors() {
    let sites = chain(2);
    let state = random_state(&sites, false, 2, 111);
    let operator = random_operator(&sites, &[0], &[], false, 1, 112);
    let error = apply_linear_operator_certified_to_indices(
        &operator,
        &state,
        &[(DynIndex::new_dyn(2), site(&sites, 0))],
        &[],
        CertifiedApplyOptions::new(1.0),
    )
    .unwrap_err();
    assert!(matches!(error, CertifiedApplyError::BindIndices(_)));
}
