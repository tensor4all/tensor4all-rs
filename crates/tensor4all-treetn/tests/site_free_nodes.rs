//! Regression tests for tree tensor networks with site-free nodes.
//!
//! A site-free node carries no site index, for example the leaf that a direct
//! sum leaves on a node without sites. Canonicalization, norms, truncation,
//! fitting and site swaps must preserve the represented tensor and report its
//! norm whatever the bond dimension of such a node. Every check materializes
//! each network once and compares it with a dense reference via `maxabs()`.

use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TagSet, TensorContractionLike, TensorIndex};
use tensor4all_treetn::contraction::{contract, ContractionOptions};
use tensor4all_treetn::{
    factorize_tensor_to_treetn, fit_sum, CanonicalForm, CanonicalizationOptions, FitOptions,
    SwapOptions, TreeTN, TreeTopology, TruncationOptions,
};

type Network = TreeTN<IdxTensor, String>;

/// Relative tolerance for exact rewrites of small networks.
const EXACT_TOL: f64 = 1e-12;

// ============================================================================
// Shared helpers
// ============================================================================

fn names(nodes: &[&str]) -> Vec<String> {
    nodes.iter().map(|node| node.to_string()).collect()
}

/// Deterministic, non-degenerate sample values.
fn sample_values(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|k| ((k as f64 + 1.0) * seed).sin() + 0.1 * k as f64)
        .collect()
}

/// Build a column-major tensor; `phase` turns the data complex.
fn dense(indices: Vec<DynIndex>, data: &[f64], phase: Option<Complex64>) -> IdxTensor {
    match phase {
        None => IdxTensor::from_dense(indices, data.to_vec()).unwrap(),
        Some(phase) => IdxTensor::from_dense(
            indices,
            data.iter()
                .map(|value| phase * *value)
                .collect::<Vec<Complex64>>(),
        )
        .unwrap(),
    }
}

fn relative_maxabs_difference(actual: &IdxTensor, expected: &IdxTensor) -> f64 {
    let scale = expected.maxabs().unwrap().max(f64::MIN_POSITIVE);
    actual.sub(expected).unwrap().maxabs().unwrap() / scale
}

fn assert_values(actual: &Network, expected: &IdxTensor, label: &str) {
    let difference = relative_maxabs_difference(&actual.to_dense().unwrap(), expected);
    assert!(
        difference < EXACT_TOL,
        "{label}: values changed, relative maxabs difference {difference}"
    );
}

/// Check `norm`, `log_norm` and `norm_squared` against `expected`.
fn assert_norms(tn: &Network, expected: f64, label: &str) {
    let norm = tn.clone().norm().unwrap();
    let log_norm = tn.clone().log_norm().unwrap();
    let norm_squared = tn.clone().norm_squared().unwrap();
    assert!(
        (norm - expected).abs() <= EXACT_TOL * expected,
        "{label}: norm {norm} != dense norm {expected}"
    );
    assert!(
        (log_norm - expected.ln()).abs() <= EXACT_TOL,
        "{label}: log_norm {log_norm} != ln(dense norm) {}",
        expected.ln()
    );
    assert!(
        (norm_squared - expected * expected).abs() <= EXACT_TOL * expected * expected,
        "{label}: norm_squared {norm_squared} != {}",
        expected * expected
    );
}

fn is_site_free(tn: &Network, node: &String) -> bool {
    tn.site_space(node).is_none_or(|sites| sites.is_empty())
}

/// Bond dimensions on every edge between a site-free leaf and its neighbor.
fn site_free_leaf_bond_dims(tn: &Network, center: &String) -> Vec<usize> {
    let mut dims = Vec::new();
    for node in tn.node_names() {
        if &node == center || !is_site_free(tn, &node) {
            continue;
        }
        let neighbors: Vec<String> = tn.site_index_network().neighbors(&node).collect();
        if let [neighbor] = neighbors.as_slice() {
            let edge = tn.edge_between(&node, neighbor).unwrap();
            dims.push(tn.bond_index(edge).unwrap().dim());
        }
    }
    dims
}

/// Canonicalize towards every node in every form, and truncate towards every
/// node exactly and with a unit bond cap; compare values and norms with the
/// dense reference.
fn assert_sweeps_exact_for_every_center(tn: &Network, label: &str) {
    let expected = tn.to_dense().unwrap();
    let expected_norm = expected.norm().unwrap();

    for center in tn.node_names() {
        for form in [CanonicalForm::Unitary, CanonicalForm::LU, CanonicalForm::CI] {
            let label = format!("{label}: canonicalize {form:?} to {center}");
            let canonical = tn
                .clone()
                .canonicalize(
                    [center.clone()],
                    CanonicalizationOptions::forced().with_form(form),
                )
                .unwrap();
            canonical.verify_internal_consistency().unwrap();
            assert!(canonical.same_topology(tn), "{label}: topology changed");
            assert_values(&canonical, &expected, &label);
            assert!(
                site_free_leaf_bond_dims(&canonical, &center)
                    .iter()
                    .all(|dim| *dim == 1),
                "{label}: a site-free leaf kept a wide bond"
            );
            if form == CanonicalForm::Unitary {
                assert_norms(&canonical, expected_norm, &label);
            }
        }

        let label_exact = format!("{label}: exact truncation to {center}");
        let truncated = tn
            .clone()
            .truncate([center.clone()], TruncationOptions::default())
            .unwrap();
        truncated.verify_internal_consistency().unwrap();
        assert_values(&truncated, &expected, &label_exact);
        assert_norms(&truncated, expected_norm, &label_exact);

        let label_capped = format!("{label}: unit-cap truncation to {center}");
        let capped = tn
            .clone()
            .truncate(
                [center.clone()],
                TruncationOptions::default().with_max_bond_dim(1),
            )
            .unwrap();
        capped.verify_internal_consistency().unwrap();
        assert!(
            capped.link_dims().iter().all(|dim| *dim <= 1),
            "{label_capped}: bond above the cap"
        );
        let capped_norm = capped.to_dense().unwrap().norm().unwrap();
        assert_norms(&capped, capped_norm, &label_capped);
    }
}

// ============================================================================
// Networks
// ============================================================================

/// `a - b - e` where `e` is a site-free leaf with a dimension-two bond. The
/// network represents `[1, 0] ⊗ [3, 4]`, whose norm is 5.
fn wide_leaf_network() -> Network {
    let (site_a, site_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (bond_ab, bond_be) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2));
    let a = dense(vec![site_a, bond_ab.clone()], &[1.0, 0.0], None);
    let b = dense(
        vec![site_b, bond_ab, bond_be.clone()],
        &[3.0, 0.0, 0.0, 4.0],
        None,
    );
    let e = dense(vec![bond_be], &[1.0, 1.0], None);
    TreeTN::from_tensors(vec![a, b, e], names(&["a", "b", "e"])).unwrap()
}

/// `x + y` with `x = [1, 0] ⊗ [3, 4]` and `y = [0, 1] ⊗ [6, 8]` (times `i`
/// when `complex`), both on `a - b - e` with a site-free leaf `e`. The sum's
/// leaf has a dimension-two bond and the norm is `sqrt(125)`.
fn direct_sum_network(complex: bool) -> Network {
    let (site_a, site_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let product = |a: [f64; 2], b: [f64; 2], phase: Option<Complex64>| {
        let (bond_ab, bond_be) = (DynIndex::new_dyn(1), DynIndex::new_dyn(1));
        TreeTN::from_tensors(
            vec![
                dense(vec![site_a.clone(), bond_ab.clone()], &a, phase),
                dense(vec![site_b.clone(), bond_ab, bond_be.clone()], &b, phase),
                dense(vec![bond_be], &[1.0], phase),
            ],
            names(&["a", "b", "e"]),
        )
        .unwrap()
    };
    let (x_phase, y_phase) = if complex {
        (
            Some(Complex64::new(1.0, 0.0)),
            Some(Complex64::new(0.0, 1.0)),
        )
    } else {
        (None, None)
    };
    let x = product([1.0, 0.0], [3.0, 4.0], x_phase);
    let y = product([0.0, 1.0], [6.0, 8.0], y_phase);
    x.add(&y).unwrap()
}

/// A tree with a degree-three site-free hub `m`, wide site-free leaves `e`
/// (on `m`) and `f` (on `b`), and a site-free chain `g1 - g2` off `a`:
///
/// ```text
/// g2 - g1 - a - m - b - f
///               |
///               e
/// ```
fn rich_network(phase: Option<Complex64>) -> Network {
    let (site_a, site_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let am = DynIndex::new_dyn(2);
    let mb = DynIndex::new_dyn(3);
    let me = DynIndex::new_dyn(3);
    let bf = DynIndex::new_dyn(2);
    let ag1 = DynIndex::new_dyn(2);
    let g1g2 = DynIndex::new_dyn(3);
    let tensors = vec![
        dense(
            vec![site_a, am.clone(), ag1.clone()],
            &sample_values(8, 0.7),
            phase,
        ),
        dense(
            vec![am, mb.clone(), me.clone()],
            &sample_values(18, 1.3),
            phase,
        ),
        dense(vec![mb, site_b, bf.clone()], &sample_values(18, 0.4), phase),
        dense(vec![me], &sample_values(3, 2.1), phase),
        dense(vec![bf], &sample_values(2, 0.9), phase),
        dense(vec![ag1, g1g2.clone()], &sample_values(6, 1.7), phase),
        dense(vec![g1g2], &sample_values(3, 0.3), phase),
    ];
    TreeTN::from_tensors(tensors, names(&["a", "m", "b", "e", "f", "g1", "g2"])).unwrap()
}

/// `a - b - e` where `b` carries a site index with the same ID as the bond
/// `k` to its site-free leaf `e`, differing only by prime level or by tags.
fn same_id_network(site_of: impl Fn(&DynIndex) -> DynIndex) -> Network {
    let k = DynIndex::new_dyn(2);
    let site_k = site_of(&k);
    assert_eq!(site_k.id(), k.id());
    assert_ne!(site_k, k);
    let site_a = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(2);
    let tn = TreeTN::from_tensors(
        vec![
            dense(vec![site_a, bond_ab.clone()], &sample_values(4, 0.5), None),
            dense(
                vec![bond_ab, site_k.clone(), k.clone()],
                &sample_values(8, 1.1),
                None,
            ),
            dense(vec![k], &[0.6, -1.2], None),
        ],
        names(&["a", "b", "e"]),
    )
    .unwrap();
    assert_eq!(tn.edge_count(), 2);
    assert!(tn.site_space(&"b".to_string()).unwrap().contains(&site_k));
    assert!(is_site_free(&tn, &"e".to_string()));
    tn
}

fn primed_site(k: &DynIndex) -> DynIndex {
    k.prime()
}

fn tagged_site(k: &DynIndex) -> DynIndex {
    DynIndex::new_with_tags(*k.id(), k.dim(), TagSet::from_tags(&["Site"]).unwrap())
}

/// Two site-free nodes joined by a dimension-three bond: a scalar network
/// whose value is `1*4 + 2*5 + 3*6 = 32`.
fn site_free_chain() -> Network {
    let bond = DynIndex::new_dyn(3);
    TreeTN::from_tensors(
        vec![
            dense(vec![bond.clone()], &[1.0, 2.0, 3.0], None),
            dense(vec![bond], &[4.0, 5.0, 6.0], None),
        ],
        names(&["left", "right"]),
    )
    .unwrap()
}

// ============================================================================
// Canonicalization, norms and truncation
// ============================================================================

#[test]
fn wide_site_free_leaf_norms_match_dense_reference() {
    let tn = wide_leaf_network();
    let expected = tn.to_dense().unwrap().norm().unwrap();
    assert_eq!(expected, 5.0);
    assert_norms(&tn, expected, "wide leaf");
}

#[test]
fn truncating_wide_site_free_leaf_keeps_values_within_unit_cap() {
    // The network has rank one across every edge, so a unit cap is exact.
    let tn = wide_leaf_network();
    let expected = tn.to_dense().unwrap();
    for center in ["a", "e"] {
        let truncated = tn
            .clone()
            .truncate(
                [center.to_string()],
                TruncationOptions::default().with_max_bond_dim(1),
            )
            .unwrap();
        assert_values(&truncated, &expected, &format!("truncate to {center}"));
        assert!(truncated.link_dims().iter().all(|dim| *dim == 1));
    }
}

#[test]
fn direct_sum_with_site_free_leaf_is_exact_for_every_center() {
    let tn = direct_sum_network(false);
    let expected = tn.to_dense().unwrap().norm().unwrap();
    assert!((expected - 125.0_f64.sqrt()).abs() < EXACT_TOL * expected);
    assert_sweeps_exact_for_every_center(&tn, "direct sum");
}

#[test]
fn complex_direct_sum_with_site_free_leaf_is_exact_for_every_center() {
    let tn = direct_sum_network(true);
    let expected = tn.to_dense().unwrap().norm().unwrap();
    assert!((expected - 125.0_f64.sqrt()).abs() < EXACT_TOL * expected);
    assert_sweeps_exact_for_every_center(&tn, "complex direct sum");
}

#[test]
fn site_free_hub_leaves_and_chain_are_exact_for_every_center() {
    assert_sweeps_exact_for_every_center(&rich_network(None), "rich tree");
}

#[test]
fn complex_site_free_hub_leaves_and_chain_are_exact_for_every_center() {
    let phase = Complex64::from_polar(1.0, 0.7);
    assert_sweeps_exact_for_every_center(&rich_network(Some(phase)), "complex rich tree");
}

#[test]
fn site_free_two_node_chain_is_exact_for_every_center() {
    let tn = site_free_chain();
    assert_eq!(tn.to_dense().unwrap().to_vec::<f64>().unwrap(), vec![32.0]);
    assert_sweeps_exact_for_every_center(&tn, "site-free chain");
}

#[test]
fn site_free_leaf_bond_sharing_id_with_primed_site_is_exact_for_every_center() {
    assert_sweeps_exact_for_every_center(&same_id_network(primed_site), "same-ID primed site");
}

#[test]
fn site_free_leaf_bond_sharing_id_with_tagged_site_is_exact_for_every_center() {
    assert_sweeps_exact_for_every_center(&same_id_network(tagged_site), "same-ID tagged site");
}

#[test]
fn same_id_site_survives_site_free_leaf_absorption() {
    for site_of in [primed_site as fn(&DynIndex) -> DynIndex, tagged_site] {
        let tn = same_id_network(site_of);
        let site = tn
            .site_space(&"b".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let canonical = tn
            .canonicalize(["a".to_string()], CanonicalizationOptions::forced())
            .unwrap();
        let b = canonical.node_index(&"b".to_string()).unwrap();
        assert!(canonical
            .tensor(b)
            .unwrap()
            .external_indices()
            .contains(&site));
        assert_eq!(site_free_leaf_bond_dims(&canonical, &"a".to_string()), [1]);
    }
}

// ============================================================================
// Variational fitting with the center at a site-free leaf
// ============================================================================

#[test]
fn fit_sum_with_center_at_site_free_leaf_reports_dense_norm() {
    let x = direct_sum_network(false);
    let expected = x.to_dense().unwrap().add(&x.to_dense().unwrap()).unwrap();
    let expected_norm = expected.norm().unwrap();
    for center in x.node_names() {
        let label = format!("fit_sum to {center}");
        let fitted = fit_sum(&[x.clone(), x.clone()], &x, &center, FitOptions::new(2)).unwrap();
        fitted.verify_internal_consistency().unwrap();
        assert_values(&fitted, &expected, &label);
        assert_norms(&fitted, expected_norm, &label);
    }
}

/// An `a - b - e` state and operator whose `e` nodes are both site-free.
fn site_free_contraction_pair() -> (Network, Network) {
    let (site_a, site_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (out_a, out_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (ab, be) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let state = TreeTN::from_tensors(
        vec![
            dense(
                vec![site_a.clone(), ab.clone()],
                &sample_values(4, 0.3),
                None,
            ),
            dense(
                vec![ab, site_b.clone(), be.clone()],
                &sample_values(8, 0.8),
                None,
            ),
            dense(vec![be], &[1.0, 0.5], None),
        ],
        names(&["a", "b", "e"]),
    )
    .unwrap();
    let (ab, be) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let operator = TreeTN::from_tensors(
        vec![
            dense(
                vec![site_a, out_a, ab.clone()],
                &sample_values(8, 1.4),
                None,
            ),
            dense(
                vec![ab, site_b, out_b, be.clone()],
                &sample_values(16, 0.6),
                None,
            ),
            dense(vec![be], &[0.7, -0.2], None),
        ],
        names(&["a", "b", "e"]),
    )
    .unwrap();
    (state, operator)
}

/// A Y-shaped state and operator around a degree-three hub `m` with sites,
/// whose leaves are `a` and `b` (with sites) and the site-free `e`:
///
/// ```text
/// a - m - b
///     |
///     e
/// ```
///
/// The tree is not a chain, so the zip-up initializer of `contract_fit`
/// takes the general tree path, where `e` leaves an empty side.
fn site_free_y_contraction_pair() -> (Network, Network) {
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let outputs: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let build = |with_outputs: bool, seed: f64| {
        let (am, mb, me) = (
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(3),
        );
        let legs = |site: usize, bonds: &[&DynIndex]| {
            let mut indices = vec![sites[site].clone()];
            if with_outputs {
                indices.push(outputs[site].clone());
            }
            indices.extend(bonds.iter().map(|bond| (*bond).clone()));
            indices
        };
        let a = legs(0, &[&am]);
        let m = legs(1, &[&am, &mb, &me]);
        let b = legs(2, &[&mb]);
        let tensors = [a, m, b, vec![me.clone()]]
            .into_iter()
            .enumerate()
            .map(|(k, indices)| {
                let len = indices.iter().map(|index| index.dim()).product();
                dense(indices, &sample_values(len, seed + 0.37 * k as f64), None)
            })
            .collect();
        TreeTN::from_tensors(tensors, names(&["a", "m", "b", "e"])).unwrap()
    };
    let state = build(false, 0.5);
    let operator = build(true, 1.1);
    assert_eq!(state.edge_count(), 3);
    assert_eq!(
        state
            .site_index_network()
            .neighbors(&"m".to_string())
            .count(),
        3
    );
    assert!(is_site_free(&state, &"e".to_string()));
    assert!(is_site_free(&operator, &"e".to_string()));
    (state, operator)
}

/// Run `contract_fit` towards every center, as the zip-up initializer alone
/// (`nfullsweeps = 0`) and with variational sweeps, and compare each result
/// with the dense contraction.
fn assert_contract_fit_matches_dense(state: &Network, operator: &Network, label: &str) {
    let expected = state
        .to_dense()
        .unwrap()
        .contract_pair(&operator.to_dense().unwrap())
        .unwrap();
    let expected_norm = expected.norm().unwrap();
    for center in state.node_names() {
        for nfullsweeps in [0, 2] {
            let label = format!("{label}: contract_fit ({nfullsweeps} sweeps) to {center}");
            let result = contract(
                state,
                operator,
                &center,
                ContractionOptions::fit().with_nfullsweeps(nfullsweeps),
            )
            .unwrap();
            result.verify_internal_consistency().unwrap();
            assert!(result.same_topology(state), "{label}: topology changed");
            assert_values(&result, &expected, &label);
            assert_norms(&result, expected_norm, &label);
        }
    }
}

#[test]
fn contract_fit_with_center_at_site_free_leaf_reports_dense_norm() {
    let (state, operator) = site_free_contraction_pair();
    assert_contract_fit_matches_dense(&state, &operator, "chain");
}

#[test]
fn contract_fit_on_branched_tree_with_site_free_leaf_matches_dense_reference() {
    let (state, operator) = site_free_y_contraction_pair();
    assert_contract_fit_matches_dense(&state, &operator, "Y tree");
}

// ============================================================================
// Site swaps onto and off site-free nodes
// ============================================================================

#[test]
fn swapping_site_onto_and_off_site_free_node_keeps_values_and_norm() {
    // Moving `s` from `a` to the site-free `b` leaves `a`'s side empty;
    // moving it back leaves `b`'s side empty.
    let site = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);
    let mut tn = TreeTN::from_tensors(
        vec![
            dense(
                vec![site.clone(), bond.clone()],
                &sample_values(6, 0.9),
                None,
            ),
            dense(vec![bond], &[0.4, -1.1], None),
        ],
        names(&["a", "b"]),
    )
    .unwrap();
    let expected = tn.to_dense().unwrap();
    let expected_norm = expected.norm().unwrap();

    for target in ["b", "a"] {
        let label = format!("swap site to {target}");
        let assignment = HashMap::from([(site.clone(), target.to_string())]);
        tn.swap_site_indices(&assignment, &SwapOptions::default())
            .unwrap();
        tn.verify_internal_consistency().unwrap();
        assert!(tn.site_space(&target.to_string()).unwrap().contains(&site));
        assert_eq!(tn.link_dims(), vec![1], "{label}");
        assert_values(&tn, &expected, &label);
        assert_norms(&tn, expected_norm, &label);
    }
}

/// A zero site-free leaf goes through the same factorization as any other
/// node. The unitary form handles it; LU and CI reject it, as checked by
/// `lu_and_ci_canonicalization_reports_zero_tensor_cause`.
#[test]
fn zero_site_free_leaf_canonicalizes_to_zero_network() {
    let (site_a, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let tn: Network = TreeTN::from_tensors(
        vec![
            dense(vec![site_a, bond.clone()], &[1.0, 2.0, 3.0, 4.0], None),
            dense(vec![bond], &[0.0, 0.0], None),
        ],
        names(&["a", "e"]),
    )
    .unwrap();
    let canonical = tn
        .canonicalize(["a".to_string()], CanonicalizationOptions::forced())
        .unwrap();
    assert_eq!(canonical.link_dims(), vec![1]);
    assert_eq!(canonical.to_dense().unwrap().maxabs().unwrap(), 0.0);
    assert_eq!(canonical.clone().norm().unwrap(), 0.0);
}

/// LU and CI canonicalization fails on a zero tensor that has to be factorized,
/// whether the node is site-free or not, because a rank-zero split has no
/// valid bond. The error keeps the root cause in its message and leaves the
/// network unchanged.
#[test]
fn lu_and_ci_canonicalization_reports_zero_tensor_cause() {
    let (site_a, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let site_free_zero: Network = TreeTN::from_tensors(
        vec![
            dense(
                vec![site_a.clone(), bond.clone()],
                &[1.0, 2.0, 3.0, 4.0],
                None,
            ),
            dense(vec![bond.clone()], &[0.0, 0.0], None),
        ],
        names(&["a", "e"]),
    )
    .unwrap();
    let site_zero: Network = TreeTN::from_tensors(
        vec![
            dense(vec![site_a, bond.clone()], &[0.0; 4], None),
            dense(vec![bond], &[0.4, -1.1], None),
        ],
        names(&["a", "e"]),
    )
    .unwrap();

    for (network, center, zero_node) in [(&site_free_zero, "a", "e"), (&site_zero, "e", "a")] {
        for form in [CanonicalForm::LU, CanonicalForm::CI] {
            let label = format!("{form:?} with zero node {zero_node}");
            let options = CanonicalizationOptions::forced().with_form(form);
            let message = network
                .clone()
                .canonicalize([center.to_string()], options)
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("canonicalize: factorization failed: invalid dimension 0"),
                "{label}: {message}"
            );

            let mut in_place = network.clone();
            assert!(
                in_place
                    .canonicalize_mut([center.to_string()], options)
                    .is_err(),
                "{label}"
            );
            assert_eq!(in_place.link_dims(), network.link_dims(), "{label}");
            assert!(!in_place.is_canonicalized(), "{label}");
            assert_values(&in_place, &network.to_dense().unwrap(), &label);
        }
    }
}

// ============================================================================
// Known failures with site-free nodes, tracked in issue #797
// ============================================================================

#[test]
#[ignore = "known bug, tracked in #797: inner fails with \"Disconnected tensor network\" on site-free nodes"]
fn inner_with_site_free_leaf_matches_dense_reference() {
    let x = direct_sum_network(false);
    let expected = x.to_dense().unwrap().norm().unwrap().powi(2);
    let inner = x.inner(&x).unwrap();
    assert!((inner.real() - expected).abs() <= EXACT_TOL * expected);
}

#[test]
#[ignore = "known bug, tracked in #797: SRC contraction fails on site-free nodes"]
fn src_contraction_with_site_free_leaf_matches_dense_reference() {
    let (state, operator) = site_free_contraction_pair();
    let expected = state
        .to_dense()
        .unwrap()
        .contract_pair(&operator.to_dense().unwrap())
        .unwrap();
    for center in state.node_names() {
        let result = contract(
            &state,
            &operator,
            &center,
            ContractionOptions::src().with_max_bond_dim(4),
        )
        .unwrap();
        assert_values(&result, &expected, &format!("SRC to {center}"));
    }
}

#[test]
#[ignore = "known bug, tracked in #797: factorize_tensor_to_treetn rejects site-free nodes"]
fn factorize_tensor_to_treetn_with_site_free_leaf_matches_dense_reference() {
    let (site_a, site_b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let expected = dense(
        vec![site_a.clone(), site_b.clone()],
        &sample_values(4, 0.9),
        None,
    );
    let topology = TreeTopology::new(
        HashMap::from([
            ("a".to_string(), vec![site_a]),
            ("b".to_string(), vec![site_b]),
            ("e".to_string(), Vec::new()),
        ]),
        vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "e".to_string()),
        ],
    );
    for root in ["a", "e"] {
        let tn = factorize_tensor_to_treetn(&expected, &topology, &root.to_string()).unwrap();
        assert_values(&tn, &expected, &format!("decompose from {root}"));
    }
}
