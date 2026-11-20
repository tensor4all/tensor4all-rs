use super::*;
use crate::operator::IndexMapping;
use crate::random::{random_treetn, LinkSpace};
use num_complex::Complex64;
use rand::SeedableRng;
use std::collections::HashMap;
use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike};

type Network = SiteIndexNetwork<String, DynIndex>;

/// A site network with one dimension-2 site index per node.
fn site_network(nodes: &[&str], edges: &[(&str, &str)]) -> Network {
    let mut network = Network::new();
    for &node in nodes {
        network
            .add_node(node.to_string(), HashSet::from([DynIndex::new_dyn(2)]))
            .unwrap();
    }
    for &(a, b) in edges {
        network.add_edge(&a.to_string(), &b.to_string()).unwrap();
    }
    network
}

/// The chain `0 - 1 - 2 - 3 - 4`.
fn chain() -> Network {
    site_network(
        &["0", "1", "2", "3", "4"],
        &[("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")],
    )
}

/// `A - C - B` and `C - D - E`: `C` has degree 3.
fn tree() -> Network {
    site_network(
        &["A", "B", "C", "D", "E"],
        &[("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")],
    )
}

fn site(network: &Network, node: &str) -> DynIndex {
    network
        .site_space(&node.to_string())
        .and_then(|space| space.iter().next().cloned())
        .unwrap()
}

/// A random operator on `nodes` with MPO `edges`, mapping each node's state
/// site index to itself (the `A x` convention). The MPO edges need not be state
/// edges: an edge between two nodes that are not adjacent in the state makes
/// the support two disjoint subtrees joined by a bond.
fn random_operator(
    state: &Network,
    nodes: &[&str],
    edges: &[(&str, &str)],
    complex: bool,
    seed: u64,
) -> LinearOperator<IdxTensor, String> {
    let mut mpo_sites = Network::new();
    let mut inputs = HashMap::new();
    let mut outputs = HashMap::new();
    for &node in nodes {
        let true_index = site(state, node);
        let (input, output) = (true_index.sim(), true_index.sim());
        mpo_sites
            .add_node(
                node.to_string(),
                HashSet::from([input.clone(), output.clone()]),
            )
            .unwrap();
        inputs.insert(
            node.to_string(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: input,
            },
        );
        outputs.insert(
            node.to_string(),
            IndexMapping {
                true_index,
                internal_index: output,
            },
        );
    }
    for &(a, b) in edges {
        mpo_sites.add_edge(&a.to_string(), &b.to_string()).unwrap();
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mpo = if complex {
        random_treetn::<Complex64, _, _>(&mut rng, &mpo_sites, LinkSpace::uniform(2))
    } else {
        random_treetn::<f64, _, _>(&mut rng, &mpo_sites, LinkSpace::uniform(2))
    }
    .unwrap();
    LinearOperator::new(mpo, inputs, outputs)
}

/// `A x` computed from the operator's dense tensor alone: contract its inputs
/// with the state's site indices, then rename its outputs to the declared
/// indices. Nodes outside the support keep the state's legs (identity).
fn dense_reference(operator: &LinearOperator<IdxTensor, String>, state: &IdxTensor) -> IdxTensor {
    let mut dense = operator.mpo().to_dense().unwrap();
    let mut renames = Vec::new();
    for name in operator.node_names() {
        let input = operator.get_input_mapping(&name).unwrap();
        let output = operator.get_output_mapping(&name).unwrap();
        let primed = output.true_index.sim();
        dense = dense
            .replaceind(&input.internal_index, &input.true_index)
            .unwrap()
            .replaceind(&output.internal_index, &primed)
            .unwrap();
        renames.push((primed, output.true_index.clone()));
    }
    let mut result = IdxTensor::contract(&[state, &dense]).unwrap();
    for (primed, declared) in renames {
        result = result.replaceind(&primed, &declared).unwrap();
    }
    result
}

fn any_complex(operator: &LinearOperator<IdxTensor, String>) -> bool {
    let mpo = operator.mpo();
    mpo.node_indices()
        .into_iter()
        .any(|node| mpo.tensor(node).unwrap().is_complex())
}

/// Convert `operator` on `state_sites`, check the network's structure and
/// dtype, and check that contracting it with a random state equals the dense
/// reference.
fn check_conversion(state_sites: &Network, operator: &LinearOperator<IdxTensor, String>) {
    let complex = any_complex(operator);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
    let state = if complex {
        random_treetn::<Complex64, _, _>(&mut rng, state_sites, LinkSpace::uniform(2))
    } else {
        random_treetn::<f64, _, _>(&mut rng, state_sites, LinkSpace::uniform(2))
    }
    .unwrap();

    let converted = operator
        .to_state_network(state.site_index_network())
        .unwrap();
    let network = converted.network();
    assert!(network.same_topology(&state));
    for node in network.node_indices() {
        let tensor = network.tensor(node).unwrap();
        assert_eq!(tensor.is_complex(), complex, "node {node:?}");
    }
    // One output per state site index, each a fresh leg (the declared outputs
    // are the state's own indices here).
    assert_eq!(converted.outputs().len(), state_sites.node_count());
    for (declared, leg) in converted.outputs() {
        assert_ne!(declared, leg);
        assert_eq!(declared.dim(), leg.dim());
    }

    let state_dense = state.to_dense().unwrap();
    let mut result = IdxTensor::contract(&[&state_dense, &network.to_dense().unwrap()]).unwrap();
    for (declared, leg) in converted.outputs() {
        result = result.replaceind(leg, declared).unwrap();
    }
    let expected = dense_reference(operator, &state_dense);
    let error = result.sub(&expected).unwrap().maxabs().unwrap();
    let scale = expected.maxabs().unwrap().max(1.0);
    assert!(error < 1e-12 * scale, "error {error} at scale {scale}");
}

#[test]
fn a_connected_subtree_operator_is_converted_on_a_chain_and_a_tree() {
    for complex in [false, true] {
        let chain = chain();
        check_conversion(
            &chain,
            &random_operator(&chain, &["1", "2"], &[("1", "2")], complex, 1),
        );
        let tree = tree();
        // The support contains the degree-3 node.
        check_conversion(
            &tree,
            &random_operator(
                &tree,
                &["A", "C", "D"],
                &[("A", "C"), ("C", "D")],
                complex,
                2,
            ),
        );
    }
}

#[test]
fn an_operator_on_two_disjoint_subtrees_is_converted_on_a_chain_and_a_tree() {
    for complex in [false, true] {
        let chain = chain();
        // {0} and {3, 4}, joined in the MPO across the gaps 1 and 2.
        check_conversion(
            &chain,
            &random_operator(
                &chain,
                &["0", "3", "4"],
                &[("0", "3"), ("3", "4")],
                complex,
                3,
            ),
        );
        let tree = tree();
        // {A} and {D, E}, joined across the degree-3 gap C.
        check_conversion(
            &tree,
            &random_operator(
                &tree,
                &["A", "D", "E"],
                &[("A", "D"), ("D", "E")],
                complex,
                4,
            ),
        );
    }
}

#[test]
fn a_full_coverage_operator_needs_the_state_edges() {
    let chain = chain();
    let nodes = ["0", "1", "2", "3", "4"];
    // The state's own edges: the operator is used as it is.
    check_conversion(
        &chain,
        &random_operator(
            &chain,
            &nodes,
            &[("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")],
            true,
            5,
        ),
    );
    // A star around node 2 would put two bonds on the edges 1-2 and 2-3.
    let star = random_operator(
        &chain,
        &nodes,
        &[("2", "0"), ("2", "1"), ("2", "3"), ("2", "4")],
        false,
        6,
    );
    let error = star.to_state_network(&chain).unwrap_err();
    assert!(matches!(
        error,
        TreeTNOperationError::InvalidConfiguration { .. }
    ));
    assert!(error.to_string().contains("must have the state's edges"));
}

#[test]
fn a_partial_operator_whose_bonds_would_share_a_state_edge_is_rejected() {
    let chain = chain();
    // Bonds 0-2 and 1-3 both cross the state edge 1-2.
    let operator = random_operator(
        &chain,
        &["0", "1", "2", "3"],
        &[("0", "2"), ("2", "1"), ("1", "3")],
        false,
        10,
    );
    let error = operator.to_state_network(&chain).unwrap_err();
    assert!(matches!(
        error,
        TreeTNOperationError::InvalidConfiguration { .. }
    ));
}

#[test]
fn an_output_outside_the_state_keeps_its_index() {
    let chain = chain();
    let input = site(&chain, "2");
    let declared = DynIndex::new_dyn(3);
    let (internal_in, internal_out) = (input.sim(), declared.sim());
    // A 3 x 2 operator on node 2 into a new space.
    let values: Vec<f64> = (0..6).map(|v| v as f64 + 1.0).collect();
    let tensor =
        IdxTensor::from_dense(vec![internal_out.clone(), internal_in.clone()], values).unwrap();
    let operator = LinearOperator::new(
        TreeTN::from_tensors(vec![tensor], vec!["2".to_string()]).unwrap(),
        HashMap::from([(
            "2".to_string(),
            IndexMapping {
                true_index: input,
                internal_index: internal_in,
            },
        )]),
        HashMap::from([(
            "2".to_string(),
            IndexMapping {
                true_index: declared.clone(),
                internal_index: internal_out,
            },
        )]),
    );
    let converted = operator.to_state_network(&chain).unwrap();
    assert!(converted
        .outputs()
        .contains(&(declared.clone(), declared.clone())));
    assert!(converted
        .network()
        .site_index_network()
        .contains_index(&declared));
    // The other four outputs are fresh legs for the gap identities.
    assert_eq!(
        converted
            .outputs()
            .iter()
            .filter(|(declared, leg)| declared != leg)
            .count(),
        4
    );
    let (network, outputs) = converted.into_parts();
    assert_eq!(network.node_count(), 5);
    assert_eq!(outputs.len(), 5);
}

#[test]
fn conversion_rejects_operators_that_do_not_fit_the_state() {
    let chain = chain();
    let error = |operator: &LinearOperator<IdxTensor, String>, network: &Network| {
        let error = operator.to_state_network(network).unwrap_err();
        assert!(matches!(error, TreeTNOperationError::InvalidGraph { .. }));
        error.to_string()
    };

    // A node the state does not have.
    let other = site_network(&["0", "9"], &[("0", "9")]);
    let operator = random_operator(&other, &["9"], &[], false, 7);
    assert!(error(&operator, &chain).contains("is not a node of the state"));

    // An input that is not the state's site index at that node.
    let foreign = random_operator(&super::tests::chain(), &["1"], &[], false, 8);
    assert!(error(&foreign, &chain).contains("is not a site index of the state"));

    // A node of the operator with a state site index it does not map.
    let mut two_sites = chain.clone();
    two_sites
        .add_site_index(&"1".to_string(), DynIndex::new_dyn(2))
        .unwrap();
    let operator = random_operator(&chain, &["1"], &[], false, 9);
    assert!(error(&operator, &two_sites).contains("is not an operator input"));
}
