// A `LinearOperator` on part of a tree, applied to a patched state through
// `contract_adaptive` via `LinearOperator::to_state_network` (#1298).

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use rand::SeedableRng;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorContractionLike};
use tensor4all_partitionedtreetn::{
    contract_adaptive, PartitionedTreeTN, PatchingOptions, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{
    contraction::ContractionOptions, random_treetn, IndexMapping, LinearOperator, LinkSpace,
    Operator, SiteIndexNetwork, TreeTN,
};

type Network = SiteIndexNetwork<String, DynIndex>;

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

fn site(network: &Network, node: &str) -> DynIndex {
    network
        .site_space(&node.to_string())
        .and_then(|space| space.iter().next().cloned())
        .unwrap()
}

fn random_tree(network: &Network, complex: bool, seed: u64) -> TreeTN<IdxTensor, String> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    if complex {
        random_treetn::<Complex64, _, _>(&mut rng, network, LinkSpace::uniform(2))
    } else {
        random_treetn::<f64, _, _>(&mut rng, network, LinkSpace::uniform(2))
    }
    .unwrap()
}

/// A random operator on `nodes` (MPO `edges`) mapping each state site index
/// to itself, and its dense tensor on (state index, primed output) pairs.
fn random_operator(
    state: &Network,
    nodes: &[&str],
    edges: &[(&str, &str)],
    complex: bool,
) -> LinearOperator<IdxTensor, String> {
    let mut mpo_sites = Network::new();
    let (mut inputs, mut outputs) = (HashMap::new(), HashMap::new());
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
    LinearOperator::new(random_tree(&mpo_sites, complex, 17), inputs, outputs)
}

/// `A x` from the operator's dense tensor and the dense state.
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

/// Apply the operator to a two-patch state with `contract_adaptive` (no
/// truncation) and compare the summed output patches with the dense product.
fn check(network: &Network, nodes: &[&str], edges: &[(&str, &str)], split: &str, center: &str) {
    for complex in [false, true] {
        let state = random_tree(network, complex, 5);
        let split_index = site(network, split);
        let patches = (0..2)
            .map(|value| {
                SubDomainTreeTN::new(
                    state.clone(),
                    Projector::from_pairs([(split_index.clone(), value)]).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let patched_state = PartitionedTreeTN::from_subdomains(patches).unwrap();
        let operator = random_operator(network, nodes, edges, complex);

        let converted = operator
            .to_state_network(state.site_index_network())
            .unwrap();
        let (operator_network, outputs) = converted.into_parts();
        // One dtype per network is what PartitionedTreeTN accepts.
        let patched_operator = PartitionedTreeTN::from_subdomain(
            SubDomainTreeTN::from_treetn(operator_network).unwrap(),
        )
        .unwrap();
        let options = PatchingOptions {
            cutoff: 0.0,
            max_bond_dim: None,
            ..PatchingOptions::default()
        };
        let result = contract_adaptive(
            &patched_operator,
            &patched_state,
            &center.to_string(),
            &ContractionOptions::default(),
            &options,
        )
        .unwrap();

        let mut dense = result
            .values()
            .map(|patch| patch.data().to_dense().unwrap())
            .reduce(|a, b| a.add(&b).unwrap())
            .unwrap();
        for (declared, leg) in &outputs {
            dense = dense.replaceind(leg, declared).unwrap();
        }
        let expected = dense_reference(&operator, &state.to_dense().unwrap());
        let error = dense.sub(&expected).unwrap().maxabs().unwrap();
        let scale = expected.maxabs().unwrap();
        assert!(
            error < 1e-11 * scale,
            "complex={complex}, support {nodes:?}: error {error} at scale {scale}"
        );
    }
}

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

#[test]
fn a_connected_subtree_operator_applies_through_contract_adaptive() {
    check(&chain(), &["1", "2"], &[("1", "2")], "0", "2");
    check(
        &tree(),
        &["A", "C", "D"],
        &[("A", "C"), ("C", "D")],
        "B",
        "C",
    );
}

#[test]
fn an_operator_on_two_disjoint_subtrees_applies_through_contract_adaptive() {
    check(
        &chain(),
        &["0", "3", "4"],
        &[("0", "3"), ("3", "4")],
        "2",
        "2",
    );
    check(
        &tree(),
        &["A", "D", "E"],
        &[("A", "D"), ("D", "E")],
        "B",
        "C",
    );
}
