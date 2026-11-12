use std::collections::HashSet;

use tensor4all_treetn::SiteIndexNetwork;

use super::{balanced_center, refuse_large_tree_block, tree_zipup_peak, EXACT_TREE_BLOCK_LIMIT};
use crate::{DynIndex, PartitionedTreeTNError};

/// A network with one site of each listed dimension per node and the given edges.
fn network(
    nodes: &[(usize, &[usize])],
    edges: &[(usize, usize)],
) -> SiteIndexNetwork<usize, DynIndex> {
    let mut network = SiteIndexNetwork::new();
    for &(name, dims) in nodes {
        let sites: HashSet<DynIndex> = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
        network.add_node(name, sites).unwrap();
    }
    for &(left, right) in edges {
        network.add_edge(&left, &right).unwrap();
    }
    network
}

fn chain(dims: &[&[usize]]) -> SiteIndexNetwork<usize, DynIndex> {
    let nodes: Vec<(usize, &[usize])> = dims.iter().copied().enumerate().collect();
    let edges: Vec<(usize, usize)> = (1..dims.len()).map(|node| (node - 1, node)).collect();
    network(&nodes, &edges)
}

#[test]
fn an_empty_network_has_no_center() {
    assert_eq!(balanced_center(&network(&[], &[])), None);
}

#[test]
fn a_single_node_is_its_own_center() {
    assert_eq!(balanced_center(&chain(&[&[2]])), Some(0));
}

#[test]
fn a_uniform_chain_is_centred_in_the_middle() {
    let one: &[usize] = &[2];
    assert_eq!(balanced_center(&chain(&[one; 5])), Some(2));
    // Nodes 1 and 2 both leave two bits on their larger side; the smaller name wins.
    assert_eq!(balanced_center(&chain(&[one; 4])), Some(1));
}

#[test]
fn bits_not_node_counts_place_the_center() {
    // Node 0 carries 4 bits and node 1 two sites; nodes 2 to 5 one bit each.
    // Node 1 leaves 4 bits on either side, every other node more.
    let dims: [&[usize]; 6] = [&[16], &[2, 2], &[2], &[2], &[2], &[2]];
    assert_eq!(balanced_center(&chain(&dims)), Some(1));
    // Non-binary sites count by log2 of their dimension: 8 is three bits, so
    // node 1 leaves three on each side, where counting nodes would pick node 2.
    let dims: [&[usize]; 5] = [&[8], &[2], &[2], &[2], &[2]];
    assert_eq!(balanced_center(&chain(&dims)), Some(1));
}

#[test]
fn a_branched_tree_is_centred_at_the_hub_whatever_the_root() {
    // Hub 5 with branches {0}, {1} and {2 - 3}; the smallest name, the
    // traversal root, is a leaf.
    let tree = network(
        &[(0, &[2]), (1, &[2]), (2, &[2]), (3, &[2]), (5, &[2])],
        &[(5, 0), (5, 1), (5, 2), (2, 3)],
    );
    assert_eq!(balanced_center(&tree), Some(5));
    // A heavy branch pulls the centre into it: nodes 2 and 3 then leave at
    // most four bits on any side, the hub five; the smaller name wins.
    let tree = network(
        &[(0, &[2]), (1, &[2]), (2, &[2]), (3, &[16]), (5, &[2])],
        &[(5, 0), (5, 1), (5, 2), (2, 3)],
    );
    assert_eq!(balanced_center(&tree), Some(2));
}

/// Hub 0 with branches {1}, {2} and {3 - 4}, one binary site per node.
fn hub_tree() -> SiteIndexNetwork<usize, DynIndex> {
    let binary: &[usize] = &[2];
    network(
        &[
            (0, binary),
            (1, binary),
            (2, binary),
            (3, binary),
            (4, binary),
        ],
        &[(0, 1), (0, 2), (0, 3), (3, 4)],
    )
}

#[test]
fn a_chain_needs_no_block_estimate() {
    let one: &[usize] = &[2];
    let operator: HashSet<usize> = (0..5).collect();
    assert_eq!(
        tree_zipup_peak(&chain(&[one; 5]), |_, _| 1000, &operator, 1000, &2),
        None
    );
}

#[test]
fn the_tree_estimate_follows_the_zip_up_and_skips_spectator_subtrees() {
    // State bonds 3, operator bond 2, centre 0, operator on nodes 0, 1 and 3.
    // Leaf 1 passes min(2, 3 * 2) = 2 on (block 2 * 6 = 12); spectator leaves 2
    // and 4 are emitted, so their parents keep the state bond 3; node 3 forms
    // (2 * 3) * (3 * 2) = 36 and passes 6; the centre forms 2 * 2 * 3 * 6 = 72.
    let operator = HashSet::from([0, 1, 3]);
    assert_eq!(
        tree_zipup_peak(&hub_tree(), |_, _| 3, &operator, 2, &0),
        Some(72.0)
    );
    // With node 4 acted on as well, it is processed like leaf 1: node 3 forms
    // (2 * 2) * 6 = 24 and passes 4, and the centre forms 2 * 2 * 3 * 4 = 48.
    let operator = HashSet::from([0, 1, 3, 4]);
    assert_eq!(
        tree_zipup_peak(&hub_tree(), |_, _| 3, &operator, 2, &0),
        Some(48.0)
    );
}

#[test]
fn deep_branches_with_wide_bonds_exceed_the_limit() {
    // Hub 0 with three branches of 16 binary nodes, state and operator bonds
    // 128: each branch forms blocks of 2^15 * 2^14 = 2^29 elements near the hub
    // and hands it a remainder of 2^14, so the centre contracts three of them
    // with its own site into 2 * 2^42 = 2^43 elements. A chain never joins more
    // than one remainder.
    let binary: &[usize] = &[2];
    let mut nodes: Vec<(usize, &[usize])> = vec![(0, binary)];
    let mut edges = Vec::new();
    for branch in 0..3 {
        for depth in 1..=16 {
            let node = branch * 16 + depth;
            nodes.push((node, binary));
            edges.push((if depth == 1 { 0 } else { node - 1 }, node));
        }
    }
    let tree = network(&nodes, &edges);
    let operator: HashSet<usize> = (0..=48).collect();
    let peak = tree_zipup_peak(&tree, |_, _| 128, &operator, 128, &0);
    assert_eq!(peak, Some(2.0_f64.powi(43)));
    assert!(matches!(
        refuse_large_tree_block(peak, EXACT_TREE_BLOCK_LIMIT),
        Err(PartitionedTreeTNError::ResourceLimit { value, .. }) if value == 1 << 43
    ));
}

#[test]
fn the_refusal_applies_only_above_the_limit() {
    assert!(refuse_large_tree_block(None, 0).is_ok());
    assert!(refuse_large_tree_block(Some(72.0), 72).is_ok());
    assert!(matches!(
        refuse_large_tree_block(Some(72.0), 71),
        Err(PartitionedTreeTNError::ResourceLimit { value: 72, .. })
    ));
}
