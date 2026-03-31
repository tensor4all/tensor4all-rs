use super::{TreeTciEdge, TreeTciGraph};
use crate::SubtreeKey;
use std::collections::{BTreeMap, BTreeSet};

fn sample_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ],
    )
    .unwrap()
}

#[test]
fn tree_graph_utils_match_julia_reference_tree() {
    let graph = sample_graph();
    let edge = TreeTciEdge::new(1, 3);

    assert_eq!(graph.separate_vertices(edge).unwrap(), (1, 3));
    assert_eq!(
        graph.subtree_vertices(3, &[1]).unwrap(),
        SubtreeKey::new(vec![0, 1, 2])
    );
    assert_eq!(
        graph.subtree_vertices(1, &[3]).unwrap(),
        SubtreeKey::new(vec![3, 4, 5, 6])
    );
    assert_eq!(
        graph.subregion_vertices(edge).unwrap(),
        (
            SubtreeKey::new(vec![0, 1, 2]),
            SubtreeKey::new(vec![3, 4, 5, 6]),
        )
    );

    assert_eq!(
        graph
            .adjacent_edges(3, &[])
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([TreeTciEdge::new(1, 3), TreeTciEdge::new(3, 4)])
    );
    assert_eq!(
        graph
            .candidate_edges(edge)
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(3, 4),
        ])
    );
    assert_eq!(
        graph.distance_edges(edge).unwrap(),
        BTreeMap::from([
            (TreeTciEdge::new(1, 3), 0),
            (TreeTciEdge::new(0, 1), 1),
            (TreeTciEdge::new(1, 2), 1),
            (TreeTciEdge::new(3, 4), 1),
            (TreeTciEdge::new(4, 5), 2),
            (TreeTciEdge::new(4, 6), 2),
        ])
    );
}

#[test]
fn test_linear_chain() {
    let graph = TreeTciGraph::linear_chain(5).unwrap();
    assert_eq!(graph.n_sites(), 5);
    let edges = graph.edges();
    assert_eq!(edges.len(), 4);
    assert_eq!(edges[0], TreeTciEdge::new(0, 1));
    assert_eq!(edges[1], TreeTciEdge::new(1, 2));
    assert_eq!(edges[2], TreeTciEdge::new(2, 3));
    assert_eq!(edges[3], TreeTciEdge::new(3, 4));
}

#[test]
fn test_linear_chain_single_site() {
    let graph = TreeTciGraph::linear_chain(1).unwrap();
    assert_eq!(graph.n_sites(), 1);
    assert_eq!(graph.edges().len(), 0);
}

#[test]
fn test_linear_chain_two_sites() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    assert_eq!(graph.n_sites(), 2);
    assert_eq!(graph.edges().len(), 1);
}
