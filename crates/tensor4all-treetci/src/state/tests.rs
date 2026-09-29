use super::TreeTCI2;
use crate::{SubtreeKey, TreeTciEdge, TreeTciGraph};
use tensor4all_core::ColMajorArray;

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
fn simple_tree_tci_requires_local_dims_to_match_graph_size() {
    let result = TreeTCI2::<f64>::new(vec![2; 6], sample_graph());
    assert!(result.is_err());
}

#[test]
fn add_global_pivots_projects_to_each_edge_bipartition() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    // [n_subtree_sites, n_pivots] = [3, 2]
    // Column 0 = [0,0,0], Column 1 = [1,0,1]
    assert_eq!(
        *tci.ijset.get(&SubtreeKey::new(vec![0, 1, 2])).unwrap(),
        ColMajorArray::new(vec![0, 0, 0, 1, 0, 1], vec![3, 2]).unwrap()
    );
    // [4, 2]: Column 0 = [0,0,0,0], Column 1 = [0,1,0,1]
    assert_eq!(
        *tci.ijset.get(&SubtreeKey::new(vec![3, 4, 5, 6])).unwrap(),
        ColMajorArray::new(vec![0, 0, 0, 0, 0, 1, 0, 1], vec![4, 2]).unwrap()
    );
    // Full key: [7, 0] - empty (no columns)
    assert_eq!(
        *tci.ijset
            .get(&SubtreeKey::new(vec![0, 1, 2, 3, 4, 5, 6]))
            .unwrap(),
        ColMajorArray::new(vec![], vec![7, 0]).unwrap()
    );
}

// Regression for #692: pivots injected by the automatic global search must
// not grow either side of an edge past the edge's maximal achievable rank
// (the smaller subtree dimension product), while the public
// `add_global_pivots` used for initial pivots keeps every distinct projection.
#[test]
fn inject_global_pivots_bounds_each_side_by_edge_rank() {
    // Chain 0 - 1 - 2 with local dims [2, 3, 3]:
    // edge (0, 1): {0} has 2 states, {1, 2} has 9 -> maximal rank 2;
    // edge (1, 2): {0, 1} has 6 states, {2} has 3 -> maximal rank 3.
    let graph = TreeTciGraph::linear_chain(3).unwrap();
    let pivots = [vec![1, 1, 1], vec![0, 2, 2], vec![1, 0, 1], vec![0, 1, 2]];

    let mut injected = TreeTCI2::<f64>::new(vec![2, 3, 3], graph.clone()).unwrap();
    injected.add_global_pivots(&[vec![0, 0, 0]]).unwrap();
    injected.inject_global_pivots(&pivots).unwrap();
    let key = |sites: Vec<usize>| SubtreeKey::new(sites);
    let expected = [
        // Deduplication alone bounds the 2-state side.
        (
            key(vec![0]),
            ColMajorArray::new(vec![0, 1], vec![1, 2]).unwrap(),
        ),
        // Bounded at 2: [0, 0] (existing), [1, 1]; the rest is skipped.
        (
            key(vec![1, 2]),
            ColMajorArray::new(vec![0, 0, 1, 1], vec![2, 2]).unwrap(),
        ),
        // Bounded at 3: [0, 0], [1, 1], [0, 2]; [1, 0] and [0, 1] skipped.
        (
            key(vec![0, 1]),
            ColMajorArray::new(vec![0, 0, 1, 1, 0, 2], vec![2, 3]).unwrap(),
        ),
        (
            key(vec![2]),
            ColMajorArray::new(vec![0, 1, 2], vec![1, 3]).unwrap(),
        ),
    ];
    for (subtree, columns) in &expected {
        assert_eq!(&injected.ijset[subtree], columns, "subtree {subtree:?}");
    }

    // Initial pivots are not bounded: all five distinct projections stay.
    let mut added = TreeTCI2::<f64>::new(vec![2, 3, 3], graph).unwrap();
    added.add_global_pivots(&[vec![0, 0, 0]]).unwrap();
    added.add_global_pivots(&pivots).unwrap();
    assert_eq!(
        added.ijset[&key(vec![1, 2])],
        ColMajorArray::new(vec![0, 0, 1, 1, 2, 2, 0, 1, 1, 2], vec![2, 5]).unwrap()
    );
}
