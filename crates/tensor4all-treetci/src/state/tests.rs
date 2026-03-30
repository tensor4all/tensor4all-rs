use super::SimpleTreeTci;
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
    let result = SimpleTreeTci::<f64>::new(vec![2; 6], sample_graph());
    assert!(result.is_err());
}

#[test]
fn add_global_pivots_projects_to_each_edge_bipartition() {
    let mut tci = SimpleTreeTci::<f64>::new(vec![2; 7], sample_graph()).unwrap();
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
