use super::SimpleTreeTci;
use crate::{SubtreeKey, TreeTciEdge, TreeTciGraph};

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

    assert_eq!(
        tci.ijset
            .get(&SubtreeKey::new(vec![0, 1, 2]))
            .cloned()
            .unwrap(),
        vec![vec![0, 0, 0], vec![1, 0, 1]]
    );
    assert_eq!(
        tci.ijset
            .get(&SubtreeKey::new(vec![3, 4, 5, 6]))
            .cloned()
            .unwrap(),
        vec![vec![0, 0, 0, 0], vec![0, 1, 0, 1]]
    );
    assert_eq!(
        tci.ijset
            .get(&SubtreeKey::new(vec![0, 1, 2, 3, 4, 5, 6]))
            .cloned()
            .unwrap(),
        Vec::<Vec<usize>>::new()
    );
}
