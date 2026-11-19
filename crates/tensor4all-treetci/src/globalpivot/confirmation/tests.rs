use super::*;
use crate::{TreeTciEdge, TreeTciGraph};

#[test]
fn confirmation_pairs_preserve_both_unit_orientations_and_visit_earlier_pivots() {
    let mut state =
        TreeTCI2::<f64>::new(vec![2, 1, 3], TreeTciGraph::linear_chain(3).unwrap()).unwrap();
    let points = [vec![1, 0, 2], vec![0, 0, 1]];
    state.add_global_pivots(&points).unwrap();
    let retained = retained_pivot_edges(&state).unwrap();
    assert_eq!(retained.len(), 2);
    let mut scratch = vec![usize::MAX; 3];
    for anchor in &retained {
        assert_eq!(anchor.rank, points.len());
        for pair in (0..anchor.rank).rev() {
            anchor.write_point(pair, &mut scratch).unwrap();
            assert_eq!(scratch, points[pair]);
        }
        assert!(anchor.write_point(anchor.rank, &mut scratch).is_err());
    }
    assert_eq!(retained[0].left_sites.as_slice(), &[0]);
    assert_eq!(retained[0].right_sites.as_slice(), &[1, 2]);
    assert_eq!(retained[1].left_sites.as_slice(), &[0, 1]);
    assert_eq!(retained[1].right_sites.as_slice(), &[2]);
    let mut ordinary =
        TreeTCI2::<f64>::new(vec![2; 3], TreeTciGraph::linear_chain(3).unwrap()).unwrap();
    ordinary.add_global_pivots(&[vec![0; 3]]).unwrap();
    assert_eq!(
        retained_pivot_edges(&ordinary).unwrap().len(),
        ordinary.graph.edges().len()
    );
}

#[test]
fn confirmation_includes_every_unit_edge() {
    let edges: Vec<_> = (1..=5).map(|site| TreeTciEdge::new(0, site)).collect();
    let mut state = TreeTCI2::<f64>::new(
        vec![1, 2, 2, 2, 2, 2],
        TreeTciGraph::new(6, &edges).unwrap(),
    )
    .unwrap();
    let point = vec![0, 1, 1, 1, 1, 1];
    state
        .add_global_pivots(std::slice::from_ref(&point))
        .unwrap();
    let retained = retained_pivot_edges(&state).unwrap();
    assert_eq!(retained.len(), edges.len());
    let mut scratch = vec![usize::MAX; 6];
    for (anchor, physical_site) in retained.iter().zip(1..=5) {
        assert_eq!(anchor.rank, 1);
        anchor.write_point(0, &mut scratch).unwrap();
        assert_eq!(scratch, point);
        assert_eq!(anchor.right_sites.as_slice(), &[physical_site]);
        assert_eq!(anchor.left_sites.as_slice().len(), 5);
    }
    // Two site-free endpoints are valid too; their only point is zero.
    let mut units =
        TreeTCI2::<f64>::new(vec![1, 1], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    units.add_global_pivots(&[vec![0, 0]]).unwrap();
    let retained = retained_pivot_edges(&units).unwrap();
    let mut scratch = [usize::MAX; 2];
    retained[0].write_point(0, &mut scratch).unwrap();
    assert_eq!(scratch, [0, 0]);
    assert_eq!(retained[0].left_sites.as_slice(), &[0]);
    assert_eq!(retained[0].right_sites.as_slice(), &[1]);
}

#[test]
fn confirmation_rejects_missing_empty_or_mismatched_pivot_sets() {
    let mut state =
        TreeTCI2::<f64>::new(vec![1, 2], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    assert!(retained_pivot_edges(&state).is_err());
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    let (left, right) = state
        .graph
        .subregion_vertices(TreeTciEdge::new(0, 1))
        .unwrap();
    let saved = state.ijset.remove(&right).unwrap();
    assert!(retained_pivot_edges(&state).is_err());
    state.ijset.insert(right.clone(), saved);
    state.ijset.remove(&left);
    assert!(retained_pivot_edges(&state).is_err());
    state.ijset.insert(
        left.clone(),
        tensor4all_core::ColMajorArray::zeros(vec![1, 0]).unwrap(),
    );
    assert!(retained_pivot_edges(&state).is_err());
    state.ijset.insert(
        left,
        tensor4all_core::ColMajorArray::zeros(vec![1, 2]).unwrap(),
    );
    assert!(retained_pivot_edges(&state).is_err());
}
