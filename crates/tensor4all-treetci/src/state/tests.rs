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

fn star_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        4,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(0, 2),
            TreeTciEdge::new(0, 3),
        ],
    )
    .unwrap()
}

fn pivot_count(tci: &TreeTCI2<f64>, edge: TreeTciEdge) -> (usize, usize) {
    let (left, right) = tci.graph.subregion_vertices(edge).unwrap();
    (
        tci.pivot_set(&left).unwrap().shape()[1],
        tci.pivot_set(&right).unwrap().shape()[1],
    )
}

#[test]
fn enrich_adds_every_value_pair_of_sites_adjacent_to_a_unit_vertex() {
    // Vertex 0 has local dimension 1; its neighbours 1, 2, 3 have local
    // dimensions 2, 3, 2. Pairs (1,2), (1,3), (2,3) give 6 + 4 + 6 pivots.
    let mut tci = TreeTCI2::<f64>::new(vec![1, 2, 3, 2], star_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 1, 2, 1]]).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 16);
    // The leaf side of edge (0, 2) holds every value of site 2; the junction
    // side holds every value pair of sites 1 and 3.
    let edge = TreeTciEdge::new(0, 2);
    let (junction, leaf) = pivot_count(&tci, edge);
    assert_eq!(leaf, 3);
    assert_eq!(junction, 4);
    // The base pivot comes first and every pivot keeps it off the pair.
    let (_, leaf_key) = tci.graph.subregion_vertices(edge).unwrap();
    assert_eq!(tci.pivot_set(&leaf_key).unwrap().column(0), Some(&[2][..]));
}

#[test]
fn enrich_is_a_no_op_when_it_cannot_or_need_not_act() {
    // No vertex of local dimension 1.
    let mut tci = TreeTCI2::<f64>::new(vec![2; 4], star_graph()).unwrap();
    tci.add_global_pivots(&[vec![0; 4]]).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 0);

    // Only one adjacent site of local dimension greater than 1.
    let chain = TreeTciGraph::linear_chain(3).unwrap();
    let mut tci = TreeTCI2::<f64>::new(vec![1, 1, 2], chain).unwrap();
    tci.add_global_pivots(&[vec![0; 3]]).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 0);

    // No pivots yet.
    let mut tci = TreeTCI2::<f64>::new(vec![1, 2, 2, 2], star_graph()).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 0);

    // Every bond around the unit vertex already has more than one pivot.
    tci.add_global_pivots(&[vec![0; 4], vec![0, 1, 1, 1]])
        .unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 0);
    assert_eq!(pivot_count(&tci, TreeTciEdge::new(0, 1)), (2, 2));
}

/// Sorted columns of the enrichment candidates recorded for `key`.
fn enrichment_columns(tci: &TreeTCI2<f64>, key: &SubtreeKey) -> Vec<Vec<usize>> {
    let array = &tci.enrichment_candidates[key];
    let mut columns: Vec<Vec<usize>> = (0..array.shape()[1])
        .map(|j| array.column(j).unwrap().to_vec())
        .collect();
    columns.sort();
    columns
}

#[test]
fn enrich_records_junction_side_candidates_once() {
    // Star with junction 0 of local dimension 1 and leaves 1, 2, 3 of local
    // dimension 2, base pivot (0, 1, 0, 1).
    let mut tci = TreeTCI2::<f64>::new(vec![1, 2, 2, 2], star_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 1, 0, 1]]).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 12);
    // Only the junction side of each of the three bonds is recorded.
    assert_eq!(tci.enrichment_candidates.len(), 3);
    let edge = TreeTciEdge::new(0, 1);
    let (junction_key, leaf_key) = tci.graph.subregion_vertices(edge).unwrap();
    assert_eq!(junction_key.as_slice(), &[0, 2, 3]);
    assert!(!tci.enrichment_candidates.contains_key(&leaf_key));
    // Pairs (1, 2) and (1, 3) vary one of sites 2, 3 around the base; pair
    // (2, 3) varies both: every value pair of sites 2 and 3.
    assert_eq!(
        enrichment_columns(&tci, &junction_key),
        vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 1]]
    );

    // A later call adds pivots again (the bonds are pinned) but keeps the
    // recorded candidates, so the candidate count stays fixed.
    let sentinel = ColMajorArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
    tci.enrichment_candidates
        .insert(junction_key.clone(), sentinel.clone());
    tci.ijset.clear();
    tci.add_global_pivots(&[vec![0, 0, 1, 1]]).unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 12);
    assert_eq!(enrichment_columns(&tci, &junction_key), vec![vec![0, 1, 1]]);
}

#[test]
fn enrich_records_candidates_without_adding_pivots_when_bonds_are_not_pinned() {
    // Every bond around the unit vertex has two pivots: no pivots are added,
    // but the junction sides still get their permanent candidates.
    let mut tci = TreeTCI2::<f64>::new(vec![1, 2, 2, 2], star_graph()).unwrap();
    tci.add_global_pivots(&[vec![0; 4], vec![0, 1, 1, 1]])
        .unwrap();
    assert_eq!(tci.enrich_pivots_around_unit_vertices().unwrap(), 0);
    assert_eq!(pivot_count(&tci, TreeTciEdge::new(0, 1)), (2, 2));
    assert_eq!(tci.enrichment_candidates.len(), 3);

    // Nothing is recorded where the enrichment rule does not apply.
    let mut tci = TreeTCI2::<f64>::new(vec![2; 4], star_graph()).unwrap();
    tci.add_global_pivots(&[vec![0; 4]]).unwrap();
    tci.enrich_pivots_around_unit_vertices().unwrap();
    assert!(tci.enrichment_candidates.is_empty());
}
