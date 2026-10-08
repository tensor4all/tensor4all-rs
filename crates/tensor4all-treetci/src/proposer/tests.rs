use super::{
    sample_ordered_candidates, union_with_pivots, DefaultProposer, PivotCandidateProposer,
    SimpleProposer, TruncatedDefaultProposer,
};
use crate::{
    column_2d, ncols_2d, AllEdges, EdgeVisitor, SubtreeKey, TreeTCI2, TreeTciEdge, TreeTciGraph,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
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
fn all_edges_visits_all_edges_in_sorted_order() {
    let tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    assert_eq!(
        AllEdges.visit_order(&tci),
        vec![
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ]
    );
}

#[test]
fn default_proposer_matches_neighbor_product_assembly() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let (iset, jset) = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(1, 3))
        .unwrap();

    assert_eq!(
        iset,
        vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![1, 1, 1],
        ]
    );
    assert_eq!(
        jset,
        vec![
            vec![0, 0, 0, 0],
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 1],
            vec![1, 1, 0, 1],
        ]
    );
}

#[test]
fn default_proposer_retains_current_edge_pivots() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();
    // Current edge pivots: ColMajorArray shape [4, 1], single column [1, 1, 1, 1]
    tci.ijset.insert(
        SubtreeKey::new(vec![0, 1, 2, 3]),
        ColMajorArray::new(vec![1, 1, 1, 1], vec![4, 1]).unwrap(),
    );

    let (iset, _jset) = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(3, 4))
        .unwrap();

    assert!(iset.contains(&vec![1, 1, 1, 1]));
}

#[test]
fn union_with_pivots_dedups_in_first_occurrence_order() {
    let key = SubtreeKey::new(vec![0, 1, 2]);
    let pivots = HashMap::from([(
        key.clone(),
        ColMajorArray::new(vec![1, 1, 1, 0, 1, 2], vec![3, 2]).unwrap(), // [1,1,1], [0,1,2]
    )]);

    // `values` contains duplicates (the cartesian pivot expansion can produce
    // them); current pivots add two columns, one of which is already present.
    let values = vec![vec![0, 0, 0], vec![0, 0, 0], vec![1, 1, 1], vec![2, 2, 2]];
    let out = union_with_pivots(values, &pivots, &key).unwrap();

    // First occurrence wins, both within `values` and across current pivots: the
    // already-seen [1,1,1] is not appended again, [0,1,2] is new and appended.
    assert_eq!(
        out,
        vec![vec![0, 0, 0], vec![1, 1, 1], vec![2, 2, 2], vec![0, 1, 2]]
    );
}

#[test]
fn simple_proposer_is_deterministic_for_a_fixed_seed() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let proposer = SimpleProposer::seeded(7);
    let first = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();
    let second = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();

    assert_eq!(first, second);
    assert!(!first.0.is_empty());
    assert!(!first.1.is_empty());
    assert!(first.0.iter().all(|candidate| candidate.len() == 3));
    assert!(first.1.iter().all(|candidate| candidate.len() == 4));
}

#[test]
fn truncated_default_proposer_truncates_default_candidates_in_order() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let default_candidates = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(1, 3))
        .unwrap();
    let proposer = TruncatedDefaultProposer::seeded(7);
    let first = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();
    let second = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.0.len(), 4);
    assert_eq!(first.1.len(), 4);
    assert_eq!(first.1, default_candidates.1);
    assert!(first
        .0
        .iter()
        .all(|candidate| default_candidates.0.contains(candidate)));

    let default_positions = first
        .0
        .iter()
        .map(|candidate| {
            default_candidates
                .0
                .iter()
                .position(|value| value == candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(default_positions
        .windows(2)
        .all(|window| window[0] < window[1]));
}

/// Star with junction 0 and leaves 1, 2, 3 of local dimension 4.
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

fn star_state(junction_dim: usize, pivots: &[Vec<usize>]) -> TreeTCI2<f64> {
    let mut tci = TreeTCI2::<f64>::new(vec![junction_dim, 4, 4, 4], star_graph()).unwrap();
    tci.add_global_pivots(pivots).unwrap();
    tci
}

#[test]
fn truncated_default_proposer_lets_site_free_junction_grow() {
    // Rank 2 on every bond. On edge (0, 1) the junction side offers the
    // product of the two other bonds' pivots (2 * 2 = 4 candidates); the old
    // `local_dim * rank = 1 * 2` budget kept the bond at rank 2.
    let tci = star_state(1, &[vec![0, 0, 0, 0], vec![0, 1, 1, 1]]);
    let edge = TreeTciEdge::new(0, 1);
    let (ikey, _) = tci.graph.subregion_vertices(edge).unwrap();
    assert_eq!(ncols_2d(&tci.ijset[&ikey]).unwrap(), 2);

    let (default_i, _) = DefaultProposer.candidates(&tci, edge).unwrap();
    assert_eq!(default_i.len(), 4);
    let (truncated_i, truncated_j) = TruncatedDefaultProposer::seeded(5)
        .candidates(&tci, edge)
        .unwrap();
    assert_eq!(truncated_i, default_i);
    // The leaf side has a site of dimension 4: budget 4 * 2 = 8 >= its 4
    // Kronecker candidates, so it is not truncated either.
    assert_eq!(truncated_j.len(), 4);
}

#[test]
fn truncated_default_proposer_keeps_site_vertex_budget() {
    // A junction carrying a site of dimension 2 keeps the `d * rank` budget.
    let pivots = [vec![0, 0, 0, 0], vec![1, 1, 1, 1], vec![0, 2, 2, 2]];
    let tci = star_state(2, &pivots);
    let edge = TreeTciEdge::new(0, 1);
    let (default_i, _) = DefaultProposer.candidates(&tci, edge).unwrap();
    assert_eq!(default_i.len(), 2 * 3 * 3);
    for seed in 0..8 {
        let (truncated_i, _) = TruncatedDefaultProposer::seeded(seed)
            .candidates(&tci, edge)
            .unwrap();
        assert_eq!(truncated_i.len(), 2 * 3);
    }
}

#[test]
fn truncated_default_proposer_keeps_previous_pivots_when_truncating() {
    let pivots = [vec![0, 0, 0, 0], vec![1, 1, 1, 1], vec![0, 2, 2, 2]];
    let tci = star_state(2, &pivots);
    let edge = TreeTciEdge::new(0, 1);
    let (ikey, _) = tci.graph.subregion_vertices(edge).unwrap();
    let previous: Vec<Vec<usize>> = (0..ncols_2d(&tci.ijset[&ikey]).unwrap())
        .map(|j| column_2d(&tci.ijset[&ikey], j).unwrap().to_vec())
        .collect();
    assert_eq!(previous.len(), 3);

    let (default_i, _) = DefaultProposer.candidates(&tci, edge).unwrap();
    for seed in 0..32 {
        let (truncated_i, _) = TruncatedDefaultProposer::seeded(seed)
            .candidates(&tci, edge)
            .unwrap();
        // Budget 2 * 3 out of 18 candidates, always including the previous
        // pivots, in the default order.
        assert_eq!(truncated_i.len(), 6);
        for pivot in &previous {
            assert!(truncated_i.contains(pivot), "seed {seed} dropped {pivot:?}");
        }
        let positions: Vec<usize> = truncated_i
            .iter()
            .map(|candidate| {
                default_i
                    .iter()
                    .position(|value| value == candidate)
                    .unwrap()
            })
            .collect();
        assert!(positions.windows(2).all(|window| window[0] < window[1]));
    }
}

#[test]
fn sample_ordered_candidates_samples_keep_set_when_it_exceeds_budget() {
    let candidates: Vec<Vec<usize>> = (0..6).map(|value| vec![value]).collect();
    let keep: HashSet<Vec<usize>> = [vec![1], vec![3], vec![4]].into_iter().collect();
    // The generic sampler consumes the named caller-owned stream directly.

    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let sampled = sample_ordered_candidates(&candidates, &keep, 2, &mut rng);
    assert_eq!(sampled.len(), 2);
    assert!(sampled.iter().all(|candidate| keep.contains(candidate)));
    assert!(sampled.windows(2).all(|window| window[0] < window[1]));

    // Within budget: returned unchanged.
    let all = sample_ordered_candidates(&candidates, &keep, 6, &mut rng);
    assert_eq!(all, candidates);
}

#[test]
fn caller_stream_matches_direct_simple_candidate_draws() {
    use rand::RngCore;
    let mut state = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    state
        .add_global_pivots(&[vec![0; 7], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();
    let edge = TreeTciEdge::new(1, 3);
    let (left, right) = state.graph.subregion_vertices(edge).unwrap();
    let mut actual_rng = ChaCha8Rng::seed_from_u64(73);
    let mut reference_rng = actual_rng.clone();
    for seed in [0, 999] {
        let actual = SimpleProposer::seeded(seed)
            .candidates_with_rng(&state, edge, &mut actual_rng)
            .unwrap();
        let expected_left =
            super::random_candidates(&mut reference_rng, &state.local_dims, &left, 4);
        let expected_right =
            super::random_candidates(&mut reference_rng, &state.local_dims, &right, 4);
        assert_eq!(
            actual,
            (
                union_with_pivots(expected_left, &state.ijset, &left).unwrap(),
                union_with_pivots(expected_right, &state.ijset, &right).unwrap()
            )
        );
    }
    assert_eq!(actual_rng.next_u64(), reference_rng.next_u64());
}

#[test]
fn caller_stream_matches_direct_truncated_sampling() {
    use rand::RngCore;
    let mut state = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    state
        .add_global_pivots(&[vec![0; 7], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();
    let edge = TreeTciEdge::new(1, 3);
    let (u, v) = state.graph.separate_vertices(edge).unwrap();
    let (left, right) = state.graph.subregion_vertices(edge).unwrap();
    let default = DefaultProposer.candidates(&state, edge).unwrap();
    let mut actual_rng = ChaCha8Rng::seed_from_u64(73);
    let mut reference_rng = actual_rng.clone();
    for seed in [0, 999] {
        let actual = TruncatedDefaultProposer::seeded(seed)
            .candidates_with_rng(&state, edge, &mut actual_rng)
            .unwrap();
        let expected_left = sample_ordered_candidates(
            &default.0,
            &super::pivot_columns(&state.ijset, &left).unwrap(),
            super::truncated_candidate_budget(&state, u, &left).unwrap(),
            &mut reference_rng,
        );
        let expected_right = sample_ordered_candidates(
            &default.1,
            &super::pivot_columns(&state.ijset, &right).unwrap(),
            super::truncated_candidate_budget(&state, v, &right).unwrap(),
            &mut reference_rng,
        );
        assert_eq!(actual, (expected_left, expected_right));
    }
    assert_eq!(actual_rng.next_u64(), reference_rng.next_u64());
}

#[test]
fn deterministic_proposer_does_not_consume_caller_randomness() {
    use rand::RngCore;
    let mut state = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    state.add_global_pivots(&[vec![0; 7]]).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(73);
    let mut reference = rng.clone();
    let edge = TreeTciEdge::new(1, 3);
    assert_eq!(
        DefaultProposer
            .candidates_with_rng(&state, edge, &mut rng)
            .unwrap(),
        DefaultProposer.candidates(&state, edge).unwrap()
    );
    assert_eq!(rng.next_u64(), reference.next_u64());
}
