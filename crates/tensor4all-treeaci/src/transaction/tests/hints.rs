use super::super::local_pivot_hints;
use super::*;

#[test]
fn hints_remap_after_candidate_reordering_and_drop_missing_components() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
    let bonds: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(1)).collect();
    let input = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![sites[0].clone(), bonds[0].clone()], vec![1.0_f64; 3])
                .unwrap(),
            IdxTensor::from_dense(
                vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
                vec![1.0; 3],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bonds[1].clone(), sites[2].clone()], vec![1.0; 3]).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let inputs = vec![input];
    let options = TreeAciOptions::default();
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    // Use global seeds to get two distinct nested components on every cut.
    let (arena, candidates) = crate::samples::SampleArena::from_global_seeds(
        &state.problem,
        &[vec![0, 0, 0], vec![1, 1, 1]],
    )
    .unwrap();
    state.sample_arena = arena;
    state.candidates = candidates;
    let forward = state
        .problem
        .directed_edges
        .iter()
        .position(|e| e.from == 1 && e.to == 2)
        .unwrap();
    let reverse = state.problem.directed_edges[forward].reverse;
    let first = state.candidates.ids[forward][0];
    let last = state.candidates.ids[reverse][1];
    let pair = if forward.is_multiple_of(2) {
        (first, last)
    } else {
        (last, first)
    };
    state.pivots.per_edge[forward / 2] = vec![pair];
    let row = state.sample_arena.record(forward, first).unwrap();
    let col = state.sample_arena.record(reverse, last).unwrap();
    let expected_row = row.local_coordinate;
    let expected_col = col.local_coordinate;
    assert_eq!(
        local_pivot_hints(&state, forward, usize::MAX).unwrap(),
        vec![(expected_row, expected_col)]
    );
    assert_eq!(
        local_pivot_hints(&state, reverse, usize::MAX).unwrap(),
        vec![(expected_col, expected_row)]
    );
    let incoming = state.problem.directed_edges[forward].incoming_to_from[0];
    state.candidates.ids[incoming].reverse();
    assert_eq!(
        local_pivot_hints(&state, forward, usize::MAX).unwrap(),
        vec![(expected_row + 3, expected_col)]
    );
    assert!(matches!(
        local_pivot_hints(&state, forward, 0),
        Err(TreeAciError::ResourceLimit {
            resource: "working bytes",
            ..
        })
    ));
    let required = row.incoming[0].1;
    let source = state.problem.node_positions[&state.problem.directed_edges[forward].from];
    state.problem.physical[source].local_dim = usize::MAX;
    // The selected incoming position is one after reversing candidates.
    assert!(matches!(
        local_pivot_hints(&state, forward, usize::MAX),
        Err(TreeAciError::SizeOverflow { .. })
    ));
    state.problem.physical[source].local_dim = 3;

    state.candidates.ids[incoming].retain(|&id| id != required);
    assert!(local_pivot_hints(&state, forward, 0).unwrap().is_empty());
    state.pivots.per_edge[forward / 2] = vec![(usize::MAX, usize::MAX)];
    assert!(matches!(
        local_pivot_hints(&state, forward, 0),
        Err(TreeAciError::InternalInvariant { .. })
    ));
}
