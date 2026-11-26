//! The data-dependent branch of the #1435 expansion predicate.

use super::*;
use crate::samples::ComponentProjectionScratch;
use crate::unit_vertex::retained_pivots_agree;

/// Install actual component pairs of global points, independent of candidate
/// lists: this isolates the predicate's committed-pivot metadata contract.
fn pivots(state: &mut TreeAciState<'_, f64, usize>, side: usize, points: &[[usize; 5]]) {
    let even = side & !1;
    let mut scratch = ComponentProjectionScratch::new(&state.problem).unwrap();
    let pairs = points
        .iter()
        .map(|point| {
            let a = state
                .sample_arena
                .project_point_onto_edge(&state.problem, even, point, &mut scratch)
                .unwrap();
            let b = state
                .sample_arena
                .project_point_onto_edge(&state.problem, even + 1, point, &mut scratch)
                .unwrap();
            (a, b)
        })
        .collect();
    state.pivots.set(side / 2, pairs);
}

#[test]
fn agreement_uses_committed_pairs_on_the_requested_side() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    let reverse = state.problem.directed_edges[forward].reverse;
    // Both component samples differ on each side. Node 1 stays fixed, node 3
    // changes, so using the wrong pair entry reverses the answer.
    pivots(&mut state, forward, &[[0, 0, 0, 0, 0], [1, 0, 0, 1, 1]]);
    assert!(retained_pivots_agree(&state, forward).unwrap());
    assert!(!retained_pivots_agree(&state, reverse).unwrap());
    // Reverse the local agreement, exercising both pair parities.
    pivots(&mut state, forward, &[[0, 0, 0, 0, 0], [1, 1, 0, 0, 1]]);
    assert!(!retained_pivots_agree(&state, forward).unwrap());
    assert!(retained_pivots_agree(&state, reverse).unwrap());
    pivots(&mut state, forward, &[[0, 1, 0, 1, 0], [1, 1, 0, 1, 1]]);
    assert!(retained_pivots_agree(&state, forward).unwrap());
    assert!(retained_pivots_agree(&state, reverse).unwrap());
    // Agreement must include the last pivot, not only the first two.
    pivots(
        &mut state,
        forward,
        &[[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],
    );
    assert!(!retained_pivots_agree(&state, forward).unwrap());
}

#[test]
fn fewer_than_two_pivots_never_establish_agreement() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    for points in [&[][..], &[[0; 5]][..]] {
        pivots(&mut state, forward, points);
        assert!(!retained_pivots_agree(&state, forward).unwrap());
    }
}

#[test]
fn a_fixed_branching_side_uses_the_same_slack_and_widest_bond() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    let reverse = state.problem.directed_edges[forward].reverse;
    pivots(&mut state, forward, &[[0; 5], [1, 0, 0, 0, 1]]);
    // Node 1 has degree 3, node 3 degree 2. Only node 1 qualifies, despite
    // identical local agreement on both sides.
    for other_rank in [1, 2, 4, 5] {
        let ranks = ranks(
            &state,
            &[(0, 1, other_rank), (1, 2, 1), (1, 3, 2), (3, 4, 1)],
        );
        for direction in [forward, reverse] {
            let targets =
                unit_vertex_expansion_targets(&state.problem, &ranks, direction, |side| {
                    retained_pivots_agree(&state, side)
                })
                .unwrap();
            let incoming = &state.problem.directed_edges[forward].incoming_to_from;
            let expected = if other_rank < 4 {
                vec![if other_rank == 1 {
                    incoming[0]
                } else {
                    directed(&state, 0, 1)
                }]
            } else {
                vec![]
            };
            assert_eq!(
                targets, expected,
                "other rank {other_rank}, direction {direction}"
            );
        }
    }
    pivots(&mut state, forward, &[[0; 5], [1, 1, 0, 0, 1]]);
    let low = ranks(&state, &[(0, 1, 1), (1, 2, 1), (1, 3, 2), (3, 4, 1)]);
    assert!(
        unit_vertex_expansion_targets(&state.problem, &low, forward, |side| {
            retained_pivots_agree(&state, side)
        },)
        .unwrap()
        .is_empty()
    );
}

#[test]
fn chain_sites_and_unit_vertices_do_not_read_the_junction_predicate() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    let incoming = state.problem.directed_edges[forward].incoming_to_from[0];
    let targets = unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward, |_| {
        panic!("unit node 1 and chain node 3 must not inspect pivot agreement")
    })
    .unwrap();
    assert_eq!(targets, [incoming]);
}

#[test]
fn two_locally_fixed_junctions_expand_in_commit_order() {
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)];
    let (inputs, options) = unit_fixture(&edges, 6, &[]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    let reverse = state.problem.directed_edges[forward].reverse;
    let mut visited = Vec::new();
    let targets =
        unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward, |side| {
            visited.push(side);
            Ok(true)
        })
        .unwrap();
    assert_eq!(visited, [forward, reverse]);
    assert_eq!(
        targets,
        [
            state.problem.directed_edges[forward].incoming_to_from[0],
            state.problem.directed_edges[reverse].incoming_to_from[0],
        ]
    );
}

#[test]
fn unknown_pivots_and_bad_component_ids_return_internal_errors() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    assert!(matches!(
        retained_pivots_agree(&state, usize::MAX),
        Err(TreeAciError::InternalInvariant { .. })
    ));
    let forward = directed(&state, 1, 3);
    state
        .pivots
        .set(forward / 2, vec![(usize::MAX, usize::MAX); 2]);
    assert!(matches!(
        retained_pivots_agree(&state, forward),
        Err(TreeAciError::InternalInvariant { .. })
    ));
    assert!(matches!(
        unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward, |side| {
            retained_pivots_agree(&state, side)
        },),
        Err(TreeAciError::InternalInvariant { .. })
    ));
}

#[test]
fn expansion_keeps_every_local_value_and_cuts_back_the_neighbour() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 3);
    pivots(&mut state, forward, &[[0; 5], [1, 0, 0, 0, 1]]);
    let before = state.candidates.clone();
    let expansion = expand_unit_vertices(
        &mut state,
        forward,
        &options,
        &mut crate::exec::Exec::Global,
    )
    .unwrap()
    .expect("the fixed junction expands");
    let rows = enumerate_candidates(
        &state.problem,
        &state.candidates,
        forward,
        "test",
        usize::MAX,
    )
    .unwrap();
    let mut coordinates = [0usize; 2];
    for row in rows {
        coordinates[row.local_coordinate] += 1;
    }
    assert_eq!(
        coordinates,
        [2, 2],
        "both local values use the expanded bond"
    );
    expansion.finish(&mut state);
    assert_eq!(state.candidates.ids, before.ids);
}
