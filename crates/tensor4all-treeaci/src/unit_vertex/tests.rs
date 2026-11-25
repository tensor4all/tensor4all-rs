//! Tests of the unit-vertex expansion (#1435).

use std::cell::Cell;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{
    expand_unit_vertices, expansion_debug_log, unit_vertex_expansion_targets,
    UNIT_VERTEX_EXPANSION_FACTOR,
};
use crate::{
    local_update::enumerate_candidates,
    problem::DirectedEdgeId,
    state::TreeAciState,
    test_support::{random_decaying_tree, SplitMix},
    transaction::update_edge_transaction,
    TreeAciError, TreeAciOptions,
};

/// Edges of a five-node tree: node 1 has degree 3 (neighbours 0, 2 and 3).
const EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (1, 3), (3, 4)];

/// A rank-2 input on `edges` whose nodes in `unit` have one site of dimension
/// 1 and every other node a binary site, with options whose rank-1 initial
/// guess starts every bond at rank 1.
fn unit_fixture(
    edges: &[(usize, usize)],
    node_count: usize,
    unit: &[usize],
) -> (Vec<TreeTN<IdxTensor, usize>>, TreeAciOptions<usize>) {
    let physical = (0..node_count)
        .map(|node| DynIndex::new_dyn(if unit.contains(&node) { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let mut rng = SplitMix(1435);
    let input = random_decaying_tree::<f64>(edges, &physical, 2, 0.5, &mut rng);
    let guess = random_decaying_tree::<f64>(edges, &physical, 1, 0.5, &mut rng);
    let options = TreeAciOptions {
        initial_guess: Some(guess),
        ..TreeAciOptions::default()
    };
    (vec![input], options)
}

fn directed(state: &TreeAciState<'_, f64, usize>, from: usize, to: usize) -> DirectedEdgeId {
    state
        .problem
        .directed_edges
        .iter()
        .position(|edge| edge.from == from && edge.to == to)
        .unwrap()
}

/// Bond ranks indexed like `edge_ranks`, from `(a, b, rank)` triples.
fn ranks(state: &TreeAciState<'_, f64, usize>, entries: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut ranks = vec![0; state.edge_ranks.len()];
    for &(a, b, rank) in entries {
        ranks[directed(state, a, b) / 2] = rank;
    }
    assert!(
        ranks.iter().all(|&rank| rank > 0),
        "every bond needs a rank"
    );
    ranks
}

fn identity(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)?;
    }
    Ok(())
}

#[test]
fn the_factor_is_the_untuned_treetci_constant() {
    assert_eq!(UNIT_VERTEX_EXPANSION_FACTOR, 2);
}

/// Below the factor, the unit side of the edge expands its largest-rank
/// other bond; the leaf side has no other bond.
#[test]
fn a_unit_side_below_the_factor_expands_its_largest_rank_other_bond() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let problem = &state.problem;
    // Rank 3 on 0-1: the other bonds offer 1 * 2 = 2 < 2 * 3.
    let edge_ranks = ranks(&state, &[(0, 1, 3), (1, 2, 1), (1, 3, 2), (3, 4, 2)]);
    for forward in [directed(&state, 1, 0), directed(&state, 0, 1)] {
        assert_eq!(
            unit_vertex_expansion_targets(problem, &edge_ranks, forward).unwrap(),
            vec![directed(&state, 3, 1)],
        );
    }
    // Bond 1-3: the other bonds offer 3 * 1 = 3 < 2 * 2, the largest is
    // 0 -> 1; side 3 has a binary site.
    assert_eq!(
        unit_vertex_expansion_targets(problem, &edge_ranks, directed(&state, 1, 3)).unwrap(),
        vec![directed(&state, 0, 1)],
    );
}

/// At or above the factor, and on edges without a unit side, nothing
/// expands.
#[test]
fn a_unit_side_at_the_factor_or_a_site_side_does_not_expand() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let problem = &state.problem;
    // The other bonds of 0-1 offer exactly 2 * 2 = 4.
    let at_factor = ranks(&state, &[(0, 1, 2), (1, 2, 2), (1, 3, 2), (3, 4, 2)]);
    for forward in [directed(&state, 1, 0), directed(&state, 0, 1)] {
        assert!(unit_vertex_expansion_targets(problem, &at_factor, forward)
            .unwrap()
            .is_empty());
    }
    // Bond 3-4 joins a binary node and a leaf, whatever the ranks.
    let low = ranks(&state, &[(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 2)]);
    for forward in [directed(&state, 3, 4), directed(&state, 4, 3)] {
        assert!(unit_vertex_expansion_targets(problem, &low, forward)
            .unwrap()
            .is_empty());
    }
    // A tree without a unit vertex never expands.
    let (plain, plain_options) = unit_fixture(&EDGES, 5, &[]);
    let plain_state = TreeAciState::<f64, usize>::initialize(&plain, &plain_options).unwrap();
    let ones = vec![1; plain_state.edge_ranks.len()];
    for forward in 0..plain_state.problem.directed_edges.len() {
        assert!(
            unit_vertex_expansion_targets(&plain_state.problem, &ones, forward)
                .unwrap()
                .is_empty()
        );
    }
}

/// Equal other ranks keep the first bond in `incoming_to_from` order.
#[test]
fn ties_expand_the_first_incoming_bond() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let edge_ranks = ranks(&state, &[(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1)]);
    let forward = directed(&state, 1, 0);
    let incoming = &state.problem.directed_edges[forward].incoming_to_from;
    assert_eq!(incoming.len(), 2);
    assert_eq!(
        unit_vertex_expansion_targets(&state.problem, &edge_ranks, forward).unwrap(),
        vec![incoming[0]],
    );
}

/// Both unit sides of one edge expand; a unit leaf has no other bond.
#[test]
fn two_unit_sides_both_expand_and_unit_leaves_never_do() {
    // Nodes 1 and 3 are unit vertices of degree 3; leaf 5 is a unit leaf.
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)];
    let (inputs, options) = unit_fixture(&edges, 6, &[1, 3, 5]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let edge_ranks = ranks(
        &state,
        &[(0, 1, 2), (1, 2, 1), (1, 3, 2), (3, 4, 1), (3, 5, 1)],
    );
    // Bond 1-3: side 1 offers 2 * 1 and side 3 offers 1 * 1, both below
    // 2 * 2. Side 1 expands its rank-2 bond 0 -> 1, side 3 its first bond.
    let forward = directed(&state, 1, 3);
    let reverse = state.problem.directed_edges[forward].reverse;
    assert_eq!(
        unit_vertex_expansion_targets(&state.problem, &edge_ranks, forward).unwrap(),
        vec![
            directed(&state, 0, 1),
            state.problem.directed_edges[reverse].incoming_to_from[0]
        ],
    );
    // Bond 3-5: the leaf side has no other bond, and side 3 offers
    // 2 * 1 = 2, not below 2 * 1.
    for forward in [directed(&state, 5, 3), directed(&state, 3, 5)] {
        assert!(
            unit_vertex_expansion_targets(&state.problem, &edge_ranks, forward)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn unknown_edges_and_ranks_are_internal_errors() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let edges = state.problem.directed_edges.len();
    assert!(matches!(
        unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, edges),
        Err(TreeAciError::InternalInvariant { .. })
    ));
    assert!(matches!(
        unit_vertex_expansion_targets(&state.problem, &[], 0),
        Err(TreeAciError::InternalInvariant { .. })
    ));
}

/// The expansion appends the neighbour's side samples after the current
/// entries, extends the frames over them, and `finish` cuts them back.
#[test]
fn expansion_appends_the_neighbour_side_samples_and_finish_cuts_them_back() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    assert!(state.edge_ranks.iter().all(|&rank| rank == 1));
    // Bond 1-2 at rank 1: the other bonds of node 1 offer 1 < 2, and the
    // first of the two rank-1 bonds expands. Its other side is a leaf.
    let forward = directed(&state, 1, 2);
    let target = state.problem.directed_edges[forward].incoming_to_from[0];
    assert_eq!(
        unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward).unwrap(),
        vec![target]
    );
    let before = state.candidates.ids[target].clone();
    assert_eq!(before.len(), 1);
    let offered = enumerate_candidates(
        &state.problem,
        &state.candidates,
        target,
        "test",
        usize::MAX,
    )
    .unwrap();
    // Node 0 is a binary leaf; node 3 is binary with one rank-1 bond.
    assert_eq!(offered.len(), 2);

    let expansion = expand_unit_vertices(
        &mut state,
        forward,
        &options,
        &mut crate::exec::Exec::Global,
    )
    .unwrap()
    .expect("bond 1-2 is below the factor");
    assert_eq!(expansion.restore(), &[(target, 1)]);
    let grown = state.candidates.ids[target].clone();
    assert_eq!(grown[0], before[0], "the prefix stays first");
    assert_eq!(grown.len(), 2);
    for sample in offered {
        let id = state
            .sample_arena
            .intern_component(&state.problem, target, sample)
            .unwrap();
        assert!(grown.contains(&id), "offered sample {id} is missing");
    }
    assert_eq!(
        state.input_frames.frames[0][target].sample_count,
        state.sample_arena.directed_record_count(target).unwrap(),
        "the frames cover every interned sample"
    );

    expansion.finish(&mut state);
    assert_eq!(state.candidates.ids[target], before);
}

/// A commit reads the expanded rows, keeps the rank they reveal, and leaves
/// the expanded set as it was before.
#[test]
fn a_commit_reads_the_expanded_set_and_cuts_it_back() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 2);
    let reverse = state.problem.directed_edges[forward].reverse;
    let target = state.problem.directed_edges[forward].incoming_to_from[0];
    let other = state.problem.directed_edges[forward].incoming_to_from[1];
    let before = state.candidates.ids[target].clone();
    assert_eq!(
        (before.len(), state.candidates.ids[other].len()),
        (1, 1),
        "without the expansion the rows are 1 x 1"
    );
    assert!(state.problem.directed_edges[reverse]
        .incoming_to_from
        .is_empty());
    expansion_debug_log::take();

    let report = update_edge_transaction(
        &mut state,
        forward,
        &options,
        true,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    // Rows: the two expanded samples of `target` times one of `other`;
    // columns: leaf 2's two coordinates.
    assert_eq!(report.evaluated_points, 2 * 2);
    assert_eq!(report.new_rank, 2, "a 1 x 2 plain matrix has rank 1");
    assert_eq!(state.candidates.ids[target], before);
    assert_eq!(state.candidates.ids[forward].len(), 2);
    assert_eq!(
        expansion_debug_log::take(),
        vec![(forward, vec![(target, 1)])]
    );
}

/// A proposal that fails after the expansion leaves the state as before it.
#[test]
fn a_failed_proposal_abandons_the_expansion() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 2);
    let candidates = state.candidates.clone();
    let pivots = state.pivots.clone();
    let records = state.sample_arena.record_count();
    let bytes = state.sample_arena.retained_bytes();
    // Directed frames only: the candidate-frame memo is shared by every
    // clone of the store and keeps what the failed evaluation computed.
    let frame_records = state.input_frames.records();
    let generation = state.generation;
    expansion_debug_log::take();
    let mut failing = |_: crate::TreeElementwiseBatch<'_, f64>, _: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "stop".into(),
        })
    };

    let result = update_edge_transaction(
        &mut state,
        forward,
        &options,
        true,
        false,
        &mut failing,
        &mut crate::exec::Exec::Global,
    );

    assert!(matches!(result, Err(TreeAciError::Callback { .. })));
    assert_eq!(expansion_debug_log::take().len(), 1, "the expansion ran");
    assert_eq!(state.candidates, candidates);
    assert_eq!(state.pivots, pivots);
    assert_eq!(state.sample_arena.record_count(), records);
    assert_eq!(state.sample_arena.retained_bytes(), bytes);
    assert_eq!(state.input_frames.records(), frame_records);
    for (edge, ids) in state.candidates.ids.iter().enumerate() {
        let frame = &state.input_frames.frames[0][edge];
        assert_eq!(
            frame.sample_count,
            state.sample_arena.directed_record_count(edge).unwrap(),
            "edge {edge} ({} candidates)",
            ids.len()
        );
    }
    assert_eq!(state.generation, generation);
}

/// An expansion that cannot be staged fails before any operator call and
/// leaves the state unchanged.
#[test]
fn a_failed_expansion_leaves_the_state_unchanged() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 2);
    let candidates = state.candidates.clone();
    let records = state.sample_arena.record_count();
    let bytes = state.sample_arena.retained_bytes();
    // The expansion interns new samples, whose frames no longer fit.
    state.problem.max_frame_bytes = state.input_frames.retained_bytes();
    expansion_debug_log::take();
    let calls = Cell::new(0);
    let mut counting = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        identity(batch, output)
    };

    let result = update_edge_transaction(
        &mut state,
        forward,
        &options,
        true,
        false,
        &mut counting,
        &mut crate::exec::Exec::Global,
    );

    assert!(matches!(
        result,
        Err(TreeAciError::ResourceLimit {
            resource: "frame bytes",
            ..
        })
    ));
    assert_eq!(calls.get(), 0, "the failure came before the proposal");
    assert!(expansion_debug_log::take().is_empty());
    assert_eq!(state.candidates, candidates);
    assert_eq!(state.sample_arena.record_count(), records);
    assert_eq!(state.sample_arena.retained_bytes(), bytes);
}

/// An expanded side is charged against the candidate limit of the side it
/// feeds, before anything is interned.
#[test]
fn an_expanded_side_respects_the_candidate_limit_of_its_side() {
    let (inputs, base) = unit_fixture(&EDGES, 5, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &base).unwrap();
    let records = state.sample_arena.record_count();
    // 1 -> 2 reads the unit side as rows, 2 -> 1 as columns.
    for (from, to, resource) in [(1, 2, "candidate rows"), (2, 1, "candidate columns")] {
        let options = TreeAciOptions {
            max_candidate_rows: if from == 1 { 1 } else { 1 << 20 },
            max_candidate_cols: if from == 1 { 1 << 20 } else { 1 },
            ..base.clone()
        };
        let forward = directed(&state, from, to);
        let result = expand_unit_vertices(
            &mut state,
            forward,
            &options,
            &mut crate::exec::Exec::Global,
        );
        match result {
            Err(TreeAciError::ResourceLimit {
                resource: actual, ..
            }) => assert_eq!(actual, resource),
            Err(other) => panic!("unexpected error {other:?}"),
            Ok(_) => panic!("{from} -> {to} expanded past its limit"),
        }
        assert_eq!(state.sample_arena.record_count(), records);
    }
}

/// A run on a tree without a unit vertex never expands.
#[test]
fn runs_without_unit_vertices_never_expand() {
    let (_, inputs) = crate::test_support::branched_product_inputs();
    expansion_debug_log::take();
    crate::tree_elementwise_batched(
        crate::test_support::product_of_two,
        &inputs,
        &TreeAciOptions::default(),
    )
    .unwrap();
    assert!(expansion_debug_log::take().is_empty());
}
