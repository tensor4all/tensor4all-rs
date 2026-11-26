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
    // clone of the store and keeps what the failed evaluation computed over
    // retained samples (see the next test for the rolled-back ones).
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

/// A failed proposal memoizes candidate frames over the expanded samples;
/// `abandon` must forget them, because the rollback hands their ids out
/// again. Otherwise a later sample with a reused id would get the frame of
/// the sample it replaced (#1464).
#[test]
fn a_reused_sample_id_does_not_return_a_stale_memoized_frame() {
    // Node 1 is the unit vertex; node 0 has a leaf child 5, so the expanded
    // bond out of node 0 or 3 has samples outside what the expansion reads.
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4), (0, 5)];
    let (inputs, options) = unit_fixture(&edges, 6, &[1]);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let forward = directed(&state, 1, 2);
    let [target, other] = state.problem.directed_edges[forward].incoming_to_from[..] else {
        panic!("node 1 has two other bonds");
    };
    let [child] = state.problem.directed_edges[target].incoming_to_from[..] else {
        panic!("the expanded bond's source has one other bond");
    };
    let freed = state.sample_arena.directed_record_count(target).unwrap();
    let mut failing = |_: crate::TreeElementwiseBatch<'_, f64>, _: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "stop".into(),
        })
    };
    expansion_debug_log::take();
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
    assert_eq!(
        expansion_debug_log::take(),
        vec![(forward, vec![(target, 1)])]
    );
    assert_eq!(
        state.sample_arena.directed_record_count(target).unwrap(),
        freed
    );
    let kept = (0..state.problem.directed_edges.len())
        .map(|edge| state.sample_arena.directed_record_count(edge).unwrap())
        .collect::<Vec<_>>();
    assert!(
        state
            .input_frames
            .candidate_cache_names_only_kept(&state.problem, &kept),
        "the memo still names a rolled-back sample"
    );

    // A sample the expansion never offered: the child's coordinate that is
    // not among its candidates. It takes the first freed id.
    let unused = enumerate_candidates(&state.problem, &state.candidates, child, "test", 2)
        .unwrap()
        .into_iter()
        .map(|sample| {
            state
                .sample_arena
                .intern_component(&state.problem, child, sample)
                .unwrap()
        })
        .find(|id| !state.candidates.ids[child].contains(id))
        .unwrap();
    let sample = crate::samples::ComponentSample {
        local_coordinate: 0,
        incoming: vec![(child, unused)],
    };
    let reused = state
        .sample_arena
        .intern_component(&state.problem, target, sample)
        .unwrap();
    assert_eq!(reused, freed, "the rollback hands the id out again");
    state.input_frames = state
        .input_frames
        .extend(
            &inputs,
            &state.problem,
            &state.sample_arena,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();

    // The row of the commit that names the reused id.
    let row = crate::samples::ComponentSample {
        local_coordinate: 0,
        incoming: vec![(target, reused), (other, state.candidates.ids[other][0])],
    };
    let memoized = state
        .input_frames
        .candidate_frame(&inputs, &state.problem, 0, forward, &row, &mut None)
        .unwrap();
    let fresh = crate::frames::InputFrameStore::<f64>::from_samples(
        &inputs,
        &state.problem,
        &state.sample_arena,
        &mut crate::exec::Exec::Global,
    )
    .unwrap()
    .candidate_frame(&inputs, &state.problem, 0, forward, &row, &mut None)
    .unwrap();
    assert_eq!(memoized, fresh);
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

/// Runs one transaction on `forward` with the identity operator and returns
/// its report together with the expansions it made.
fn commit(
    state: &mut TreeAciState<'_, f64, usize>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<usize>,
) -> (
    crate::transaction::EdgeCommitReport,
    Vec<expansion_debug_log::Entry>,
) {
    expansion_debug_log::take();
    let report = update_edge_transaction(
        state,
        forward,
        options,
        true,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    (report, expansion_debug_log::take())
}

/// Appends every side sample of `edge` to its candidate set, with frames.
fn widen(state: &mut TreeAciState<'_, f64, usize>, edge: DirectedEdgeId) {
    state
        .grow_candidate_sets(
            &mut crate::exec::Exec::Global,
            |arena, problem, candidates| {
                let mut grown = candidates.clone();
                for sample in enumerate_candidates(problem, candidates, edge, "test", usize::MAX)? {
                    let id = arena.intern_component(problem, edge, sample)?;
                    grown.push_unique(edge, id);
                }
                Ok(Some((grown, ())))
            },
        )
        .unwrap()
        .unwrap();
}

/// An expansion whose enumeration alone exceeds the candidate limit of the
/// side it feeds is skipped before anything is interned, and the commit runs
/// on the plain sets instead of failing (#1464).
#[test]
fn an_enumeration_over_the_candidate_limit_of_its_side_is_skipped() {
    let (inputs, base) = unit_fixture(&EDGES, 5, &[1]);
    // 1 -> 2 reads the unit side as rows, 2 -> 1 as columns. The plain unit
    // side has one candidate; the expanded bond enumerates two.
    for (from, to) in [(1, 2), (2, 1)] {
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &base).unwrap();
        let records = state.sample_arena.record_count();
        let candidates = state.candidates.clone();
        let options = TreeAciOptions {
            max_candidate_rows: if from == 1 { 1 } else { 1 << 20 },
            max_candidate_cols: if from == 1 { 1 << 20 } else { 1 },
            ..base.clone()
        };
        let forward = directed(&state, from, to);
        assert_eq!(
            unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward)
                .unwrap()
                .len(),
            1,
            "{from} -> {to} qualifies for an expansion"
        );
        let expansion = expand_unit_vertices(
            &mut state,
            forward,
            &options,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        assert!(
            expansion.is_none(),
            "{from} -> {to} expanded past its limit"
        );
        assert_eq!(state.sample_arena.record_count(), records);
        assert_eq!(state.candidates, candidates);

        // The commit sees the 1 x 2 (or 2 x 1) plain matrix.
        let (report, expansions) = commit(&mut state, forward, &options);
        assert_eq!(report.evaluated_points, 2);
        assert!(expansions.is_empty());
    }
}

/// An expansion that fits its enumeration but leaves the enlarged side over
/// the limit is rolled back and skipped; one exactly at the limit stays.
#[test]
fn an_enlarged_side_over_the_candidate_limit_is_rolled_back() {
    let (inputs, base) = unit_fixture(&EDGES, 5, &[1]);
    let fresh = || {
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &base).unwrap();
        let forward = directed(&state, 1, 2);
        let target = state.problem.directed_edges[forward].incoming_to_from[0];
        let other = state.problem.directed_edges[forward].incoming_to_from[1];
        // Two candidates on the other bond: the rows are 1 x 2 plain and
        // 2 x 2 expanded, while the expanded bond enumerates only two.
        widen(&mut state, other);
        assert_eq!(state.candidates.ids[other].len(), 2);
        (state, forward, target)
    };
    let (mut state, forward, _) = fresh();
    let records = state.sample_arena.record_count();
    let candidates = state.candidates.clone();
    let tight = TreeAciOptions {
        max_candidate_rows: 3,
        ..base.clone()
    };
    let expansion =
        expand_unit_vertices(&mut state, forward, &tight, &mut crate::exec::Exec::Global).unwrap();
    assert!(expansion.is_none(), "4 rows exceed the limit of 3");
    assert_eq!(state.sample_arena.record_count(), records);
    assert_eq!(state.candidates, candidates);
    let (report, expansions) = commit(&mut state, forward, &tight);
    assert_eq!(report.evaluated_points, 2 * 2, "plain rows times leaf 2");
    assert!(expansions.is_empty());

    let (mut state, forward, target) = fresh();
    let at_limit = TreeAciOptions {
        max_candidate_rows: 4,
        ..base.clone()
    };
    let (report, expansions) = commit(&mut state, forward, &at_limit);
    assert_eq!(report.evaluated_points, 4 * 2);
    assert_eq!(expansions, vec![(forward, vec![(target, 1)])]);
}

/// An expansion that would push the local matrix past the element ceiling is
/// skipped, and the commit runs on the plain matrix.
#[test]
fn an_expansion_over_the_local_matrix_ceiling_is_skipped() {
    let (inputs, options) = unit_fixture(&EDGES, 5, &[1]);
    for (ceiling, expected_points, expands) in [(2, 2, false), (4, 4, true)] {
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let forward = directed(&state, 1, 2);
        let target = state.problem.directed_edges[forward].incoming_to_from[0];
        // Plain 1 x 2, expanded 2 x 2.
        state.problem.max_local_matrix_elements = ceiling;
        let (report, expansions) = commit(&mut state, forward, &options);
        assert_eq!(
            report.evaluated_points, expected_points,
            "ceiling {ceiling}"
        );
        if expands {
            assert_eq!(expansions, vec![(forward, vec![(target, 1)])]);
        } else {
            assert!(expansions.is_empty(), "ceiling {ceiling}");
        }
    }
}

/// Each target is charged on its own: a column target over its limit is
/// skipped while the row target of the same commit still expands.
#[test]
fn a_skipped_target_leaves_the_other_side_expanded() {
    // Nodes 1 and 3 are unit vertices of degree 3; every leaf is binary.
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)];
    let (inputs, base) = unit_fixture(&edges, 6, &[1, 3]);
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &base).unwrap();
    let forward = directed(&state, 1, 3);
    let reverse = state.problem.directed_edges[forward].reverse;
    let row_target = state.problem.directed_edges[forward].incoming_to_from[0];
    let col_target = state.problem.directed_edges[reverse].incoming_to_from[0];
    drop(state);
    for (max_candidate_cols, expected) in [
        (1 << 20, vec![(row_target, 1), (col_target, 1)]),
        (1, vec![(row_target, 1)]),
    ] {
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &base).unwrap();
        let options = TreeAciOptions {
            max_candidate_cols,
            ..base.clone()
        };
        let expansion = expand_unit_vertices(
            &mut state,
            forward,
            &options,
            &mut crate::exec::Exec::Global,
        )
        .unwrap()
        .expect("the row side expands");
        assert_eq!(expansion.restore(), expected.as_slice());
        expansion.finish(&mut state);
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
