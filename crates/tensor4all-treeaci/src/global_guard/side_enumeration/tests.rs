//! Tests of the third Guard stage (#1466).

use std::collections::HashSet;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{
    enumeration_chunk_points, has_unit_vertex, maximal_sides, SideEnumeration,
    SideEnumerationReport, SIDE_ENUMERATION_BUDGET,
};
use crate::{
    global_guard::{find_global_pivots, InputEvaluators},
    state::TreeAciState,
    test_support::unit_junction,
    TreeAciError, TreeAciOptions,
};

/// Edges of a five-node chain `0 - 1 - 2 - 3 - 4`.
const CHAIN: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 4)];

/// Local dimension of every node by position.
fn local_dims(state: &TreeAciState<'_, f64, usize>) -> Vec<usize> {
    state
        .problem
        .physical
        .iter()
        .map(|physical| physical.local_dim)
        .collect()
}

/// Node names of the side nodes, ascending.
fn side_names(state: &TreeAciState<'_, f64, usize>, nodes: &[usize]) -> Vec<usize> {
    let mut names = nodes
        .iter()
        .map(|&position| state.problem.node_order[position])
        .collect::<Vec<_>>();
    names.sort_unstable();
    names
}

/// A rank-1 tree on `edges` with one site per node: node `k` has a site of
/// dimension `dims[k]` and its tensor is the one-hot vector at `target[k]`,
/// so the tree is `scale` at the point `target` and 0 elsewhere.
fn one_hot_tree(
    edges: &[(usize, usize)],
    sites: &[DynIndex],
    target: &[usize],
    scale: f64,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = sites
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(a, b)) in edges.iter().enumerate() {
                if a == node || b == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let mut data = vec![0.0; tensor4all_core::IndexLike::dim(site)];
            data[target[node]] = if node == 0 { scale } else { 1.0 };
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
}

fn sites(dims: &[usize]) -> Vec<DynIndex> {
    dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
}

/// Every point the stage generates, in generation order, with repeats.
fn generated_points(
    anchors: &[Vec<usize>],
    sides: &[Vec<usize>],
    local_dims: &[usize],
) -> Vec<Vec<usize>> {
    let mut points = Vec::new();
    for anchor in anchors {
        for side in sides {
            let configurations: usize = side.iter().map(|&node| local_dims[node]).product();
            for code in 0..configurations {
                let mut point = anchor.clone();
                let mut rest = code;
                for &node in side {
                    point[node] = rest % local_dims[node];
                    rest /= local_dims[node];
                }
                points.push(point);
            }
        }
    }
    points
}

/// Runs `enumeration` and returns its report and every evaluated chunk.
fn run_recording(
    enumeration: &SideEnumeration<'_>,
    chunk_points: usize,
    threshold: f64,
    max_pivots: usize,
    residual: impl Fn(&[usize]) -> f64,
) -> (SideEnumerationReport, Vec<Vec<Vec<usize>>>) {
    let mut chunks = Vec::new();
    let report = enumeration
        .run(chunk_points, threshold, max_pivots, |points| {
            chunks.push(points.to_vec());
            Ok(points.iter().map(|point| residual(point)).collect())
        })
        .unwrap();
    (report, chunks)
}

#[test]
fn the_budget_is_the_treetci_bound_and_chunks_hold_two_to_the_twenty_entries() {
    assert_eq!(SIDE_ENUMERATION_BUDGET, 32);
    assert_eq!(enumeration_chunk_points(16), (1 << 20) / 16);
    assert_eq!(enumeration_chunk_points(0), 1 << 20);
    assert_eq!(enumeration_chunk_points(1 << 21), 1);
}

#[test]
fn only_trees_with_a_vertex_of_local_dimension_one_trigger_the_stage() {
    assert!(has_unit_vertex(&[2, 1, 2]));
    assert!(!has_unit_vertex(&[2, 3, 2]));
    assert!(!has_unit_vertex(&[]));
}

/// On the #1435 table the maximal sides are the three 5-site branches, and
/// the unit centre is in none of them.
#[test]
fn the_maximal_sides_of_the_unit_junction_are_its_three_branches() {
    let inputs = vec![unit_junction::input().unwrap()];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let dims = local_dims(&state);
    let sides = maximal_sides(&state.problem, &dims, SIDE_ENUMERATION_BUDGET).unwrap();

    let mut names = sides
        .iter()
        .map(|side| side_names(&state, &side.nodes))
        .collect::<Vec<_>>();
    names.sort();
    let branch = |branch: usize| {
        (0..unit_junction::R)
            .map(|level| 3 * level + branch)
            .collect::<Vec<_>>()
    };
    assert_eq!(names, vec![branch(0), branch(1), branch(2)]);
    assert!(sides.iter().all(|side| side.configurations == 32));
}

/// A side over the budget is not enumerated, a side contained in a larger
/// fitting side is dropped, and a side of unit vertices only never fits.
#[test]
fn maximal_sides_respect_the_budget_containment_and_unit_vertices() {
    // 0 - 1 - 2 - 3 - 4 with node 2 of local dimension 1.
    let dims = [2, 2, 1, 2, 2];
    let sites = sites(&dims);
    let inputs = vec![one_hot_tree(&CHAIN, &sites, &[0; 5], 1.0)];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let local = local_dims(&state);

    // Budget 4: {0, 1} (through the unit 2 as well) and {3, 4}.
    let mut sides = maximal_sides(&state.problem, &local, 4)
        .unwrap()
        .into_iter()
        .map(|side| (side_names(&state, &side.nodes), side.configurations))
        .collect::<Vec<_>>();
    sides.sort();
    assert_eq!(sides, vec![(vec![0, 1], 4), (vec![3, 4], 4)]);

    // Budget 1: nothing has 2 or more configurations within it.
    assert!(maximal_sides(&state.problem, &local, 1).unwrap().is_empty());

    // Budget 8: the two overlapping 3-site sides around the unit vertex.
    let mut sides = maximal_sides(&state.problem, &local, 8)
        .unwrap()
        .into_iter()
        .map(|side| side_names(&state, &side.nodes))
        .collect::<Vec<_>>();
    sides.sort();
    assert_eq!(sides, vec![vec![0, 1, 3], vec![1, 3, 4]]);
}

/// Every distinct generated point is evaluated exactly once, at the place of
/// its first generator, whatever the anchors share.
#[test]
fn deduplication_evaluates_each_point_once_in_first_generator_order() {
    // 0 - 1 - 2 - 3 with node 3 of local dimension 1; budget 4 gives the
    // overlapping sides {0, 1} and {1, 2}.
    let edges = [(0, 1), (1, 2), (2, 3)];
    let dims = [2, 2, 2, 1];
    let sites = sites(&dims);
    let inputs = vec![one_hot_tree(&edges, &sites, &[0; 4], 1.0)];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let local = local_dims(&state);
    let position = |name: usize| state.problem.node_order.iter().position(|&n| n == name);
    let at = |coordinates: [usize; 4]| {
        let mut point = vec![0; 4];
        for (name, coordinate) in coordinates.into_iter().enumerate() {
            point[position(name).unwrap()] = coordinate;
        }
        point
    };
    // The second anchor differs from the first inside {0, 1} only, the third
    // repeats the first, the fourth differs on node 0 and node 2.
    let anchors = vec![
        at([0, 0, 0, 0]),
        at([1, 1, 0, 0]),
        at([0, 0, 0, 0]),
        at([1, 0, 1, 0]),
    ];
    let enumeration = SideEnumeration::new(&state.problem, &local, &anchors, 4).unwrap();
    assert_eq!(
        enumeration.anchors.len(),
        3,
        "the repeated anchor is kept once"
    );
    assert_eq!(enumeration.point_bound().unwrap(), 3 * (4 + 4));

    let (report, chunks) = run_recording(&enumeration, 1 << 10, f64::INFINITY, 5, |_| 0.0);

    let side_nodes = enumeration
        .sides
        .iter()
        .map(|side| side.nodes.clone())
        .collect::<Vec<_>>();
    let mut seen = HashSet::new();
    let expected = generated_points(&anchors, &side_nodes, &local)
        .into_iter()
        .filter(|point| seen.insert(point.clone()))
        .collect::<Vec<_>>();
    let evaluated = chunks.concat();
    assert_eq!(evaluated, expected);
    assert_eq!(report.evaluated_points, expected.len());
    // 6 points around the first anchor, 2 new ones around the second (its
    // {0, 1} block is the first anchor's) and none around the fourth: the 8
    // points of the three binary nodes, from 3 * 8 generated.
    assert_eq!(expected.len(), 8);
}

/// Splitting the stage into chunks changes neither the points, their order,
/// nor the report; every chunk but the last is full.
#[test]
fn chunk_boundaries_do_not_change_the_points_or_the_result() {
    let inputs = vec![unit_junction::input().unwrap()];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let local = local_dims(&state);
    let mut rng = ChaCha8Rng::seed_from_u64(1466);
    let anchors = (0..5)
        .map(|_| {
            local
                .iter()
                .map(|&dim| rng.random_range(0..dim))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let enumeration = SideEnumeration::new(&state.problem, &local, &anchors, 32).unwrap();
    // A residual with many ties and a spread of values.
    let residual = |point: &[usize]| point.iter().sum::<usize>() as f64;

    let (whole, whole_chunks) = run_recording(&enumeration, usize::MAX, 6.0, 4, residual);
    assert_eq!(whole_chunks.len(), 1);
    let points = whole_chunks.concat();
    // 5 anchors, 3 sides of 32 configurations each, minus repeats.
    assert!(points.len() <= 5 * 3 * 32);
    assert_eq!(whole.evaluated_points, points.len());
    for chunk_points in [1, 2, 7, 32, points.len() - 1, points.len()] {
        let (report, chunks) = run_recording(&enumeration, chunk_points, 6.0, 4, residual);
        assert_eq!(report, whole, "chunk length {chunk_points}");
        assert_eq!(chunks.concat(), points, "chunk length {chunk_points}");
        let (last, full) = chunks.split_last().unwrap();
        assert!(full.iter().all(|chunk| chunk.len() == chunk_points));
        assert!(!last.is_empty() && last.len() <= chunk_points);
    }
    // Zero is read as one point per chunk.
    let (report, chunks) = run_recording(&enumeration, 0, 6.0, 4, residual);
    assert_eq!(report, whole);
    assert!(chunks.iter().all(|chunk| chunk.len() == 1));
}

/// At most `max_pivots` points are kept, strongest first, ties in
/// evaluation order, and only strictly above the threshold.
#[test]
fn the_report_keeps_the_strongest_points_up_to_the_cap() {
    let dims = [2, 2, 1, 2, 2];
    let sites = sites(&dims);
    let inputs = vec![one_hot_tree(&CHAIN, &sites, &[0; 5], 1.0)];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let local = local_dims(&state);
    let anchors = vec![vec![0; 5]];
    let enumeration = SideEnumeration::new(&state.problem, &local, &anchors, 4).unwrap();
    let residual = |point: &[usize]| point.iter().sum::<usize>() as f64;

    let (all, chunks) = run_recording(&enumeration, 3, 0.0, usize::MAX, residual);
    let evaluated = chunks.concat();
    let mut expected = evaluated
        .iter()
        .map(|point| (residual(point), point.clone()))
        .filter(|(residual, _)| *residual > 0.0)
        .collect::<Vec<_>>();
    expected.sort_by(|(left, _), (right, _)| right.total_cmp(left));
    assert_eq!(all.found, expected);

    for cap in [1, 2, 3] {
        let (report, _) = run_recording(&enumeration, 3, 0.0, cap, residual);
        assert_eq!(report.found, expected[..cap], "cap {cap}");
        assert_eq!(report.evaluated_points, evaluated.len());
    }
    // A residual equal to the threshold does not qualify.
    let (report, _) = run_recording(&enumeration, 3, 2.0, usize::MAX, residual);
    assert!(report.found.is_empty());
    // A zero cap evaluates nothing.
    let (report, chunks) = run_recording(&enumeration, 3, 0.0, 0, residual);
    assert_eq!(report, SideEnumerationReport::default());
    assert!(chunks.is_empty());
}

#[test]
fn an_evaluator_returning_the_wrong_count_is_an_internal_error() {
    let dims = [2, 2, 1, 2, 2];
    let sites = sites(&dims);
    let inputs = vec![one_hot_tree(&CHAIN, &sites, &[0; 5], 1.0)];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let local = local_dims(&state);
    let anchors = vec![vec![0; 5]];
    let enumeration = SideEnumeration::new(&state.problem, &local, &anchors, 4).unwrap();
    let error = enumeration
        .run(4, 0.0, 1, |points| Ok(vec![0.0; points.len() + 1]))
        .unwrap_err();
    assert!(matches!(error, TreeAciError::InternalInvariant { .. }));
    let error = SideEnumeration::new(&state.problem, &local, &[vec![0; 4]], 4).unwrap_err();
    assert!(matches!(error, TreeAciError::InternalInvariant { .. }));
}

/// The Guard of one call on a chain `y0 - y1 - y2 - c - x` whose target is 1
/// at one point and 0 elsewhere, with an exactly zero output and one random
/// start. `y_flips` sets how many `y` bits of the start the support differs
/// in; `unit` gives `c` local dimension 1 (else a binary site whose target
/// coordinate is the start's). Returns the Guard pivots, the support point,
/// and the largest batch the operator saw.
struct GuardCall {
    pivots: Vec<Vec<usize>>,
    support: Vec<usize>,
    largest_batch: usize,
    result: crate::Result<()>,
}

fn guard_call(y_flips: usize, unit: bool, retain_support: bool, working_bytes: usize) -> GuardCall {
    // Node names: y0 = 0, y1 = 1, y2 = 2, c = 3, x = 4.
    let dims = [2, 2, 2, if unit { 1 } else { 2 }, 2];
    let sites = sites(&dims);
    let placeholder = vec![one_hot_tree(&CHAIN, &sites, &[0; 5], 1.0)];
    let options = TreeAciOptions {
        nsearch_global_pivots: 1,
        max_nglobal_pivots: 2,
        nsweeps_global_search: 4,
        global_tolerance_margin: 1.0,
        ..TreeAciOptions::default()
    };
    // The start depends only on the RNG and the local dimensions: draw it as
    // the Guard does, then put the support relative to it.
    let probe = TreeAciState::<f64, usize>::initialize(&placeholder, &options).unwrap();
    let local = local_dims(&probe);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let start = local
        .iter()
        .map(|&dim| rng.random_range(0..dim))
        .collect::<Vec<_>>();
    let mut support = start.clone();
    let mut target = vec![0; 5];
    for (position, &name) in probe.problem.node_order.iter().enumerate() {
        let flip = name < y_flips;
        if flip {
            support[position] = 1 - start[position];
        }
        target[name] = support[position];
    }
    drop(probe);
    let inputs = vec![one_hot_tree(&CHAIN, &sites, &target, 1.0)];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = one_hot_tree(&CHAIN, &sites, &target, 0.0);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    evaluators.max_working_bytes = working_bytes;
    let mut largest_batch = 0;
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        largest_batch = largest_batch.max(output.len());
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };
    let known = if retain_support {
        vec![support.clone()]
    } else {
        Vec::new()
    };
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut ChaCha8Rng::seed_from_u64(0),
        &mut operator,
        &known,
        &mut crate::exec::Exec::Global,
    );
    let (pivots, result) = match report {
        Ok(report) => (report.pivots, Ok(())),
        Err(error) => (Vec::new(), Err(error)),
    };
    GuardCall {
        pivots,
        support,
        largest_batch,
        result,
    }
}

/// Walks and the start batch evaluate at most one point per batch here, so a
/// larger batch is the third stage.
const WALK_BATCH: usize = 1;

/// A support three `y` flips from the only start: the walk sees only zero
/// residuals, so on a tree with a unit vertex the stage runs and finds it.
#[test]
fn the_stage_finds_a_support_the_walks_cannot_reach_on_a_unit_vertex_tree() {
    let call = guard_call(3, true, false, usize::MAX);
    call.result.unwrap();
    assert!(call.largest_batch > WALK_BATCH, "the stage ran");
    assert_eq!(call.pivots, vec![call.support]);
}

/// The same target on a tree without a unit vertex: the stage does not run,
/// and the Guard returns nothing.
#[test]
fn the_stage_does_not_run_on_a_tree_without_a_unit_vertex() {
    let call = guard_call(3, false, false, usize::MAX);
    call.result.unwrap();
    assert_eq!(call.largest_batch, WALK_BATCH, "the stage did not run");
    assert!(call.pivots.is_empty());
}

/// When a walk finds the support (one flip away), the stage does not run.
#[test]
fn the_stage_does_not_run_when_a_walk_found_something() {
    let call = guard_call(1, true, false, usize::MAX);
    call.result.unwrap();
    assert_eq!(call.largest_batch, WALK_BATCH, "the stage did not run");
    assert_eq!(call.pivots, vec![call.support]);
}

/// When a revalidated retained pivot still fails, the stage does not run.
#[test]
fn the_stage_does_not_run_when_a_retained_pivot_still_fails() {
    let call = guard_call(3, true, true, usize::MAX);
    call.result.unwrap();
    // The start batch holds the random start and the retained point.
    assert_eq!(call.largest_batch, 2, "the stage did not run");
    assert_eq!(call.pivots, vec![call.support]);
}

/// The stage's largest chunk is charged against the working budget before it
/// evaluates anything: just below what the call needs, the start batch and
/// the walk run and the stage fails with `ResourceLimit`.
#[test]
fn the_stage_is_checked_against_the_working_budget_ahead() {
    // The smallest budget the whole call fits.
    let (mut low, mut high) = (0usize, 1usize << 20);
    assert!(guard_call(3, true, false, high).result.is_ok());
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if guard_call(3, true, false, middle).result.is_ok() {
            high = middle;
        } else {
            low = middle;
        }
    }
    let call = guard_call(3, true, false, high - 1);
    assert!(matches!(
        call.result,
        Err(TreeAciError::ResourceLimit {
            resource: "working bytes",
            requested,
            limit,
        }) if requested == high && limit == high - 1
    ));
    // The start batch and the walk ran; the stage evaluated nothing.
    assert_eq!(call.largest_batch, WALK_BATCH);
}
