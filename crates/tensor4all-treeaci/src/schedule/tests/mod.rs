use std::{cell::Cell, collections::HashSet};

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::{
    convergence_criterion, current_state_is_rank_limited, global_injection_capacities,
    run_directional_pass, run_local_sweeps, track_rank_stability, PassDirection,
};
use crate::global_guard::input_evaluator_debug_stats;
use crate::residual_refresh::refresh_stale_residuals;
use crate::test_support::{branched_product_inputs, product_of_two};
use crate::transaction::update_edge_transaction;
use crate::{state::TreeAciState, TreeAciError, TreeAciOptions, TreeAciTermination};

fn seeded_rng() -> rand_chacha::ChaCha8Rng {
    <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(0)
}

fn product_tree(edges: &[(usize, usize)], node_count: usize) -> TreeTN<IdxTensor, usize> {
    let physical = (0..node_count)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = (0..node_count)
        .map(|node| {
            let mut indices = vec![physical[node].clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            let values = (0..len)
                .map(|linear| (node + 1) as f64 + linear as f64)
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..node_count).collect()).unwrap()
}

fn identity(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)?;
    }
    Ok(())
}

fn rank_two_delta_tree() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left =
        IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn product_tree_with_physical(
    edges: &[(usize, usize)],
    physical: &[DynIndex],
    offset: f64,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = physical
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            IdxTensor::from_dense(
                indices,
                (0..len)
                    .map(|coordinate| offset + node as f64 + coordinate as f64)
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}

#[test]
fn path_pass_matches_train_endpoint_order_and_exact_reverse() {
    let options = TreeAciOptions::default();
    let inputs = vec![product_tree(&[(0, 1), (1, 2), (2, 3)], 4)];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();

    let forward = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Forward,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    let reverse = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Reverse,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    assert_eq!(forward.updated_edges, vec![5, 3, 1]);
    assert_eq!(reverse.updated_edges, vec![0, 2, 4]);
    assert_eq!(forward.update_count(), 3);
    assert_eq!(reverse.update_count(), 3);
    assert_eq!(state.output.canonical_region(), &HashSet::from([3]));
    assert_eq!(state.output.canonical_form(), None);
}

#[test]
fn branched_topologies_cover_every_edge_with_optimal_retracing() {
    for (edges, expected_forward_updates, expected_reverse_updates) in [
        (vec![(0, 1), (0, 2), (0, 3)], 4, 2),
        (vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 8, 4),
        (vec![(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)], 7, 3),
    ] {
        let options = TreeAciOptions::default();
        let inputs = vec![product_tree(&edges, edges.len() + 1)];
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let forward = run_directional_pass(
            &mut state,
            &options,
            PassDirection::Forward,
            false,
            &mut identity,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        let reverse = run_directional_pass(
            &mut state,
            &options,
            PassDirection::Reverse,
            false,
            &mut identity,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        assert_eq!(forward.update_count(), expected_forward_updates);
        assert_eq!(reverse.update_count(), expected_reverse_updates);
        let mut round = forward.updated_edges.clone();
        round.extend(&reverse.updated_edges);
        round.sort_unstable();
        assert_eq!(round, (0..2 * edges.len()).collect::<Vec<_>>());
        state.output.verify_internal_consistency().unwrap();
    }
}

#[test]
fn failed_update_preserves_all_commits_before_the_failing_edge() {
    let options = TreeAciOptions::default();
    let inputs = vec![product_tree(&[(0, 1), (1, 2), (2, 3)], 4)];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let generation_before = state.generation;
    let calls = Cell::new(0usize);
    let mut fail_after_one = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        if calls.get() > 1 {
            return Err(TreeAciError::Callback {
                message: "stop after the first committed edge".into(),
            });
        }
        identity(batch, output)
    };
    let result = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Forward,
        false,
        &mut fail_after_one,
        &mut crate::exec::Exec::Global,
    );

    assert!(matches!(result, Err(TreeAciError::Callback { .. })));
    assert_eq!(state.generation, generation_before + 1);
    assert_eq!(state.output.canonical_region(), &HashSet::from([2]));
    assert_eq!(state.output.canonical_form(), None);
    state.output.verify_internal_consistency().unwrap();
}

#[test]
fn convergence_requires_stable_edge_ranks_and_minimum_dwell() {
    let errors = vec![0.0, 0.0];
    let global = vec![0, 0];

    // Below the minimum dwell (only 1 completed sweep): never converges.
    assert!(!convergence_criterion(
        1,
        1,
        &errors[..1],
        &global[..1],
        2,
        1.0e-12
    ));

    // Two sweeps, stable edge ranks, error at tolerance, no global pivots
    // found in the window: must converge.
    assert!(convergence_criterion(2, 2, &errors, &global, 2, 1.0e-12));

    // Growth on any edge restarts the rank-stability window.
    assert!(!convergence_criterion(2, 1, &errors, &global, 2, 1.0e-12));

    // Error above tolerance still blocks convergence, independent of rank.
    assert!(!convergence_criterion(
        2,
        2,
        &[0.0, 1.0e-6],
        &global,
        2,
        1.0e-12
    ));

    // A global pivot found within the window still blocks convergence
    // (its error estimate is stale with respect to the injected pivot).
    assert!(!convergence_criterion(2, 2, &errors, &[1, 0], 2, 1.0e-12));
}

#[test]
fn smaller_cut_growth_blocks_convergence_even_when_the_maximum_does_not_grow() {
    for largest in [233, 232] {
        let mut previous = vec![192, 229, 233, 18];
        let current = [195, 229, largest, 19];
        let stable = track_rank_stability(&mut previous, &current, 4);
        assert_eq!(stable, 1);
        assert_eq!(previous, current);
        assert!(!convergence_criterion(
            2, stable, &[0.0; 2], &[0; 2], 2, 1e-12
        ));
    }
}

#[test]
fn rank_stability_allows_shrinkage_but_restarts_on_regrowth() {
    let mut previous = vec![4, 3];
    let mut stable = track_rank_stability(&mut previous, &[4, 3], 0);
    assert_eq!(stable, 1);
    stable = track_rank_stability(&mut previous, &[4, 2], stable);
    assert_eq!(stable, 2);
    stable = track_rank_stability(&mut previous, &[4, 3], stable);
    assert_eq!(stable, 1);
    stable = track_rank_stability(&mut previous, &[4, 3], stable);
    assert_eq!(stable, 2);
    assert!(convergence_criterion(
        2, stable, &[0.0; 2], &[0; 2], 2, 1e-12
    ));
}

#[test]
fn relative_and_absolute_tolerance_policy_uses_consistent_units() {
    let relative = TreeAciOptions::<usize>::default().tolerance_policy();
    let absolute = TreeAciOptions::<usize> {
        scale_tolerance: false,
        ..TreeAciOptions::default()
    }
    .tolerance_policy();

    assert_eq!(relative.local_normalizer(10.0), 10.0);
    assert_eq!(relative.local_normalizer(0.0), 1.0);
    assert_eq!(absolute.local_normalizer(10.0), 1.0);
    assert_eq!(relative.local_threshold(), 1.0e-12);
    assert_eq!(relative.absolute_threshold(10.0), 1.0e-11);
    assert_eq!(relative.absolute_threshold(0.0), 1.0e-12);
    assert_eq!(absolute.absolute_threshold(10.0), 1.0e-12);
    assert_eq!(relative.error_metric(2.0, 10.0), 0.2);
    assert_eq!(absolute.error_metric(2.0, 10.0), 2.0);
    assert_eq!(relative.error_metric(2.0, 0.0), 2.0);
    assert!(relative.exceeds(2.0e-11, 10.0));
    assert!(!relative.exceeds(1.0e-11, 10.0));
    assert!(absolute.exceeds(2.0e-12, 10.0));
}

#[test]
fn algebraically_saturated_cut_cannot_receive_global_rank_growth() {
    let options = TreeAciOptions::default();
    let inputs = vec![rank_two_delta_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    assert_eq!(state.edge_ranks, vec![2]);
    assert_eq!(state.algebraic_edge_bounds, vec![2]);
    assert_eq!(global_injection_capacities(&state, &options), vec![0]);

    state.edge_errors[0] = 1.0;
    state.edge_scales[0] = 1.0;
    assert!(current_state_is_rank_limited(&state, &options));
}

#[test]
fn local_sweeps_honor_convergence_and_rank_limit_dwell() {
    let converged_options = TreeAciOptions {
        enable_global_guard: false,
        min_sweeps: 2,
        max_sweeps: 5,
        ..TreeAciOptions::default()
    };
    let converged_inputs = vec![product_tree(&[(0, 1), (1, 2)], 3)];
    let mut converged_state =
        TreeAciState::<f64, usize>::initialize(&converged_inputs, &converged_options).unwrap();
    input_evaluator_debug_stats::reset();
    let converged = run_local_sweeps(
        &mut converged_state,
        &converged_options,
        &mut identity,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(converged.termination, TreeAciTermination::Converged);
    assert_eq!(converged.max_ranks.len(), 2);
    assert_eq!(converged.max_errors.len(), 2);
    assert_eq!(input_evaluator_debug_stats::constructions(), 0);

    let limited_options = TreeAciOptions {
        enable_global_guard: false,
        max_bond_dim: Some(1),
        min_sweeps: 2,
        max_sweeps: 5,
        ..TreeAciOptions::default()
    };
    let limited_inputs = vec![rank_two_delta_tree()];
    let mut limited_state =
        TreeAciState::<f64, usize>::initialize(&limited_inputs, &limited_options).unwrap();
    let limited = run_local_sweeps(
        &mut limited_state,
        &limited_options,
        &mut identity,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(limited.termination, TreeAciTermination::RankLimited);
    assert_eq!(limited.max_ranks, vec![1, 1]);
    assert!(limited.max_errors.iter().all(|error| *error > 1.0e-12));
}

#[test]
fn continuous_minimum_retracing_walk_preserves_pivot_gauges_on_binary_tree() {
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
    let physical = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let left = product_tree_with_physical(&edges, &physical, 1.0);
    let right = product_tree_with_physical(&edges, &physical, 2.0);
    let expected_left = left.to_dense().unwrap();
    let expected_right = right
        .to_dense()
        .unwrap()
        .permute_indices(expected_left.indices())
        .unwrap();
    let expected = expected_left
        .to_vec::<f64>()
        .unwrap()
        .into_iter()
        .zip(expected_right.to_vec::<f64>().unwrap())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let options = TreeAciOptions {
        root: Some(3),
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let inputs = vec![left, right];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut add = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)? + batch.get(1, point)?;
        }
        Ok(())
    };
    let forward = [5, 6, 7, 1, 2, 10, 11, 8];
    for directed in forward {
        update_edge_transaction(
            &mut state,
            directed,
            &options,
            true,
            false,
            &mut add,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
    }
    for directed in forward.into_iter().rev().map(|directed| directed ^ 1) {
        update_edge_transaction(
            &mut state,
            directed,
            &options,
            true,
            false,
            &mut add,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
    }
    let actual = state
        .output
        .to_dense()
        .unwrap()
        .permute_indices(expected_left.indices())
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-9));
}

/// Classification and injection capacity answer different questions. A
/// saturated bad cut may be locally rank-limited even if an accurate cut has
/// headroom. Retain the early classification and lazy capacity scan contract.
#[test]
fn rank_limit_and_capacity_policy_preserve_mixed_cut_and_tolerance_semantics() {
    let inputs = vec![product_tree(&[(0, 1), (1, 2), (2, 3)], 4)];
    let options = TreeAciOptions {
        tolerance: 1e-3,
        max_bond_dim: Some(4),
        ..TreeAciOptions::default()
    };
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    // This test isolates the scheduler's per-cut metadata policy. No local
    // contraction is performed with these deliberately selected dimensions.
    state.edge_ranks = vec![2, 2, 3];
    state.algebraic_edge_bounds = vec![2, 5, 3];
    state.edge_scales = vec![1.0; 3];
    for errors in [
        vec![0.0; 3],
        vec![0.01, 0.0, 0.0],
        vec![0.0, 0.01, 0.0],
        vec![0.01, 0.01, 0.0],
    ] {
        let expected_limited = errors == [0.01, 0.0, 0.0];
        state.edge_errors = errors;
        for guard in [false, true] {
            for searches in [0, 2] {
                let configured = TreeAciOptions {
                    enable_global_guard: guard,
                    nsearch_global_pivots: searches,
                    ..options.clone()
                };
                assert_eq!(
                    current_state_is_rank_limited(&state, &configured),
                    expected_limited
                );
                assert_eq!(global_injection_capacities(&state, &configured), [0, 2, 0]);
            }
        }
    }
    state.edge_errors = vec![0.05, 0.0, 0.0];
    state.edge_scales = vec![100.0; 3];
    assert!(!current_state_is_rank_limited(&state, &options));
    let absolute = TreeAciOptions {
        scale_tolerance: false,
        ..options.clone()
    };
    assert!(current_state_is_rank_limited(&state, &absolute));
    let uncapped = TreeAciOptions {
        max_bond_dim: None,
        ..options.clone()
    };
    assert_eq!(global_injection_capacities(&state, &uncapped), [0, 3, 0]);
    let saturated = TreeAciOptions {
        max_bond_dim: Some(1),
        ..absolute
    };
    assert_eq!(global_injection_capacities(&state, &saturated), [0, 0, 0]);
    assert!(current_state_is_rank_limited(&state, &saturated));
    state.edge_errors.fill(0.0);
    assert!(!current_state_is_rank_limited(&state, &saturated));
}

/// Directed edge `from -> to` of a prepared problem with `usize` node names.
fn directed_between(state: &TreeAciState<'_, f64, usize>, from: usize, to: usize) -> usize {
    state
        .problem
        .directed_edges
        .iter()
        .position(|edge| edge.from == from && edge.to == to)
        .unwrap()
}

#[test]
fn return_pass_recommits_off_spine_bonds_the_guard_enlarged() {
    // Spine 0-1-2-3 from root 0, with the branch 1-4-{5, 6} and the leaf 2-7.
    let edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (4, 6), (2, 7)];
    let physical = (0..8).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let inputs = vec![crate::test_support::random_decaying_tree::<f64>(
        &edges,
        &physical,
        2,
        0.5,
        &mut crate::test_support::SplitMix(5),
    )];
    let options = TreeAciOptions {
        root: Some(0),
        tolerance: 0.3,
        ..TreeAciOptions::default()
    };
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    run_directional_pass(
        &mut state,
        &options,
        PassDirection::Forward,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(crate::test_support::pivot_basis_deviation(&state), 0.0);
    let capacities = global_injection_capacities(&state, &options);
    let injected = crate::global_guard::inject_global_pivots(
        &mut state,
        &[vec![0, 1, 0, 1, 1, 1, 0, 1]],
        &capacities,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(injected, 1);
    let step = |from, to| directed_between(&state, from, to);
    let grown = |state: &TreeAciState<'_, f64, usize>, from, to| {
        state
            .candidate_set_outgrew_basis(directed_between(state, from, to))
            .unwrap()
    };
    // The injection enlarged the off-spine bonds 4 -> 1 and 5 -> 4, but not
    // 6 -> 4 or 7 -> 2.
    assert!(grown(&state, 4, 1) && grown(&state, 5, 4));
    assert!(!grown(&state, 6, 4) && !grown(&state, 7, 2));
    let expected = vec![
        step(3, 2),
        step(2, 1),
        step(1, 4),
        step(4, 5),
        step(5, 4),
        step(4, 1),
        step(1, 0),
    ];

    // The stored spine-only walk (`main` before #1450) breaks the pivot basis.
    let mut spine_only = state.clone();
    for phase in spine_only.problem.schedule.reverse.clone().iter() {
        super::run_phase_serial(
            &mut spine_only,
            &options,
            phase,
            false,
            &mut identity,
            &mut Vec::new(),
            &mut 0,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
    }
    let broken = crate::test_support::pivot_basis_deviation(&spine_only);
    assert!(broken > 0.5, "spine-only max|M - I| = {broken:e}");

    let reverse = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Reverse,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(reverse.updated_edges, expected);
    let deviation = crate::test_support::pivot_basis_deviation(&state);
    assert!(deviation <= 1e-12, "max|M - I| = {deviation:e}");
    assert!(!grown(&state, 4, 1) && !grown(&state, 5, 4));
    assert_eq!(state.output.canonical_region(), &HashSet::from([0]));
}

/// A return-pass re-commit (#1450) is an ordinary commit, so it expands
/// across a unit vertex like any other (#1435). The expanded set is cut back
/// before the next step, so the walk is the one without the expansion and the
/// pivot basis holds after the pass.
#[test]
fn return_pass_recommits_expand_across_a_unit_vertex() {
    // The tree of the test above, with the spine node 1 a unit vertex.
    let edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (4, 6), (2, 7)];
    let physical = (0..8)
        .map(|node| DynIndex::new_dyn(if node == 1 { 1 } else { 2 }))
        .collect::<Vec<_>>();
    let inputs = vec![crate::test_support::random_decaying_tree::<f64>(
        &edges,
        &physical,
        2,
        0.5,
        &mut crate::test_support::SplitMix(5),
    )];
    let options = TreeAciOptions {
        root: Some(0),
        tolerance: 0.3,
        ..TreeAciOptions::default()
    };
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    run_directional_pass(
        &mut state,
        &options,
        PassDirection::Forward,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    let capacities = global_injection_capacities(&state, &options);
    let injected = crate::global_guard::inject_global_pivots(
        &mut state,
        &[vec![1, 0, 0, 0, 1, 1, 0, 0]],
        &capacities,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(injected, 1);
    let step = |from, to| directed_between(&state, from, to);
    let grown = |state: &TreeAciState<'_, f64, usize>, from, to| {
        state
            .candidate_set_outgrew_basis(directed_between(state, from, to))
            .unwrap()
    };
    // The injection enlarged the off-spine bond 4 -> 1 only.
    assert!(grown(&state, 4, 1));
    assert!(!grown(&state, 7, 2) && !grown(&state, 5, 4) && !grown(&state, 6, 4));
    let neighbour = step(0, 1);
    let neighbour_len = state.candidates.ids[neighbour].len();
    let (down, up) = (step(1, 4), step(4, 1));
    let expected_walk = vec![step(3, 2), step(2, 1), down, up, step(1, 0)];
    crate::unit_vertex::expansion_debug_log::take();

    let reverse = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Reverse,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    // The excursion 1 -> 4, 4 -> 1 re-commits the enlarged bond before the
    // spine step 1 -> 0, as without the expansion.
    assert_eq!(reverse.updated_edges, expected_walk);
    // Both re-commits read the unit vertex 1 (rows of 1 -> 4, columns of
    // 4 -> 1) and expand its rank-1 bond 0 -> 1 by leaf 0's other
    // coordinate.
    let log = crate::unit_vertex::expansion_debug_log::take();
    let recommit_expansions = log
        .iter()
        .filter(|(forward, _)| [down, up].contains(forward))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        recommit_expansions,
        vec![
            (down, vec![(neighbour, neighbour_len)]),
            (up, vec![(neighbour, neighbour_len)]),
        ]
    );
    assert_eq!(state.candidates.ids[neighbour].len(), neighbour_len);
    assert!(!grown(&state, 4, 1));
    let deviation = crate::test_support::pivot_basis_deviation(&state);
    assert!(deviation <= 1e-12, "max|M - I| = {deviation:e}");
}

fn branched_product_options(max_bond_dim: Option<usize>) -> TreeAciOptions<usize> {
    TreeAciOptions {
        enable_global_guard: false,
        tolerance: 1.0e-3,
        scale_tolerance: false,
        min_sweeps: 1,
        max_sweeps: 12,
        max_bond_dim,
        ..TreeAciOptions::default()
    }
}

/// The first pass of a fresh run, replayed on its own state.
fn replay_first_pass<'a>(
    inputs: &'a [TreeTN<IdxTensor, usize>],
    options: &TreeAciOptions<usize>,
) -> (TreeAciState<'a, f64, usize>, super::PassReport) {
    let mut state = TreeAciState::<f64, usize>::initialize(inputs, options).unwrap();
    let report = run_directional_pass(
        &mut state,
        options,
        PassDirection::Forward,
        false,
        &mut product_of_two,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    (state, report)
}

#[test]
fn stale_residuals_that_hide_an_error_do_not_certify_a_stop() {
    let (_, inputs) = branched_product_inputs();
    let options = branched_product_options(None);
    // Every recorded residual of pass 1 is within tolerance, and the Guard is
    // off, so the old stop rule would stop here.
    let (mut replay, first) = replay_first_pass(&inputs, &options);
    assert!(first.max_error <= options.tolerance);
    let refresh = refresh_stale_residuals(
        &mut replay,
        &options,
        &mut product_of_two,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(refresh.refreshed_edges, replay.edge_ranks.len() - 1);
    let refreshed = super::recorded_max_error(&replay, &options);
    assert!(refreshed > options.tolerance, "refreshed {refreshed:e}");

    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let history = run_local_sweeps(
        &mut state,
        &options,
        &mut product_of_two,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    // Pass 1 records the refreshed maximum and does not stop; pass 2 stops
    // on residuals that were re-measured on the output it returns.
    assert_eq!(history.max_errors[0], refreshed);
    assert_eq!(history.termination, TreeAciTermination::Converged);
    assert_eq!(history.max_errors.len(), 2);
    assert!(history.max_errors[1] <= options.tolerance);
    assert_eq!(history.global_pivots_found, vec![0, 0]);
    assert!(!history.needs_cleanup);
    for edge in 0..state.edge_ranks.len() {
        assert!(!state.residual_is_stale(edge).unwrap());
    }
    // Both stop attempts paid for a refresh on top of the two passes.
    assert!(history.evaluated_points > first.evaluated_points + refresh.evaluated_points);
}

#[test]
fn a_stop_that_survives_the_refresh_counts_its_points() {
    let edges = [(0, 1), (1, 2), (1, 3)];
    let inputs = vec![product_tree(&edges, 4)];
    let options = TreeAciOptions {
        enable_global_guard: false,
        min_sweeps: 1,
        ..TreeAciOptions::default()
    };
    let mut replay = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let first = run_directional_pass(
        &mut replay,
        &options,
        PassDirection::Forward,
        false,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    let refresh = refresh_stale_residuals(
        &mut replay,
        &options,
        &mut identity,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(refresh.refreshed_edges, edges.len() - 1);
    assert!(refresh.evaluated_points > 0);

    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let history = run_local_sweeps(
        &mut state,
        &options,
        &mut identity,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    // The rank-1 input is reproduced exactly, so the refreshed residuals are
    // within tolerance and pass 1 stops.
    assert_eq!(history.termination, TreeAciTermination::Converged);
    assert_eq!(history.max_errors.len(), 1);
    assert!(history.max_errors[0] <= options.tolerance);
    assert_eq!(
        history.evaluated_points,
        first.evaluated_points + refresh.evaluated_points
    );
}

#[test]
fn stop_rules_that_fail_on_recorded_residuals_skip_the_refresh() {
    let (_, inputs) = branched_product_inputs();
    // Rank-limited: the cap of 3 leaves a recorded residual above tolerance.
    let limited = branched_product_options(Some(3));
    let (_, first) = replay_first_pass(&inputs, &limited);
    assert!(first.max_error > limited.tolerance);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &limited).unwrap();
    let history = run_local_sweeps(
        &mut state,
        &limited,
        &mut product_of_two,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(history.termination, TreeAciTermination::RankLimited);
    assert_eq!(history.max_errors, vec![first.max_error]);
    assert_eq!(history.evaluated_points, first.evaluated_points);

    // Within tolerance but short of the dwell: pass 1 cannot stop, so its
    // residuals stay as recorded (a refresh would raise them above the
    // tolerance; see the test above).
    let dwell = TreeAciOptions {
        min_sweeps: 2,
        ..branched_product_options(None)
    };
    let (_, first) = replay_first_pass(&inputs, &dwell);
    assert!(first.max_error <= dwell.tolerance);
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &dwell).unwrap();
    let history = run_local_sweeps(
        &mut state,
        &dwell,
        &mut product_of_two,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(history.max_errors[0], first.max_error);
    assert!(history.max_errors.len() > 1);
}
