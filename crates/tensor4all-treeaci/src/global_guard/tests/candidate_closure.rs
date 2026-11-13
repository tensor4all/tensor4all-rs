//! The fixture-first #1265 cases now assert useful candidate closure.
use super::*;
use std::collections::HashSet;

fn chain(bond_dims: &[usize]) -> TreeTN<IdxTensor, usize> {
    let n = bond_dims.len() + 1;
    let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = bond_dims.iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let tensors = (0..n)
        .map(|node| {
            let left = if node == 0 { 1 } else { bond_dims[node - 1] };
            let right = if node + 1 == n { 1 } else { bond_dims[node] };
            let mut indices = Vec::new();
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            indices.push(sites[node].clone());
            if node + 1 < n {
                indices.push(bonds[node].clone());
            }
            let values = if left * right == 2 {
                vec![1.0, 0.0, 0.0, 1.0]
            } else {
                vec![node as f64 + 1.0, node as f64 + 2.0]
            };
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..n).collect()).unwrap()
}

/// Missing immediate child references in candidates directed toward an endpoint.
fn inward_holes(
    state: &TreeAciState<'_, f64, usize>,
    reflected: bool,
) -> Vec<(usize, usize, usize, usize)> {
    let mut holes = Vec::new();
    for edge in &state.problem.directed_edges {
        if (edge.from < edge.to) == reflected {
            continue;
        }
        for &id in &state.candidates.ids[edge.id] {
            for &(child, sample) in &state.sample_arena.record(edge.id, id).unwrap().incoming {
                if !state.candidates.ids[child].contains(&sample) {
                    holes.push((edge.id, id, child, sample));
                }
            }
        }
    }
    holes
}

#[test]
fn useful_query_candidates_grow_independently_of_the_selected_output_basis() {
    for reflected in [false, true] {
        let inputs = vec![chain(&[1, 1, 1])];
        let options = TreeAciOptions {
            root: Some(if reflected { 0 } else { 3 }),
            max_bond_dim: Some(4),
            ..Default::default()
        };
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let output = state.output.to_dense().unwrap();
        let pivots = state.pivots.clone();
        let mut first = vec![0, 1, 0, 0];
        let mut second = vec![1, 1, 0, 0];
        if reflected {
            first.reverse();
            second.reverse();
        }
        assert_eq!(
            inject_global_pivots(
                &mut state,
                &[first],
                &[1, 3, 1],
                &mut crate::exec::Exec::Global
            )
            .unwrap(),
            1
        );
        assert!(inward_holes(&state, reflected).is_empty());
        let child = if reflected { 5 } else { 0 };
        assert_eq!(state.edge_ranks[child / 2], 1);
        assert_eq!(state.candidates.ids[child].len(), 1);
        assert_eq!(
            state.candidates.ids[child]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            inject_global_pivots(
                &mut state,
                &[second],
                &[1, 3, 1],
                &mut crate::exec::Exec::Global
            )
            .unwrap(),
            1
        );
        assert!(inward_holes(&state, reflected).is_empty());
        assert_eq!(state.candidates.ids[child].len(), 2);
        assert_eq!(
            state.candidates.ids[child]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            2
        );
        assert_eq!(state.edge_ranks, vec![1, 1, 1]);
        assert_eq!(state.output.link_dims(), vec![1, 1, 1]);
        assert_eq!(state.pivots, pivots);
        assert_eq!(
            state
                .output
                .to_dense()
                .unwrap()
                .sub(&output)
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
    }
}

#[test]
fn a_genuinely_full_child_cannot_publish_an_unusable_parent() {
    for reflected in [false, true] {
        let dims = if reflected {
            vec![1, 1, 2, 1]
        } else {
            vec![1, 2, 1, 1]
        };
        let input = chain(&dims);
        let options = TreeAciOptions {
            root: Some(if reflected { 0 } else { 4 }),
            max_bond_dim: Some(2),
            initial_guess: Some(input.clone()),
            ..Default::default()
        };
        let inputs = vec![input];
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        // Guard runs after local sweeps, which establish the selected-pivot
        // gauge that initialization deliberately defers for an explicit guess.
        for direction in [PassDirection::Forward, PassDirection::Reverse] {
            run_directional_pass(
                &mut state,
                &options,
                direction,
                &mut identity,
                &mut crate::exec::Exec::Global,
            )
            .unwrap();
        }
        assert_eq!(state.edge_ranks, dims);
        assert!(inward_holes(&state, reflected).is_empty());
        let output = state.output.to_dense().unwrap();
        let pivots = state.pivots.clone();
        let child = if reflected { 5 } else { 2 };
        assert_eq!(state.candidates.ids[child].len(), 2);
        assert_eq!(
            state.candidates.ids[child]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            2
        );
        let capacities: Vec<_> = state.edge_ranks.iter().map(|&rank| 2 - rank).collect();
        assert_eq!(capacities[child / 2], 0);
        let leaf = if reflected { 7 } else { 0 };
        let leaf_node = if reflected { 4 } else { 0 };
        let mut point = vec![1; 5];
        point[leaf_node] = 1 - state
            .sample_arena
            .record(leaf, state.candidates.ids[leaf][0])
            .unwrap()
            .local_coordinate;
        assert_eq!(
            inject_global_pivots(
                &mut state,
                &[point.clone()],
                &capacities,
                &mut crate::exec::Exec::Global
            )
            .unwrap(),
            1
        );
        assert!(inward_holes(&state, reflected).is_empty());
        let parent = if reflected { 3 } else { 4 };
        assert_eq!(
            state.candidates.ids[parent].len(),
            1,
            "the blocked parent must not become a useful candidate"
        );
        assert_eq!(state.pivots, pivots);
        assert!(state.edge_ranks.iter().all(|&rank| rank <= 2));
        assert_eq!(
            state
                .output
                .to_dense()
                .unwrap()
                .sub(&output)
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
        let guard_options = TreeAciOptions {
            nsearch_global_pivots: 1,
            max_nglobal_pivots: 1,
            nsweeps_global_search: 0,
            ..options.clone()
        };
        let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
        let mut target = |_: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
            out.fill(4096.0);
            Ok(())
        };
        let pending = vec![point];
        let found = find_global_pivots(
            &state,
            &mut evaluators,
            &guard_options,
            &mut seeded_rng(),
            &mut target,
            &pending,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        assert_eq!(
            found.pivots, pending,
            "a partially blocked activation cannot resolve known residual evidence"
        );
    }
}

#[test]
fn projection_and_frame_extension_failures_roll_back_staged_query_candidates() {
    for frame_failure in [false, true] {
        let inputs = vec![chain(&[1, 1, 1])];
        let options = TreeAciOptions {
            root: Some(3),
            max_bond_dim: Some(4),
            ..Default::default()
        };
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        inject_global_pivots(
            &mut state,
            &[vec![0, 1, 0, 0]],
            &[1, 3, 1],
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        let before = state.candidates.clone();
        let pivots = state.pivots.clone();
        let ranks = state.edge_ranks.clone();
        let generation = state.generation;
        let records = state.sample_arena.record_count();
        let bytes = state.sample_arena.retained_bytes();
        let frames = state.input_frames.records();
        let frame_bytes = state.input_frames.retained_bytes();
        let expected = state.output.to_dense().unwrap();
        let mut points = vec![vec![1, 1, 0, 0]];
        if frame_failure {
            state.problem.max_frame_bytes = 0;
        } else {
            points.push(vec![2, 0, 0, 0]);
        }
        let error = inject_global_pivots(
            &mut state,
            &points,
            &[1, 3, 1],
            &mut crate::exec::Exec::Global,
        )
        .unwrap_err();
        if frame_failure {
            assert!(
                matches!(
                    error,
                    crate::TreeAciError::ResourceLimit {
                        resource: "frame bytes",
                        ..
                    }
                ),
                "{error}"
            );
        }
        assert_eq!(state.candidates, before);
        assert_eq!(state.pivots, pivots);
        assert_eq!(state.edge_ranks, ranks);
        assert_eq!(state.generation, generation);
        assert_eq!(state.sample_arena.record_count(), records);
        assert_eq!(state.sample_arena.retained_bytes(), bytes);
        assert_eq!(state.input_frames.records(), frames);
        assert_eq!(state.input_frames.retained_bytes(), frame_bytes);
        assert_eq!(
            state
                .output
                .to_dense()
                .unwrap()
                .sub(&expected)
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
    }
}

/// Guard query pools must not become the selected child basis while sweeping.
#[test]
fn guard_proposals_preserve_selected_partial_nesting_at_every_chain_commit() {
    for reflected in [false, true] {
        let inputs = vec![chain(&[1, 1, 1])];
        let options = TreeAciOptions {
            root: Some(if reflected { 0 } else { 3 }),
            max_bond_dim: Some(4),
            ..Default::default()
        };
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let mut point = vec![1, 1, 0, 0];
        if reflected {
            point.reverse();
        }
        inject_global_pivots(
            &mut state,
            &[point],
            &[1, 3, 1],
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
            for (point, value) in out.iter_mut().enumerate() {
                *value = 1.0 / (1.0 + batch.get(0, point)?);
            }
            Ok(())
        };
        for direction in [PassDirection::Forward, PassDirection::Reverse] {
            let phases = match direction {
                PassDirection::Forward => state.problem.schedule.forward.clone(),
                PassDirection::Reverse => state.problem.schedule.reverse.clone(),
            };
            for phase in phases.iter() {
                for path in &phase.paths {
                    for step in &path.steps {
                        let cut = state
                            .problem
                            .directed_edges
                            .iter()
                            .find(|e| e.from == step.from && e.to == step.to)
                            .unwrap()
                            .id;
                        update_edge_transaction(
                            &mut state,
                            cut,
                            &options,
                            true,
                            &mut operator,
                            &mut crate::exec::Exec::Global,
                        )
                        .unwrap();
                        let center = *state.output.canonical_region().iter().next().unwrap();
                        for edge in &state.problem.directed_edges {
                            if edge.from.abs_diff(center) < edge.to.abs_diff(center) {
                                continue;
                            }
                            for &(left, right) in &state.pivots.per_edge[edge.id / 2] {
                                let id = if edge.id % 2 == 0 { left } else { right };
                                for &(child, sample) in
                                    &state.sample_arena.record(edge.id, id).unwrap().incoming
                                {
                                    let selected: Vec<_> = if child % 2 == 0 {
                                        state.pivots.forward_ids(child / 2)
                                    } else {
                                        state.pivots.reverse_ids(child / 2)
                                    };
                                    assert!(
                                        selected.contains(&sample),
                                        "center={center}, cut={}, child={child}, sample={sample}",
                                        edge.id
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn compact_candidates_reconstruct_a_nonlinear_target_over_the_complete_small_grid() {
    let inputs = vec![chain(&[1, 1, 1])];
    let options = TreeAciOptions {
        root: Some(3),
        max_bond_dim: Some(4),
        ..Default::default()
    };
    let input = inputs[0].to_dense().unwrap();
    let exact = IdxTensor::from_dense(
        input.indices().to_vec(),
        input
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(|x| 1.0 / (1.0 + x))
            .collect(),
    )
    .unwrap();
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    inject_global_pivots(
        &mut state,
        &[vec![0, 1, 0, 0], vec![1, 1, 0, 0]],
        &[1, 3, 1],
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        for (point, value) in out.iter_mut().enumerate() {
            *value = 1.0 / (1.0 + batch.get(0, point)?);
        }
        Ok(())
    };
    let history = crate::schedule::run_local_sweeps(
        &mut state,
        &options,
        &mut operator,
        &mut seeded_rng(),
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(history.termination, crate::TreeAciTermination::Converged);
    let error = state
        .output
        .to_dense()
        .unwrap()
        .sub(&exact)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(error < 1e-12, "full-grid residual={error:e}");
}
