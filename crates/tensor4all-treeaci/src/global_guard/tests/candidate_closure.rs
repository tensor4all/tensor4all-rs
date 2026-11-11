//! Characterization fixtures for #1265, before the candidate abstraction fix.
//!
//! These tests identify the current defect, rather than treating it as an
//! accepted algorithmic contract. The fix will change the observed violations
//! into closure assertions while retaining the same fixture and invariants.
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
fn fixture_exposes_duplicate_padding_headroom_in_both_directions() {
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
        assert_eq!(
            state.edge_ranks[child / 2],
            state.algebraic_edge_bounds[child / 2]
        );
        assert_eq!(state.candidates.ids[child].len(), 2);
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
                &[0, 2, 0],
                &mut crate::exec::Exec::Global
            )
            .unwrap(),
            1
        );
        let parent = if reflected { 3 } else { 2 };
        assert!(
            inward_holes(&state, reflected)
                .iter()
                .any(|&(edge, _, incoming, _)| edge == parent && incoming == child),
            "fixture must expose the missing required child"
        );
        assert_eq!(state.edge_ranks, vec![2, 3, 2]);
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
fn fixture_exposes_a_genuinely_full_child_in_both_directions() {
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
                &[point],
                &capacities,
                &mut crate::exec::Exec::Global
            )
            .unwrap(),
            1
        );
        let parent = if reflected { 3 } else { 4 };
        assert!(
            inward_holes(&state, reflected)
                .iter()
                .any(|&(edge, _, incoming, _)| edge == parent && incoming == child),
            "fixture must expose activation past the full child"
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
    }
}
