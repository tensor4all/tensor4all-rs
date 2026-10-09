use super::*;
use crate::TreeAciScalar;
use num_complex::{Complex32, Complex64};
use tensor4all_core::Scalar;
use tensor4all_tensorbackend::mat_mul;

fn constant_input<T: TreeAciScalar>() -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(1);
    TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![DynIndex::new_dyn(3), bond.clone()],
                vec![T::from_f64(1.0); 3],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(3)], vec![T::from_f64(1.0); 3])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap()
}

fn verify_reuse<T: TreeAciScalar>(phase: T, rounding: f64) {
    let inputs = vec![constant_input::<T>()];
    for relative in [false, true] {
        let options = TreeAciOptions {
            tolerance: if relative {
                0.02 / 9.0
            } else {
                0.02 * phase.abs_val()
            },
            scale_tolerance: relative,
            ..TreeAciOptions::default()
        };
        let problem = prepare_problem::<T, _>(&inputs, &options).unwrap();
        let (arena, active) =
            SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1], vec![2, 2]])
                .unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        for forward in [0, 1] {
            for left_orthogonal in [false, true] {
                let (row, col) = if left_orthogonal { (2, 1) } else { (1, 2) };
                let (a, b) = if forward == 0 { (row, col) } else { (col, row) };
                let pairs = vec![(active.ids[0][a], active.ids[1][b])];
                // A small near-threshold perturbation admits several distinct
                // rank-one crosses. Preserve the old, non-largest pivot only
                // when its residual over every entry is below the same limit.
                let mut target = (0..3)
                    .flat_map(|col| {
                        (0..3).map(move |row| T::from_f64(((row + 1) * (col + 1)) as f64) * phase)
                    })
                    .collect::<Vec<_>>();
                target[4] = target[4] + T::from_f64(0.01) * phase;
                let mut operator = |_: crate::TreeElementwiseBatch<'_, T>, output: &mut [T]| {
                    output.copy_from_slice(&target);
                    Ok(())
                };
                let fresh = materialize_and_factor_edge(
                    &inputs,
                    &problem,
                    &active,
                    &frames,
                    forward,
                    &options,
                    left_orthogonal,
                    None,
                    &mut operator,
                )
                .unwrap();
                assert_eq!(fresh.row_samples[0].local_coordinate, 2);
                assert_eq!(fresh.col_samples[0].local_coordinate, 2);
                let reused = materialize_and_factor_edge(
                    &inputs,
                    &problem,
                    &active,
                    &frames,
                    forward,
                    &options,
                    left_orthogonal,
                    Some((&arena, &pairs)),
                    &mut operator,
                )
                .unwrap();
                assert_eq!(reused.row_samples[0].local_coordinate, row);
                assert_eq!(reused.col_samples[0].local_coordinate, col);
                let reconstructed = mat_mul(&reused.left, &reused.right).unwrap();
                let residual = target
                    .iter()
                    .zip(reconstructed.as_col_major_slice())
                    .map(|(&a, &b)| Scalar::abs_val(a - b))
                    .fold(0.0_f64, f64::max);
                assert!((residual - 0.015 * phase.abs_val()).abs() < rounding);
                assert!((reused.pivot_errors[0] - residual).abs() < rounding);
                assert!(!options
                    .tolerance_policy()
                    .exceeds(residual, reused.sampled_scale));

                // The same old pivot now amplifies an allowed perturbation at
                // its intersection into an unacceptable residual elsewhere.
                // Checking only the pivot block would incorrectly retain it.
                target[4] = target[4] - T::from_f64(0.01) * phase;
                target[row + 3 * col] = target[row + 3 * col] + T::from_f64(0.025) * phase;
                let mut operator = |_: crate::TreeElementwiseBatch<'_, T>, output: &mut [T]| {
                    output.copy_from_slice(&target);
                    Ok(())
                };
                let rejected = materialize_and_factor_edge(
                    &inputs,
                    &problem,
                    &active,
                    &frames,
                    forward,
                    &options,
                    left_orthogonal,
                    Some((&arena, &pairs)),
                    &mut operator,
                )
                .unwrap();
                assert_eq!(rejected.row_samples[0].local_coordinate, 2);
                let reconstructed = mat_mul(&rejected.left, &rejected.right).unwrap();
                let residual = target
                    .iter()
                    .zip(reconstructed.as_col_major_slice())
                    .map(|(&a, &b)| Scalar::abs_val(a - b))
                    .fold(0.0_f64, f64::max);
                assert!(residual <= 0.02 * phase.abs_val() + rounding);
            }
        }
    }
}

#[test]
fn sufficient_previous_cross_is_retained_with_full_residual_validation() {
    verify_reuse::<f32>(1.0, 2e-5);
    verify_reuse::<f64>(1.0, 1e-12);
    verify_reuse::<Complex32>(Complex32::new(1.0, 0.25), 2e-5);
    verify_reuse::<Complex64>(Complex64::new(1.0, 0.25), 1e-12);
}

fn verify_partial_restoration<T: TreeAciScalar>(phase: T, rounding: f64) {
    let inputs = vec![constant_input::<T>()];
    let metadata = 6 * std::mem::size_of::<crate::samples::ComponentSample>();
    let matrix_bytes = 9 * std::mem::size_of::<T>();
    let scratch = tensor4all_core::matrix_luci_factors_working_bytes::<T>(3, 3, 1).unwrap();
    let retention_bytes = 2 * matrix_bytes + 2 * scratch + 2 * metadata;
    for relative in [false, true] {
        let options = TreeAciOptions {
            tolerance: if relative {
                0.02 / 9.0
            } else {
                0.02 * phase.abs_val()
            },
            scale_tolerance: relative,
            max_bond_dim: Some(1),
            ..TreeAciOptions::default()
        };
        let problem = prepare_problem::<T, _>(&inputs, &options).unwrap();
        let (arena, active) =
            SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1], vec![2, 2]])
                .unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        let pairs = vec![(active.ids[0][1], active.ids[1][1])];
        let target = [1.0, 2.0, 3.0, 2.0, 4.01, 6.0, 3.0, 6.0, 9.0].map(|x| T::from_f64(x) * phase);
        for forward in [0, 1] {
            for left_orthogonal in [false, true] {
                for refine in [false, true] {
                    let budgeted = TreeAciOptions {
                        max_working_bytes: retention_bytes + if refine { scratch } else { 0 },
                        ..options.clone()
                    };
                    let mut operator = |_: crate::TreeElementwiseBatch<'_, T>, output: &mut [T]| {
                        output.copy_from_slice(&target);
                        Ok(())
                    };
                    let result = materialize_and_factor_edge(
                        &inputs,
                        &problem,
                        &active,
                        &frames,
                        forward,
                        &budgeted,
                        left_orthogonal,
                        Some((&arena, &pairs)),
                        &mut operator,
                    )
                    .unwrap();
                    // The complete old (1, 1) cross exceeds the tolerance.
                    // Only its column can be restored while preserving rank.
                    // A budget that permits full retention alone must instead
                    // keep the fresh cross, without failing the operation.
                    assert_eq!(result.left.ncols(), 1);
                    assert_eq!(result.row_samples[0].local_coordinate, 2);
                    assert_eq!(
                        result.col_samples[0].local_coordinate,
                        if refine { 1 } else { 2 }
                    );
                    let approximation = mat_mul(&result.left, &result.right).unwrap();
                    let error = target
                        .iter()
                        .zip(approximation.as_col_major_slice())
                        .map(|(&a, &b)| Scalar::abs_val(a - b))
                        .fold(0.0_f64, f64::max);
                    assert!(
                        (error - if refine { 0.015 } else { 0.01 } * phase.abs_val()).abs()
                            < rounding
                    );
                    assert!(!budgeted
                        .tolerance_policy()
                        .exceeds(error, result.sampled_scale));
                }
            }
        }
    }
}

#[test]
fn rejected_complete_cross_can_restore_partial_pivots_with_a_separate_budget() {
    verify_partial_restoration::<f32>(1.0, 2e-5);
    verify_partial_restoration::<f64>(1.0, 1e-12);
    verify_partial_restoration::<Complex32>(Complex32::new(1.0, 0.25), 2e-5);
    verify_partial_restoration::<Complex64>(Complex64::new(1.0, 0.25), 1e-12);
}

#[test]
fn previous_cross_never_prevents_rank_reduction_or_zero_target() {
    let inputs = vec![constant_input::<f64>()];
    for cap in [1, 3] {
        let options = TreeAciOptions {
            max_bond_dim: Some(cap),
            ..TreeAciOptions::default()
        };
        let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
        let (arena, active) =
            SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1]]).unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        let pairs = vec![
            (active.ids[0][0], active.ids[1][0]),
            (active.ids[0][1], active.ids[1][1]),
        ];
        for value in [0.0, 1.0] {
            let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
                output.fill(value);
                Ok(())
            };
            let result = materialize_and_factor_edge(
                &inputs,
                &problem,
                &active,
                &frames,
                0,
                &options,
                true,
                Some((&arena, &pairs)),
                &mut operator,
            )
            .unwrap();
            assert_eq!(result.left.ncols(), 1);
            let reconstructed = mat_mul(&result.left, &result.right).unwrap();
            for &entry in reconstructed.as_col_major_slice() {
                assert!((entry - value).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn singular_previous_cross_falls_back_to_fresh_luci() {
    let inputs = vec![constant_input::<f64>()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![2, 2]]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let pairs = vec![
        (active.ids[0][0], active.ids[1][0]),
        (active.ids[0][1], active.ids[1][1]),
    ];
    let target = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output.copy_from_slice(&target);
        Ok(())
    };
    let result = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        0,
        &options,
        true,
        Some((&arena, &pairs)),
        &mut operator,
    )
    .unwrap();
    assert_eq!(result.left.ncols(), 2);
    let reconstructed = mat_mul(&result.left, &result.right).unwrap();
    for (&expected, &actual) in target.iter().zip(reconstructed.as_col_major_slice()) {
        assert!((expected - actual).abs() < 1e-12);
    }
}

#[test]
fn previous_cross_matrix_is_reserved_before_allocation_with_budget_fallback() {
    let inputs = vec![constant_input::<f64>()];
    let metadata = 6 * std::mem::size_of::<crate::samples::ComponentSample>();
    let base_bytes = 9 * std::mem::size_of::<f64>()
        + tensor4all_core::matrix_luci_factors_working_bytes::<f64>(3, 3, 3).unwrap()
        + 2 * metadata;
    let options = TreeAciOptions {
        max_working_bytes: base_bytes,
        ..TreeAciOptions::default()
    };
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[vec![0, 0]]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let pairs = vec![(active.ids[0][0], active.ids[1][0])];
    let target = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    let calls = Cell::new(0);
    let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        output.copy_from_slice(&target);
        Ok(())
    };
    let fallback = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        0,
        &options,
        true,
        Some((&arena, &pairs)),
        &mut operator,
    )
    .unwrap();
    assert_eq!(fallback.row_samples[0].local_coordinate, 2);
    let reconstructed = mat_mul(&fallback.left, &fallback.right).unwrap();
    for (&expected, &actual) in target.iter().zip(reconstructed.as_col_major_slice()) {
        assert!((expected - actual).abs() < 1e-12);
    }
    let roomy = TreeAciOptions {
        max_working_bytes: base_bytes
            + 9 * std::mem::size_of::<f64>()
            + tensor4all_core::matrix_luci_factors_working_bytes::<f64>(3, 3, 3).unwrap(),
        ..options.clone()
    };
    let reused = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        0,
        &roomy,
        true,
        Some((&arena, &pairs)),
        &mut operator,
    )
    .unwrap();
    assert_eq!(reused.row_samples[0].local_coordinate, 0);
    let reconstructed = mat_mul(&reused.left, &reused.right).unwrap();
    for (&expected, &actual) in target.iter().zip(reconstructed.as_col_major_slice()) {
        assert!((expected - actual).abs() < 1e-12);
    }
    calls.set(0);
    let insufficient = TreeAciOptions {
        max_working_bytes: base_bytes - 1,
        ..options
    };
    let error = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        0,
        &insufficient,
        true,
        Some((&arena, &pairs)),
        &mut operator,
    )
    .unwrap_err();
    assert!(
        matches!(error, TreeAciError::ResourceLimit { resource: "working bytes", requested, limit }
        if requested == base_bytes && limit == base_bytes - 1)
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn absent_or_invalid_previous_samples_do_not_bypass_validation() {
    let inputs = vec![super::three_node_chain_for_batched_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, mut active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0], vec![1, 1, 1]]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let forward = problem
        .directed_edges
        .iter()
        .position(|edge| !edge.incoming_to_from.is_empty())
        .unwrap();
    let canonical = (forward / 2) * 2;
    let pairs = vec![(active.ids[canonical][1], active.ids[canonical + 1][1])];
    for ids in &mut active.ids {
        ids.truncate(1);
    }
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };
    let fresh = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        forward,
        &options,
        true,
        None,
        &mut operator,
    )
    .unwrap();
    let missing = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        forward,
        &options,
        true,
        Some((&arena, &pairs)),
        &mut operator,
    )
    .unwrap();
    assert_eq!(missing.row_samples, fresh.row_samples);
    assert_eq!(missing.col_samples, fresh.col_samples);
    let empty = materialize_and_factor_edge(
        &inputs,
        &problem,
        &active,
        &frames,
        forward,
        &options,
        true,
        Some((&arena, &[])),
        &mut operator,
    )
    .unwrap();
    assert_eq!(empty.row_samples, fresh.row_samples);
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs,
            &problem,
            &active,
            &frames,
            forward,
            &options,
            true,
            Some((&arena, &[(usize::MAX, usize::MAX)])),
            &mut operator
        ),
        Err(TreeAciError::InternalInvariant { .. })
    ));
}

#[test]
fn candidate_inverse_handles_branch_order_and_missing_dependencies() {
    let inputs = vec![super::local_update_measurement_branch(2)];
    let options = TreeAciOptions::default();
    let mut problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (_, mut active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0], vec![0, 1, 1, 1]]).unwrap();
    let forward = problem
        .directed_edges
        .iter()
        .position(|e| e.incoming_to_from.len() == 2)
        .unwrap();
    let candidates = enumerate_candidates(&problem, &active, forward, "rows", usize::MAX).unwrap();
    for (index, sample) in candidates.iter().enumerate() {
        assert_eq!(
            super::super::candidate_index(&problem, &active, forward, sample).unwrap(),
            Some(index)
        );
    }
    let mut sample = candidates[0].clone();
    sample.incoming[0].1 = usize::MAX;
    assert_eq!(
        super::super::candidate_index(&problem, &active, forward, &sample).unwrap(),
        None
    );
    sample = candidates[0].clone();
    sample.incoming.swap(0, 1);
    assert_eq!(
        super::super::candidate_index(&problem, &active, forward, &sample).unwrap(),
        None
    );
    sample = candidates[0].clone();
    sample.incoming.pop();
    assert_eq!(
        super::super::candidate_index(&problem, &active, forward, &sample).unwrap(),
        None
    );
    sample = candidates[0].clone();
    sample.local_coordinate = usize::MAX;
    assert_eq!(
        super::super::candidate_index(&problem, &active, forward, &sample).unwrap(),
        None
    );

    let node = problem.node_positions[&problem.directed_edges[forward].from];
    problem.physical[node].local_dim = usize::MAX;
    sample = candidates[0].clone();
    assert!(matches!(
        super::super::candidate_index(&problem, &active, forward, &sample),
        Err(TreeAciError::SizeOverflow { .. })
    ));
    let incoming = sample.incoming[0].0;
    active.ids[incoming] = vec![0, 1, 2];
    sample.incoming[0].1 = 2;
    assert!(matches!(
        super::super::candidate_index(&problem, &active, forward, &sample),
        Err(TreeAciError::SizeOverflow { .. })
    ));
}

#[test]
fn neighbour_replacement_can_complete_one_missing_cross_axis() {
    let inputs = vec![super::three_node_chain_for_batched_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, mut active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0], vec![1, 1, 1]]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let internal = problem
        .directed_edges
        .iter()
        .position(|e| !e.incoming_to_from.is_empty())
        .unwrap();
    let leaf = problem.directed_edges[internal].reverse;
    let pairs = if internal.is_multiple_of(2) {
        vec![(active.ids[internal][1], active.ids[leaf][0])]
    } else {
        vec![(active.ids[leaf][0], active.ids[internal][1])]
    };
    // The old internal component depends on a neighbour sample that has
    // been replaced. The leaf coordinate still belongs to the current space.
    for ids in &mut active.ids {
        ids.truncate(1);
    }
    let unchanged = active.clone();
    for forward in [internal, leaf] {
        let (nrows, ncols) = if forward == internal { (3, 2) } else { (2, 3) };
        let target = (0..ncols)
            .flat_map(|j| (0..nrows).map(move |i| ((i + 1) * (j + 1)) as f64))
            .collect::<Vec<_>>();
        let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
            output.copy_from_slice(&target);
            Ok(())
        };
        let fresh = materialize_and_factor_edge(
            &inputs,
            &problem,
            &active,
            &frames,
            forward,
            &options,
            true,
            None,
            &mut operator,
        )
        .unwrap();
        assert_eq!(fresh.row_samples[0].local_coordinate, nrows - 1);
        assert_eq!(fresh.col_samples[0].local_coordinate, ncols - 1);
        let retained = materialize_and_factor_edge(
            &inputs,
            &problem,
            &active,
            &frames,
            forward,
            &options,
            true,
            Some((&arena, &pairs)),
            &mut operator,
        )
        .unwrap();
        if forward == internal {
            assert_eq!(retained.col_samples[0].local_coordinate, 0);
            assert_eq!(retained.row_samples[0].local_coordinate, 2);
        } else {
            assert_eq!(retained.row_samples[0].local_coordinate, 0);
            assert_eq!(retained.col_samples[0].local_coordinate, 2);
        }
        assert_eq!(active, unchanged);
        assert_eq!(retained.left.nrows(), nrows);
        assert_eq!(retained.right.ncols(), ncols);
        let reconstructed = mat_mul(&retained.left, &retained.right).unwrap();
        for (&a, &b) in target.iter().zip(reconstructed.as_col_major_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
