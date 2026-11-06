use super::{
    contract_prefix_with_probed_site_pair, contract_prefix_with_probed_site_pair_batch_range,
    contract_prefix_with_site_pair, contract_retaining, contract_site_pair, local_output_indices,
    maximum_site_width, probed_site_pair, probed_site_pair_batch_range, ProbeBank, TreeTN,
};
use crate::treetn::contraction::SrcOptions;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

/// `local_output_indices` used to order same-dimension output legs by
/// `Index::id()` (via `HashSet` + `sort_indices_deterministic`), and
/// `id()` is drawn from a per-process, unseeded RNG -- so on a real
/// process restart, freshly created `out_x`/`out_y` below would have a
/// coin-flip chance of swapping places. This test can't literally
/// restart the process, but it rebuilds `out_x`/`out_y` (and therefore
/// their random ids) fresh on every iteration: under the old
/// implementation that gave roughly a 50% chance per iteration of
/// observing `[out_y, out_x]`, so 200 iterations would fail with
/// probability `1 - 2^-200`. Under the fix, order always follows
/// `tensor_a`'s own construction order, regardless of the ids involved.
#[test]
fn local_output_indices_orders_same_dimension_legs_by_construction_not_by_random_id() {
    for _ in 0..200 {
        let in_idx = DynIndex::new_dyn(2);
        let out_x = DynIndex::new_dyn(3);
        let out_y = DynIndex::new_dyn(3);

        let tensor_a = IdxTensor::from_dense(
            vec![in_idx.clone(), out_x.clone(), out_y.clone()],
            vec![0.0; 2 * 3 * 3],
        )
        .unwrap();
        let tensor_b = IdxTensor::from_dense(vec![in_idx.clone()], vec![0.0; 2]).unwrap();

        let tn_a = TreeTN::from_tensors(vec![tensor_a], vec!["X".to_string()]).unwrap();
        let tn_b = TreeTN::from_tensors(vec![tensor_b], vec!["X".to_string()]).unwrap();

        let outputs = local_output_indices(&tn_a, &tn_b, &"X".to_string()).unwrap();
        assert_eq!(outputs, vec![out_x, out_y]);
    }
}

#[test]
fn probe_bank_extension_preserves_the_existing_prefix() {
    let first = DynIndex::new_dyn(3);
    let second = DynIndex::new_dyn(2);
    let indices = vec![first.clone(), second.clone()];

    let mut extended = ProbeBank::from_seed(indices.clone(), 2, 17).unwrap();
    let prefix_first = extended.coefficients(&first).unwrap().to_vec();
    let prefix_second = extended.coefficients(&second).unwrap().to_vec();
    extended.extend_to(5).unwrap();

    let reference = ProbeBank::from_seed(indices, 5, 17).unwrap();
    assert_eq!(extended.width(), 5);
    assert_eq!(
        &extended.coefficients(&first).unwrap()[..6],
        &prefix_first[..]
    );
    assert_eq!(
        &extended.coefficients(&second).unwrap()[..4],
        &prefix_second[..]
    );
    assert_eq!(
        extended.coefficients(&first).unwrap(),
        reference.coefficients(&first).unwrap()
    );
    assert_eq!(
        extended.coefficients(&second).unwrap(),
        reference.coefficients(&second).unwrap()
    );
    assert_eq!(first.dim(), 3);
    assert_eq!(second.dim(), 2);
}

#[test]
fn probe_bank_rejects_zero_width_and_zero_dimensional_indices() {
    let index = DynIndex::new_dyn(2);
    assert!(ProbeBank::from_seed(vec![index.clone()], 0, 0).is_err());

    let zero_dimensional = DynIndex::new_dyn(0);
    assert!(ProbeBank::from_seed(vec![zero_dimensional], 1, 0).is_err());
}

#[test]
fn probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg() {
    let shared = DynIndex::new_dyn(2);
    let output_a = DynIndex::new_dyn(2);
    let output_b = DynIndex::new_dyn(2);
    let tensor_a = IdxTensor::from_dense(
        vec![output_a.clone(), shared.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let tensor_b =
        IdxTensor::from_dense(vec![shared, output_b.clone()], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 1, 23).unwrap();
    let x = probes.column(&output_a, 0).unwrap();
    let y = probes.column(&output_b, 0).unwrap();
    let a_values = tensor_a.to_vec::<f64>().unwrap();
    let b_values = tensor_b.to_vec::<f64>().unwrap();
    let expected = (0..output_a.dim())
        .map(|s| {
            (0..output_b.dim())
                .map(|t| {
                    (0..2)
                        .map(|u| {
                            a_values[s + output_a.dim() * u] * b_values[u + 2 * t] * x[s] * y[t]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum::<f64>();

    let actual = probed_site_pair(&tensor_a, &tensor_b, &[output_a, output_b], &probes, 0).unwrap();
    let value = actual.to_vec::<f64>().unwrap();
    assert_eq!(value.len(), 1);
    assert!((value[0] - expected).abs() < 1.0e-12);
}

#[test]
fn batched_probed_site_pair_keeps_independent_mpo_probes_paired() {
    let shared = DynIndex::new_dyn(2);
    let output_a = DynIndex::new_dyn(2);
    let output_b = DynIndex::new_dyn(2);
    let batch = DynIndex::new_link(2).unwrap();
    let tensor_a = IdxTensor::from_dense(
        vec![output_a.clone(), shared.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let tensor_b =
        IdxTensor::from_dense(vec![shared, output_b.clone()], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 2, 23).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let actual = probed_site_pair_batch_range(
        &tensor_a,
        &tensor_b,
        &[output_a.clone(), output_b.clone()],
        &probes,
        0,
        2,
        &batch,
        &context,
    )
    .unwrap();
    assert_eq!(actual.indices(), std::slice::from_ref(&batch));

    let a_values = tensor_a.to_vec::<f64>().unwrap();
    let b_values = tensor_b.to_vec::<f64>().unwrap();
    let expected = (0..2)
        .map(|column| {
            let x = probes.column(&output_a, column).unwrap();
            let y = probes.column(&output_b, column).unwrap();
            (0..output_a.dim())
                .map(|s| {
                    (0..output_b.dim())
                        .map(|t| {
                            (0..2)
                                .map(|u| {
                                    a_values[s + output_a.dim() * u]
                                        * b_values[u + 2 * t]
                                        * x[s]
                                        * y[t]
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let actual = actual.to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
}

#[test]
fn prefix_probe_contraction_matches_local_product_with_optimized_order() {
    let left_a = DynIndex::new_dyn(2);
    let left_b = DynIndex::new_dyn(3);
    let right_a = DynIndex::new_dyn(2);
    let right_b = DynIndex::new_dyn(2);
    let shared = DynIndex::new_dyn(2);
    let output_a = DynIndex::new_dyn(2);
    let output_b = DynIndex::new_dyn(2);
    let batch = DynIndex::new_link(2).unwrap();
    let prefix = IdxTensor::from_dense(
        vec![left_a.clone(), left_b.clone(), batch.clone()],
        (0..12).map(|value| value as f64 + 1.0).collect(),
    )
    .unwrap();
    let tensor_a = IdxTensor::from_dense(
        vec![
            left_a.clone(),
            right_a.clone(),
            shared.clone(),
            output_a.clone(),
        ],
        (0..16).map(|value| value as f64 + 1.0).collect(),
    )
    .unwrap();
    let tensor_b = IdxTensor::from_dense(
        vec![
            left_b.clone(),
            right_b.clone(),
            shared.clone(),
            output_b.clone(),
        ],
        (0..24).map(|value| value as f64 + 2.0).collect(),
    )
    .unwrap();
    let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 2, 31).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let local = probed_site_pair_batch_range(
        &tensor_a,
        &tensor_b,
        &[output_a.clone(), output_b.clone()],
        &probes,
        0,
        2,
        &batch,
        &context,
    )
    .unwrap();
    let reference = contract_retaining(&[&prefix, &local], &batch).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let actual = contract_prefix_with_probed_site_pair_batch_range(
        &prefix,
        &tensor_a,
        &tensor_b,
        &[output_a.clone(), output_b.clone()],
        &probes,
        0,
        2,
        &batch,
        &context,
    )
    .unwrap();
    let error = actual.distance(&reference).unwrap();
    assert!(
        error < 1.0e-10,
        "batched prefix contraction error is {error}"
    );

    let prefix_without_batch = prefix.select_indices(&[batch], &[0]).unwrap();
    let reference = contract_site_pair(&tensor_a, &tensor_b, &[&prefix_without_batch]).unwrap();
    let actual =
        contract_prefix_with_site_pair(&prefix_without_batch, &tensor_a, &tensor_b).unwrap();
    assert_eq!(actual.indices(), reference.indices());
    let actual_values = actual.to_vec::<f64>().unwrap();
    let reference_values = reference.to_vec::<f64>().unwrap();
    assert_eq!(actual_values.len(), reference_values.len());
    assert!(actual_values
        .iter()
        .zip(reference_values)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let local = probed_site_pair(
        &tensor_a,
        &tensor_b,
        &[output_a.clone(), output_b.clone()],
        &probes,
        0,
    )
    .unwrap();
    let reference = contract_site_pair(&prefix_without_batch, &local, &[]).unwrap();
    let actual = contract_prefix_with_probed_site_pair(
        &prefix_without_batch,
        &tensor_a,
        &tensor_b,
        &[output_a, output_b],
        &probes,
        0,
    )
    .unwrap();
    assert_eq!(actual.indices(), reference.indices());
    let actual_values = actual.to_vec::<f64>().unwrap();
    let reference_values = reference.to_vec::<f64>().unwrap();
    assert_eq!(actual_values.len(), reference_values.len());
    assert!(actual_values
        .iter()
        .zip(reference_values)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
}

#[test]
fn scalar_probed_site_pair_batch_range_broadcasts_over_the_batch_axis() {
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let batch = DynIndex::new_dyn(3);
    let bank = ProbeBank::from_seed(vec![], 3, 17).unwrap();

    let left = DynIndex::new_dyn(2);
    let shared = DynIndex::new_dyn(1);
    let local = IdxTensor::from_dense(vec![left.clone(), shared.clone()], vec![2.0, 3.0]).unwrap();
    let other = IdxTensor::from_dense(vec![shared], vec![4.0]).unwrap();
    let pair =
        probed_site_pair_batch_range(&local, &other, &[], &bank, 0, 3, &batch, &context).unwrap();
    assert_eq!(pair.indices(), &[left, batch]);
    assert_eq!(
        pair.to_vec::<f64>().unwrap(),
        vec![8.0, 12.0, 8.0, 12.0, 8.0, 12.0]
    );
}

#[test]
fn maximum_probe_width_respects_rank_row_and_cut_bounds() {
    let fixed = SrcOptions::fixed();
    assert_eq!(maximum_site_width(8, 16, 12, &fixed), 8);
    assert_eq!(maximum_site_width(12, 16, 12, &fixed), 12);
    assert_eq!(maximum_site_width(32, 16, 12, &fixed), 12);

    let oversampled = SrcOptions::fixed().with_final_svd(true);
    assert_eq!(maximum_site_width(8, 16, 12, &oversampled), 12);

    let adaptive = SrcOptions::adaptive(1.0e-6, 4096);
    assert_eq!(maximum_site_width(4096, 1024, 256, &adaptive), 256);
}

#[test]
fn final_svd_adaptive_sketch_uses_the_paper_safety_factor() {
    let adaptive = SrcOptions::adaptive(1.0e-6, 64).with_final_svd(true);
    assert_eq!(adaptive.sketch_options(true).rtol, Some(1.0e-7));
    assert_eq!(adaptive.sketch_options(true).atol, adaptive.atol);

    let fixed = SrcOptions::fixed().with_final_svd(true);
    assert_eq!(fixed.sketch_options(true), fixed);
}

#[test]
fn final_svd_without_a_tolerance_policy_keeps_the_requested_sketch_tolerance() {
    let adaptive = SrcOptions::adaptive(1.0e-6, 64).with_final_svd(true);

    assert_eq!(adaptive.sketch_options(false).rtol, Some(1.0e-6));
}

#[test]
fn adaptive_src_defaults_to_the_requested_tolerance_without_final_round() {
    let adaptive = SrcOptions::adaptive(1.0e-6, 64);

    assert!(!adaptive.final_svd);
    assert_eq!(adaptive.sketch_options(false).rtol, Some(1.0e-6));
}

#[test]
fn factorize_probe_batches_grows_by_rank_increment_and_stops_on_error_estimate() {
    use super::factorize_probe_batches;

    let row = DynIndex::new_dyn(4);
    // A 4x4 identity-ish sketch (well-conditioned, rank 4) split into two
    // width-2 batches, so a rank_increment of 2 should need exactly two
    // `make_batch` calls to reach the full rank-4 estimate below tolerance.
    let batch0 = DynIndex::new_dyn(2);
    let batch1 = DynIndex::new_dyn(2);
    let full = IdxTensor::from_dense(
        vec![row.clone(), DynIndex::new_dyn(4)],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let column_index = full.indices()[1].clone();
    let col = |position: usize| {
        full.select_indices(std::slice::from_ref(&column_index), &[position])
            .unwrap()
    };
    let first_two =
        IdxTensor::stack_along_new_index(&[&col(0), &col(1)], batch0.clone(), -1).unwrap();
    let last_two =
        IdxTensor::stack_along_new_index(&[&col(2), &col(3)], batch1.clone(), -1).unwrap();

    let mut calls = Vec::new();
    let src_options = SrcOptions::adaptive(1.0e-10, 4)
        .with_min_rank(1)
        .with_rank_increment(2);
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let (result, result_batch) = factorize_probe_batches::<IdxTensor, _>(
        &[row],
        2,
        4,
        &src_options,
        "test",
        |start, width| {
            calls.push((start, width));
            if start == 0 {
                Ok((first_two.clone(), batch0.clone()))
            } else {
                Ok((last_two.clone(), batch1.clone()))
            }
        },
        &context,
    )
    .unwrap();

    assert_eq!(calls, vec![(0, 2), (2, 2)]);
    assert_eq!(result_batch.dim(), 4);
    let values = result.to_vec::<f64>().unwrap();
    assert_eq!(values.len(), 16);
}
