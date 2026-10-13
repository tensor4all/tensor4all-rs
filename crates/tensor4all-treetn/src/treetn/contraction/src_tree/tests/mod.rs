use super::super::src_probe::{
    local_output_indices, local_site_pairs, probe_batch_tensors, probed_site_pair_batch_range,
    ProbeBank,
};
use super::{
    directed_messages_batched, directed_messages_batched_from_factors, uncontracted_indices,
    EnvironmentCache,
};
use crate::treetn::TreeTN;
use std::collections::HashSet;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

/// `uncontracted_indices` used to order same-dimension survivors by
/// `Index::id()` (via `HashMap` + `sort_indices_deterministic`), and
/// `id()` is drawn from a per-process, unseeded RNG that `contract()`
/// re-invokes on every call (`sim_internal_inds()` mints fresh bond ids
/// each time). Rebuilding `out_a`/`out_b` fresh each iteration mimics
/// that: under the old implementation this had roughly a 50% chance per
/// iteration of returning `[out_b, out_a]` instead, so 200 iterations
/// would fail with probability `1 - 2^-200`. Under the fix, order always
/// follows `factors`' own order, regardless of the ids involved.
#[test]
fn uncontracted_indices_orders_same_dimension_survivors_by_factor_order_not_random_id() {
    for _ in 0..200 {
        let shared = DynIndex::new_dyn(2);
        let out_a = DynIndex::new_dyn(3);
        let out_b = DynIndex::new_dyn(3);

        let factor_1 =
            IdxTensor::from_dense(vec![shared.clone(), out_a.clone()], vec![0.0; 2 * 3]).unwrap();
        let factor_2 =
            IdxTensor::from_dense(vec![shared.clone(), out_b.clone()], vec![0.0; 2 * 3]).unwrap();

        let result = uncontracted_indices(&[&factor_1, &factor_2], &HashSet::new());
        assert_eq!(result, vec![out_a.clone(), out_b.clone()]);
    }
}

fn three_node_path(offset: f64) -> TreeTN<IdxTensor, String> {
    let dim = 3usize;
    let ab = DynIndex::new_dyn(dim);
    let bc = DynIndex::new_dyn(dim);
    let a_out = DynIndex::new_dyn(dim);
    let b_out = DynIndex::new_dyn(dim);
    let c_out = DynIndex::new_dyn(dim);
    let a = IdxTensor::from_dense(
        vec![a_out, ab.clone()],
        (0..dim * dim)
            .map(|i| offset + f64::from(i as i32) / 10.0)
            .collect(),
    )
    .unwrap();
    let b = IdxTensor::from_dense(
        vec![ab, b_out, bc.clone()],
        (0..dim * dim * dim)
            .map(|i| offset + f64::from(i as i32) / 11.0)
            .collect(),
    )
    .unwrap();
    let c = IdxTensor::from_dense(
        vec![bc, c_out],
        (0..dim * dim)
            .map(|i| offset + f64::from(i as i32) / 12.0)
            .collect(),
    )
    .unwrap();
    TreeTN::from_tensors(vec![a, b, c], vec!["A".into(), "B".into(), "C".into()]).unwrap()
}

#[test]
fn request_grows_a_fresh_segment_and_re_reads_an_aligned_one() {
    let tn_a = three_node_path(1.0);
    let tn_b = three_node_path(2.0);
    let mut nodes = tn_a.node_names();
    nodes.sort();
    let edges = tn_a
        .edges_to_canonicalize_by_names(&"B".to_string())
        .unwrap();

    let local_values = local_site_pairs(&tn_a, &tn_b, &nodes).unwrap();
    let local = nodes
        .iter()
        .cloned()
        .zip(local_values)
        .collect::<std::collections::HashMap<_, _>>();
    let outputs = nodes
        .iter()
        .map(|node| {
            (
                node.clone(),
                local_output_indices(&tn_a, &tn_b, node).unwrap(),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut probe_indices = nodes
        .iter()
        .flat_map(|node| outputs[node].iter().cloned())
        .collect::<Vec<_>>();
    let mut probes = ProbeBank::from_seed(std::mem::take(&mut probe_indices), 1, 99).unwrap();
    // The first width-3 request is narrower than batch_size 5, so this
    // exercises the permitted short initial segment and its direct replay.
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let mut cache = EnvironmentCache::new(
        &tn_a,
        &edges,
        &nodes,
        &local,
        &outputs,
        &mut probes,
        5,
        &context,
    );
    assert!(cache
        .request(&"B".to_string(), &"A".to_string(), 0, 0)
        .is_err());

    let (first, first_batch) = cache
        .request(&"B".to_string(), &"A".to_string(), 0, 3)
        .unwrap();
    assert_eq!(first_batch.dim(), 3);

    let (again, again_batch) = cache
        .request(&"B".to_string(), &"A".to_string(), 0, 3)
        .unwrap();
    assert_eq!(again_batch.dim(), first_batch.dim());
    let difference = again.sub(&first).unwrap().maxabs().unwrap();
    assert!(
        difference < 1e-12,
        "re-read segment tensor should exactly match the grown one, got difference {difference}"
    );
    assert_eq!(
        cache.segments.len(),
        1,
        "re-reading an aligned segment must not grow a new one"
    );
}

#[test]
fn request_shared_across_edges_replays_a_misaligned_range_exactly() {
    // `EnvironmentCache` is shared across every edge in the tree, and
    // different edges generally need different widths (different ranks).
    // This reproduces exactly that: edge (B, A) requests width 5 on a
    // batch_size-2 grid. Later segments stay full, producing [0,2),
    // [2,2), [4,2), so future rank increments remain aligned. Edge (B, C)
    // then requests [0,3), which still genuinely spans segment boundaries
    // and exercises the split/re-stack fallback. Re-reading the same range
    // must reproduce the cached columns bit-for-bit.
    let tn_a = three_node_path(1.0);
    let tn_b = three_node_path(2.0);
    let mut nodes = tn_a.node_names();
    nodes.sort();
    let edges = tn_a
        .edges_to_canonicalize_by_names(&"B".to_string())
        .unwrap();

    let local_values = local_site_pairs(&tn_a, &tn_b, &nodes).unwrap();
    let local = nodes
        .iter()
        .cloned()
        .zip(local_values)
        .collect::<std::collections::HashMap<_, _>>();
    let outputs = nodes
        .iter()
        .map(|node| {
            (
                node.clone(),
                local_output_indices(&tn_a, &tn_b, node).unwrap(),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let index_list = nodes
        .iter()
        .flat_map(|node| outputs[node].iter().cloned())
        .collect::<Vec<_>>();

    let mut probes = ProbeBank::from_seed(index_list, 1, 7).unwrap();
    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let mut cache = EnvironmentCache::new(
        &tn_a,
        &edges,
        &nodes,
        &local,
        &outputs,
        &mut probes,
        2,
        &context,
    );

    let (_wide, wide_batch) = cache
        .request(&"B".to_string(), &"A".to_string(), 0, 5)
        .unwrap();
    assert_eq!(wide_batch.dim(), 5);
    assert_eq!(cache.segments.len(), 3);
    assert_eq!(
        cache
            .segments
            .iter()
            .map(|(_, width, _, _)| *width)
            .collect::<Vec<_>>(),
        vec![2, 2, 2],
        "later tree segments must keep the rank-increment grid aligned"
    );
    assert_eq!(cache.segment_total_width - 5, 1);
    assert!(cache.segment_total_width - 5 < cache.batch_size);

    let (narrow, narrow_batch) = cache
        .request(&"B".to_string(), &"C".to_string(), 0, 3)
        .unwrap();
    assert_eq!(narrow_batch.dim(), 3);
    assert_eq!(
        cache.segments.len(),
        3,
        "edge C's narrower request is already covered by edge A's growth and needs no new segment"
    );

    let (replayed, replayed_batch) = cache
        .request(&"B".to_string(), &"C".to_string(), 0, 3)
        .unwrap();
    assert_eq!(replayed_batch.dim(), narrow_batch.dim());
    assert_eq!(cache.segments.len(), 3);
    assert_eq!(
        narrow.to_vec::<f64>().unwrap(),
        replayed.to_vec::<f64>().unwrap(),
        "misaligned fallback must replay cached columns exactly"
    );

    let (_aligned, aligned_batch) = cache
        .request(&"B".to_string(), &"C".to_string(), 4, 2)
        .unwrap();
    assert_eq!(aligned_batch, cache.segments[2].2);
}

#[test]
fn factorized_directed_messages_match_prepaired_reference() {
    let tn_a = three_node_path(1.0);
    let tn_b = three_node_path(2.0);
    let mut nodes = tn_a.node_names();
    nodes.sort();
    let edges = tn_a
        .edges_to_canonicalize_by_names(&"B".to_string())
        .unwrap();
    let local_values = local_site_pairs(&tn_a, &tn_b, &nodes).unwrap();
    let local = nodes
        .iter()
        .cloned()
        .zip(local_values)
        .collect::<std::collections::HashMap<_, _>>();
    let outputs = nodes
        .iter()
        .map(|node| {
            (
                node.clone(),
                local_output_indices(&tn_a, &tn_b, node).unwrap(),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let probe_indices = nodes
        .iter()
        .flat_map(|node| outputs[node].iter().cloned())
        .collect::<Vec<_>>();
    let probes = ProbeBank::from_seed(probe_indices, 3, 41).unwrap();
    let batch = DynIndex::new_link(3).unwrap();

    let prepaired = nodes
        .iter()
        .map(|node| {
            let (tensor_a, tensor_b) = local[node];
            let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
                tensor4all_tensorbackend::default_cpu_execution_context(),
            );
            let tensor = probed_site_pair_batch_range(
                tensor_a,
                tensor_b,
                &outputs[node],
                &probes,
                0,
                3,
                &batch,
                &context,
            )
            .unwrap();
            (node.clone(), tensor)
        })
        .collect::<std::collections::HashMap<_, _>>();
    let reference = directed_messages_batched(&tn_a, &edges, &prepaired, &batch).unwrap();

    let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    );
    let probe_batches = nodes
        .iter()
        .map(|node| {
            (
                node.clone(),
                probe_batch_tensors(&outputs[node], &probes, 0, 3, &batch, &context).unwrap(),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let actual =
        directed_messages_batched_from_factors(&tn_a, &edges, &local, &probe_batches, &batch)
            .unwrap();

    assert_eq!(actual.len(), reference.len());
    for (edge, expected) in reference {
        let got = actual.get(&edge).unwrap();
        let residual = got.sub(&expected).unwrap().maxabs().unwrap();
        assert!(
            residual < 1.0e-10,
            "factorized directed message {edge:?} residual is {residual}"
        );
    }
}
