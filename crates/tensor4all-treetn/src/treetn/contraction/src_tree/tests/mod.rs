use super::super::src_probe::{
    local_output_indices, local_site_pairs, probe_batch_tensors, probed_site_pair_batch_range,
    ProbeBank,
};
use super::{
    directed_messages_batched, directed_messages_batched_from_factors, uncontracted_indices,
    EnvironmentCache,
};
use crate::treetn::TreeTN;
use std::collections::{HashMap, HashSet};
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
    let actual = directed_messages_batched_from_factors(
        &tn_a,
        &edges,
        &local,
        &probe_batches,
        &batch,
        &context,
    )
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

/// Sorted node list, rooted edges, local pairs and outputs of one pair of
/// networks: the inputs every `EnvironmentCache` test builds.
type TreeInputs<'a> = (
    Vec<String>,
    Vec<(String, String)>,
    HashMap<String, (&'a IdxTensor, &'a IdxTensor)>,
    HashMap<String, Vec<DynIndex>>,
);

fn tree_inputs<'a>(
    tn_a: &'a TreeTN<IdxTensor, String>,
    tn_b: &'a TreeTN<IdxTensor, String>,
    center: &str,
) -> TreeInputs<'a> {
    let mut nodes = tn_a.node_names();
    nodes.sort();
    let edges = tn_a
        .edges_to_canonicalize_by_names(&center.to_string())
        .unwrap();
    let local_values = local_site_pairs(tn_a, tn_b, &nodes).unwrap();
    let local = nodes
        .iter()
        .cloned()
        .zip(local_values)
        .collect::<HashMap<_, _>>();
    let outputs = nodes
        .iter()
        .map(|node| {
            (
                node.clone(),
                local_output_indices(tn_a, tn_b, node).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
    (nodes, edges, local, outputs)
}

fn probe_indices(nodes: &[String], outputs: &HashMap<String, Vec<DynIndex>>) -> Vec<DynIndex> {
    nodes
        .iter()
        .flat_map(|node| outputs[node].iter().cloned())
        .collect::<Vec<_>>()
}

fn cpu_context() -> tensor4all_tensorbackend::ExecutionContext {
    tensor4all_tensorbackend::ExecutionContext::Cpu(
        tensor4all_tensorbackend::default_cpu_execution_context(),
    )
}

/// Issue #696 Finding B: `batch(parent, child, width)` used to key its cache
/// by the exact width and rebuild *every* probe column from 0 on each miss,
/// so a run of edge-specific widths paid `sum(width)` columns of batched
/// message work. Growing incrementally must pay only the missing columns.
#[test]
fn batch_materializes_only_the_missing_probe_columns() {
    let tn_a = three_node_path(1.0);
    let tn_b = three_node_path(2.0);
    let (nodes, edges, local, outputs) = tree_inputs(&tn_a, &tn_b, "B");
    let mut probes = ProbeBank::from_seed(probe_indices(&nodes, &outputs), 1, 17).unwrap();
    let context = cpu_context();
    // Every width below is on the segment grid this cache builds, so each
    // miss is expected to add exactly `width - already` columns.
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

    let parent = "B".to_string();
    let child = "C".to_string();
    let mut already = 0usize;
    for width in [2usize, 4, 6] {
        let before = cache.columns_computed;
        let (_environment, batch) = cache.batch(&parent, &child, width).unwrap();
        assert_eq!(batch.dim(), width);
        let materialized = cache.columns_computed - before;
        assert_eq!(
            materialized,
            width - already,
            "batch({width}) must materialize width - already = {} columns, not {materialized}",
            width - already
        );
        already = cache.columns_computed;
    }
    assert_eq!(
        cache.columns_computed, 6,
        "growing 2 -> 4 -> 6 must materialize 6 columns in total, not 12"
    );
}

/// Largest per-column deviation between two batched environments, comparing
/// `actual_columns` of `actual` against `expected` from `expected_offset` on.
fn max_column_difference(
    actual: &IdxTensor,
    actual_batch: &DynIndex,
    expected: &IdxTensor,
    expected_batch: &DynIndex,
    actual_columns: std::ops::Range<usize>,
    expected_offset: usize,
) -> f64 {
    let mut max_difference = 0.0f64;
    for (actual_column, expected_column) in actual_columns.zip(expected_offset..) {
        let actual_column_tensor = actual
            .select_indices(std::slice::from_ref(actual_batch), &[actual_column])
            .unwrap();
        let expected_column_tensor = expected
            .select_indices(std::slice::from_ref(expected_batch), &[expected_column])
            .unwrap();
        max_difference = max_difference.max(
            actual_column_tensor
                .sub(&expected_column_tensor)
                .unwrap()
                .maxabs()
                .unwrap(),
        );
    }
    max_difference
}

/// The incremental cache must return what a from-zero rebuild returns: same
/// probe columns, same per-column contraction values. The reference below is
/// the pre-fix `batch` body (probe columns `[0, width)` rebuilt into one fresh
/// batch) and stays local to this test.
///
/// Equality is claimed within 1e-12, not bit equality: a probe column computed
/// inside a wider batch can round differently from the same column computed in
/// a narrower one because the contraction kernel blocks over the batch axis.
/// Measured on the development machine, max |difference| is 0 for width 2 and
/// 1.07e-14 for every other width below, against tensor values of order 1 to
/// 100.
#[test]
fn incremental_batch_matches_a_from_zero_rebuild_reference() {
    let tn_a = three_node_path(1.0);
    let tn_b = three_node_path(2.0);
    let (nodes, edges, local, outputs) = tree_inputs(&tn_a, &tn_b, "B");
    let context = cpu_context();
    let mut reference_probes =
        ProbeBank::from_seed(probe_indices(&nodes, &outputs), 1, 23).unwrap();
    let mut incremental_probes =
        ProbeBank::from_seed(probe_indices(&nodes, &outputs), 1, 23).unwrap();
    let mut cache = EnvironmentCache::new(
        &tn_a,
        &edges,
        &nodes,
        &local,
        &outputs,
        &mut incremental_probes,
        2,
        &context,
    );
    let parent = "B".to_string();
    let child = "C".to_string();

    let mut rebuild_from_zero = |width: usize| -> (IdxTensor, DynIndex) {
        reference_probes.extend_to(width).unwrap();
        let reference_batch = DynIndex::new_link(width).unwrap();
        let probe_batches = nodes
            .iter()
            .map(|node| {
                (
                    node.clone(),
                    probe_batch_tensors(
                        &outputs[node],
                        &reference_probes,
                        0,
                        width,
                        &reference_batch,
                        &context,
                    )
                    .unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        let directed = directed_messages_batched_from_factors(
            &tn_a,
            &edges,
            &local,
            &probe_batches,
            &reference_batch,
            &context,
        )
        .unwrap();
        (
            directed[&(parent.clone(), child.clone())].clone(),
            reference_batch,
        )
    };

    // Widths 2, 3 and 6 grow the segments; 4 is requested only afterwards and
    // was never cached, so it must be assembled from the existing segments
    // (wider cached width, narrower request) without growing a new one.
    let mut width_four_batch = None;
    for width in [2usize, 3, 6, 4] {
        let (expected, expected_batch) = rebuild_from_zero(width);
        let (actual, actual_batch) = cache.batch(&parent, &child, width).unwrap();
        assert_eq!(actual_batch.dim(), width);
        let max_difference = max_column_difference(
            &actual,
            &actual_batch,
            &expected,
            &expected_batch,
            0..width,
            0,
        );
        assert!(
            max_difference <= 1.0e-12,
            "batch({width}) must reproduce the from-zero rebuild, got max |diff| {max_difference:e}"
        );
        if width == 4 {
            width_four_batch = Some(actual_batch);
        }
    }

    // The exact-width entry now holds this edge at width 4, so the repeat must
    // take the fast path: the same batch index comes back instead of a freshly
    // assembled one.
    let (_, cached_batch) = cache.batch(&parent, &child, 4).unwrap();
    assert_eq!(
        Some(cached_batch),
        width_four_batch,
        "a repeated width must be served from the exact-width entry"
    );

    // A range that cuts both ends of the cached segments: [1, 5) is assembled
    // from the tail of one segment, whole segments, and the head of another,
    // which is the split-and-concatenate path `batch` never takes.
    let (expected, expected_batch) = rebuild_from_zero(6);
    let (actual, actual_batch) = cache.request(&parent, &child, 1, 4).unwrap();
    assert_eq!(actual_batch.dim(), 4);
    let max_difference =
        max_column_difference(&actual, &actual_batch, &expected, &expected_batch, 0..4, 1);
    assert!(
        max_difference <= 1.0e-12,
        "request(1, 4) must reproduce the from-zero columns 1..5, got max |diff| {max_difference:e}"
    );
}

#[test]
#[ignore = "manual measurement harness for issue #696 Finding B"]
fn bench_batch_cost_per_call_by_width() {
    // Path A-B-C-D-E rooted at C: four edges, each with its own width, which
    // is the heterogeneous-branch case the issue describes.
    fn path_tree(dim: usize, len: usize, offset: f64) -> TreeTN<IdxTensor, String> {
        let bonds = (0..len - 1)
            .map(|_| DynIndex::new_dyn(dim))
            .collect::<Vec<_>>();
        let outs = (0..len).map(|_| DynIndex::new_dyn(dim)).collect::<Vec<_>>();
        let mut names = Vec::new();
        let mut tensors = Vec::new();
        for site in 0..len {
            let mut index_list = Vec::new();
            if site > 0 {
                index_list.push(bonds[site - 1].clone());
            }
            index_list.push(outs[site].clone());
            if site + 1 < len {
                index_list.push(bonds[site].clone());
            }
            let size = index_list
                .iter()
                .map(|index| index.dim())
                .product::<usize>();
            tensors.push(
                IdxTensor::from_dense(
                    index_list,
                    (0..size)
                        .map(|i| offset + f64::from(i as i32) / 13.0)
                        .collect(),
                )
                .unwrap(),
            );
            names.push(format!("N{site}"));
        }
        TreeTN::from_tensors(tensors, names).unwrap()
    }

    // (label, dimension, path length, batch_size, four per-edge widths)
    let configs = [
        ("dim3-5nodes-batch8", 3usize, 5usize, 8usize, 8usize, 8usize),
        ("dim4-7nodes-batch3", 4, 7, 3, 12, 6),
        ("dim4-7nodes-batch3-wide", 4, 7, 3, 24, 12),
    ];
    println!(
        "cpu execution context threads: {}",
        tensor4all_tensorbackend::default_cpu_execution_context()
            .with_backend(|backend| backend.num_threads())
    );
    let context = cpu_context();
    let repetitions = 15;
    for (label, dim, len, batch_size, first_width, step) in configs {
        let tn_a = path_tree(dim, len, 1.0);
        let tn_b = path_tree(dim, len, 2.0);
        let center = format!("N{}", len / 2);
        let (nodes, edges, local, outputs) = tree_inputs(&tn_a, &tn_b, &center);
        println!("--- {label}: edges {edges:?}, batch_size {batch_size}");
        let mut requests = Vec::new();
        for (slot, (child, parent)) in edges.iter().enumerate() {
            requests.push((parent.clone(), child.clone(), first_width + step * slot));
        }
        let mut per_call = vec![Vec::new(); requests.len()];
        let mut pass = Vec::new();
        for _ in 0..repetitions {
            let mut probes = ProbeBank::from_seed(probe_indices(&nodes, &outputs), 1, 31).unwrap();
            let mut cache = EnvironmentCache::new(
                &tn_a,
                &edges,
                &nodes,
                &local,
                &outputs,
                &mut probes,
                batch_size,
                &context,
            );
            let start_time = std::time::Instant::now();
            for (slot, (parent, child, width)) in requests.iter().enumerate() {
                let call = std::time::Instant::now();
                let (_environment, batch_index) = cache.batch(parent, child, *width).unwrap();
                assert_eq!(batch_index.dim(), *width);
                per_call[slot].push(call.elapsed().as_secs_f64() * 1e6);
            }
            pass.push(start_time.elapsed().as_secs_f64() * 1e6);
        }
        let median = |values: &mut Vec<f64>| {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values[values.len() / 2]
        };
        for (slot, (parent, child, width)) in requests.iter().enumerate() {
            println!(
                "batch({parent}->{child}, width {width}): median {:.1} us",
                median(&mut per_call[slot])
            );
        }
        println!(
            "one pass over {} edges: median {:.1} us",
            requests.len(),
            median(&mut pass)
        );
    }
}
