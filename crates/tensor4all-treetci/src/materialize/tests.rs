use super::{fill_tensor_values, site_side_key, to_treetn, to_treetn_chunked, FullPivLuScalar};
use crate::batch::EVALUATION_CHUNK_POINTS;
use crate::test_support::{assert_complex_slice_close, assert_scalar_close};
use crate::{
    column_2d, ncols_2d, optimize_default, GlobalIndexBatch, MultiIndex, SubtreeKey, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use num_complex::Complex64;
use std::cell::RefCell;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn to_treetn_preserves_two_site_identity_evaluations() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: None,
            normalize_error: true,
            ..Default::default()
        },
    )
    .unwrap();

    let tn = to_treetn(&tci, batch_eval, Some(0)).unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos0 = {
        let site_index = tn.site_space(&0usize).unwrap().iter().next().unwrap();
        indices
            .iter()
            .position(|index| index == site_index)
            .unwrap()
    };
    let pos1 = {
        let site_index = tn.site_space(&1usize).unwrap().iter().next().unwrap();
        indices
            .iter()
            .position(|index| index == site_index)
            .unwrap()
    };

    let eval = |i: usize, j: usize| -> f64 {
        let mut data = vec![0usize; indices.len()];
        data[pos0] = i;
        data[pos1] = j;
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        tn.evaluate(&indices, values).unwrap()[0].real()
    };

    assert_scalar_close(eval(0, 0), 1.0, 1.0, 1e-12);
    assert_scalar_close(eval(0, 1), 0.0, 1.0, 1e-12);
    assert_scalar_close(eval(1, 0), 0.0, 1.0, 1e-12);
    assert_scalar_close(eval(1, 1), 1.0, 1.0, 1e-12);
}

fn chain_graph() -> TreeTciGraph {
    let edges: Vec<TreeTciEdge> = (0..5).map(|i| TreeTciEdge::new(i, i + 1)).collect();
    TreeTciGraph::new(6, &edges).unwrap()
}

/// Sites 1 and 4 have degree 3.
fn branched_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ],
    )
    .unwrap()
}

/// A smooth, non-separable function of the point.
fn smooth_value(point: &[usize]) -> f64 {
    let x: f64 = point
        .iter()
        .enumerate()
        .map(|(site, &value)| value as f64 / (site as f64 + 2.0))
        .sum();
    1.0 / (1.0 + x * x)
}

fn smooth_eval(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    Ok(batch
        .data()
        .chunks(batch.n_sites())
        .map(smooth_value)
        .collect())
}

/// A converged state whose pivot sets have several columns on every bond.
fn converged_state(graph: TreeTciGraph, local_dims: Vec<usize>) -> TreeTCI2<f64> {
    let n_sites = local_dims.len();
    let mut tci = TreeTCI2::<f64>::new(local_dims, graph).unwrap();
    tci.add_global_pivots(&[vec![0; n_sites]]).unwrap();
    tci.max_sample_value = smooth_value(&vec![0; n_sites]);
    optimize_default(
        &mut tci,
        smooth_eval,
        &TreeTciOptions {
            tolerance: 1e-13,
            max_iter: 10,
            seed: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    tci
}

/// Points of `fill_tensor_values` as the previous implementation enumerated
/// them: one separately assembled point per entry of the cartesian product
/// `out_keys x in_keys x central_sites`, the first key and the last central
/// site varying fastest.
fn reference_fill_points(
    state: &TreeTCI2<f64>,
    in_keys: &[SubtreeKey],
    out_keys: &[SubtreeKey],
    central_sites: &[usize],
) -> Vec<usize> {
    fn combos(state: &TreeTCI2<f64>, keys: &[SubtreeKey]) -> Vec<Vec<MultiIndex>> {
        let mut combos = vec![Vec::new()];
        for key in keys {
            let pivots = &state.ijset[key];
            let mut next = Vec::new();
            for j in 0..ncols_2d(pivots).unwrap() {
                for combo in &combos {
                    let mut extended: Vec<MultiIndex> = combo.clone();
                    extended.push(column_2d(pivots, j).unwrap().to_vec());
                    next.push(extended);
                }
            }
            combos = next;
        }
        combos
    }
    let mut central_combos: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
    for &site in central_sites {
        let mut next = Vec::new();
        for combo in &central_combos {
            for value in 0..state.local_dims[site] {
                let mut extended = combo.clone();
                extended.push((site, value));
                next.push(extended);
            }
        }
        central_combos = next;
    }

    let mut points = Vec::new();
    for out_combo in combos(state, out_keys) {
        for in_combo in combos(state, in_keys) {
            for central in &central_combos {
                let mut point = vec![usize::MAX; state.local_dims.len()];
                let subtree_values = in_keys
                    .iter()
                    .zip(in_combo.iter())
                    .chain(out_keys.iter().zip(out_combo.iter()))
                    .flat_map(|(key, values)| key.as_slice().iter().zip(values.iter()));
                let central_values = central.iter().map(|(site, value)| (site, value));
                for (&site, &value) in subtree_values.chain(central_values) {
                    assert_eq!(point[site], usize::MAX, "site {site} assigned twice");
                    point[site] = value;
                }
                assert!(point.iter().all(|&value| value != usize::MAX));
                points.extend(point);
            }
        }
    }
    points
}

/// Every `(in_keys, out_keys, central_sites)` combination `to_treetn` can
/// request, for any choice of root.
fn fill_configurations(
    state: &TreeTCI2<f64>,
) -> Vec<(Vec<SubtreeKey>, Vec<SubtreeKey>, Vec<usize>)> {
    let graph = &state.graph;
    let mut configurations = Vec::new();
    for site in 0..graph.n_sites() {
        let all_edges = graph.adjacent_edges(site, &[]);
        let in_keys = graph.edge_in_ij_keys(site, &all_edges).unwrap();
        configurations.push((in_keys, Vec::new(), vec![site]));
        for &parent_edge in &all_edges {
            let incoming = graph.adjacent_edges(site, &[parent_edge]);
            let in_keys = graph.edge_in_ij_keys(site, &incoming).unwrap();
            let out_keys = graph.edge_in_ij_keys(site, &[parent_edge]).unwrap();
            let side_key = site_side_key(state, site, parent_edge).unwrap();
            configurations.push((in_keys, out_keys.clone(), vec![site]));
            configurations.push((vec![side_key], out_keys, Vec::new()));
        }
    }
    configurations
}

#[test]
fn fill_tensor_values_matches_reference_point_order() {
    for (graph, local_dims) in [
        (chain_graph(), vec![2, 3, 2, 3, 2, 3]),
        (branched_graph(), vec![2, 3, 2, 2, 3, 2, 3]),
    ] {
        let state = converged_state(graph, local_dims);
        assert!(state.max_bond_dim() >= 2);
        let n_sites = state.local_dims.len();
        for (in_keys, out_keys, central) in fill_configurations(&state) {
            let reference = reference_fill_points(&state, &in_keys, &out_keys, &central);
            for chunk_points in [1, 7, EVALUATION_CHUNK_POINTS] {
                let recorded = RefCell::new(Vec::new());
                let record = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                    recorded.borrow_mut().extend_from_slice(batch.data());
                    smooth_eval(batch)
                };
                let values = fill_tensor_values(
                    &state,
                    &in_keys,
                    &out_keys,
                    &central,
                    &record,
                    chunk_points,
                )
                .unwrap();
                assert_eq!(
                    recorded.into_inner(),
                    reference,
                    "in {in_keys:?} out {out_keys:?} central {central:?} chunk {chunk_points}"
                );
                assert_eq!(values.len() * n_sites, reference.len());
            }
        }
    }
}

#[test]
fn fill_tensor_values_rejects_invalid_partitions() {
    let state = converged_state(two_site_graph(), vec![2, 2]);
    let left = SubtreeKey::new(vec![0]);
    let right = SubtreeKey::new(vec![1]);
    let fill = |in_keys: &[SubtreeKey], out_keys: &[SubtreeKey], central: &[usize]| {
        fill_tensor_values(&state, in_keys, out_keys, central, &smooth_eval, 4)
    };

    assert!(fill(&[left.clone(), right.clone()], &[], &[]).is_ok());
    // Site 0 twice, site 1 never, an out-of-range central site, and a key
    // without pivots.
    assert!(fill(std::slice::from_ref(&left), &[], &[0])
        .unwrap_err()
        .to_string()
        .contains("assigned more than once"));
    assert!(fill(std::slice::from_ref(&left), &[], &[])
        .unwrap_err()
        .to_string()
        .contains("unassigned"));
    assert!(fill(std::slice::from_ref(&left), &[], &[9])
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
    assert!(fill(&[SubtreeKey::new(vec![7])], &[], &[])
        .unwrap_err()
        .to_string()
        .contains("missing pivot set"));

    // A pivot set whose rows do not match its key, and an empty one.
    let mut bad = state.clone();
    bad.ijset.insert(
        left.clone(),
        ColMajorArray::new(vec![0, 1], vec![2, 1]).unwrap(),
    );
    assert!(fill_tensor_values(
        &bad,
        std::slice::from_ref(&left),
        &[],
        &[1],
        &smooth_eval,
        4
    )
    .unwrap_err()
    .to_string()
    .contains("does not match subtree key"));
    bad.ijset.insert(
        left.clone(),
        ColMajorArray::new(vec![], vec![1, 0]).unwrap(),
    );
    assert!(fill_tensor_values(&bad, &[left], &[], &[1], &smooth_eval, 4).is_err());
}

/// Per-node dimensions and raw value bits, independent of bond index ids.
fn tensor_bits(tn: &TreeTN<IdxTensor, usize>, n_sites: usize) -> Vec<(Vec<usize>, Vec<u64>)> {
    (0..n_sites)
        .map(|site| {
            let tensor = tn.tensor(tn.node_index(&site).unwrap()).unwrap();
            let bits = tensor
                .to_vec::<f64>()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            (tensor.dims(), bits)
        })
        .collect()
}

#[test]
fn to_treetn_is_independent_of_chunk_size_on_chain_and_branched_tree() {
    for (graph, local_dims) in [
        (chain_graph(), vec![2, 3, 2, 3, 2, 3]),
        (branched_graph(), vec![2, 3, 2, 2, 3, 2, 3]),
    ] {
        let state = converged_state(graph, local_dims.clone());
        let n_sites = local_dims.len();
        for center in [0, 1, n_sites - 1] {
            let tn = to_treetn(&state, smooth_eval, Some(center)).unwrap();
            let reference_bits = tensor_bits(&tn, n_sites);
            for chunk_points in [1, 7] {
                let chunked =
                    to_treetn_chunked(&state, &smooth_eval, Some(center), chunk_points).unwrap();
                assert_eq!(
                    tensor_bits(&chunked, n_sites),
                    reference_bits,
                    "center {center} chunk {chunk_points}"
                );
            }

            // The interpolation is exact here: compare with a dense reference.
            let indices: Vec<DynIndex> = (0..n_sites)
                .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
                .collect();
            let total: usize = local_dims.iter().product();
            let mut point = vec![0usize; n_sites];
            let mut values = Vec::with_capacity(total);
            for _ in 0..total {
                values.push(smooth_value(&point));
                for (slot, &dim) in point.iter_mut().zip(&local_dims) {
                    *slot += 1;
                    if *slot < dim {
                        break;
                    }
                    *slot = 0;
                }
            }
            let expected = IdxTensor::from_dense(indices, values).unwrap();
            let residual = tn
                .to_dense()
                .unwrap()
                .sub(&expected)
                .unwrap()
                .maxabs()
                .unwrap();
            assert!(residual < 1e-10, "center {center}: residual {residual:e}");
        }
    }
}

#[test]
fn solve_right_full_piv_lu_preserves_complex_entries_for_identity_pivot() {
    let lhs = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(5.0, -6.0),
        Complex64::new(-7.0, -8.0),
    ];
    let pivot = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let solved = Complex64::solve_right_full_piv_lu(&lhs, 2, 2, &pivot, 2, 2).unwrap();

    let max_sample = lhs.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
    assert_complex_slice_close(&solved, &lhs, max_sample, 1e-12);
}

#[test]
fn solve_right_full_piv_lu_recovers_complex_rhs_for_nontrivial_pivot() {
    let target = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(5.0, -6.0),
        Complex64::new(-7.0, -8.0),
    ];
    let pivot = vec![
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 3.0),
        Complex64::new(4.0, -2.0),
        Complex64::new(3.0, 5.0),
    ];
    let pi1 = vec![
        target[0] * pivot[0] + target[2] * pivot[1],
        target[1] * pivot[0] + target[3] * pivot[1],
        target[0] * pivot[2] + target[2] * pivot[3],
        target[1] * pivot[2] + target[3] * pivot[3],
    ];

    let solved = Complex64::solve_right_full_piv_lu(&pi1, 2, 2, &pivot, 2, 2).unwrap();

    let max_sample = target
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    assert_complex_slice_close(&solved, &target, max_sample, 1e-12);
}

#[test]
fn to_treetn_emits_zero_core_for_zero_pivot_matrix() {
    // A pivot whose sampled function values are all exactly zero produces a
    // singular pivot matrix in `site_tensor_with_parent`; materialization
    // must emit a zero core instead of failing the solve.
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    // Evaluator that is exactly zero at every pivot cross-section.
    let batch_eval =
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0; batch.n_points()]) };

    let tn = to_treetn(&tci, batch_eval, Some(0)).unwrap();
    let dense = tn.to_dense().unwrap();
    let values = dense.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![0.0; 4]);
}
