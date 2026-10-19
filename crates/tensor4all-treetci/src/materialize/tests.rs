use super::{
    fill_tensor_values, site_side_key, to_named_treetn, to_treetn, to_treetn_chunked,
    FullPivLuScalar,
};
use crate::batch::EVALUATION_CHUNK_POINTS;
use crate::test_support::{assert_complex_slice_close, assert_scalar_close};
use crate::{
    column_2d, ncols_2d, optimize_default, GlobalIndexBatch, MultiIndex, SubtreeKey, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use num_complex::{Complex32, Complex64};
use std::cell::RefCell;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;


/// The context-free materialization route, with the evaluator wrapped as the run's
/// execution owner.
fn materialize_compat(
    state: &TreeTCI2<f64>,
    mut evaluate: impl FnMut(GlobalIndexBatch<'_>) -> Result<Vec<f64>>,
    center: Option<usize>,
    chunk_points: usize,
) -> crate::TreeTciResult<TreeTN<IdxTensor, usize>> {
    let mut execution = crate::evaluation::Compatibility::new(&mut evaluate);
    to_treetn_chunked(state, &mut execution, center, chunk_points)
}

/// The same materialization on one concrete session of `context`.
#[cfg(test)]
fn materialize_in_context(
    state: &TreeTCI2<f64>,
    mut evaluate: impl FnMut(GlobalIndexBatch<'_>) -> Result<Vec<f64>>,
    center: Option<usize>,
    chunk_points: usize,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> crate::TreeTciResult<TreeTN<IdxTensor, usize>> {
    context
        .with_concrete_session(|session| {
            let mut contextual =
                |batch: GlobalIndexBatch<'_>, _ctx: &mut tensor4all_treetn::interpolation::EvalContext<'_>| {
                    evaluate(batch)
                };
            let mut ctx = crate::evaluation::serial_context(session);
            let mut execution = crate::evaluation::Explicit::new(&mut ctx, &mut contextual);
            to_treetn_chunked(state, &mut execution, center, chunk_points)
        })
        .map_err(|error| crate::TreeTciError::from(anyhow::Error::new(error)))?
}

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
            scale_tolerance: true,
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
                let mut record = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                    recorded.borrow_mut().extend_from_slice(batch.data());
                    smooth_eval(batch)
                };
                let values = fill_tensor_values(
                    &state,
                    &in_keys,
                    &out_keys,
                    &central,
                    &mut crate::evaluation::Compatibility::new(&mut record),
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
        let mut evaluate = smooth_eval;
        fill_tensor_values(
            &state,
            in_keys,
            out_keys,
            central,
            &mut crate::evaluation::Compatibility::new(&mut evaluate),
            4,
        )
    };

    assert!(fill(&[left.clone(), right.clone()], &[], &[]).is_ok());
    // Site 0 twice, site 1 never, an out-of-range central site, and a key
    // without pivots.
    let typed = |error: anyhow::Error| {
        matches!(
            error.downcast_ref::<crate::TreeTciError>(),
            Some(crate::TreeTciError::IndexOutOfBounds { .. })
        )
    };
    let duplicate = fill(std::slice::from_ref(&left), &[], &[0]).unwrap_err();
    assert!(duplicate.to_string().contains("assigned more than once"));
    assert!(typed(duplicate));
    assert!(typed(
        fill(std::slice::from_ref(&left), &[], &[]).unwrap_err()
    ));
    assert!(typed(
        fill(std::slice::from_ref(&left), &[], &[9]).unwrap_err()
    ));
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
    let mut evaluate = smooth_eval;
    assert!(fill_tensor_values(
        &bad,
        std::slice::from_ref(&left),
        &[],
        &[1],
        &mut crate::evaluation::Compatibility::new(&mut evaluate),
        4
    )
    .unwrap_err()
    .to_string()
    .contains("does not match subtree key"));
    bad.ijset.insert(
        left.clone(),
        ColMajorArray::new(vec![], vec![1, 0]).unwrap(),
    );
    let mut evaluate = smooth_eval;
    assert!(fill_tensor_values(
        &bad,
        &[left],
        &[],
        &[1],
        &mut crate::evaluation::Compatibility::new(&mut evaluate),
        4
    )
    .is_err());
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
                let chunked = materialize_compat(&state, smooth_eval, Some(center), chunk_points)
                    .unwrap();
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

fn zero_pivot_matrix<T>()
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
{
    // A pivot whose sampled function values are all exactly zero produces a
    // singular pivot matrix in `site_tensor_with_parent`; materialization
    // must emit a zero core instead of failing the solve.
    let mut tci = TreeTCI2::<T>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    // Evaluator that is exactly zero at every pivot cross-section.
    for zero in [T::zero(), T::from_f64(-0.0)] {
        let batch_eval =
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> { Ok(vec![zero; batch.n_points()]) };
        for center in [0, 1] {
            let tn = to_treetn(&tci, batch_eval, Some(center)).unwrap();
            assert_eq!(tn.to_dense().unwrap().maxabs().unwrap(), 0.0);
        }
    }
}

#[test]
fn to_treetn_emits_zero_core_for_zero_pivot_matrix() {
    zero_pivot_matrix::<f64>();
    zero_pivot_matrix::<Complex64>();
    zero_pivot_matrix::<f32>();
    zero_pivot_matrix::<Complex32>();
}

/// Check a whole dense reconstruction relative to the oracle's scale. An
/// all-zero materialization must fail even when every value is tiny.
fn assert_scaled_materialization<T, F>(state: &TreeTCI2<T>, value: F, tolerance: f64)
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(&[usize]) -> T,
{
    let n_sites = state.local_dims.len();
    for scale in [1.0, 1e-16, 1e-20, 1e-31] {
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            Ok(batch
                .data()
                .chunks(batch.n_sites())
                .map(|point| value(point) * T::from_f64(scale))
                .collect())
        };
        // None exercises the default root; explicit roots exercise every
        // direction of the pivot solve, including site-free junctions.
        for center in std::iter::once(None).chain((0..n_sites).map(Some)) {
            let tn = to_treetn(state, evaluate, center).unwrap();
            let indices: Vec<DynIndex> = (0..n_sites)
                .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
                .collect();
            let total: usize = state.local_dims.iter().product();
            let mut point = vec![0; n_sites];
            let mut values = Vec::with_capacity(total);
            for _ in 0..total {
                values.push(value(&point) * T::from_f64(scale));
                for (slot, &dim) in point.iter_mut().zip(&state.local_dims) {
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
                .unwrap()
                / scale;
            assert!(
                residual < tolerance,
                "scale {scale:e}, center {center:?}: scaled residual {residual:e}"
            );
        }
    }
}

fn small_full_rank_pivot<T>(value: impl Fn(&[usize]) -> T, tolerance: f64)
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
{
    // Seed both sides completely so the test isolates materialization from
    // the independent small-pivot floor in the optimizer (#819 / #779).
    let mut state = TreeTCI2::<T>::new(vec![2, 2], two_site_graph()).unwrap();
    state.add_global_pivots(&[vec![0, 0], vec![1, 1]]).unwrap();
    assert_eq!(state.max_bond_dim(), 2);
    assert_scaled_materialization(&state, &value, tolerance);

    // Rank-one constant oracles exercise a rectangular site block and the
    // scalar pivot solve, including the 1e-31 initialization reproducer.
    let mut constant_state = TreeTCI2::<T>::new(vec![3, 2], two_site_graph()).unwrap();
    constant_state.add_global_pivots(&[vec![0, 0]]).unwrap();
    let constant = value(&[0, 0]);
    assert_scaled_materialization(&constant_state, |_| constant, tolerance);
}

#[test]
fn to_treetn_preserves_small_nonzero_full_rank_pivots() {
    let real = |point: &[usize]| 1.0 + point[0] as f64 + 2.0 * point[1] as f64;
    let imag = |point: &[usize]| 0.25 + 0.5 * point[0] as f64 - 0.75 * point[1] as f64;
    small_full_rank_pivot(real, 1e-12);
    small_full_rank_pivot(|point| Complex64::new(real(point), imag(point)), 1e-12);
    small_full_rank_pivot(|point| real(point) as f32, 2e-6);
    small_full_rank_pivot(
        |point| Complex32::new(real(point) as f32, imag(point) as f32),
        2e-6,
    );
}

fn small_tree_pivots<T>(value: impl Fn(&[usize]) -> T)
where
    T: FullPivLuScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + tensor4all_core::CommonScalar
        + crate::globalpivot::ScalarParts,
{
    for (graph, dims) in [
        (chain_graph(), vec![2; 6]),
        // Degree-three junctions at sites 1 and 4 have no physical variable.
        (branched_graph(), vec![2, 1, 2, 2, 1, 2, 2]),
    ] {
        let n_sites = dims.len();
        let mut state = TreeTCI2::<T>::new(dims, graph).unwrap();
        state.add_global_pivots(&[vec![0; n_sites]]).unwrap();
        state.max_sample_value = value(&vec![0; n_sites]).abs_val();
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            Ok(batch.data().chunks(batch.n_sites()).map(&value).collect())
        };
        // Obtain well-conditioned pivots at unit scale, then materialize the
        // same state at each scale, without relying on the core floor fix.
        optimize_default(
            &mut state,
            evaluate,
            &TreeTciOptions {
                tolerance: 1e-13,
                max_iter: 10,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(state.max_bond_dim() >= 2);
        assert_scaled_materialization(&state, &value, 1e-10);
    }
}

#[test]
fn to_treetn_preserves_small_nonzero_chain_and_branched_tree_values() {
    small_tree_pivots(smooth_value);
    small_tree_pivots(|point| {
        let real = smooth_value(point);
        let imag = real * (0.5 + point.iter().sum::<usize>() as f64 / 8.0);
        Complex64::new(real, imag)
    });
}

#[test]
fn to_named_treetn_rejects_mismatched_names_and_site_dimensions() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    let constant =
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![1.0; batch.n_points()]) };
    optimize_default(&mut tci, constant, &TreeTciOptions::default()).unwrap();

    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let error = to_named_treetn(&tci, constant, None, &["a"], &sites).unwrap_err();
    assert!(error.to_string().contains("1 names, 2 site-index lists"));

    let wrong = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(3)]];
    let error = to_named_treetn(&tci, constant, None, &["a", "b"], &wrong).unwrap_err();
    assert!(error.to_string().contains("dimension product 3"));

    let overflow = vec![
        vec![DynIndex::new_dyn(2)],
        vec![DynIndex::new_dyn(usize::MAX), DynIndex::new_dyn(2)],
    ];
    let error = to_named_treetn(&tci, constant, None, &["a", "b"], &overflow).unwrap_err();
    assert!(error.to_string().contains("overflowed usize"));

    // Valid names and a site split into a dimension-one leg plus a
    // dimension-two leg; the site indices are used as given.
    let (first, unit, second) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(1),
        DynIndex::new_dyn(2),
    );
    let split = vec![vec![first.clone()], vec![unit.clone(), second.clone()]];
    let tree = to_named_treetn(&tci, constant, None, &["a", "b"], &split).unwrap();
    assert_eq!(tree.site_space(&"a").unwrap().len(), 1);
    assert!(tree.site_space(&"b").unwrap().contains(&unit));
    let value = tree
        .evaluate_point(&[first, unit, second], &[1, 0, 1])
        .unwrap();
    assert_scalar_close(value.real(), 1.0, 1.0, 1e-12);
}

// --- F3 step 3d: the materialization solve in a caller-owned context ---

mod materialization_context {
    use super::*;
    use std::time::Duration;
    use tenferro_cpu::CpuBackend;
    use tensor4all_tensorbackend::{
        with_default_backend, CpuExecutionContext, CpuExecutionContextError,
    };

    /// A 4-site chain whose nonzero, nonsingular pivot data really reaches the
    /// solve (`site_tensor_with_parent`), so the fixtures below exercise the
    /// context-taking route rather than the zero-pivot early return.
    fn nonzero_state() -> TreeTCI2<f64> {
        let mut tci = TreeTCI2::<f64>::new(vec![3, 2, 2, 3], four_site_graph()).unwrap();
        tci.add_global_pivots(&[vec![0, 0, 0, 0]]).unwrap();
        tci.flush_pivot_errors();
        tci
    }

    fn four_site_graph() -> TreeTciGraph {
        TreeTciGraph::new(
            4,
            &[
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(1, 2),
                TreeTciEdge::new(2, 3),
            ],
        )
        .unwrap()
    }

    fn smooth(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let mut total = 1.0;
            for site in 0..4 {
                let value = batch.get(site, point).unwrap() as f64;
                total += value * 0.5 + (site as f64) * value * value;
            }
            values.push(total);
        }
        Ok(values)
    }

    fn one_worker_context() -> CpuExecutionContext {
        CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("1T CPU backend"))
    }

    fn dense_values(tn: &TreeTN<IdxTensor, usize>) -> Vec<f64> {
        tn.to_dense().unwrap().to_vec::<f64>().unwrap()
    }

    #[test]
    fn to_treetn_chunked_with_a_context_ignores_the_process_global_backend() {
        // Warm the process-global eager runtime first: publishing a physical
        // tensor adopts into it, and its *cold* initialization takes the
        // global context's backend lock, which this test holds. Warm
        // publication is a cached handle and takes no such lock.
        let warm_state = nonzero_state();
        let warm_context = one_worker_context();
        let _ = materialize_in_context(
            &warm_state,
            smooth,
            Some(1),
            EVALUATION_CHUNK_POINTS,
            &warm_context,
        )
        .expect("warm-up materialization");

        let legacy = materialize_compat(&warm_state, smooth, Some(1), EVALUATION_CHUNK_POINTS)
            .expect("legacy materialization");

        // The scoped route must complete *while* the process-global backend is
        // held by this scope: a materialization that resolved its solve
        // through the global session would block here, which is the dispatch
        // evidence. The run happens on a worker thread so a wrong route fails
        // this test instead of hanging the suite; the panic releases the lock.
        let scoped = with_default_backend(|_global| {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let state = nonzero_state();
                let context = one_worker_context();
                let result = materialize_in_context(
                    &state,
                    smooth,
                    Some(1),
                    EVALUATION_CHUNK_POINTS,
                    &context,
                );
                let _ = done_tx.send(());
                result
            });
            match done_rx.recv_timeout(std::time::Duration::from_secs(120)) {
                Ok(()) => worker.join().expect("worker thread").expect("scoped materialization"),
                Err(_) => panic!(
                    "the scoped materialization did not finish while the process-global backend was held"
                ),
            }
        });

        let scoped_values = dense_values(&scoped);
        let legacy_values = dense_values(&legacy);
        assert_eq!(scoped_values.len(), legacy_values.len());
        for (index, (left, right)) in scoped_values.iter().zip(legacy_values.iter()).enumerate() {
            assert!(
                (left - right).abs() <= 1e-12 * right.abs().max(1.0),
                "entry {index}: {left} != {right}"
            );
        }
    }

    #[test]
    /// The negative control for the dispatch test above: under the same
    /// held-global condition, the legacy route really does block on that
    /// resource and then completes once the lock is released, so "the scoped
    /// route completed" is discriminating evidence. Ignored by default because
    /// it waits out a bounded timeout.
    #[ignore = "negative control; waits out a bounded timeout"]
    fn negative_control_legacy_route_blocks_while_the_global_backend_is_held() {
        let warm_state = nonzero_state();
        let warm_context = one_worker_context();
        let _ = materialize_in_context(
            &warm_state,
            smooth,
            Some(1),
            EVALUATION_CHUNK_POINTS,
            &warm_context,
        )
        .expect("warm-up materialization");

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = with_default_backend(|_global| {
            let worker = std::thread::spawn(move || {
                let state = nonzero_state();
                let _ = started_tx.send(());
                let result = materialize_compat(&state, smooth, Some(1), EVALUATION_CHUNK_POINTS);
                let _ = done_tx.send(());
                result
            });
            started_rx
                .recv_timeout(Duration::from_secs(30))
                .expect("the worker starts while the global backend is held");
            match done_rx.recv_timeout(Duration::from_secs(8)) {
                Ok(()) => panic!("the legacy route completed while the global backend was held"),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("the worker died before reporting: it neither blocked nor completed")
                }
            }
            worker
        });

        // Released now: the blocked legacy route must have been waiting for this
        // resource, not failing for another reason.
        let materialized = worker.join().expect("the worker thread lives");
        let materialized =
            materialized.expect("the legacy route completes once the lock is released");
        assert!(materialized.to_dense().unwrap().maxabs().unwrap() > 0.0);
    }

    #[test]
    fn a_zero_pivot_matrix_takes_the_early_return_without_a_solve() {
        // The zero-pivot early return sits above the solve, so the context route emits a
        // zero site tensor without a factorization.
        let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        state.add_global_pivots(&[vec![0, 0]]).unwrap();
        let zero =
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0; batch.n_points()]) };
        let context = one_worker_context();
        let tn = crate::to_treetn_in(&state, zero, Some(0), &context)
            .expect("a zero pivot matrix takes the early return");
        assert_eq!(tn.to_dense().unwrap().maxabs().unwrap(), 0.0);
    }

    #[test]
    fn a_worker_cannot_enter_the_context_route_even_for_zero_pivots() {
        // The context-taking entry enters one session for the whole materialization, so
        // the refusal happens up front instead of at the first nonzero solve: a worker is
        // refused typed even when the zero-pivot shortcut would have avoided the solve.
        let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
        state.add_global_pivots(&[vec![0, 0]]).unwrap();
        let zero =
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0; batch.n_points()]) };
        let context = one_worker_context();
        let pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = pool
            .install(|| crate::to_treetn_in(&state, zero, Some(0), &context))
            .expect_err("a worker may not enter the caller's concrete session");
        assert!(
            format!("{error}").contains("explicit concrete session"),
            "the refusal must stay visible: {error}"
        );
    }

    #[test]
    fn a_nonzero_solve_from_a_rayon_worker_is_refused_typed() {
        // The counterpart of the previous test: once a nonzero solve is
        // reached, the concrete session entry is entered and a worker is
        // refused rather than silently rerouted.
        let state = nonzero_state();
        let context = one_worker_context();
        let pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = pool
            .install(|| {
                materialize_in_context(
                    &state,
                    smooth,
                    Some(1),
                    EVALUATION_CHUNK_POINTS,
                    &context,
                )
            })
            .expect_err("a worker may not enter the caller's concrete session");

        // The refusal must survive as a typed entry error rather than as text:
        // walk the source chain to the concrete-session error.
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut found = false;
        while let Some(candidate) = source {
            if let Some(refusal) = candidate.downcast_ref::<CpuExecutionContextError>() {
                found = matches!(refusal, CpuExecutionContextError::SessionEntry { .. });
                break;
            }
            source = candidate.source();
        }
        assert!(
            found,
            "the entry refusal must stay in the source chain: {error}"
        );
        assert!(
            format!("{error}").contains("explicit concrete session"),
            "the refusal must stay visible: {error}"
        );
    }
}
