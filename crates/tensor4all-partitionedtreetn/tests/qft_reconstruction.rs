//! Subset QFT integration for orthogonal-target reconstruction.
//!
//! These tests bind a real `tensor4all_quanticstransform` Fourier operator to an
//! ordered subset of site indices and check the documented bit-significance and
//! bit-reversal conventions against a dense oracle.

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
    TreeTN,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{
    apply_linear_operator_to_indices, ApplyOptions, IndexMapping, LinearOperator,
    RestructureOptions, SiteIndexNetwork,
};

const TOL: f64 = 1e-8;

fn reconstruct_exact(target: &ReconstructionTarget) -> ReconstructedTreeTN {
    reconstruct(
        target,
        &0,
        ReconstructionTolerance {
            rtol: 1e-12,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            max_regions: 256,
            ..Default::default()
        },
    )
    .expect("reconstruction failed")
}

fn single_term_partition(target: &ReconstructionTarget) -> PartitionedTreeTN {
    reconstruct_exact(target)
        .into_partition()
        .expect("single-term regions expected")
}

/// Build a chain MPS over `sites`, with `values` stored most-significant site
/// first: `values[x]`, `x = sum_j x_j 2^(R-1-j)` and site `j` carries `x_j`.
fn mps(sites: &[DynIndex], values: &[Complex64]) -> TreeTN<IdxTensor, usize> {
    let r = sites.len();
    assert_eq!(values.len(), 1usize << r);
    let bonds: Vec<DynIndex> = (1..r).map(|i| DynIndex::new_dyn(1usize << i)).collect();
    let mut tensors = Vec::with_capacity(r);
    for (i, site) in sites.iter().enumerate() {
        let left_dim = 1usize << i;
        let has_right = i + 1 < r;
        let right_dim = if has_right { 1usize << (i + 1) } else { 1 };
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site.clone());
        if has_right {
            indices.push(bonds[i].clone());
        }
        // Payloads are column-major over `indices`.
        let mut data = vec![Complex64::new(0.0, 0.0); left_dim * 2 * right_dim];
        for left in 0..left_dim {
            for bit in 0..2 {
                if has_right {
                    let right = left * 2 + bit;
                    let offset = if i > 0 {
                        left + left_dim * (bit + 2 * right)
                    } else {
                        bit + 2 * right
                    };
                    data[offset] = Complex64::new(1.0, 0.0);
                } else {
                    let offset = if i > 0 { left + left_dim * bit } else { bit };
                    data[offset] = values[left * 2 + bit];
                }
            }
        }
        tensors.push(IdxTensor::from_dense(indices, data).expect("tensor"));
    }
    TreeTN::from_tensors(tensors, (0..r).collect()).expect("chain")
}

fn target_of(tree: TreeTN<IdxTensor, usize>) -> ReconstructionTarget {
    ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree).expect("subdomain"))
            .expect("partition"),
    )
    .expect("target")
}

/// Regroup `tree`'s sites onto the given nodes, chained in the supplied order.
///
/// Every site must appear exactly once. Regrouping is exact, so the dense value
/// of the returned chain equals the dense value of `tree`.
fn regroup(
    tree: &TreeTN<IdxTensor, usize>,
    groups: &[(usize, Vec<DynIndex>)],
) -> TreeTN<IdxTensor, usize> {
    let mut target: SiteIndexNetwork<usize, DynIndex> = SiteIndexNetwork::new();
    for (node, sites) in groups {
        target
            .add_node(*node, sites.iter().cloned().collect::<HashSet<_>>())
            .expect("target node");
    }
    for pair in groups.windows(2) {
        target
            .add_edge(&pair[0].0, &pair[1].0)
            .expect("target edge");
    }
    tree.restructure_to(&target, &RestructureOptions::default())
        .expect("regroup")
}

fn dense_of(partition: &PartitionedTreeTN) -> (Vec<DynIndex>, Vec<Complex64>) {
    let dense = partition
        .to_treetn()
        .expect("treetn")
        .to_dense()
        .expect("dense");
    (
        dense.indices().to_vec(),
        dense.to_vec::<Complex64>().expect("vector"),
    )
}

fn max_error(left: &[Complex64], right: &[Complex64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max)
}

/// Dense oracle for a subset DFT with the documented conventions: selected bits
/// are ordered most significant first, and the frequency bit `t` lands on
/// selected position `t` (bit-reversed output placement).
fn dft_oracle(
    indices: &[DynIndex],
    sites: &[DynIndex],
    selected: &[DynIndex],
    values: &[Complex64],
) -> Vec<Complex64> {
    let total = indices.len();
    assert_eq!(total, sites.len());
    let k_bits = selected.len();
    let n = 1usize << k_bits;
    let scale = 1.0 / (n as f64).sqrt();
    let site_of = |index: &DynIndex| {
        sites
            .iter()
            .position(|candidate| candidate == index)
            .unwrap_or_else(|| panic!("index {index:?} absent from dense result"))
    };
    // `values` is stored most-significant site first; dense storage is
    // column-major, so dense position `p` is the p-th index and varies fastest.
    let site_bit = |x: usize, index: &DynIndex| {
        let j = site_of(index);
        (x >> (total - 1 - j)) & 1
    };

    let mut expected = vec![Complex64::new(0.0, 0.0); 1usize << total];
    for (x, value) in values.iter().enumerate() {
        let mut selected_input = 0usize;
        for site in selected {
            selected_input = (selected_input << 1) | site_bit(x, site);
        }
        for k in 0..n {
            let angle = -2.0 * std::f64::consts::PI * (k * selected_input) as f64 / n as f64;
            let mut output_storage = 0usize;
            for (position, index) in indices.iter().enumerate() {
                let bit = match selected.iter().position(|site| site == index) {
                    Some(t) => (k >> t) & 1,
                    None => site_bit(x, index),
                };
                output_storage |= bit << position;
            }
            expected[output_storage] += value * Complex64::from_polar(scale, angle);
        }
    }
    expected
}

/// Reorder a most-significant-site-first input vector into the dense index order
/// of `indices`, so the result can be compared index-aligned.
fn values_in_index_order(
    indices: &[DynIndex],
    sites: &[DynIndex],
    values: &[Complex64],
) -> Vec<Complex64> {
    let total = indices.len();
    let mut out = vec![Complex64::new(0.0, 0.0); 1usize << total];
    for (storage, value) in out.iter_mut().enumerate() {
        let mut source = 0usize;
        for (position, index) in indices.iter().enumerate() {
            let site = sites
                .iter()
                .position(|candidate| candidate == index)
                .unwrap_or_else(|| panic!("unexpected index {index:?}"));
            let bit = (storage >> position) & 1;
            source |= bit << (total - 1 - site);
        }
        *value = values[source];
    }
    out
}

/// A one-site operator `M` written as a one-node MPO. `data` is column-major
/// over `[input, output]`, matching the crate's payload convention.
fn one_site_matrix(site: &DynIndex, data: [f64; 4]) -> LinearOperator<IdxTensor, usize> {
    let internal_input = DynIndex::new_dyn(2);
    let internal_output = DynIndex::new_dyn(2);
    let mpo = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![internal_input.clone(), internal_output.clone()],
            data.to_vec(),
        )
        .expect("mpo tensor")],
        vec![0usize],
    )
    .expect("mpo");
    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        0usize,
        IndexMapping {
            true_index: site.clone(),
            internal_index: internal_input,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        0usize,
        IndexMapping {
            true_index: site.clone(),
            internal_index: internal_output,
        },
    );
    LinearOperator::new(mpo, input_mapping, output_mapping)
}

/// Split `tree` into two disjoint patches that fix `site` to 0 and 1.
fn partitioned_target(tree: &TreeTN<IdxTensor, usize>, site: &DynIndex) -> ReconstructionTarget {
    let patch = |value| {
        SubDomainTreeTN::new(
            tree.clone(),
            Projector::from_pairs([(site.clone(), value)]).expect("projector"),
        )
        .expect("patch")
    };
    ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![patch(0), patch(1)]).expect("partition"),
    )
    .expect("target")
}

/// Apply `operator` to the whole `tree` for an independent dense reference.
fn applied_dense(
    operator: &LinearOperator<IdxTensor, usize>,
    tree: &TreeTN<IdxTensor, usize>,
    selection: &[DynIndex],
) -> (Vec<DynIndex>, Vec<Complex64>) {
    let mut nodes = operator.mpo().node_names();
    nodes.sort();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (node, index) in nodes.iter().zip(selection) {
        inputs.push((
            operator
                .get_input_mapping(node)
                .expect("input mapping")
                .true_index
                .clone(),
            index.clone(),
        ));
        outputs.push((
            operator
                .get_output_mapping(node)
                .expect("output mapping")
                .true_index
                .clone(),
            index.clone(),
        ));
    }
    let result =
        apply_linear_operator_to_indices(operator, tree, &inputs, &outputs, ApplyOptions::naive())
            .expect("reference apply");
    let dense = result.to_dense().expect("dense");
    (
        dense.indices().to_vec(),
        dense.to_vec::<Complex64>().expect("vector"),
    )
}

fn norm_of(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// Reorder a column-major dense vector from one index order to another.
fn reorder(values: &[Complex64], from: &[DynIndex], to: &[DynIndex]) -> Vec<Complex64> {
    let n = to.len();
    assert_eq!(from.len(), n);
    let mut out = vec![Complex64::new(0.0, 0.0); 1usize << n];
    for (storage, slot) in out.iter_mut().enumerate() {
        let mut source = 0usize;
        for (position, index) in to.iter().enumerate() {
            let q = from
                .iter()
                .position(|candidate| candidate == index)
                .unwrap_or_else(|| panic!("index {index:?} absent from reference"));
            source |= ((storage >> position) & 1) << q;
        }
        *slot = values[source];
    }
    out
}

#[test]
fn qft_subset_matches_dense_dft_convention() {
    for r in [2usize, 3] {
        let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
        let values: Vec<Complex64> = (0..1usize << r)
            .map(|m| Complex64::new(m as f64 + 1.0, 0.5 * m as f64))
            .collect();
        let preimage = target_of(mps(&sites, &values));

        let operator =
            quantics_fourier_operator(r, FourierOptions::default()).expect("fourier operator");
        let target = ReconstructionTarget::from_subset_operator(
            &preimage,
            &0,
            &operator,
            &sites,
            &SubsetOperatorOptions { unitary: true },
        )
        .expect("subset transform");

        let (indices, dense) = dense_of(&single_term_partition(&target));
        let expected = dft_oracle(&indices, &sites, &sites, &values);
        let error = max_error(&dense, &expected);
        assert!(error < TOL, "r = {r}: max error {error:e}");
    }
}

#[test]
fn qft_subset_supports_noncontiguous_selection_with_spectators() {
    let r = 4;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new(m as f64 - 1.0, 0.25 * m as f64))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = target_of(tree.clone());

    // Transform sites 0 and 2 only; sites 1 and 3 are spectators.
    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");
    let selected = [sites[0].clone(), sites[2].clone()];
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &selected,
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("subset transform");

    let partition = single_term_partition(&target);
    let (indices, dense) = dense_of(&partition);
    let expected = dft_oracle(&indices, &sites, &selected, &values);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");

    // Spectators keep identity, dimension, and node assignment.
    let transformed = partition.to_treetn().expect("treetn");
    let mut transformed_nodes = transformed.node_names();
    let mut original_nodes = tree.node_names();
    transformed_nodes.sort();
    original_nodes.sort();
    assert_eq!(transformed_nodes, original_nodes);
    for node in tree.node_names() {
        assert_eq!(transformed.site_space(&node), tree.site_space(&node));
    }
}

#[test]
fn qft_forward_then_inverse_restores_the_target() {
    let r = 3;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new((m as f64 * 0.7).sin() + 1.0, (m as f64).cos()))
        .collect();
    let preimage = target_of(mps(&sites, &values));

    let forward = quantics_fourier_operator(r, FourierOptions::forward()).expect("forward");
    let inverse = quantics_fourier_operator(r, FourierOptions::inverse()).expect("inverse");
    // A second Fourier operator reads its input most significant bit first, so
    // the inverse takes the reversed operator-node-to-site selection to undo the
    // forward transform's bit-reversed output placement.
    let reverse: Vec<DynIndex> = sites.iter().rev().cloned().collect();
    let transformed = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &forward,
        &sites,
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("forward target");
    let restored = ReconstructionTarget::from_subset_operator(
        &transformed,
        &0,
        &inverse,
        &reverse,
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("inverse target");

    // The forward-then-inverse round trip restores the original tensor on the
    // original full indices; it does not return a bit-reversed tensor.
    let (indices, dense) = dense_of(&single_term_partition(&restored));
    let expected = values_in_index_order(&indices, &sites, &values);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");
}

#[test]
fn subset_operator_default_scales_by_the_frobenius_norm() {
    let site = DynIndex::new_dyn(2);
    let state = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0]).expect("state")],
        vec![0usize],
    )
    .expect("tree");
    let preimage = target_of(state);
    assert!((preimage.reference_scale() - 5.0).abs() < 1e-12);

    // A 2x2 identity has Frobenius norm sqrt(2), not amplification 1.
    let identity = one_site_matrix(&site, [1.0, 0.0, 0.0, 1.0]);
    let scaled = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &identity,
        std::slice::from_ref(&site),
        &SubsetOperatorOptions::default(),
    )
    .expect("default scale");
    assert!(
        (scaled.reference_scale() - 5.0 * 2.0_f64.sqrt()).abs() < 1e-12,
        "default scale {}",
        scaled.reference_scale()
    );

    let unitary = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &identity,
        &[site],
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("unitary scale");
    assert!((unitary.reference_scale() - 5.0).abs() < 1e-12);
}

#[test]
fn subset_operator_scale_propagates_through_successive_applications() {
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let preimage = target_of(mps(&sites, &values));
    let input_scale = preimage.reference_scale();
    assert!((input_scale - 30.0_f64.sqrt()).abs() < 1e-12);

    let scaling = one_site_matrix(&sites[0], [0.1, 0.0, 0.0, 0.1]);
    let first = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &scaling,
        &[sites[0].clone()],
        &SubsetOperatorOptions::default(),
    )
    .expect("first");
    let factor = 0.1 * 2.0_f64.sqrt();
    assert!((first.reference_scale() - factor * input_scale).abs() < 1e-12);

    let second = ReconstructionTarget::from_subset_operator(
        &first,
        &0,
        &scaling,
        &[sites[0].clone()],
        &SubsetOperatorOptions::default(),
    )
    .expect("second");
    assert!(
        (second.reference_scale() - factor * factor * input_scale).abs() < 1e-12,
        "propagated scale {}",
        second.reference_scale()
    );

    // A unitary application contributes exactly a factor of one.
    let identity = one_site_matrix(&sites[0], [1.0, 0.0, 0.0, 1.0]);
    let third = ReconstructionTarget::from_subset_operator(
        &second,
        &0,
        &identity,
        &[sites[0].clone()],
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("unitary");
    assert!((third.reference_scale() - second.reference_scale()).abs() < 1e-12);
}

#[test]
fn subset_operator_scale_is_not_measured_under_destructive_interference() {
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    // The two `s0` branches are exact negatives, so an operator that adds them
    // produces the zero function even though every image is nonzero.
    let values: Vec<Complex64> = (0..4)
        .map(|m| {
            let sign = if m < 2 { 1.0 } else { -1.0 };
            Complex64::new(sign * (m % 2 + 1) as f64, 0.0)
        })
        .collect();
    let tree = mps(&sites, &values);
    let preimage = partitioned_target(&tree, &sites[0]);
    let input_scale = preimage.reference_scale();
    assert!((input_scale - 10.0_f64.sqrt()).abs() < 1e-12);

    // M = [[1, 1], [0, 0]] with Frobenius norm sqrt(2).
    let operator = one_site_matrix(&sites[0], [1.0, 1.0, 0.0, 0.0]);
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone()],
        &SubsetOperatorOptions::default(),
    )
    .expect("subset transform");
    assert!(
        (target.reference_scale() - 2.0_f64.sqrt() * input_scale).abs() < 1e-12,
        "scale {}",
        target.reference_scale()
    );

    // The true output is the zero function: the scale is the operator-based
    // upper bound, not a measured output norm.
    let (_, applied) = applied_dense(&operator, &tree, &[sites[0].clone()]);
    assert!(
        norm_of(&applied) < 1e-12,
        "applied norm {}",
        norm_of(&applied)
    );

    // Separate-term retention: preparation stores the two images independently,
    // so reconstruction at zero tolerance keeps both. This observes the retained
    // representation only; it cannot see a temporary network built and discarded
    // during preparation.
    let output = reconstruct(
        &target,
        &0,
        ReconstructionTolerance {
            rtol: 0.0,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: None,
            ..Default::default()
        },
    )
    .expect("reconstruction");
    assert_eq!(output.report().term_count, 2);
    assert_eq!(output.report().region_count, 1);
    assert_eq!(output.report().reference_scale, target.reference_scale());
}

#[test]
fn qft_subset_rejects_invalid_selections() {
    let r = 2;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values = vec![Complex64::new(1.0, 0.0); 4];
    let preimage = target_of(mps(&sites, &values));
    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");

    // Wrong selection length.
    assert!(ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone()],
        &SubsetOperatorOptions::default(),
    )
    .is_err());

    // Repeated index.
    assert!(ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone(), sites[0].clone()],
        &SubsetOperatorOptions::default(),
    )
    .is_err());

    // Absent index.
    assert!(ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone(), DynIndex::new_dyn(2)],
        &SubsetOperatorOptions::default(),
    )
    .is_err());
}

#[test]
fn qft_subset_supports_two_selected_indices_on_one_node() {
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(2);
    let sites = [a.clone(), b.clone(), c.clone()];
    let values: Vec<Complex64> = (0..8)
        .map(|m| Complex64::new((m as f64 * 0.4).sin() + 1.0, 0.3 * m as f64))
        .collect();
    // One node owns bits `a` and `b`; the operator's two nodes are fused into it.
    let tree = regroup(
        &mps(&sites, &values),
        &[
            (0usize, vec![a.clone(), b.clone()]),
            (1usize, vec![c.clone()]),
        ],
    );
    let preimage = target_of(tree);

    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[a.clone(), b.clone()],
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("merged subset transform");

    let (indices, dense) = dense_of(&single_term_partition(&target));
    let expected = dft_oracle(&indices, &sites, &[a, b], &values);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");
}

#[test]
fn qft_subset_merges_selected_indices_that_share_a_spectator_node() {
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(2);
    let sites = [a.clone(), b.clone(), spectator.clone(), c.clone()];
    let values: Vec<Complex64> = (0..16)
        .map(|m| Complex64::new(m as f64 - 2.0, (m as f64 * 0.7).cos()))
        .collect();
    let tree = regroup(
        &mps(&sites, &values),
        &[
            (0usize, vec![a.clone(), b.clone(), spectator.clone()]),
            (1usize, vec![c.clone()]),
        ],
    );
    let preimage = target_of(tree);

    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[a.clone(), b.clone()],
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("merged subset transform");

    let partition = single_term_partition(&target);
    let (indices, dense) = dense_of(&partition);
    let expected = dft_oracle(&indices, &sites, &[a, b], &values);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");

    // The spectator keeps its identity, dimension, and node assignment.
    let transformed = partition.to_treetn().expect("treetn");
    assert_eq!(transformed.node_count(), 2);
    assert!(transformed
        .site_space(&0)
        .is_some_and(|space| space.contains(&spectator)));
    assert!(transformed
        .site_space(&1)
        .is_some_and(|space| space.contains(&c)));
}

#[test]
fn qft_subset_rejects_selected_indices_on_a_disconnected_operator_group() {
    // Node 0 owns `a` and `c`, which are the operator's first and third nodes;
    // the group is threaded through another owner's node, so it cannot be fused.
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(2);
    let d = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![a.clone(), c.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    )
    .expect("tensor");
    let right =
        IdxTensor::from_dense(vec![bond, b.clone(), d.clone()], vec![1.0; 8]).expect("tensor");
    let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).expect("tree");
    let preimage = target_of(tree);

    let operator = quantics_fourier_operator(4, FourierOptions::default()).expect("fourier");
    let error = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[a, b, c, d],
        &SubsetOperatorOptions::default(),
    )
    .expect_err("a disconnected operator group cannot be fused");
    assert!(
        error.to_string().contains("connected group"),
        "unexpected error: {error}"
    );
}

#[test]
fn qft_subset_multi_patch_matches_dense_reference() {
    let r = 3;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new((m as f64 * 0.3).cos(), (m as f64 * 0.9).sin()))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = partitioned_target(&tree, &sites[0]);

    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
    )
    .expect("subset transform");
    let (reference_indices, reference) = applied_dense(&operator, &tree, &sites);
    assert!(
        (target.reference_scale() / norm_of(&reference) - 1.0).abs() < 1e-10,
        "reference norm {} vs dense {}",
        target.reference_scale(),
        norm_of(&reference)
    );

    let (indices, dense) = dense_of(&single_term_partition(&target));
    let expected = reorder(&reference, &reference_indices, &indices);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");
}

// ---------------------------------------------------------------------------
// Level-coupled merge-refine schedule
// ---------------------------------------------------------------------------

/// The projector fixing `sites` to the most-significant-first coordinate `leaf`.
fn input_leaf_projector(sites: &[DynIndex], leaf: usize) -> Projector {
    let pairs = sites
        .iter()
        .enumerate()
        .map(|(position, site)| (site.clone(), (leaf >> (sites.len() - 1 - position)) & 1));
    Projector::from_pairs(pairs).expect("projector")
}

/// Build the `2^depth` dyadic input leaves of `sites` as an immutable target.
fn dyadic_input_leaves(
    tree: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
) -> ReconstructionTarget {
    let full = SubDomainTreeTN::from_treetn(tree.clone()).expect("subdomain");
    let leaves = (0..1usize << sites.len())
        .map(|leaf| input_leaf_projector(sites, leaf))
        .map(|projector| {
            full.project(&projector)
                .expect("project")
                .expect("a dyadic leaf of a generic state is nonzero")
        })
        .collect::<Vec<_>>();
    ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(leaves).expect("partition"),
    )
    .expect("target")
}

/// Bit `index` of a column-major dense `storage` position.
fn dense_bit(indices: &[DynIndex], storage: usize, index: &DynIndex) -> usize {
    let position = indices
        .iter()
        .position(|candidate| candidate == index)
        .unwrap_or_else(|| panic!("index {index:?} absent from the dense result"));
    (storage >> position) & 1
}

/// Zero every dense coordinate that `projector` does not select.
fn mask_by_projector(
    values: &[Complex64],
    indices: &[DynIndex],
    projector: &Projector,
) -> Vec<Complex64> {
    values
        .iter()
        .enumerate()
        .map(|(storage, value)| {
            let selected = projector
                .iter()
                .all(|(index, coordinate)| dense_bit(indices, storage, index) == *coordinate);
            if selected {
                *value
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect()
}

/// Zero every most-significant-site-first input coordinate that `projector` does
/// not select.
fn mask_input_leaves(
    values: &[Complex64],
    sites: &[DynIndex],
    projector: &Projector,
) -> Vec<Complex64> {
    values
        .iter()
        .enumerate()
        .map(|(x, value)| {
            let selected = projector.iter().all(|(index, coordinate)| {
                let position = sites
                    .iter()
                    .position(|candidate| candidate == index)
                    .unwrap_or_else(|| panic!("site {index:?} absent from the input"));
                ((x >> (sites.len() - 1 - position)) & 1) == *coordinate
            });
            if selected {
                *value
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect()
}

fn schedule_exact(
    preimage: &ReconstructionTarget,
    operator: &LinearOperator<IdxTensor, usize>,
    selection: &[DynIndex],
    output_depth: Option<usize>,
    max_work_items: usize,
) -> MergeRefineResult<usize> {
    schedule_merge_refine(
        preimage,
        &0,
        operator,
        selection,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-12,
            atol: 0.0,
        },
        &MergeRefineOptions {
            target_bond_dim: None,
            max_terms: 1 << 20,
            apply_options: None,
            output_depth,
            max_work_items,
        },
    )
    .expect("schedule")
}

#[test]
fn merge_refine_schedule_matches_every_intermediate_against_dense_oracles() {
    for r in [2usize, 3] {
        let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
        let values: Vec<Complex64> = (0..1usize << r)
            .map(|m| Complex64::new((m as f64 * 0.8).sin() + 1.0, (m as f64 * 0.5).cos()))
            .collect();
        let tree = mps(&sites, &values);
        let preimage = dyadic_input_leaves(&tree, &sites);
        let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");
        let leaves = 1usize << r;

        for depth in 0..=r {
            let result = schedule_exact(&preimage, &operator, &sites, Some(depth), 4 * leaves);
            let report = result.report();
            assert_eq!(report.level_count, depth);
            assert_eq!(report.applied_operator_count, leaves);
            assert_eq!(report.additions, leaves * depth);
            assert_eq!(report.peak_work_items, leaves);
            assert_eq!(report.work_items_per_level, vec![leaves; depth + 1]);
            assert_eq!(report.error_bound, 0.0);

            // Every retained work item is one intermediate `P_B F P_A w`.
            let mut canonical: Option<Vec<DynIndex>> = None;
            let mut items = 0usize;
            for (input, output, term) in result.items() {
                items += 1;
                let dense = term.data().to_dense().expect("dense");
                let term_indices = dense.indices().to_vec();
                let term_values = dense.to_vec::<Complex64>().expect("vector");
                let indices = canonical.get_or_insert_with(|| term_indices.clone());
                let aligned = reorder(&term_values, &term_indices, indices);

                let masked_input = mask_input_leaves(&values, &sites, input);
                let expected = dft_oracle(indices, &sites, &sites, &masked_input);
                let expected = mask_by_projector(&expected, indices, output);
                let error = max_error(&aligned, &expected);
                assert!(error < TOL, "r = {r} depth = {depth}: max error {error:e}");
            }
            // Only the executed final level is retained.
            assert_eq!(items, leaves);

            if depth == r {
                // The fully refined trajectory is the strict partition of the
                // normalized DFT, one region per output coordinate.
                assert_eq!(report.region_count, leaves);
                assert_eq!(report.term_count, leaves);
                let partition = result.into_partition().expect("fully refined partition");
                let (indices, dense) = dense_of(&partition);
                let expected = dft_oracle(&indices, &sites, &sites, &values);
                let error = max_error(&dense, &expected);
                assert!(error < TOL, "r = {r}: max error {error:e}");
            } else if depth > 0 {
                // A partially refined level keeps one term per input region, so
                // the regions are a superposition and not a strict partition.
                assert_eq!(report.term_count, leaves);
                assert_eq!(report.region_count, 1usize << depth);
                assert!(result.into_partition().is_err());
            }
        }
    }
}

#[test]
fn merge_refine_schedule_refines_one_selected_coordinate_of_several() {
    // Only the first two sites are selected; the third is a spectator.
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..8)
        .map(|m| Complex64::new(m as f64 - 2.0, 0.5 * m as f64))
        .collect();
    let tree = mps(&sites, &values);
    let selected = [sites[0].clone(), sites[1].clone()];
    let preimage = dyadic_input_leaves(&tree, &selected);
    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");

    let result = schedule_exact(&preimage, &operator, &selected, None, 32);
    let report = result.report();
    assert_eq!(report.applied_operator_count, 4);
    assert_eq!(report.additions, 8);
    assert_eq!(report.work_items_per_level, vec![4, 4, 4]);
    assert_eq!(report.region_count, 4);

    // Every item is `P_B F P_A w` on the selected pair, with the spectator kept.
    let mut canonical: Option<Vec<DynIndex>> = None;
    for (input, output, term) in result.items() {
        let dense = term.data().to_dense().expect("dense");
        let term_indices = dense.indices().to_vec();
        let term_values = dense.to_vec::<Complex64>().expect("vector");
        let indices = canonical.get_or_insert_with(|| term_indices.clone());
        let aligned = reorder(&term_values, &term_indices, indices);

        let masked_input = mask_input_leaves(&values, &sites, input);
        let expected = dft_oracle(indices, &sites, &selected, &masked_input);
        let expected = mask_by_projector(&expected, indices, output);
        let error = max_error(&aligned, &expected);
        assert!(error < TOL, "max error {error:e}");
    }

    // Spectators keep their identity, dimension, and node assignment.
    let partition = result.into_partition().expect("partition");
    let transformed = partition.to_treetn().expect("treetn");
    assert_eq!(transformed.node_count(), 3);
    for node in tree.node_names() {
        assert_eq!(transformed.site_space(&node), tree.site_space(&node));
    }
}

#[test]
fn merge_refine_schedule_rejects_invalid_geometry() {
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..4)
        .map(|m| Complex64::new(m as f64 + 1.0, 0.0))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");

    // A selected index of dimension three is not dyadic.
    let ternary = DynIndex::new_dyn(3);
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &[ternary],
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions::default(),
    )
    .expect_err("non-binary selection");
    assert!(error.to_string().contains("binary selected indices"));

    // The output depth cannot exceed the selection depth.
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions {
            output_depth: Some(3),
            max_work_items: 16,
            ..Default::default()
        },
    )
    .expect_err("output depth beyond the selection");
    assert!(error.to_string().contains("output_depth"));

    // The work limit must cover the input leaves.
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions {
            output_depth: None,
            max_work_items: 3,
            ..Default::default()
        },
    )
    .expect_err("work limit below the leaf count");
    assert!(error.to_string().contains("max_work_items"));

    // A zero soft rank goal is invalid.
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions {
            target_bond_dim: Some(0),
            ..Default::default()
        },
    )
    .expect_err("zero rank goal");
    assert!(error.to_string().contains("target_bond_dim"));

    // A preimage that misses one dyadic leaf is not a coverage contract.
    let full = SubDomainTreeTN::from_treetn(tree.clone()).expect("subdomain");
    let partial = full
        .project(&input_leaf_projector(&sites, 0))
        .expect("project")
        .expect("nonzero leaf");
    let incomplete = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![partial]).expect("partition"),
    )
    .expect("target");
    let error = schedule_merge_refine(
        &incomplete,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions::default(),
    )
    .expect_err("missing input leaf");
    assert!(error
        .to_string()
        .contains("cover every selected-coordinate"));

    // A patch that leaves a selected index free is not an input leaf.
    let unrestricted = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![full]).expect("partition"),
    )
    .expect("target");
    let error = schedule_merge_refine(
        &unrestricted,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions::default(),
    )
    .expect_err("unconstrained patch");
    assert!(error
        .to_string()
        .contains("must fix all selected indices to one coordinate"));
}

#[test]
fn merge_refine_schedule_rejects_leaf_spectator_mismatch() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let values: Vec<Complex64> = (0..4)
        .map(|m| Complex64::new(m as f64 + 1.0, 0.0))
        .collect();
    let tree = mps(&[selected.clone(), spectator.clone()], &values);
    let full = SubDomainTreeTN::from_treetn(tree).expect("subdomain");
    let leaf = |selected_value: usize, spectator_value: usize| {
        let projector = Projector::from_pairs([
            (selected.clone(), selected_value),
            (spectator.clone(), spectator_value),
        ])
        .expect("projector");
        full.project(&projector)
            .expect("project")
            .expect("nonzero leaf")
    };
    // The two input leaves disagree on the spectator coordinate.
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![leaf(0, 0), leaf(1, 1)]).expect("partition"),
    )
    .expect("target");
    let operator = one_site_matrix(&selected, [1.0, 0.0, 0.0, 1.0]);

    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &[selected],
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance::default(),
        &MergeRefineOptions::default(),
    )
    .expect_err("spectator constraints must agree across leaves");
    assert!(matches!(
        error,
        tensor4all_partitionedtreetn::PartitionedTreeTNError::ProjectorMismatch
    ));
}

#[test]
fn merge_refine_schedule_keeps_shared_spectator_constraints() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let values: Vec<Complex64> = (0..4)
        .map(|m| Complex64::new(1.0 + m as f64, 0.5 * m as f64))
        .collect();
    let tree = mps(&[selected.clone(), spectator.clone()], &values);
    let full = SubDomainTreeTN::from_treetn(tree).expect("subdomain");
    let leaf = |selected_value: usize| {
        let projector =
            Projector::from_pairs([(selected.clone(), selected_value), (spectator.clone(), 1)])
                .expect("projector");
        full.project(&projector)
            .expect("project")
            .expect("nonzero leaf")
    };
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![leaf(0), leaf(1)]).expect("partition"),
    )
    .expect("target");
    let operator = one_site_matrix(&selected, [1.0, 0.0, 0.0, 1.0]);

    let result = schedule_exact(&preimage, &operator, &[selected], None, 8);
    let report = result.report();
    assert_eq!(report.applied_operator_count, 2);
    assert_eq!(report.additions, 2);
    assert_eq!(report.region_count, 2);
    // Each region keeps the shared spectator constraint.
    let partition = result.into_partition().expect("partition");
    for patch in partition.values() {
        assert_eq!(patch.projector().get(&spectator), Some(1));
    }
    // The identity is exact on the retained spectator-constrained leaves:
    // `values` is most-significant-selected-first, so the spectator bit is the
    // least significant one and only `values[1]` and `values[3]` survive.
    let expected: f64 = values
        .iter()
        .skip(1)
        .step_by(2)
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!((partition.norm().expect("norm") - expected).abs() < 1e-12);
}

#[test]
fn merge_refine_schedule_handles_exact_cancellation() {
    // The two input leaves are exact negatives under a rank-deficient operator
    // that adds them, so every transformed image and every merged sum is zero.
    let site = DynIndex::new_dyn(2);
    let values = vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let tree = mps(std::slice::from_ref(&site), &values);
    let preimage = dyadic_input_leaves(&tree, std::slice::from_ref(&site));
    let operator = one_site_matrix(&site, [1.0, 1.0, 0.0, 0.0]);

    let result = schedule_exact(&preimage, &operator, &[site], None, 8);
    assert_eq!(result.report().applied_operator_count, 2);
    assert_eq!(result.report().additions, 2);
    for (_, _, term) in result.items() {
        assert!(term.norm().expect("norm") < 1e-12);
    }
    let partition = result.into_partition().expect("partition");
    assert!(partition.norm().expect("norm") < 1e-12);
}

/// Run the fully refined schedule with a soft rank goal and a relative tolerance.
fn schedule_rank_limited(
    preimage: &ReconstructionTarget,
    operator: &LinearOperator<IdxTensor, usize>,
    selection: &[DynIndex],
    rtol: f64,
    target_bond_dim: usize,
) -> MergeRefineResult<usize> {
    let leaves = 1usize << selection.len();
    schedule_merge_refine(
        preimage,
        &0,
        operator,
        selection,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance { rtol, atol: 0.0 },
        &MergeRefineOptions {
            output_depth: None,
            max_work_items: 4 * leaves,
            target_bond_dim: Some(target_bond_dim),
            ..Default::default()
        },
    )
    .expect("schedule")
}

/// Dense value of a schedule result with all of its retained terms summed, in one
/// canonical index order. Adaptive results keep overlapping terms in one region,
/// so they are not always a strict partition.
fn dense_of_regions(result: &MergeRefineResult<usize>) -> (Vec<DynIndex>, Vec<Complex64>) {
    let mut canonical: Option<Vec<DynIndex>> = None;
    let mut total: Option<Vec<Complex64>> = None;
    for (_, terms) in result.regions() {
        for term in terms {
            let dense = term.data().to_dense().expect("dense");
            let term_indices = dense.indices().to_vec();
            let values = dense.to_vec::<Complex64>().expect("vector");
            match &canonical {
                None => {
                    canonical = Some(term_indices);
                    total = Some(values);
                }
                Some(indices) => {
                    let aligned = reorder(&values, &term_indices, indices);
                    let accumulator = total.as_mut().expect("accumulator");
                    for (target, value) in accumulator.iter_mut().zip(aligned) {
                        *target += value;
                    }
                }
            }
        }
    }
    (
        canonical.expect("at least one retained term"),
        total.expect("accumulator"),
    )
}

#[test]
fn merge_refine_schedule_truncates_within_the_global_allowance() {
    let r = 4usize;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new(1.0 + (m as f64).sin(), (m as f64).cos()))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");

    let limited = schedule_rank_limited(&preimage, &operator, &sites, 0.3, 2);
    let report = limited.report().clone();

    // Truncation happened, respected the soft goal, and measured a residual that
    // stayed inside the global allowance.
    assert!(report.compression_attempts > 0, "{report:?}");
    assert!(report.compressions > 0, "{report:?}");
    assert!(report.max_bond_dim <= 2, "{:?}", report.max_bond_dim);
    assert!(
        report.max_transient_bond_dim > report.max_bond_dim,
        "transient {} retained {}",
        report.max_transient_bond_dim,
        report.max_bond_dim
    );
    assert!(report.error_bound > 0.0);
    assert!(report.error_bound <= report.absolute_tolerance);

    // The reported bound covers the deviation from the dense normalized DFT.
    let (indices, dense) = dense_of_regions(&limited);
    let expected = dft_oracle(&indices, &sites, &sites, &values);
    let oracle_error = max_error(&dense, &expected);
    assert!(
        oracle_error <= report.error_bound + 1e-12,
        "oracle residual {oracle_error:e} exceeds the bound {:e}",
        report.error_bound
    );
}

#[test]
fn merge_refine_schedule_zero_tolerance_keeps_the_exact_trajectory() {
    let r = 3usize;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new(m as f64 - 2.0, 0.5 * m as f64))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");

    let exact = schedule_exact(&preimage, &operator, &sites, None, 32);
    let limited = schedule_rank_limited(&preimage, &operator, &sites, 0.0, 1);
    let report = limited.report().clone();

    // A zero allowance leaves no budget, so every probe is rejected for free.
    assert!(report.compression_attempts > 0);
    assert_eq!(report.compressions, 0);
    assert_eq!(report.error_bound, 0.0);
    assert_eq!(report.absolute_tolerance, 0.0);
    // No truncation is affordable, so the merge policy keeps the operands as
    // separate terms; their ranks are at most the summed ranks of the exact run.
    assert!(report.max_bond_dim <= exact.report().max_bond_dim);

    let (indices, dense) = dense_of_regions(&limited);
    let (exact_indices, exact_dense) = dense_of(&exact.into_partition().expect("partition"));
    let exact_dense = reorder(&exact_dense, &exact_indices, &indices);
    assert!(max_error(&dense, &exact_dense) < 1e-12);
}

#[test]
fn merge_refine_schedule_keeps_exact_cancellation_with_a_rank_goal() {
    let site = DynIndex::new_dyn(2);
    let values = vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let tree = mps(std::slice::from_ref(&site), &values);
    let preimage = dyadic_input_leaves(&tree, std::slice::from_ref(&site));
    let operator = one_site_matrix(&site, [1.0, 1.0, 0.0, 0.0]);

    let result = schedule_rank_limited(&preimage, &operator, &[site], 0.5, 1);
    let report = result.report();
    // Exact cancellation costs no accuracy budget: the retained superposition is
    // the zero function even though its individual terms are not zero.
    assert_eq!(report.error_bound, 0.0);
    assert!(report.term_count > 0);
    let (_, dense) = dense_of_regions(&result);
    assert!(norm_of(&dense) < 1e-12, "residual {}", norm_of(&dense));
}

#[test]
fn merge_refine_schedule_stops_refinement_once_the_rank_goal_is_met() {
    let r = 3usize;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| {
            if m == 3 {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");

    let result = schedule_rank_limited(&preimage, &operator, &sites, 1e-3, 1);
    let report = result.report().clone();

    // The refined plane wave reaches rank one after one level, so the schedule
    // stops there instead of forcing the preset `2^r` output blocks.
    assert_eq!(report.refined_regions, 1, "{report:?}");
    assert!(report.stopped_regions >= 2, "{report:?}");
    assert_eq!(report.region_count, 2);
    assert!(report.region_count < 1usize << r);
    assert_eq!(report.max_bond_dim, 1);

    // The retained superposition is still the exact normalized DFT.
    let (indices, dense) = dense_of_regions(&result);
    let expected = dft_oracle(&indices, &sites, &sites, &values);
    let residual = max_error(&dense, &expected);
    assert!(
        residual <= report.error_bound + 1e-12,
        "residual {residual:e} exceeds the bound {:e}",
        report.error_bound
    );
}

#[test]
fn merge_refine_schedule_rejects_a_term_budget_it_cannot_keep() {
    let r = 2usize;
    let sites: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..1usize << r)
        .map(|m| Complex64::new(m as f64 + 1.0, 0.0))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = quantics_fourier_operator(r, FourierOptions::default()).expect("fourier");

    // Four input leaves cannot be represented within a two-term budget.
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &sites,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-3,
            atol: 0.0,
        },
        &MergeRefineOptions {
            output_depth: None,
            max_work_items: 16,
            target_bond_dim: Some(1),
            max_terms: 2,
            apply_options: None,
        },
    )
    .expect_err("two terms cannot hold four leaves");
    assert!(matches!(
        error,
        PartitionedTreeTNError::ResourceLimit {
            limit: "max_terms",
            ..
        }
    ));
    assert!(error.to_string().contains("max_terms"));
}

/// Schedule with truncating application options and a relative tolerance.
fn schedule_with_application(
    preimage: &ReconstructionTarget,
    operator: &LinearOperator<IdxTensor, usize>,
    selection: &[DynIndex],
    rtol: f64,
    apply_options: ApplyOptions,
) -> Result<MergeRefineResult<usize>, PartitionedTreeTNError> {
    let leaves = 1usize << selection.len();
    schedule_merge_refine(
        preimage,
        &0,
        operator,
        selection,
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance { rtol, atol: 0.0 },
        &MergeRefineOptions {
            output_depth: None,
            max_work_items: 4 * leaves,
            apply_options: Some(apply_options),
            ..Default::default()
        },
    )
}

/// `I ⊗ I + scale · X ⊗ X` as a two-node MPO of bond dimension two.
///
/// Its image of a dyadic input leaf is a rank-two superposition
/// (`|00> + scale · |11>`), so truncating the application to bond dimension one
/// has a measurable effect whose size the `scale` controls, unlike a Fourier
/// operator whose leaf images are already low rank.
fn identity_plus_scaled_xx(sites: &[DynIndex], scale: f64) -> LinearOperator<IdxTensor, usize> {
    let (in0, out0) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (in1, out1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let bond = DynIndex::new_dyn(2);
    // Column-major over `[in, out, bond]`: bond 0 is the identity, bond 1 the X.
    let left = IdxTensor::from_dense(
        vec![in0.clone(), out0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    )
    .expect("left tensor");
    // Column-major over `[bond, in, out]`: bond 0 is the identity, bond 1 the
    // scaled X.
    let right = IdxTensor::from_dense(
        vec![bond, in1.clone(), out1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, scale, scale, 0.0],
    )
    .expect("right tensor");
    let mpo = TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).expect("mpo");
    let mut input = HashMap::new();
    input.insert(
        0usize,
        IndexMapping {
            true_index: sites[0].clone(),
            internal_index: in0,
        },
    );
    input.insert(
        1usize,
        IndexMapping {
            true_index: sites[1].clone(),
            internal_index: in1,
        },
    );
    let mut output = HashMap::new();
    output.insert(
        0usize,
        IndexMapping {
            true_index: sites[0].clone(),
            internal_index: out0,
        },
    );
    output.insert(
        1usize,
        IndexMapping {
            true_index: sites[1].clone(),
            internal_index: out1,
        },
    );
    LinearOperator::new(mpo, input, output)
}

#[test]
fn merge_refine_schedule_charges_a_truncating_application_to_the_bound() {
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..4)
        .map(|m| Complex64::new(1.0 + (m as f64).sin(), (m as f64).cos()))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    // A small second term keeps the truncation error affordable at `rtol = 0.5`.
    let operator = identity_plus_scaled_xx(&sites, 0.1);

    // Truncating each application to bond dimension one removes a real part of
    // every leaf image, which the schedule must charge before any compression.
    let result = schedule_with_application(
        &preimage,
        &operator,
        &sites,
        0.5,
        ApplyOptions::zipup().with_max_bond_dim(1),
    )
    .expect("approximate application");
    let report = result.report().clone();
    assert!(report.error_bound > 1e-6, "bound {:e}", report.error_bound);
    assert!(report.error_bound <= report.absolute_tolerance);

    // The bound covers the deviation from the exact trajectory.
    let exact = schedule_exact(&preimage, &operator, &sites, None, 16);
    let (indices, dense) = dense_of_regions(&result);
    let (exact_indices, exact_dense) = dense_of(&exact.into_partition().expect("partition"));
    let exact_dense = reorder(&exact_dense, &exact_indices, &indices);
    let deviation = max_error(&dense, &exact_dense);
    assert!(
        deviation <= report.error_bound + 1e-12,
        "deviation {deviation:e} exceeds the bound {:e}",
        report.error_bound
    );
}

#[test]
fn merge_refine_schedule_rejects_an_unaffordable_application_error() {
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let values: Vec<Complex64> = (0..4)
        .map(|m| Complex64::new(1.0 + m as f64, 0.5 * m as f64))
        .collect();
    let tree = mps(&sites, &values);
    let preimage = dyadic_input_leaves(&tree, &sites);
    let operator = identity_plus_scaled_xx(&sites, 0.1);

    // A bond-dimension-one application cannot fit a near-exact tolerance.
    let error = schedule_with_application(
        &preimage,
        &operator,
        &sites,
        1e-12,
        ApplyOptions::zipup().with_max_bond_dim(1),
    )
    .expect_err("the application error exceeds the allowance");
    assert!(
        error.to_string().contains("application error"),
        "unexpected error: {error}"
    );
}
