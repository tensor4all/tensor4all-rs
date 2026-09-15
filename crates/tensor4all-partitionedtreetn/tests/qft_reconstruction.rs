//! Subset QFT integration for orthogonal-target reconstruction.
//!
//! These tests bind a real `tensor4all_quanticstransform` Fourier operator to an
//! ordered subset of site indices and check the documented bit-significance and
//! bit-reversal conventions against a dense oracle.

use std::collections::HashMap;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{reconstruction::*, PartitionedTreeTN, SubDomainTreeTN, TreeTN};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{IndexMapping, LinearOperator};

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

/// A one-site `scale * I` operator written as a one-node MPO.
fn scaled_identity(scale: f64, site: &DynIndex) -> LinearOperator<IdxTensor, usize> {
    let internal_input = DynIndex::new_dyn(2);
    let internal_output = DynIndex::new_dyn(2);
    let mpo = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![internal_input.clone(), internal_output.clone()],
            vec![scale, 0.0, 0.0, scale],
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
        let target = ReconstructionTarget::from_subset_operator(&preimage, &0, &operator, &sites)
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
    let target = ReconstructionTarget::from_subset_operator(&preimage, &0, &operator, &selected)
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
    let transformed = ReconstructionTarget::from_subset_operator(&preimage, &0, &forward, &sites)
        .expect("forward target");
    let restored = ReconstructionTarget::from_subset_operator(&transformed, &0, &inverse, &reverse)
        .expect("inverse target");

    // The forward-then-inverse round trip restores the original tensor on the
    // original full indices; it does not return a bit-reversed tensor.
    let (indices, dense) = dense_of(&single_term_partition(&restored));
    let expected = values_in_index_order(&indices, &sites, &values);
    let error = max_error(&dense, &expected);
    assert!(error < TOL, "max error {error:e}");
}

#[test]
fn subset_operator_measures_the_transformed_norm() {
    // A scaled identity is not unitary: inheriting the preimage norm would leave
    // the target's global allowance at ~5e-6 while its actual norm is ~5e-9, so
    // reconstruction would discard the whole nonzero target.
    let site = DynIndex::new_dyn(2);
    let state = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0]).expect("state")],
        vec![0usize],
    )
    .expect("tree");
    let preimage = target_of(state);
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &scaled_identity(1e-9, &site),
        &[site],
    )
    .expect("subset transform");
    assert!(
        (target.reference_norm() / 5e-9 - 1.0).abs() < 1e-12,
        "reference norm {}",
        target.reference_norm()
    );

    let output = reconstruct(
        &target,
        &0,
        ReconstructionTolerance {
            rtol: 1e-6,
            atol: 0.0,
        },
        &Default::default(),
    )
    .expect("reconstruction");
    assert_eq!(output.report().region_count, 1);
    assert_eq!(output.report().term_count, 1);
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
    )
    .is_err());

    // Repeated index.
    assert!(ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone(), sites[0].clone()],
    )
    .is_err());

    // Absent index.
    assert!(ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        &[sites[0].clone(), DynIndex::new_dyn(2)],
    )
    .is_err());
}

#[test]
fn qft_subset_rejects_two_selected_indices_on_one_node() {
    // Node 0 owns two site indices; node 1 owns the third.
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![a.clone(), b.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .expect("tensor");
    let right = IdxTensor::from_dense(vec![bond, c], vec![1.0, 0.0, 0.0, 1.0]).expect("tensor");
    let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).expect("tree");
    let preimage = target_of(tree);

    let operator = quantics_fourier_operator(2, FourierOptions::default()).expect("fourier");
    let error = ReconstructionTarget::from_subset_operator(&preimage, &0, &operator, &[a, b])
        .expect_err("indices sharing a node are not supported");
    assert!(error.to_string().contains("distinct tree nodes"));
}
