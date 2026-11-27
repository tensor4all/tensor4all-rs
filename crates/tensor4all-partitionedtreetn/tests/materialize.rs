use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

fn assert_c64_slice_close(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).norm();
        assert!(
            diff <= tol,
            "entry {idx} mismatch: actual {a:?}, expected {e:?}, |diff| = {diff:e} > {tol:e}"
        );
    }
}

fn assert_f64_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "entry {idx} mismatch: actual {a}, expected {e}, |diff| = {diff:e} > {tol:e}"
        );
    }
}

/// Build a 3-node chain `0 - 1 - 2` with site indices:
/// - node 0: `s0` (dim 2)
/// - node 1: `s1a` (dim 2), `s1b` (dim 3)
/// - node 2: `s2` (dim 2)
///
/// Nodes are inserted in non-sorted order `[2, 0, 1]` to verify that `to_dense`
/// uses sorted node-name order `[s0, s1a, s1b, s2]` rather than insertion order.
fn build_complex_three_node_chain(
    s0: &DynIndex,
    s1a: &DynIndex,
    s1b: &DynIndex,
    s2: &DynIndex,
    scale: Complex64,
    phase: Complex64,
) -> TreeTN<IdxTensor, usize> {
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);

    // Node 0: [s0(2), b01(2)]
    let t0_data: Vec<Complex64> = (0..4)
        .map(|k| scale * Complex64::new((k + 1) as f64, 0.25 * (k as f64)))
        .collect();
    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], t0_data).unwrap();

    // Node 1: [b01(2), s1a(2), s1b(3), b12(2)]
    let t1_data: Vec<Complex64> = (0..24)
        .map(|k| phase * Complex64::new(0.5 + 0.1 * (k as f64), -0.2 * ((k % 5) as f64)))
        .collect();
    let t1 = IdxTensor::from_dense(
        vec![b01.clone(), s1a.clone(), s1b.clone(), b12.clone()],
        t1_data,
    )
    .unwrap();

    // Node 2: [b12(2), s2(2)]
    let t2_data: Vec<Complex64> = (0..4)
        .map(|k| Complex64::new(1.0 + 0.3 * (k as f64), 0.4 * ((k + 1) as f64)))
        .collect();
    let t2 = IdxTensor::from_dense(vec![b12.clone(), s2.clone()], t2_data).unwrap();

    // Insert in non-sorted node-name order [2, 0, 1].
    TreeTN::from_tensors(vec![t2, t0, t1], vec![2usize, 0usize, 1usize]).unwrap()
}

#[test]
fn dense_partition_scatter_preserves_axis_order() {
    let s0 = DynIndex::new_dyn(2);
    let s1a = DynIndex::new_dyn(2);
    let s1b = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(2);

    let expected_axes = vec![s0.clone(), s1a.clone(), s1b.clone(), s2.clone()];

    // Patch A fixes non-leading index `s1b = 0` (free: s0, s1a, s2).
    let tree_a = build_complex_three_node_chain(
        &s0,
        &s1a,
        &s1b,
        &s2,
        Complex64::new(1.0, 0.5),
        Complex64::new(0.8, -0.3),
    );
    let proj_a = Projector::from_pairs([(s1b.clone(), 0)]).unwrap();
    let patch_a = SubDomainTreeTN::new(tree_a.clone(), proj_a).unwrap();

    // Patch B fixes two non-leading indices `s1b = 2` and `s2 = 1` (free: s0, s1a).
    let tree_b = build_complex_three_node_chain(
        &s0,
        &s1a,
        &s1b,
        &s2,
        Complex64::new(-0.7, 1.2),
        Complex64::new(0.4, 0.9),
    );
    let proj_b = Projector::from_pairs([(s1b.clone(), 2), (s2.clone(), 1)]).unwrap();
    let patch_b = SubDomainTreeTN::new(tree_b.clone(), proj_b).unwrap();

    // Leave `s1b = 1` and `(s1b = 2, s2 = 0)` uncovered: they must be implicit zeros.
    let partition =
        PartitionedTreeTN::from_subdomains(vec![patch_b.clone(), patch_a.clone()]).unwrap();

    // Build expected column-major entries independently from the raw trees.
    let raw_a = tree_a.to_dense().unwrap();
    let raw_b = tree_b.to_dense().unwrap();
    assert_eq!(raw_a.external_indices(), expected_axes);
    assert_eq!(raw_b.external_indices(), expected_axes);
    let raw_a_vec = raw_a.to_vec::<Complex64>().unwrap();
    let raw_b_vec = raw_b.to_vec::<Complex64>().unwrap();

    let total_len = 2 * 2 * 3 * 2;
    let mut expected_column_major_entries = vec![Complex64::new(0.0, 0.0); total_len];
    let mut expected_patch_entries = Vec::with_capacity(2 * 2 * 2);

    // Column-major linear index for [s0(2), s1a(2), s1b(3), s2(2)]:
    // idx = i0 + 2 * (i1a + 2 * (i1b + 3 * i2))
    for i2 in 0..2 {
        for i1b in 0..3 {
            for i1a in 0..2 {
                for i0 in 0..2 {
                    let linear = i0 + 2 * (i1a + 2 * (i1b + 3 * i2));
                    if i1b == 0 {
                        expected_column_major_entries[linear] = raw_a_vec[linear];
                        expected_patch_entries.push(raw_a_vec[linear]);
                    } else if i1b == 2 && i2 == 1 {
                        expected_column_major_entries[linear] = raw_b_vec[linear];
                    }
                }
            }
        }
    }

    let dense = partition.to_dense::<Complex64>().unwrap();
    let axes = dense.external_indices();
    let entries = dense.to_vec::<Complex64>().unwrap();

    let one_patch = patch_a.to_dense::<Complex64>().unwrap();
    let one_patch_dense = one_patch.to_vec::<Complex64>().unwrap();

    assert_eq!(axes, expected_axes);
    assert_c64_slice_close(&entries, &expected_column_major_entries, 1e-12);
    assert_eq!(
        one_patch.external_indices(),
        vec![s0.clone(), s1a.clone(), s2.clone()]
    );
    assert_c64_slice_close(&one_patch_dense, &expected_patch_entries, 1e-12);

    // Also verify agreement with `to_treetn()?.to_dense()` and sliced `data().to_dense()`.
    let via_treetn = partition.to_treetn().unwrap().to_dense().unwrap();
    assert_eq!(via_treetn.external_indices(), expected_axes);
    assert_c64_slice_close(&entries, &via_treetn.to_vec::<Complex64>().unwrap(), 1e-12);

    let sliced_a = patch_a
        .data()
        .to_dense()
        .unwrap()
        .select_indices(std::slice::from_ref(&s1b), &[0])
        .unwrap();
    assert_eq!(sliced_a.external_indices(), one_patch.external_indices());
    assert_c64_slice_close(
        &one_patch_dense,
        &sliced_a.to_vec::<Complex64>().unwrap(),
        1e-12,
    );
}

#[test]
fn subdomain_to_dense_all_projected_returns_rank_zero() {
    // Multi-node tree where every site index is projected.
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![0.5_f64, -1.0, 2.0, 0.25, -0.5, 1.5],
    )
    .unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap();
    let projector = Projector::from_pairs([(s0.clone(), 1), (s1.clone(), 2)]).unwrap();
    let patch = SubDomainTreeTN::new(tree, projector).unwrap();

    let dense = patch.to_dense::<f64>().unwrap();
    assert!(dense.external_indices().is_empty());
    assert!(dense.dims().is_empty());

    let sliced = patch
        .data()
        .to_dense()
        .unwrap()
        .select_indices(&[s0, s1], &[1, 2])
        .unwrap();
    assert_f64_slice_close(
        &dense.to_vec::<f64>().unwrap(),
        &sliced.to_vec::<f64>().unwrap(),
        1e-12,
    );

    // Single-node tree where its only site index is projected.
    let s_single = DynIndex::new_dyn(3);
    let single_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![s_single.clone()], vec![10.0_f64, 20.0, 30.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let single_patch =
        SubDomainTreeTN::new(single_tree, Projector::from_pairs([(s_single, 1)]).unwrap()).unwrap();
    let single_dense = single_patch.to_dense::<f64>().unwrap();
    assert!(single_dense.external_indices().is_empty());
    assert_f64_slice_close(&single_dense.to_vec::<f64>().unwrap(), &[20.0], 1e-12);
}

#[test]
fn empty_partition_to_dense_returns_zero_of_requested_dtype() {
    let empty = PartitionedTreeTN::<usize>::new();

    let dense_f64 = empty.to_dense::<f64>().unwrap();
    assert!(dense_f64.external_indices().is_empty());
    assert!(dense_f64.is_f64());
    assert_eq!(dense_f64.to_vec::<f64>().unwrap(), vec![0.0]);

    let dense_c64 = empty.to_dense::<Complex64>().unwrap();
    assert!(dense_c64.external_indices().is_empty());
    assert!(dense_c64.is_c64());
    assert_eq!(
        dense_c64.to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(0.0, 0.0)]
    );
}

#[test]
fn to_dense_dtype_mismatch_returns_typed_error() {
    let s = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let patch = SubDomainTreeTN::new(tree, Projector::from_pairs([(s, 0)]).unwrap()).unwrap();
    let partition = PartitionedTreeTN::from_subdomain(patch.clone()).unwrap();

    assert!(matches!(
        patch.to_dense::<Complex64>(),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        partition.to_dense::<Complex64>(),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
}

#[test]
fn partition_to_dense_avoids_summed_bond_blowup() {
    // 16 disjoint patches over 4 binary projected indices, each with bond dim 4
    // between two free-site nodes. `to_treetn()` would sum bonds to 16 * 4 = 64,
    // while `to_dense()` contracts each patch at bond dim 4 into its 2x2 subdomain.
    let p0 = DynIndex::new_dyn(2);
    let p1 = DynIndex::new_dyn(2);
    let p2 = DynIndex::new_dyn(2);
    let p3 = DynIndex::new_dyn(2);
    let f0 = DynIndex::new_dyn(2);
    let f1 = DynIndex::new_dyn(2);

    let chi = 4usize;
    let mut patches = Vec::with_capacity(16);
    let mut expected = vec![0.0_f64; 1 << 6];

    for patch_idx in 0..16usize {
        let v0 = patch_idx & 1;
        let v1 = (patch_idx >> 1) & 1;
        let v2 = (patch_idx >> 2) & 1;
        let v3 = (patch_idx >> 3) & 1;

        let b0 = DynIndex::new_dyn(chi);
        let b1 = DynIndex::new_dyn(chi);

        // Node 0: [p0, p1, b0]
        let t0 = IdxTensor::from_dense(
            vec![p0.clone(), p1.clone(), b0.clone()],
            (0..4 * chi)
                .map(|k| ((patch_idx + k + 1) as f64) * 0.05)
                .collect(),
        )
        .unwrap();
        // Node 1: [b0, p2, p3, f0, b1]
        let t1 = IdxTensor::from_dense(
            vec![b0.clone(), p2.clone(), p3.clone(), f0.clone(), b1.clone()],
            (0..chi * 8 * chi)
                .map(|k| (((patch_idx * 7 + k) % 13) as f64 - 6.0) * 0.1)
                .collect(),
        )
        .unwrap();
        // Node 2: [b1, f1]
        let t2 = IdxTensor::from_dense(
            vec![b1.clone(), f1.clone()],
            (0..chi * 2)
                .map(|k| 1.0 + 0.1 * ((patch_idx + k) as f64))
                .collect(),
        )
        .unwrap();

        let tree = TreeTN::from_tensors(vec![t0, t1, t2], vec![0usize, 1usize, 2usize]).unwrap();
        let proj = Projector::from_pairs([
            (p0.clone(), v0),
            (p1.clone(), v1),
            (p2.clone(), v2),
            (p3.clone(), v3),
        ])
        .unwrap();
        let patch = SubDomainTreeTN::new(tree, proj).unwrap();

        // Expected 2x2 block for this patch from its own `to_dense`.
        let block = patch.to_dense::<f64>().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(block.len(), 4);
        for i_f1 in 0..2 {
            for i_f0 in 0..2 {
                let linear = v0 + 2 * (v1 + 2 * (v2 + 2 * (v3 + 2 * (i_f0 + 2 * i_f1))));
                expected[linear] = block[i_f0 + 2 * i_f1];
            }
        }
        patches.push(patch);
    }

    let partition = PartitionedTreeTN::from_subdomains(patches).unwrap();
    let summed = partition.to_treetn().unwrap();
    let summed_max_bond = summed.link_dims().into_iter().max().unwrap_or(1);
    assert_eq!(summed_max_bond, 16 * chi);

    let dense = partition.to_dense::<f64>().unwrap();
    assert_eq!(dense.external_indices(), vec![p0, p1, p2, p3, f0, f1]);
    assert_f64_slice_close(&dense.to_vec::<f64>().unwrap(), &expected, 1e-11);
}
