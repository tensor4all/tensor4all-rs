//! Tests for `PartitionedTreeTN::add_on_lanes` and `add_truncated_on_lanes`
//! (#1495, #1498): bit-for-bit parity with serial `add` / `add_truncated`,
//! coverage intersection and clipping (#1494), `item_sites` grouping, and
//! typed error reporting.

use hataori_tenferro::TenferroLanes;
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::{TreeTN, TruncationOptions};

fn make_lanes(count: usize) -> TenferroLanes {
    let engines = (0..count).map(|_| CpuEngine::serial()).collect();
    TenferroLanes::new(engines).expect("one serial CpuEngine per lane")
}

/// Build a 3-node complex chain `(0 -- 1 -- 2)` with site indices `k` (node 0),
/// `s` (node 1), and `x` (node 2), and bond dimension `bond_dim`.
fn make_complex_chain(
    k: &DynIndex,
    s: &DynIndex,
    x: &DynIndex,
    bond_dim: usize,
    seed: f64,
) -> TreeTN<IdxTensor, usize> {
    let b01 = DynIndex::new_dyn(bond_dim);
    let b12 = DynIndex::new_dyn(bond_dim);

    let mut d0 = Vec::with_capacity(k.dim * bond_dim);
    for b in 0..bond_dim {
        for ik in 0..k.dim {
            let re = seed + 0.13 * (ik as f64) + 0.37 * (b as f64);
            let im = -0.21 * seed + 0.19 * (ik as f64) - 0.11 * (b as f64);
            d0.push(Complex64::new(re, im));
        }
    }

    let mut d1 = Vec::with_capacity(bond_dim * s.dim * bond_dim);
    for br in 0..bond_dim {
        for is in 0..s.dim {
            for bl in 0..bond_dim {
                let re = 0.5 + 0.07 * seed + 0.29 * (is as f64) + 0.17 * (bl as f64)
                    - 0.23 * (br as f64);
                let im = 0.11 * (is as f64) - 0.05 * (bl as f64) + 0.13 * (br as f64);
                d1.push(Complex64::new(re, im));
            }
        }
    }

    let mut d2 = Vec::with_capacity(bond_dim * x.dim);
    for ix in 0..x.dim {
        for b in 0..bond_dim {
            let re = 0.3 * seed - 0.41 * (ix as f64) + 0.31 * (b as f64);
            let im = 0.27 * (ix as f64) + 0.09 * (b as f64) - 0.15 * seed;
            d2.push(Complex64::new(re, im));
        }
    }

    let t0 = IdxTensor::from_dense(vec![k.clone(), b01.clone()], d0).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01, s.clone(), b12.clone()], d1).unwrap();
    let t2 = IdxTensor::from_dense(vec![b12, x.clone()], d2).unwrap();
    TreeTN::from_tensors(vec![t0, t1, t2], vec![0usize, 1, 2]).unwrap()
}

/// Assert two partitions are bit-for-bit identical in keys, coverage, link dims,
/// and every complex tensor entry.
fn assert_partitions_bit_identical(
    actual: &PartitionedTreeTN<usize>,
    expected: &PartitionedTreeTN<usize>,
) {
    assert_eq!(actual.coverage(), expected.coverage(), "coverage mismatch");
    assert_eq!(actual.len(), expected.len(), "patch count mismatch");

    let expected_keys: std::collections::HashSet<Projector> =
        expected.projectors().cloned().collect();
    let actual_keys: std::collections::HashSet<Projector> = actual.projectors().cloned().collect();
    assert_eq!(actual_keys, expected_keys, "projector keys mismatch");

    for key in &expected_keys {
        let a_patch = actual.get(key).expect("missing actual patch");
        let e_patch = expected.get(key).expect("missing expected patch");
        assert_eq!(
            a_patch.data().link_dims(),
            e_patch.data().link_dims(),
            "link_dims mismatch for {key:?}"
        );
        for node in 0..3usize {
            let a_idx = a_patch.data().node_index(&node).unwrap();
            let e_idx = e_patch.data().node_index(&node).unwrap();
            let a_vals = a_patch
                .data()
                .tensor(a_idx)
                .unwrap()
                .to_vec::<Complex64>()
                .unwrap();
            let e_vals = e_patch
                .data()
                .tensor(e_idx)
                .unwrap()
                .to_vec::<Complex64>()
                .unwrap();
            assert_eq!(a_vals.len(), e_vals.len(), "tensor len mismatch");
            for (i, (av, ev)) in a_vals.iter().zip(e_vals.iter()).enumerate() {
                assert_eq!(
                    av.re.to_bits(),
                    ev.re.to_bits(),
                    "real bit mismatch at patch {key:?} node {node} entry {i}"
                );
                assert_eq!(
                    av.im.to_bits(),
                    ev.im.to_bits(),
                    "imag bit mismatch at patch {key:?} node {node} entry {i}"
                );
            }
        }
    }
}

/// Assert two partitions match in keys, coverage, link dims, and dense values
/// within relative tolerance `rtol` (used across the eager vs concrete-session
/// ~1 ULP `S * V^H` rounding boundary per D4).
fn assert_partitions_close_to_serial(
    actual: &PartitionedTreeTN<usize>,
    expected: &PartitionedTreeTN<usize>,
    rtol: f64,
) {
    assert_eq!(actual.coverage(), expected.coverage(), "coverage mismatch");
    assert_eq!(actual.len(), expected.len(), "patch count mismatch");

    let expected_keys: std::collections::HashSet<Projector> =
        expected.projectors().cloned().collect();
    let actual_keys: std::collections::HashSet<Projector> = actual.projectors().cloned().collect();
    assert_eq!(actual_keys, expected_keys, "projector keys mismatch");

    for key in &expected_keys {
        let a_patch = actual.get(key).expect("missing actual patch");
        let e_patch = expected.get(key).expect("missing expected patch");
        assert_eq!(
            a_patch.data().link_dims(),
            e_patch.data().link_dims(),
            "link_dims mismatch for {key:?}"
        );
        let a_dense = a_patch
            .data()
            .contract_to_tensor()
            .unwrap()
            .to_vec::<Complex64>()
            .unwrap();
        let e_dense = e_patch
            .data()
            .contract_to_tensor()
            .unwrap()
            .to_vec::<Complex64>()
            .unwrap();
        assert_eq!(a_dense.len(), e_dense.len());
        let diff_norm: f64 = a_dense
            .iter()
            .zip(e_dense.iter())
            .map(|(a, e)| (a - e).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = e_dense.iter().map(|e| e.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            diff_norm <= rtol * ref_norm.max(1.0),
            "patch {key:?} relative difference {diff_norm:e} / {ref_norm:e} exceeds {rtol:e}"
        );
    }
}

#[test]
fn lane_add_matches_serial() {
    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);

    // Patches of different cost: bond_dim 1 (cheap) vs bond_dim 3 (heavier),
    // plus one patch present only on the left and one only on the right.
    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
    let p10 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 0)]).unwrap();
    let p11 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 1)]).unwrap();

    let left = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 1.0), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 2.0), p01.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 3.0), p10.clone()).unwrap(),
    ])
    .unwrap();

    let right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 4.0), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 5.0), p01.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 6.0), p11.clone()).unwrap(),
    ])
    .unwrap();

    let serial_exact = left.add(&right).unwrap();
    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);
    let serial_trunc = left.add_truncated(&right, &0, trunc_opts).unwrap();

    let groupings = vec![
        None,
        Some(vec![]),
        Some(vec![k.clone()]),
        Some(vec![s.clone()]),
        Some(vec![k.clone(), s.clone()]),
        Some(vec![s.clone(), k.clone()]),
    ];

    let mut lanes_ref = make_lanes(1);
    let lane1_trunc_ref = left
        .add_truncated_on_lanes(&right, &0, trunc_opts, None, &mut lanes_ref)
        .unwrap();
    assert_partitions_close_to_serial(&lane1_trunc_ref, &serial_trunc, 1e-12);

    for lane_count in [1usize, 2, 4] {
        let mut lanes = make_lanes(lane_count);
        assert_eq!(lanes.lane_count(), lane_count);
        for item_sites in &groupings {
            let exact = left
                .add_on_lanes(&right, item_sites.clone(), &mut lanes)
                .unwrap();
            assert_partitions_bit_identical(&exact, &serial_exact);

            let trunc = left
                .add_truncated_on_lanes(&right, &0, trunc_opts, item_sites.clone(), &mut lanes)
                .unwrap();
            assert_partitions_bit_identical(&trunc, &lane1_trunc_ref);
            assert_partitions_close_to_serial(&trunc, &serial_trunc, 1e-12);
        }
    }
}

#[test]
fn lane_add_preserves_coverage_intersection_and_clipping() {
    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);

    let pk0 = Projector::from_pairs([(k.clone(), 0)]).unwrap();
    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
    let p10 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 0)]).unwrap();

    // Left covers {k=0} with one coarse patch; right covers {k=0,s=0} and {k=1,s=0}.
    // Intersection is {k=0,s=0}, so left's coarse patch must be clipped.
    let left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_complex_chain(&k, &s, &x, 2, 1.5),
        pk0.clone(),
    )
    .unwrap()])
    .unwrap()
    .with_coverage(vec![pk0])
    .unwrap();

    let right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 2.5), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 3.5), p10.clone()).unwrap(),
    ])
    .unwrap()
    .with_coverage(vec![p00.clone(), p10])
    .unwrap();

    let serial_exact = left.add(&right).unwrap();
    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);
    let serial_trunc = left.add_truncated(&right, &0, trunc_opts).unwrap();

    assert_eq!(serial_exact.coverage(), Some(&[p00.clone()][..]));
    assert!(serial_exact.get(&p01).is_none());

    let mut lanes_ref = make_lanes(1);
    let lane1_trunc_ref = left
        .add_truncated_on_lanes(
            &right,
            &0,
            trunc_opts,
            Some(vec![k.clone()]),
            &mut lanes_ref,
        )
        .unwrap();
    assert_partitions_close_to_serial(&lane1_trunc_ref, &serial_trunc, 1e-12);

    for lane_count in [1usize, 2] {
        let mut lanes = make_lanes(lane_count);
        let exact = left
            .add_on_lanes(&right, Some(vec![k.clone()]), &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&exact, &serial_exact);

        let trunc = left
            .add_truncated_on_lanes(&right, &0, trunc_opts, Some(vec![k.clone()]), &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&trunc, &lane1_trunc_ref);
        assert_partitions_close_to_serial(&trunc, &serial_trunc, 1e-12);
    }
}

#[test]
fn lane_add_rejects_invalid_item_sites() {
    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);
    let foreign = DynIndex::new_dyn(2);

    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
    let pk1 = Projector::from_pairs([(k.clone(), 1)]).unwrap();

    // `k` is projected on every patch, `s` is only projected on {k=0} patches,
    // and `foreign` is not in the layout at all.
    let left = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 1.0), p00).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 2.0), p01).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 3.0), pk1).unwrap(),
    ])
    .unwrap();

    let mut lanes = make_lanes(2);
    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);

    // Foreign site outside layout.
    let err = left
        .add_on_lanes(&left, Some(vec![foreign.clone()]), &mut lanes)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &foreign
    ));

    let err = left
        .add_truncated_on_lanes(
            &left,
            &0,
            trunc_opts,
            Some(vec![foreign.clone()]),
            &mut lanes,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &foreign
    ));

    // Site not projected by all patches (`s` is absent from `pk1`).
    let err = left
        .add_on_lanes(&left, Some(vec![s.clone()]), &mut lanes)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &s
    ));

    let err = left
        .add_truncated_on_lanes(&left, &0, trunc_opts, Some(vec![s.clone()]), &mut lanes)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &s
    ));
}

#[test]
fn lane_add_rejects_invalid_item_sites_on_empty_clipped_intersection_and_dim_mismatch() {
    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);
    let foreign = DynIndex::new_dyn(2);

    let pk0 = Projector::from_pairs([(k.clone(), 0)]).unwrap();
    let pk1 = Projector::from_pairs([(k.clone(), 1)]).unwrap();

    // Disjoint coverages: intersection is empty, so clipping empties all_keys.
    let left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_complex_chain(&k, &s, &x, 1, 1.0),
        pk0.clone(),
    )
    .unwrap()])
    .unwrap()
    .with_coverage(vec![pk0])
    .unwrap();

    let right = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_complex_chain(&k, &s, &x, 1, 2.0),
        pk1.clone(),
    )
    .unwrap()])
    .unwrap()
    .with_coverage(vec![pk1])
    .unwrap();

    let mut lanes = make_lanes(2);
    let err = left
        .add_on_lanes(&right, Some(vec![foreign.clone()]), &mut lanes)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &foreign
    ));

    // Template site `s` is in the layout, but not projected by the patches;
    // must still be rejected even when coverage clipping empties the result.
    let err = left
        .add_on_lanes(&right, Some(vec![s.clone()]), &mut lanes)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::InvalidItemSites { ref index } if index == &s
    ));

    // Same DynIndex ID/tags/plev as `k`, but mismatched dimension (larger and smaller).
    let mut k_larger = k.clone();
    k_larger.dim = 4;
    let err = left
        .add_on_lanes(&left, Some(vec![k_larger]), &mut lanes)
        .unwrap_err();
    assert!(matches!(err, PartitionedTreeTNError::SiteIndexMismatch));

    let mut k_smaller = k.clone();
    k_smaller.dim = 1;
    let err = left
        .add_on_lanes(&left, Some(vec![k_smaller]), &mut lanes)
        .unwrap_err();
    assert!(matches!(err, PartitionedTreeTNError::SiteIndexMismatch));
}

#[test]
fn lane_add_handles_empty_or_cheap_subproblem_item() {
    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);

    // Sub-problem k=0 has two heavier patches present in both operands;
    // sub-problem k=1 has one cheap rank-1 patch present on only one side.
    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
    let p10 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 0)]).unwrap();

    let left = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 1.0), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 2.0), p01.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 1, 3.0), p10).unwrap(),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 4.0), p00).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 5.0), p01).unwrap(),
    ])
    .unwrap();

    let serial_exact = left.add(&right).unwrap();
    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);
    let serial_trunc = left.add_truncated(&right, &0, trunc_opts).unwrap();

    let mut lanes = make_lanes(2);
    let lane_trunc_ref = left
        .add_truncated_on_lanes(&right, &0, trunc_opts, None, &mut lanes)
        .unwrap();
    assert_partitions_close_to_serial(&lane_trunc_ref, &serial_trunc, 1e-12);

    for item_sites in [None, Some(vec![k])] {
        let exact = left
            .add_on_lanes(&right, item_sites.clone(), &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&exact, &serial_exact);

        let trunc = left
            .add_truncated_on_lanes(&right, &0, trunc_opts, item_sites, &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&trunc, &lane_trunc_ref);
        assert_partitions_close_to_serial(&trunc, &serial_trunc, 1e-12);
    }
}

#[test]
fn lane_add_and_truncated_add_never_enter_global_backend() {
    use std::sync::mpsc;
    use std::time::Duration;
    use tensor4all_tensorbackend::with_default_backend;

    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);

    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
    let p10 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 0)]).unwrap();

    let left = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 1.0), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 2.0), p01.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 3.0), p10.clone()).unwrap(),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 4.0), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 3, 5.0), p01).unwrap(),
    ])
    .unwrap();

    // Case (ii): 1-node partition with permuted site order between operands.
    let t1_left = IdxTensor::from_dense(
        vec![k.clone(), s.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(3.0, 0.25),
            Complex64::new(4.0, -0.25),
        ],
    )
    .unwrap();
    let t1_right = IdxTensor::from_dense(
        vec![s.clone(), k.clone()],
        vec![
            Complex64::new(0.5, 1.0),
            Complex64::new(-0.5, 2.0),
            Complex64::new(0.25, 3.0),
            Complex64::new(-0.25, 4.0),
        ],
    )
    .unwrap();
    let one_node_left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        TreeTN::from_tensors(vec![t1_left], vec![0usize]).unwrap(),
        p00.clone(),
    )
    .unwrap()])
    .unwrap();
    let one_node_right = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        TreeTN::from_tensors(vec![t1_right], vec![0usize]).unwrap(),
        p00.clone(),
    )
    .unwrap()])
    .unwrap();

    // Case (iii): overlapping patches + coverage clipping.
    let pk0 = Projector::from_pairs([(k.clone(), 0)]).unwrap();
    let clip_left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_complex_chain(&k, &s, &x, 2, 1.5),
        pk0.clone(),
    )
    .unwrap()])
    .unwrap()
    .with_coverage(vec![pk0])
    .unwrap();
    let clip_right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 2.5), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_complex_chain(&k, &s, &x, 2, 3.5), p10.clone()).unwrap(),
    ])
    .unwrap()
    .with_coverage(vec![p00.clone(), p10])
    .unwrap();

    // Case (iv): 2-node tree with a site-free endpoint node (node 0 has only bond).
    let make_site_free = |seed: f64| {
        let bond = DynIndex::new_dyn(3);
        let t0 = IdxTensor::from_dense(
            vec![bond.clone()],
            vec![
                Complex64::new(seed, 0.1),
                Complex64::new(seed + 1.0, -0.2),
                Complex64::new(seed + 2.0, 0.3),
            ],
        )
        .unwrap();
        let t1 = IdxTensor::from_dense(
            vec![bond, k.clone()],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.25),
                Complex64::new(-0.5, 0.5),
                Complex64::new(0.25, -0.5),
                Complex64::new(2.0, 0.1),
                Complex64::new(-1.0, 0.2),
            ],
        )
        .unwrap();
        TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap()
    };
    let sf_left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_site_free(1.0),
        Projector::from_pairs([(k.clone(), 0)]).unwrap(),
    )
    .unwrap()])
    .unwrap();
    let sf_right = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_site_free(2.0),
        Projector::from_pairs([(k.clone(), 0)]).unwrap(),
    )
    .unwrap()])
    .unwrap();

    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);

    // Holding the process-global backend makes any route that enters it block.
    with_default_backend(|_global| {
        let (done_tx, done_rx) = mpsc::channel();
        let k_worker = k.clone();
        let worker = std::thread::spawn(move || {
            let mut lanes = make_lanes(2);
            // (i) Multi-node disjoint.
            let exact = left
                .add_on_lanes(&right, Some(vec![k_worker.clone()]), &mut lanes)
                .unwrap();
            let trunc = left
                .add_truncated_on_lanes(
                    &right,
                    &0,
                    trunc_opts,
                    Some(vec![k_worker.clone()]),
                    &mut lanes,
                )
                .unwrap();
            // (ii) 1-node partition with permuted site order.
            let one_exact = one_node_left
                .add_on_lanes(&one_node_right, Some(vec![k_worker.clone()]), &mut lanes)
                .unwrap();
            let one_trunc = one_node_left
                .add_truncated_on_lanes(
                    &one_node_right,
                    &0,
                    trunc_opts,
                    Some(vec![k_worker.clone()]),
                    &mut lanes,
                )
                .unwrap();
            // (iii) Overlapping patches + coverage clipping.
            let clip_exact = clip_left
                .add_on_lanes(&clip_right, Some(vec![k_worker.clone()]), &mut lanes)
                .unwrap();
            let clip_trunc = clip_left
                .add_truncated_on_lanes(
                    &clip_right,
                    &0,
                    trunc_opts,
                    Some(vec![k_worker.clone()]),
                    &mut lanes,
                )
                .unwrap();
            // (iv) Site-free endpoint node.
            let sf_trunc = sf_left
                .add_truncated_on_lanes(&sf_right, &1, trunc_opts, Some(vec![k_worker]), &mut lanes)
                .unwrap();
            done_tx
                .send((
                    exact.len(),
                    trunc.len(),
                    one_exact.len(),
                    one_trunc.len(),
                    clip_exact.len(),
                    clip_trunc.len(),
                    sf_trunc.len(),
                ))
                .unwrap();
        });
        let (
            exact_len,
            trunc_len,
            one_exact_len,
            one_trunc_len,
            clip_exact_len,
            clip_trunc_len,
            sf_trunc_len,
        ) = done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("lane add/add_truncated blocked on the process-global backend");
        worker.join().unwrap();
        assert_eq!(exact_len, 3);
        assert_eq!(trunc_len, 3);
        assert_eq!(
            (
                one_exact_len,
                one_trunc_len,
                clip_exact_len,
                clip_trunc_len,
                sf_trunc_len
            ),
            (1, 1, 1, 1, 1)
        );
    });
}

#[test]
fn lane_add_is_independent_of_index_allocation_and_item_sites_order() {
    let build_pair = |k: &DynIndex, s: &DynIndex, x: &DynIndex| {
        let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
        let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();
        let p10 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 0)]).unwrap();
        let p11 = Projector::from_pairs([(k.clone(), 1), (s.clone(), 1)]).unwrap();

        let left = PartitionedTreeTN::from_subdomains(vec![
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 1.0), p00.clone()).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 2.0), p01.clone()).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 3.0), p10.clone()).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 4.0), p11.clone()).unwrap(),
        ])
        .unwrap();
        let right = PartitionedTreeTN::from_subdomains(vec![
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 5.0), p00).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 6.0), p01).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 7.0), p10).unwrap(),
            SubDomainTreeTN::new(make_complex_chain(k, s, x, 2, 8.0), p11).unwrap(),
        ])
        .unwrap();
        (left, right)
    };

    // Allocation order A: k before s.
    let k_a = DynIndex::new_dyn(2);
    let s_a = DynIndex::new_dyn(2);
    let x_a = DynIndex::new_dyn(2);
    let (left_a, right_a) = build_pair(&k_a, &s_a, &x_a);

    // Allocation order B: s before k.
    let s_b = DynIndex::new_dyn(2);
    let k_b = DynIndex::new_dyn(2);
    let x_b = DynIndex::new_dyn(2);
    let (left_b, right_b) = build_pair(&k_b, &s_b, &x_b);

    let trunc_opts = TruncationOptions::default().with_max_bond_dim(2);
    let mut lanes = make_lanes(2);

    let exact_a = left_a
        .add_on_lanes(&right_a, Some(vec![k_a.clone()]), &mut lanes)
        .unwrap();
    let exact_b = left_b
        .add_on_lanes(&right_b, Some(vec![k_b.clone()]), &mut lanes)
        .unwrap();
    let trunc_a = left_a
        .add_truncated_on_lanes(
            &right_a,
            &0,
            trunc_opts,
            Some(vec![k_a.clone()]),
            &mut lanes,
        )
        .unwrap();
    let trunc_b = left_b
        .add_truncated_on_lanes(
            &right_b,
            &0,
            trunc_opts,
            Some(vec![k_b.clone()]),
            &mut lanes,
        )
        .unwrap();

    // Slice order [k, s] vs [s, k] produces bit-identical results.
    let exact_ks = left_a
        .add_on_lanes(&right_a, Some(vec![k_a.clone(), s_a.clone()]), &mut lanes)
        .unwrap();
    let exact_sk = left_a
        .add_on_lanes(&right_a, Some(vec![s_a.clone(), k_a.clone()]), &mut lanes)
        .unwrap();
    assert_partitions_bit_identical(&exact_ks, &exact_sk);

    let trunc_ks = left_a
        .add_truncated_on_lanes(
            &right_a,
            &0,
            trunc_opts,
            Some(vec![k_a.clone(), s_a.clone()]),
            &mut lanes,
        )
        .unwrap();
    let trunc_sk = left_a
        .add_truncated_on_lanes(
            &right_a,
            &0,
            trunc_opts,
            Some(vec![s_a.clone(), k_a.clone()]),
            &mut lanes,
        )
        .unwrap();
    assert_partitions_bit_identical(&trunc_ks, &trunc_sk);

    // Every logical (k, s) patch matches bit-for-bit between allocation orders A and B.
    for ik in 0..2 {
        for is in 0..2 {
            let pa = Projector::from_pairs([(k_a.clone(), ik), (s_a.clone(), is)]).unwrap();
            let pb = Projector::from_pairs([(k_b.clone(), ik), (s_b.clone(), is)]).unwrap();
            for (res_a, res_b, label) in [
                (&exact_a, &exact_b, "exact"),
                (&trunc_a, &trunc_b, "truncated"),
            ] {
                let patch_a = res_a.get(&pa).unwrap();
                let patch_b = res_b.get(&pb).unwrap();
                assert_eq!(
                    patch_a.data().link_dims(),
                    patch_b.data().link_dims(),
                    "{label} link_dims mismatch at (k={ik}, s={is})"
                );
                for node in 0..3usize {
                    let va = patch_a
                        .data()
                        .tensor(patch_a.data().node_index(&node).unwrap())
                        .unwrap()
                        .to_vec::<Complex64>()
                        .unwrap();
                    let vb = patch_b
                        .data()
                        .tensor(patch_b.data().node_index(&node).unwrap())
                        .unwrap()
                        .to_vec::<Complex64>()
                        .unwrap();
                    assert_eq!(va.len(), vb.len());
                    for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
                        assert_eq!(
                            a.re.to_bits(),
                            b.re.to_bits(),
                            "{label} real bit mismatch at (k={ik}, s={is}) node {node} entry {i}"
                        );
                        assert_eq!(
                            a.im.to_bits(),
                            b.im.to_bits(),
                            "{label} imag bit mismatch at (k={ik}, s={is}) node {node} entry {i}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lane_add_preserves_f32_and_c32_under_refinement_and_single_node_on_session() {
    use num_complex::Complex32;
    use std::sync::mpsc;
    use std::time::Duration;
    use tensor4all_tensorbackend::with_default_backend;

    let k = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let pk0 = Projector::from_pairs([(k.clone(), 0)]).unwrap();
    let p00 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 0)]).unwrap();
    let p01 = Projector::from_pairs([(k.clone(), 0), (s.clone(), 1)]).unwrap();

    // Multi-node f32 partition with overlapping projectors (triggers project_on_session).
    let make_f32_chain = |seed: f32| {
        let b = DynIndex::new_dyn(1);
        let t0 = IdxTensor::from_dense(vec![k.clone(), b.clone()], vec![seed, seed + 1.0]).unwrap();
        let t1 = IdxTensor::from_dense(vec![b, s.clone()], vec![1.0_f32, 2.0]).unwrap();
        TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1]).unwrap()
    };
    let f32_left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        make_f32_chain(1.5),
        pk0.clone(),
    )
    .unwrap()])
    .unwrap();
    let f32_right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(make_f32_chain(2.5), p00.clone()).unwrap(),
        SubDomainTreeTN::new(make_f32_chain(3.5), p01.clone()).unwrap(),
    ])
    .unwrap();
    let f32_serial = f32_left.add(&f32_right).unwrap();

    // 1-node Complex32 partition with permuted site order and refinement.
    let c32_t_left = IdxTensor::from_dense(
        vec![k.clone(), s.clone()],
        vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -0.5),
            Complex32::new(3.0, 0.25),
            Complex32::new(4.0, -0.25),
        ],
    )
    .unwrap();
    let c32_t_right = IdxTensor::from_dense(
        vec![s.clone(), k.clone()],
        vec![
            Complex32::new(0.5, 1.0),
            Complex32::new(-0.5, 2.0),
            Complex32::new(0.25, 3.0),
            Complex32::new(-0.25, 4.0),
        ],
    )
    .unwrap();
    let c32_left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        TreeTN::from_tensors(vec![c32_t_left], vec![0usize]).unwrap(),
        pk0,
    )
    .unwrap()])
    .unwrap();
    let c32_right = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        TreeTN::from_tensors(vec![c32_t_right], vec![0usize]).unwrap(),
        p00.clone(),
    )
    .unwrap()])
    .unwrap();
    let c32_serial = c32_left.add(&c32_right).unwrap();

    let (f32_lanes, c32_lanes) = with_default_backend(|_global| {
        let (tx, rx) = mpsc::channel();
        let k_worker = k.clone();
        let worker = std::thread::spawn(move || {
            let mut lanes = make_lanes(2);
            let f32_res = f32_left
                .add_on_lanes(&f32_right, Some(vec![k_worker.clone()]), &mut lanes)
                .unwrap();
            let c32_res = c32_left
                .add_on_lanes(&c32_right, Some(vec![k_worker]), &mut lanes)
                .unwrap();
            tx.send((f32_res, c32_res)).unwrap();
        });
        let pair = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("f32/c32 add_on_lanes blocked on global backend");
        worker.join().unwrap();
        pair
    });

    for key in [&p00, &p01] {
        let a_patch = f32_lanes.get(key).unwrap();
        let e_patch = f32_serial.get(key).unwrap();
        for node in 0..2usize {
            let at = a_patch
                .data()
                .tensor(a_patch.data().node_index(&node).unwrap())
                .unwrap();
            let et = e_patch
                .data()
                .tensor(e_patch.data().node_index(&node).unwrap())
                .unwrap();
            assert!(at.is_f32(), "expected f32 tensor after refinement");
            let av: Vec<u32> = at
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let ev: Vec<u32> = et
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(av, ev);
        }
    }

    for key in [&p00, &p01] {
        let a_patch = c32_lanes.get(key).unwrap();
        let e_patch = c32_serial.get(key).unwrap();
        let at = a_patch
            .data()
            .tensor(a_patch.data().node_index(&0).unwrap())
            .unwrap();
        let et = e_patch
            .data()
            .tensor(e_patch.data().node_index(&0).unwrap())
            .unwrap();
        assert!(
            at.is_c32(),
            "expected Complex32 tensor after 1-node refinement"
        );
        let av: Vec<(u32, u32)> = at
            .to_vec::<Complex32>()
            .unwrap()
            .into_iter()
            .map(|c| (c.re.to_bits(), c.im.to_bits()))
            .collect();
        let ev: Vec<(u32, u32)> = et
            .to_vec::<Complex32>()
            .unwrap()
            .into_iter()
            .map(|c| (c.re.to_bits(), c.im.to_bits()))
            .collect();
        assert_eq!(av, ev);
    }
}
