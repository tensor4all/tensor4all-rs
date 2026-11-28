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
            assert_partitions_bit_identical(&trunc, &serial_trunc);
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

    for lane_count in [1usize, 2] {
        let mut lanes = make_lanes(lane_count);
        let exact = left
            .add_on_lanes(&right, Some(vec![k.clone()]), &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&exact, &serial_exact);

        let trunc = left
            .add_truncated_on_lanes(&right, &0, trunc_opts, Some(vec![k.clone()]), &mut lanes)
            .unwrap();
        assert_partitions_bit_identical(&trunc, &serial_trunc);
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
