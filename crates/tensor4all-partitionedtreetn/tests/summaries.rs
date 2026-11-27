use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

fn complex_two_site_patch(
    s0: &DynIndex,
    s1: &DynIndex,
    bond_dim: usize,
    a_vals: Vec<Complex64>,
    b_vals: Vec<Complex64>,
    projector: Projector,
) -> SubDomainTreeTN {
    let link = DynIndex::new_dyn(bond_dim);
    let t0 = IdxTensor::from_dense(vec![s0.clone(), link.clone()], a_vals).unwrap();
    let t1 = IdxTensor::from_dense(vec![link, s1.clone()], b_vals).unwrap();
    let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap();
    SubDomainTreeTN::new(tree, projector).unwrap()
}

fn dense_two_site_contraction(
    d0: usize,
    d1: usize,
    bond_dim: usize,
    a_vals: &[Complex64],
    b_vals: &[Complex64],
    fixed_s0: usize,
) -> Vec<Complex64> {
    // Column-major: t0 has shape (d0, bond_dim), t1 has shape (bond_dim, d1).
    let mut out = vec![Complex64::new(0.0, 0.0); d0 * d1];
    for i1 in 0..d1 {
        let mut acc = Complex64::new(0.0, 0.0);
        for k in 0..bond_dim {
            let a = a_vals[fixed_s0 + d0 * k];
            let b = b_vals[k + bond_dim * i1];
            acc += a * b;
        }
        out[fixed_s0 + d0 * i1] = acc;
    }
    out
}

fn vec_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

fn vec_distance(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

#[test]
fn summaries_pair_identical_projectors() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let p0 = Projector::from_pairs([(s0.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(s0.clone(), 1)]).unwrap();

    // Patch 0: bond_dim = 2, scale ~ 1.0
    let a0_left = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(9.0, -9.0), // masked out by s0 = 0
        Complex64::new(-0.25, 1.25),
        Complex64::new(7.0, 7.0), // masked out by s0 = 0
    ];
    let b0_left = vec![
        Complex64::new(2.0, -1.0),
        Complex64::new(0.5, 0.75),
        Complex64::new(-1.5, 0.25),
        Complex64::new(1.0, 2.0),
    ];

    // Patch 1: bond_dim = 3, scale ~ 1e3 (nonuniform scale)
    let a1_left = vec![
        Complex64::new(5.0e3, 5.0e3), // masked out by s0 = 1
        Complex64::new(1.0e3, -2.0e3),
        Complex64::new(-3.0e3, 1.0e3), // masked out
        Complex64::new(0.5e3, 1.5e3),
        Complex64::new(2.0e3, -1.0e3), // masked out
        Complex64::new(-1.25e3, 0.75e3),
    ];
    let b1_left = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(0.5, -0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(-0.25, 1.25),
    ];

    // Right partition on the same two projectors with different bond dimensions and values.
    let a0_right = vec![
        Complex64::new(0.5, -0.5),
        Complex64::new(-4.0, 4.0), // masked out
        Complex64::new(1.0, 0.25),
        Complex64::new(3.0, -3.0), // masked out
    ];
    let b0_right = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(-0.5, 0.5),
        Complex64::new(0.25, -1.0),
        Complex64::new(2.0, 0.0),
    ];

    let a1_right = vec![
        Complex64::new(1.0e3, 0.0), // masked out
        Complex64::new(0.75e3, -1.5e3),
        Complex64::new(0.0, 1.0e3), // masked out
        Complex64::new(0.25e3, 1.0e3),
    ];
    let b1_right = vec![
        Complex64::new(1.5, -0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.5, 0.5),
        Complex64::new(1.25, -0.5),
    ];

    let expected_left_0 = dense_two_site_contraction(2, 2, 2, &a0_left, &b0_left, 0);
    let expected_left_1 = dense_two_site_contraction(2, 2, 3, &a1_left, &b1_left, 1);
    let expected_right_0 = dense_two_site_contraction(2, 2, 2, &a0_right, &b0_right, 0);
    let expected_right_1 = dense_two_site_contraction(2, 2, 2, &a1_right, &b1_right, 1);

    let expected_norms = [
        (p0.clone(), vec_norm(&expected_left_0)),
        (p1.clone(), vec_norm(&expected_left_1)),
    ];
    let expected_distances = [
        (
            p0.clone(),
            vec_distance(&expected_left_0, &expected_right_0),
        ),
        (
            p1.clone(),
            vec_distance(&expected_left_1, &expected_right_1),
        ),
    ];

    // Insert in reverse order to verify canonical iter() ordering.
    let left = PartitionedTreeTN::from_subdomains(vec![
        complex_two_site_patch(&s0, &s1, 3, a1_left, b1_left, p1.clone()),
        complex_two_site_patch(&s0, &s1, 2, a0_left, b0_left, p0.clone()),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        complex_two_site_patch(&s0, &s1, 2, a0_right, b0_right, p0.clone()),
        complex_two_site_patch(&s0, &s1, 2, a1_right, b1_right, p1.clone()),
    ])
    .unwrap();

    assert_eq!(left.max_bond_dim(), 3);
    assert_eq!(right.max_bond_dim(), 2);

    let norms = left.patch_norms().unwrap();
    assert_eq!(norms.len(), expected_norms.len());
    for ((proj, norm), (exp_proj, exp_norm)) in norms.iter().zip(expected_norms.iter()) {
        assert_eq!(proj, exp_proj);
        assert!(
            (norm - exp_norm).abs() <= 1.0e-12 * exp_norm.max(1.0),
            "norm mismatch for {proj:?}: got {norm}, expected {exp_norm}"
        );
    }

    let distances = left.patch_distances(&right).unwrap();
    assert_eq!(distances.len(), expected_distances.len());
    for ((proj, dist), (exp_proj, exp_dist)) in distances.iter().zip(expected_distances.iter()) {
        assert_eq!(proj, exp_proj);
        assert!(
            (dist - exp_dist).abs() <= 1.0e-12 * exp_dist.max(1.0),
            "distance mismatch for {proj:?}: got {dist}, expected {exp_dist}"
        );
    }

    let self_distances = left.patch_distances(&left).unwrap();
    for (proj, dist) in &self_distances {
        assert!(
            *dist <= 1.0e-12 * left.get(proj).unwrap().norm().unwrap().max(1.0),
            "self-distance for {proj:?} should be zero, got {dist}"
        );
    }
}

#[test]
fn summaries_handle_empty_partition() {
    let empty = PartitionedTreeTN::<usize>::new();
    assert_eq!(empty.max_bond_dim(), 0);
    assert!(empty.patch_norms().unwrap().is_empty());
    assert!(empty.patch_distances(&empty).unwrap().is_empty());

    let site = DynIndex::new_dyn(2);
    let p0 = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let non_empty =
        PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(tree, p0.clone()).unwrap()).unwrap();

    assert_eq!(non_empty.max_bond_dim(), 1);
    let err = empty.patch_distances(&non_empty).unwrap_err();
    assert!(
        matches!(err, PartitionedTreeTNError::LayoutMismatch { ref projector } if projector == &p0),
        "expected LayoutMismatch naming {p0:?}, got {err:?}"
    );
}

#[test]
fn summaries_handle_zero_patch_and_nonuniform_scales() {
    let site = DynIndex::new_dyn(3);
    let p0 = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site.clone(), 1)]).unwrap();
    let p2 = Projector::from_pairs([(site.clone(), 2)]).unwrap();

    let make = |vals: Vec<f64>, proj: Projector| {
        let t = IdxTensor::from_dense(vec![site.clone()], vals).unwrap();
        let tree = TreeTN::from_tensors(vec![t], vec![0usize]).unwrap();
        SubDomainTreeTN::new(tree, proj).unwrap()
    };

    let left = PartitionedTreeTN::from_subdomains(vec![
        make(vec![0.0, 0.0, 0.0], p0.clone()),
        make(vec![0.0, 1.0e-6, 0.0], p1.clone()),
        make(vec![0.0, 0.0, 3.0e6], p2.clone()),
    ])
    .unwrap();

    let right = PartitionedTreeTN::from_subdomains(vec![
        make(vec![5.0, 0.0, 0.0], p0.clone()),
        make(vec![0.0, 4.0e-6, 0.0], p1.clone()),
        make(vec![0.0, 0.0, 0.0], p2.clone()),
    ])
    .unwrap();

    let norms = left.patch_norms().unwrap();
    assert_eq!(norms.len(), 3);
    assert_eq!(norms[0].0, p0);
    assert!((norms[0].1 - 0.0).abs() <= 1.0e-15);
    assert_eq!(norms[1].0, p1);
    assert!((norms[1].1 - 1.0e-6).abs() <= 1.0e-18);
    assert_eq!(norms[2].0, p2);
    assert!((norms[2].1 - 3.0e6).abs() <= 1.0e-6);

    let distances = left.patch_distances(&right).unwrap();
    assert_eq!(distances.len(), 3);
    assert_eq!(distances[0].0, p0);
    assert!((distances[0].1 - 5.0).abs() <= 1.0e-12);
    assert_eq!(distances[1].0, p1);
    assert!((distances[1].1 - 3.0e-6).abs() <= 1.0e-18);
    assert_eq!(distances[2].0, p2);
    assert!((distances[2].1 - 3.0e6).abs() <= 1.0e-6);
}

#[test]
fn patch_distances_rejects_layout_mismatch_with_first_differing_projector() {
    let site = DynIndex::new_dyn(3);
    let p0 = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site.clone(), 1)]).unwrap();
    let p2 = Projector::from_pairs([(site.clone(), 2)]).unwrap();

    let make = |val: f64, proj: Projector| {
        let t = IdxTensor::from_dense(vec![site.clone()], vec![val, val, val]).unwrap();
        let tree = TreeTN::from_tensors(vec![t], vec![0usize]).unwrap();
        SubDomainTreeTN::new(tree, proj).unwrap()
    };

    let left =
        PartitionedTreeTN::from_subdomains(vec![make(1.0, p0.clone()), make(2.0, p1.clone())])
            .unwrap();
    let right =
        PartitionedTreeTN::from_subdomains(vec![make(1.0, p0.clone()), make(3.0, p2.clone())])
            .unwrap();

    // First differing key in `left` is p1.
    let err = left.patch_distances(&right).unwrap_err();
    assert!(
        matches!(err, PartitionedTreeTNError::LayoutMismatch { ref projector } if projector == &p1),
        "expected LayoutMismatch naming {p1:?}, got {err:?}"
    );

    // Extra key in `right` when `left` is a subset.
    let subset = PartitionedTreeTN::from_subdomains(vec![make(1.0, p0.clone())]).unwrap();
    let err = subset.patch_distances(&right).unwrap_err();
    assert!(
        matches!(err, PartitionedTreeTNError::LayoutMismatch { ref projector } if projector == &p2),
        "expected LayoutMismatch naming {p2:?}, got {err:?}"
    );
}
