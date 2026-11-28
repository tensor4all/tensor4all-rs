use std::sync::Mutex;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike, TensorIndex};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

fn make_two_site_complex_partition(
    site_a: &DynIndex,
    site_b: &DynIndex,
) -> PartitionedTreeTN<usize> {
    let bond = DynIndex::new_dyn(2);
    // Patch 0 (site_a = 0) and Patch 1 (site_a = 1) with distinct complex entries.
    let make_patch = |val: usize, offset: f64| -> SubDomainTreeTN<usize> {
        let t0 = IdxTensor::from_dense(
            vec![site_a.clone(), bond.clone()],
            vec![
                Complex64::new(1.0 + offset, 0.5 + offset),
                Complex64::new(-2.0 + offset, 1.5),
                Complex64::new(0.25, -1.25 - offset),
                Complex64::new(3.0, 0.75 + offset),
            ],
        )
        .unwrap();
        let t1 = IdxTensor::from_dense(
            vec![bond.clone(), site_b.clone()],
            vec![
                Complex64::new(0.5, -0.5 + offset),
                Complex64::new(1.5 + offset, 2.0),
                Complex64::new(-1.0, 0.25),
                Complex64::new(2.0 - offset, -1.5),
            ],
        )
        .unwrap();
        let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap();
        SubDomainTreeTN::new(
            tree,
            Projector::from_pairs([(site_a.clone(), val)]).unwrap(),
        )
        .unwrap()
    };

    PartitionedTreeTN::from_subdomains(vec![make_patch(1, 1.0), make_patch(0, 0.0)]).unwrap()
}

#[test]
fn valid_map_and_invalid_map_keeps_input() {
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let original = make_two_site_complex_partition(&site_a, &site_b);
    let before_projectors: Vec<Projector> = original.iter().map(|(p, _)| p.clone()).collect();
    let before_dense = original.to_treetn().unwrap().to_dense().unwrap();

    // 1. Complex conjugation: equals per-patch conjugate, projectors unchanged.
    let conjugated = original.conj().unwrap();
    let conj_projectors: Vec<Projector> = conjugated.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(conj_projectors, before_projectors);
    let conj_dense = conjugated.to_treetn().unwrap().to_dense().unwrap();
    let expected_conj_dense = before_dense.conj();
    let diff = conj_dense
        .sub(&expected_conj_dense)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(diff < 1e-12, "conj dense mismatch: {diff}");

    // 2. Valid projected relabel: replace site_a with site_a_new in every patch.
    let site_a_new = DynIndex::new_dyn(2);
    let relabeled = original
        .try_map_patches(|patch| {
            let new_tree = patch
                .data()
                .replaceind(&site_a, &site_a_new)
                .map_err(PartitionedTreeTNError::from)?;
            let val = patch.projector().get(&site_a).unwrap();
            let new_proj = Projector::from_pairs([(site_a_new.clone(), val)])?;
            SubDomainTreeTN::new(new_tree, new_proj)
        })
        .unwrap();
    let out_projectors: Vec<Projector> = relabeled.iter().map(|(p, _)| p.clone()).collect();
    let expected_projectors = vec![
        Projector::from_pairs([(site_a_new.clone(), 0)]).unwrap(),
        Projector::from_pairs([(site_a_new.clone(), 1)]).unwrap(),
    ];
    assert_eq!(out_projectors, expected_projectors);
    let relabeled_dense = relabeled
        .to_treetn()
        .unwrap()
        .to_dense()
        .unwrap()
        .replaceind(&site_a_new, &site_a)
        .unwrap();
    let relabel_diff = relabeled_dense
        .sub(&before_dense)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(
        relabel_diff < 1e-12,
        "relabel dense mismatch: {relabel_diff}"
    );

    // 3. Valid site removal: contract site_b with ones vector so each patch becomes a 1-site tree.
    let ones_b = IdxTensor::from_dense(
        vec![site_b.clone()],
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .unwrap();
    let reduced = original
        .try_map_patches(|patch| {
            let patch_dense = patch.data().to_dense().unwrap();
            let contracted = patch_dense.contract_pair(&ones_b).unwrap();
            let new_tree = TreeTN::from_tensors(vec![contracted], vec![0usize]).unwrap();
            SubDomainTreeTN::new(new_tree, patch.projector().clone())
        })
        .unwrap();
    let reduced_projectors: Vec<Projector> = reduced.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(reduced_projectors, before_projectors);
    let reduced_dense = reduced.to_treetn().unwrap().to_dense().unwrap();
    let expected_reduced = before_dense.contract_pair(&ones_b).unwrap();
    let reduced_diff = reduced_dense
        .sub(&expected_reduced)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(
        reduced_diff < 1e-12,
        "site-removal dense mismatch: {reduced_diff}"
    );

    // 4. Invalid map: callback changes just one patch's topology -> TopologyMismatch.
    let topo_err = original
        .try_map_patches(|patch| {
            if patch.projector().get(&site_a) == Some(1) {
                let single = IdxTensor::from_dense(
                    vec![site_a.clone()],
                    vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                )
                .unwrap();
                let tree = TreeTN::from_tensors(vec![single], vec![0usize]).unwrap();
                SubDomainTreeTN::new(tree, patch.projector().clone())
            } else {
                Ok(patch.clone())
            }
        })
        .unwrap_err();
    assert!(
        matches!(topo_err, PartitionedTreeTNError::TopologyMismatch),
        "expected TopologyMismatch, got {topo_err:?}"
    );

    // 5. Invalid map: callback changes just one patch's site index -> SiteIndexMismatch.
    let other_b = DynIndex::new_dyn(2);
    let site_err = original
        .try_map_patches(|patch| {
            if patch.projector().get(&site_a) == Some(1) {
                let new_tree = patch
                    .data()
                    .replaceind(&site_b, &other_b)
                    .map_err(PartitionedTreeTNError::from)?;
                SubDomainTreeTN::new(new_tree, patch.projector().clone())
            } else {
                Ok(patch.clone())
            }
        })
        .unwrap_err();
    assert!(
        matches!(site_err, PartitionedTreeTNError::SiteIndexMismatch),
        "expected SiteIndexMismatch, got {site_err:?}"
    );

    // 6. Invalid map: callback produces overlapping projectors -> OverlappingProjectors.
    let overlap_err = original
        .try_map_patches(|patch| {
            if patch.projector().get(&site_a) == Some(1) {
                SubDomainTreeTN::new(patch.data().clone(), Projector::new())
            } else {
                Ok(patch.clone())
            }
        })
        .unwrap_err();
    assert!(
        matches!(overlap_err, PartitionedTreeTNError::OverlappingProjectors),
        "expected OverlappingProjectors, got {overlap_err:?}"
    );

    // Input partition is completely untouched after all valid and invalid maps.
    let after_projectors: Vec<Projector> = original.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(after_projectors, before_projectors);
    let after_dense = original.to_treetn().unwrap().to_dense().unwrap();
    let untouched_diff = after_dense.sub(&before_dense).unwrap().maxabs().unwrap();
    assert_eq!(untouched_diff, 0.0);
}

#[test]
fn try_map_patches_visits_in_deterministic_order() {
    let site_a = DynIndex::new_dyn(3);
    let make = |order: &[usize]| -> PartitionedTreeTN<usize> {
        let patches = order
            .iter()
            .map(|&v| {
                let t = IdxTensor::from_dense(
                    vec![site_a.clone()],
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(2.0, 0.0),
                        Complex64::new(3.0, 0.0),
                    ],
                )
                .unwrap();
                let tree = TreeTN::from_tensors(vec![t], vec![0usize]).unwrap();
                SubDomainTreeTN::new(tree, Projector::from_pairs([(site_a.clone(), v)]).unwrap())
                    .unwrap()
            })
            .collect();
        PartitionedTreeTN::from_subdomains(patches).unwrap()
    };

    let p_fwd = make(&[0, 1, 2]);
    let p_rev = make(&[2, 0, 1]);

    let record_order = |p: &PartitionedTreeTN<usize>| -> Vec<Projector> {
        let seen = Mutex::new(Vec::new());
        let _ = p
            .try_map_patches(|patch| {
                seen.lock().unwrap().push(patch.projector().clone());
                Ok(patch.clone())
            })
            .unwrap();
        seen.into_inner().unwrap()
    };

    let order_fwd = record_order(&p_fwd);
    let order_rev = record_order(&p_rev);
    assert_eq!(order_fwd, order_rev);
    assert_eq!(order_fwd.len(), 3);
}

#[test]
fn try_map_patches_empty_and_callback_error() {
    let empty: PartitionedTreeTN<usize> = PartitionedTreeTN::new();
    let mapped = empty
        .try_map_patches(|_| Err(PartitionedTreeTNError::InvalidCenter))
        .unwrap();
    assert!(mapped.is_empty());

    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let part = make_two_site_complex_partition(&site_a, &site_b);
    let err = part
        .try_map_patches(|_| Err(PartitionedTreeTNError::InvalidCenter))
        .unwrap_err();
    assert!(matches!(err, PartitionedTreeTNError::InvalidCenter));
}

fn partition_with_coverage() -> (DynIndex, PartitionedTreeTN<usize>, Vec<Projector>) {
    let site = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![site.clone()],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        )
        .unwrap()],
        vec![0usize],
    )
    .unwrap();
    let p0 = Projector::from_pairs([(site.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site.clone(), 1)]).unwrap();
    let cover = vec![p0.clone(), p1.clone()];
    let partition = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(tree.clone(), p0).unwrap(),
        SubDomainTreeTN::new(tree, p1).unwrap(),
    ])
    .unwrap()
    .with_coverage(cover.clone())
    .unwrap();
    (site, partition, cover)
}

#[test]
fn conj_preserves_coverage() {
    let (_site, partition, cover) = partition_with_coverage();
    let conj = partition.conj().unwrap();
    assert!(conj.coverage().is_some());
    assert_eq!(conj.coverage(), partition.coverage());
    assert_eq!(conj.coverage().unwrap().len(), cover.len());
}

#[test]
fn try_map_patches_keeps_coverage_when_projectors_unchanged() {
    let (_site, partition, _cover) = partition_with_coverage();
    let mapped = partition.try_map_patches(|p| Ok(p.clone())).unwrap();
    assert_eq!(mapped.coverage(), partition.coverage());
}

#[test]
fn try_map_patches_drops_coverage_when_projectors_change() {
    let (site, partition, _cover) = partition_with_coverage();
    let new_site = DynIndex::new_dyn(2);
    let mapped = partition
        .try_map_patches(|patch| {
            let data = patch.data().replaceind(&site, &new_site).unwrap();
            let val = patch.projector().get(&site).unwrap();
            SubDomainTreeTN::new(
                data,
                Projector::from_pairs([(new_site.clone(), val)]).unwrap(),
            )
        })
        .unwrap();
    assert_eq!(mapped.coverage(), None);
}
