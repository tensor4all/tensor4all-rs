use std::sync::Mutex;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{PartitionedTreeTN, PartitionedTreeTNError, Projector};
use tensor4all_treetn::TreeTN;

fn make_complex_two_site_tree(
    site_a: &DynIndex,
    site_b: &DynIndex,
    offset: f64,
) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(
        vec![site_a.clone(), bond.clone()],
        vec![
            Complex64::new(1.0 + offset, 0.5 + offset),
            Complex64::new(-2.0 + offset, 1.5),
            Complex64::new(0.25, -1.25 - offset),
            Complex64::new(3.0, 0.75 + offset),
            Complex64::new(-0.5 - offset, 2.25),
            Complex64::new(1.25, -0.75 + offset),
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
    TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap()
}

#[test]
fn builder_preserves_absent_and_present_projectors() {
    let site_a = DynIndex::new_dyn(3);
    let site_b = DynIndex::new_dyn(2);

    let p0 = Projector::from_pairs([(site_a.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site_a.clone(), 1)]).unwrap();
    let p2 = Projector::from_pairs([(site_a.clone(), 2)]).unwrap();

    // Non-canonical order to verify `build` is called in `layout` order.
    let requested = vec![p2.clone(), p0.clone(), p1.clone()];
    let called = Mutex::new(Vec::new());

    let partition = PartitionedTreeTN::try_from_projectors(&requested, |projector| {
        called.lock().unwrap().push(projector.clone());
        match projector.get(&site_a) {
            Some(0) => Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 0.0))),
            Some(1) => Ok(None),
            Some(2) => Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 2.0))),
            _ => unreachable!("unexpected projector"),
        }
    })
    .unwrap();

    assert_eq!(*called.lock().unwrap(), requested);

    let made: Vec<Projector> = partition.iter().map(|(p, _)| p.clone()).collect();
    let expected_nonzero = vec![p0.clone(), p2.clone()];
    assert_eq!(made, expected_nonzero);
    assert!(partition.contains(&p0));
    assert!(!partition.contains(&p1));
    assert!(partition.contains(&p2));

    // Verify dense values on each present patch match the projected tree.
    for (proj, offset) in [(&p0, 0.0), (&p2, 2.0)] {
        let patch = partition.get(proj).unwrap();
        let expected_sub = tensor4all_partitionedtreetn::SubDomainTreeTN::new(
            make_complex_two_site_tree(&site_a, &site_b, offset),
            proj.clone(),
        )
        .unwrap();
        let diff = patch
            .data()
            .to_dense()
            .unwrap()
            .sub(&expected_sub.data().to_dense().unwrap())
            .unwrap()
            .maxabs()
            .unwrap();
        assert!(diff < 1e-12, "patch dense mismatch for {proj:?}: {diff}");
    }

    // Invalid output: mismatched site index between patches -> SiteIndexMismatch.
    let other_site_b = DynIndex::new_dyn(2);
    let invalid_site_err = PartitionedTreeTN::try_from_projectors(&[p0.clone(), p2.clone()], |p| {
        if p == &p0 {
            Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 0.0)))
        } else {
            Ok(Some(make_complex_two_site_tree(&site_a, &other_site_b, 1.0)))
        }
    })
    .unwrap_err();
    assert!(
        matches!(invalid_site_err, PartitionedTreeTNError::SiteIndexMismatch),
        "expected SiteIndexMismatch, got {invalid_site_err:?}"
    );

    // Invalid output: mismatched scalar dtype -> DTypeMismatch.
    let invalid_dtype_err =
        PartitionedTreeTN::try_from_projectors(&[p0.clone(), p2.clone()], |p| {
            if p == &p0 {
                Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 0.0)))
            } else {
                let bond = DynIndex::new_dyn(2);
                let t0 = IdxTensor::from_dense(
                    vec![site_a.clone(), bond.clone()],
                    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
                )
                .unwrap();
                let t1 = IdxTensor::from_dense(
                    vec![bond.clone(), site_b.clone()],
                    vec![1.0_f64, 0.0, 0.0, 1.0],
                )
                .unwrap();
                Ok(Some(
                    TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap(),
                ))
            }
        })
        .unwrap_err();
    assert!(
        matches!(invalid_dtype_err, PartitionedTreeTNError::DTypeMismatch { .. }),
        "expected DTypeMismatch, got {invalid_dtype_err:?}"
    );

    // Invalid output: mismatched topology -> TopologyMismatch.
    let invalid_topo_err =
        PartitionedTreeTN::try_from_projectors(&[p0.clone(), p2.clone()], |p| {
            if p == &p0 {
                Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 0.0)))
            } else {
                let t = IdxTensor::from_dense(
                    vec![site_a.clone(), site_b.clone()],
                    vec![Complex64::new(1.0, 0.0); 6],
                )
                .unwrap();
                Ok(Some(TreeTN::from_tensors(vec![t], vec![0usize]).unwrap()))
            }
        })
        .unwrap_err();
    assert!(
        matches!(invalid_topo_err, PartitionedTreeTNError::TopologyMismatch),
        "expected TopologyMismatch, got {invalid_topo_err:?}"
    );

    // Invalid output: projector index absent from returned TreeTN -> ProjectorIndexNotFound.
    let missing_site = DynIndex::new_dyn(2);
    let p_missing = Projector::from_pairs([(missing_site, 0)]).unwrap();
    let invalid_proj_err = PartitionedTreeTN::try_from_projectors(&[p_missing], |_| {
        Ok(Some(make_complex_two_site_tree(&site_a, &site_b, 0.0)))
    })
    .unwrap_err();
    assert!(
        matches!(
            invalid_proj_err,
            PartitionedTreeTNError::ProjectorIndexNotFound { .. }
        ),
        "expected ProjectorIndexNotFound, got {invalid_proj_err:?}"
    );
}

#[test]
fn empty_layout_returns_empty_partition() {
    let called = Mutex::new(0usize);
    let partition: PartitionedTreeTN<usize> = PartitionedTreeTN::try_from_projectors(&[], |_| {
        *called.lock().unwrap() += 1;
        Ok(None)
    })
    .unwrap();

    assert!(partition.is_empty());
    assert_eq!(partition.len(), 0);
    assert_eq!(*called.lock().unwrap(), 0);
}

#[test]
fn all_none_returns_empty_partition() {
    let site_a = DynIndex::new_dyn(2);
    let p0 = Projector::from_pairs([(site_a.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site_a.clone(), 1)]).unwrap();
    let called = Mutex::new(Vec::new());

    let partition: PartitionedTreeTN<usize> =
        PartitionedTreeTN::try_from_projectors(&[p0.clone(), p1.clone()], |p| {
            called.lock().unwrap().push(p.clone());
            Ok(None)
        })
        .unwrap();

    assert_eq!(*called.lock().unwrap(), vec![p0, p1]);
    assert!(partition.is_empty());
}

#[test]
fn duplicate_or_overlapping_layout_rejects_before_calling_build() {
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let p0 = Projector::from_pairs([(site_a.clone(), 0)]).unwrap();
    let p0_b0 = Projector::from_pairs([(site_a.clone(), 0), (site_b.clone(), 0)]).unwrap();

    // Duplicate projector.
    let dup_calls = Mutex::new(0usize);
    let dup_err: PartitionedTreeTNError =
        PartitionedTreeTN::<usize>::try_from_projectors(&[p0.clone(), p0.clone()], |_| {
            *dup_calls.lock().unwrap() += 1;
            Ok(None)
        })
        .unwrap_err();
    assert!(
        matches!(dup_err, PartitionedTreeTNError::OverlappingProjectors),
        "expected OverlappingProjectors for duplicate, got {dup_err:?}"
    );
    assert_eq!(*dup_calls.lock().unwrap(), 0);

    // Distinct but overlapping projectors.
    let overlap_calls = Mutex::new(0usize);
    let overlap_err: PartitionedTreeTNError =
        PartitionedTreeTN::<usize>::try_from_projectors(&[p0, p0_b0], |_| {
            *overlap_calls.lock().unwrap() += 1;
            Ok(None)
        })
        .unwrap_err();
    assert!(
        matches!(overlap_err, PartitionedTreeTNError::OverlappingProjectors),
        "expected OverlappingProjectors for overlap, got {overlap_err:?}"
    );
    assert_eq!(*overlap_calls.lock().unwrap(), 0);
}

#[test]
fn callback_error_propagates_and_stops() {
    let site_a = DynIndex::new_dyn(2);
    let p0 = Projector::from_pairs([(site_a.clone(), 0)]).unwrap();
    let p1 = Projector::from_pairs([(site_a.clone(), 1)]).unwrap();
    let called = Mutex::new(0usize);

    let err: PartitionedTreeTNError =
        PartitionedTreeTN::<usize>::try_from_projectors(&[p0, p1], |_| {
            *called.lock().unwrap() += 1;
            Err(PartitionedTreeTNError::InvalidCenter)
        })
        .unwrap_err();

    assert!(matches!(err, PartitionedTreeTNError::InvalidCenter));
    assert_eq!(*called.lock().unwrap(), 1);
}
