use super::*;

#[test]
fn swaps_within_and_between_nodes_are_simultaneous() {
    let a = DynIndex::new_dyn(2);
    let b = a.prime();
    let values = vec![1.0, 2.0, 3.0, 5.0];
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![a.clone(), b.clone()], values.clone()).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let swapped = source
        .replace_indices(&[a.clone(), b.clone()], &[b.clone(), a.clone()])
        .unwrap();
    let expected = IdxTensor::from_dense(vec![b.clone(), a.clone()], values.clone()).unwrap();
    assert_eq!(
        swapped
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );
    assert_eq!(
        source
            .to_dense()
            .unwrap()
            .sub(&IdxTensor::from_dense(vec![a, b], values).unwrap())
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );

    let (source, a, _, b) = make_two_node_treetn();
    let swapped = source
        .replace_indices(&[a.clone(), b.clone()], &[b.clone(), a.clone()])
        .unwrap();
    let expected = source
        .to_dense()
        .unwrap()
        .replace_indices(&[a.clone(), b.clone()], &[b.clone(), a.clone()])
        .unwrap();
    assert_eq!(
        swapped
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );
    assert_eq!(
        swapped.node_site_indices(&"A".to_string()).unwrap(),
        vec![b.clone()]
    );
    assert_eq!(
        swapped.node_site_indices(&"B".to_string()).unwrap(),
        vec![a.clone()]
    );
    // The public reindex helper uses the same owner seam, including cross-node swaps.
    let aligned = source.reindex_site_space_like(&swapped).unwrap();
    assert_eq!(
        aligned
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );
}

#[test]
fn three_cycle_and_site_bond_swap_preserve_roles_and_values() {
    let a = DynIndex::new_dyn(2);
    let b = a.prime();
    let c = b.prime();
    let values = vec![1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![a.clone(), b.clone(), c.clone()], values.clone()).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let renamed = source
        .replace_indices(
            &[a.clone(), b.clone(), c.clone()],
            &[b.clone(), c.clone(), a.clone()],
        )
        .unwrap();
    let expected = IdxTensor::from_dense(vec![b, c, a], values).unwrap();
    assert_eq!(
        renamed
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );

    let site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_bond(2).unwrap();
    let other = DynIndex::new_dyn(2);
    let source = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![site.clone(), bond.clone()], vec![1.0, 2.0, 3.0, 5.0])
                .unwrap(),
            IdxTensor::from_dense(
                vec![bond.clone(), other.clone()],
                vec![7.0, 11.0, 13.0, 17.0],
            )
            .unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let renamed = source
        .replace_indices(&[site.clone(), bond.clone()], &[bond.clone(), site.clone()])
        .unwrap();
    let expected = source.to_dense().unwrap().replaceind(&site, &bond).unwrap();
    assert_eq!(renamed.link_dims(), vec![2]);
    assert_eq!(renamed.external_indices(), vec![bond, other]);
    assert_eq!(
        renamed
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );
}

#[test]
fn invalid_batch_mappings_are_transactional_and_empty_mapping_is_identity() {
    let (source, a, _, b) = make_two_node_treetn();
    let fresh = a.sim();
    let before = source.to_dense().unwrap();
    for (old, new) in [
        (vec![a.clone(), a.clone()], vec![a.sim(), b.sim()]),
        (
            vec![a.clone(), b.clone()],
            vec![fresh.clone(), fresh.clone()],
        ),
        (vec![a.clone()], vec![b.clone()]),
        (vec![a.clone()], vec![DynIndex::new_dyn(3)]),
        (vec![fresh.clone()], vec![fresh.sim()]),
    ] {
        assert!(source.replace_indices(&old, &new).is_err());
        assert_eq!(
            source
                .to_dense()
                .unwrap()
                .sub(&before)
                .unwrap()
                .maxabs()
                .unwrap(),
            0.0
        );
    }
    assert_eq!(
        source
            .replace_indices(&[], &[])
            .unwrap()
            .to_dense()
            .unwrap()
            .sub(&before)
            .unwrap()
            .maxabs()
            .unwrap(),
        0.0
    );
}
