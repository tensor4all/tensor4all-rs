use super::*;

fn mixed(selected: &DynIndex, spectator: &DynIndex) -> Vec<Projector> {
    vec![
        Projector::from_pairs([(selected.clone(), 0), (spectator.clone(), 0)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 0), (spectator.clone(), 1)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 1)]).unwrap(),
    ]
}

#[test]
fn refinement_is_canonical_and_preserves_full_index_identity() {
    let selected = DynIndex::new_dyn(2);
    let spectator = selected.prime();
    let patches = mixed(&selected, &spectator);
    let mut first = ScheduleGeometry::new(std::slice::from_ref(&selected), None).unwrap();
    assert_eq!(
        prepare(patches.iter(), &mut first, CoverageContract::Complete, 4, 4).unwrap(),
        4
    );
    assert_eq!(
        first.spectators,
        vec![
            Projector::from_pairs([(spectator.clone(), 0)]).unwrap(),
            Projector::from_pairs([(spectator, 1)]).unwrap(),
        ]
    );
    assert_eq!(first.source_groups, vec![vec![0], vec![1], vec![0, 1]]);
    let mut reversed = ScheduleGeometry::new(&[selected], None).unwrap();
    assert_eq!(
        prepare(
            patches.iter().rev(),
            &mut reversed,
            CoverageContract::Complete,
            4,
            4
        )
        .unwrap(),
        4
    );
    assert_eq!(first.spectators, reversed.spectators);
    assert_eq!(
        first.source_groups.into_iter().rev().collect::<Vec<_>>(),
        reversed.source_groups,
    );
}

#[test]
fn refined_leaf_limits_are_checked_for_work_and_terms() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let patches = mixed(&selected, &spectator);
    for (work, terms, expected) in [(3, 4, "max_work_items"), (4, 3, "max_terms")] {
        let mut geometry = ScheduleGeometry::new(std::slice::from_ref(&selected), None).unwrap();
        assert!(
            matches!(prepare(patches.iter(), &mut geometry, CoverageContract::Complete, work, terms),
            Err(PartitionedTreeTNError::ResourceLimit { limit, value: 4, .. }) if limit == expected)
        );
    }
}

#[test]
fn varying_grid_is_refused_before_enumerating_large_dimensions() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(usize::MAX);
    let patches = [
        Projector::from_pairs([(selected.clone(), 0), (spectator, 0)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 1)]).unwrap(),
    ];
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert!(matches!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::ZeroForMissingLeaves,
            8,
            8
        ),
        Err(PartitionedTreeTNError::RefinementTooLarge {
            atoms: usize::MAX,
            limit: 8
        })
    ));
}

#[test]
fn disjoint_cells_do_not_enumerate_unused_spectator_coordinates() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(usize::MAX);
    let patches: Vec<_> = [0, 1]
        .into_iter()
        .flat_map(|s| {
            let selected = &selected;
            let spectator = &spectator;
            [0, 1].into_iter().map(move |k| {
                Projector::from_pairs([(selected.clone(), k), (spectator.clone(), s)]).unwrap()
            })
        })
        .collect();
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert_eq!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::Complete,
            4,
            4
        )
        .unwrap(),
        4
    );
    assert_eq!(geometry.spectators.len(), 2);
}

#[test]
fn common_fixed_dimensions_are_not_part_of_the_grid() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let common = DynIndex::new_dyn(usize::MAX);
    let mut patches = mixed(&selected, &spectator);
    for patch in &mut patches {
        patch.insert(common.clone(), 17).unwrap();
    }
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert_eq!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::Complete,
            4,
            4
        )
        .unwrap(),
        4
    );
    assert!(geometry
        .spectators
        .iter()
        .all(|cell| cell.get(&common) == Some(17)));
}

#[test]
fn common_grid_omits_cells_outside_the_spectator_support_union() {
    let selected = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let t = DynIndex::new_dyn(2);
    let patches = [
        Projector::from_pairs([(selected.clone(), 0), (s.clone(), 0)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 1), (t.clone(), 0)]).unwrap(),
    ];
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert_eq!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::ZeroForMissingLeaves,
            4,
            4
        )
        .unwrap(),
        4
    );
    assert_eq!(geometry.spectators.len(), 3);
    assert!(geometry
        .spectators
        .iter()
        .all(|cell| !(cell.get(&s) == Some(1) && cell.get(&t) == Some(1))));
}

#[test]
fn a_dimension_one_spectator_refines_without_creating_extra_leaves() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(1);
    let patches = [
        Projector::from_pairs([(selected.clone(), 0), (spectator.clone(), 0)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 1)]).unwrap(),
    ];
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert_eq!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::Complete,
            2,
            2
        )
        .unwrap(),
        2
    );
    assert_eq!(
        geometry.spectators,
        vec![Projector::from_pairs([(spectator, 0)]).unwrap()]
    );
}

#[test]
fn complete_coverage_is_checked_per_cell_and_sparse_coverage_is_explicit() {
    let selected = DynIndex::new_dyn(2);
    let spectator = DynIndex::new_dyn(2);
    let patches = [
        Projector::from_pairs([(selected.clone(), 0), (spectator.clone(), 0)]).unwrap(),
        Projector::from_pairs([(selected.clone(), 1), (spectator, 1)]).unwrap(),
    ];
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    assert!(prepare(
        patches.iter(),
        &mut geometry,
        CoverageContract::Complete,
        2,
        2
    )
    .unwrap_err()
    .to_string()
    .contains("cover every selected-coordinate"));
    assert_eq!(
        prepare(
            patches.iter(),
            &mut geometry,
            CoverageContract::ZeroForMissingLeaves,
            2,
            2
        )
        .unwrap(),
        2
    );
}

#[test]
fn trie_validation_rejects_nested_selected_prefixes_with_different_raw_spectators() {
    let selected = DynIndex::new_dyn(2);
    let spectator = selected.prime();
    let patches = [
        Projector::from_pairs([(selected.clone(), 0), (spectator, 0)]).unwrap(),
        Projector::new(),
    ];
    let mut geometry = ScheduleGeometry::new(&[selected], None).unwrap();
    let error = prepare(
        patches.iter(),
        &mut geometry,
        CoverageContract::ZeroForMissingLeaves,
        4,
        4,
    )
    .unwrap_err();
    assert!(error.to_string().contains("input leaves must not overlap"));
}
