use std::collections::BTreeSet;

use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

fn proj(pairs: &[(&DynIndex, usize)]) -> anyhow::Result<Projector> {
    Ok(Projector::from_pairs(
        pairs
            .iter()
            .map(|(index, value)| ((*index).clone(), *value)),
    )?)
}

fn two_patch_fixture() -> anyhow::Result<(PartitionedTreeTN<usize>, DynIndex, DynIndex)> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);

    // Deterministic complex tensors on a 2-node TreeTN with bond dimension 2.
    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![
            Complex64::new(1.25, -0.5),
            Complex64::new(-0.75, 1.5),
            Complex64::new(0.5, 2.0),
            Complex64::new(1.75, -1.25),
        ],
    )?;
    let t1 = IdxTensor::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![
            Complex64::new(0.25, 1.0),
            Complex64::new(-1.5, 0.75),
            Complex64::new(2.25, -0.5),
            Complex64::new(0.5, 1.75),
        ],
    )?;
    let tree_a = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize])?;

    let bond_b = DynIndex::new_dyn(2);
    let t0_b = IdxTensor::from_dense(
        vec![s0.clone(), bond_b.clone()],
        vec![
            Complex64::new(-0.5, 0.25),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.5, 0.5),
            Complex64::new(-1.25, 2.5),
        ],
    )?;
    let t1_b = IdxTensor::from_dense(
        vec![bond_b, s1.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.75, -2.0),
            Complex64::new(-0.25, 1.25),
            Complex64::new(1.5, -0.75),
        ],
    )?;
    let tree_b = TreeTN::from_tensors(vec![t0_b, t1_b], vec![0usize, 1usize])?;

    // Patch 1: fixed on s0 = 0, free on s1.
    let patch_1 = SubDomainTreeTN::new(tree_a, proj(&[(&s0, 0)])?)?;
    // Patch 2: fixed on s0 = 1, s1 = 0. Point (s0 = 1, s1 = 1) is uncovered.
    let patch_2 = SubDomainTreeTN::new(tree_b, proj(&[(&s0, 1), (&s1, 0)])?)?;

    let partition = PartitionedTreeTN::from_subdomains(vec![patch_1, patch_2])?;
    Ok((partition, s0, s1))
}

fn hand_written_restrict(
    partition: &PartitionedTreeTN<usize>,
    selection: &[Projector],
) -> anyhow::Result<PartitionedTreeTN<usize>> {
    let mut patches = Vec::new();
    for patch in partition.values() {
        for selected in selection {
            if let Some(projected) = patch.project(selected)? {
                patches.push(projected);
            }
        }
    }
    Ok(PartitionedTreeTN::from_subdomains(patches)?)
}

fn points_of(selection: &[Projector], s0: &DynIndex, s1: &DynIndex) -> Vec<[usize; 2]> {
    let mut points = BTreeSet::new();
    for p in selection {
        let r0: Vec<usize> = match p.get(s0) {
            Some(v) => vec![v],
            None => (0..s0.dim).collect(),
        };
        let r1: Vec<usize> = match p.get(s1) {
            Some(v) => vec![v],
            None => (0..s1.dim).collect(),
        };
        for &v0 in &r0 {
            for &v1 in &r1 {
                points.insert([v0, v1]);
            }
        }
    }
    points.into_iter().collect()
}

fn eval_points(
    partition: &PartitionedTreeTN<usize>,
    s0: &DynIndex,
    s1: &DynIndex,
    points: &[[usize; 2]],
) -> anyhow::Result<Vec<tensor4all_core::AnyScalar>> {
    let mut flat = Vec::with_capacity(points.len() * 2);
    for &[v0, v1] in points {
        flat.push(v0);
        flat.push(v1);
    }
    let shape = [2usize, points.len()];
    let array = ColMajorArrayRef::new(&flat, &shape)?;
    Ok(partition.evaluate_batched(&[s0.clone(), s1.clone()], array)?)
}

#[test]
fn restrict_is_pairwise_and_matches_the_hand_written_loop() -> anyhow::Result<()> {
    let (partition, s0, s1) = two_patch_fixture()?;

    // 1. Split case: one patch, two disjoint selected projectors -> two patches, pairwise disjoint.
    let selection = [proj(&[(&s0, 0), (&s1, 0)])?, proj(&[(&s0, 0), (&s1, 1)])?];
    let restricted = partition.restrict(&selection)?;
    let restricted_projectors: Vec<Projector> = restricted.projectors().cloned().collect();
    assert_eq!(restricted_projectors, selection.to_vec());
    assert!(Projector::are_disjoint(&restricted_projectors));

    // Bit-identical to the hand-written loop on every selected point.
    let by_hand = hand_written_restrict(&partition, &selection)?;
    let selected_points = points_of(&selection, &s0, &s1);
    let restricted_vals = eval_points(&restricted, &s0, &s1, &selected_points)?;
    let by_hand_vals = eval_points(&by_hand, &s0, &s1, &selected_points)?;
    assert_eq!(restricted_vals, by_hand_vals);

    // 2. Patch coarser than the selection on s1 and finer on s0: intersection on both patches.
    let mixed_selection = [proj(&[(&s1, 0)])?];
    let mixed = partition.restrict(&mixed_selection)?;
    let mixed_projectors: Vec<Projector> = mixed.projectors().cloned().collect();
    assert_eq!(
        mixed_projectors,
        vec![proj(&[(&s0, 0), (&s1, 0)])?, proj(&[(&s0, 1), (&s1, 0)])?]
    );
    assert!(Projector::are_disjoint(&mixed_projectors));
    let mixed_by_hand = hand_written_restrict(&partition, &mixed_selection)?;
    let mixed_points = points_of(&mixed_selection, &s0, &s1);
    assert_eq!(
        eval_points(&mixed, &s0, &s1, &mixed_points)?,
        eval_points(&mixed_by_hand, &s0, &s1, &mixed_points)?
    );

    // 3. Valid empty partition from empty selection, and from empty source + empty selection.
    let empty = partition.restrict(&[])?;
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(PartitionedTreeTN::<usize>::new().restrict(&[])?.is_empty());

    // 4. Refusal of overlapping selected projectors.
    let overlap_err = partition
        .restrict(&[selection[0].clone(), proj(&[(&s0, 0)])?])
        .unwrap_err();
    assert!(matches!(
        overlap_err,
        PartitionedTreeTNError::OverlappingProjectors
    ));

    // 5. The source has coverage `None` (defined everywhere), so a selection
    //    with no stored patch is a known zero, not an error (the merged
    //    version refused it).
    //    (a) region with no patch (s0 = 1, s1 = 1): known zero, no patch kept
    let zero_cell = proj(&[(&s0, 1), (&s1, 1)])?;
    let known_zero = partition.restrict(std::slice::from_ref(&zero_cell))?;
    assert!(known_zero.is_empty());
    assert_eq!(known_zero.coverage(), Some(&[zero_cell][..]));

    //    (b) partially patched region (s0 = 1 holds s1 = 0 only): the stored
    //        part is kept and the whole selection is declared known
    let partial_sel = proj(&[(&s0, 1)])?;
    let partial = partition.restrict(std::slice::from_ref(&partial_sel))?;
    assert_eq!(partial.len(), 1);
    assert_eq!(partial.coverage(), Some(&[partial_sel][..]));

    //    (c) index absent from the TreeTN site space
    let unknown = DynIndex::new_dyn(2);
    let unknown_err = partition.restrict(&[proj(&[(&unknown, 0)])?]).unwrap_err();
    assert!(matches!(
        unknown_err,
        PartitionedTreeTNError::ProjectorIndexNotFound { .. }
    ));

    //    (d) empty source partition (coverage `None`): every selection is a known zero
    let empty_source = PartitionedTreeTN::<usize>::new().restrict(&[selection[0].clone()])?;
    assert!(empty_source.is_empty());
    assert_eq!(empty_source.coverage(), Some(&selection[..1]));

    // Source partition is unchanged.
    assert_eq!(partition.len(), 2);
    Ok(())
}
