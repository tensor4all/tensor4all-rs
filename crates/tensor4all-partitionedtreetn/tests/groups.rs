use std::sync::Mutex;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

/// Build a 2-site complex partition with two groups on `site_g`:
/// - `site_g = 0`: 2 patches (`site_s = 0`, `site_s = 1`)
/// - `site_g = 1`: 3 patches (`site_s = 0`, `site_s = 1`, `site_s = 2`)
fn make_grouped_complex_partition(
    site_g: &DynIndex,
    site_s: &DynIndex,
    pairs: &[(usize, usize)],
) -> PartitionedTreeTN<usize> {
    let bond = DynIndex::new_dyn(2);
    let patches = pairs
        .iter()
        .map(|&(g, s)| {
            let offset = (g * 3 + s) as f64;
            let t0 = IdxTensor::from_dense(
                vec![site_g.clone(), bond.clone()],
                vec![
                    Complex64::new(1.0 + offset, 0.5),
                    Complex64::new(-0.5, 1.0 + offset),
                    Complex64::new(0.25 + offset, -0.75),
                    Complex64::new(2.0, 0.25 + offset),
                ],
            )
            .unwrap();
            let t1 = IdxTensor::from_dense(
                vec![bond.clone(), site_s.clone()],
                vec![
                    Complex64::new(0.5 + offset, -0.25),
                    Complex64::new(1.0, 0.75 + offset),
                    Complex64::new(-1.0 + offset, 0.5),
                    Complex64::new(0.25, -1.5 + offset),
                    Complex64::new(1.5 - offset, 0.25),
                    Complex64::new(-0.75, 2.0 - offset),
                ],
            )
            .unwrap();
            let tree = TreeTN::from_tensors(vec![t0, t1], vec![0usize, 1usize]).unwrap();
            let proj = Projector::from_pairs([(site_g.clone(), g), (site_s.clone(), s)]).unwrap();
            SubDomainTreeTN::new(tree, proj).unwrap()
        })
        .collect();
    PartitionedTreeTN::from_subdomains(patches).unwrap()
}

#[test]
fn visit_each_projector_group_once() {
    let site_g = DynIndex::new_dyn(2);
    let site_s = DynIndex::new_dyn(3);
    // Scrambled insertion order: 2 patches in group 0, 3 patches in group 1.
    let pairs = [(1, 2), (0, 1), (1, 0), (0, 0), (1, 1)];
    let partition = make_grouped_complex_partition(&site_g, &site_s, &pairs);
    let before_dense = partition.to_treetn().unwrap().to_dense().unwrap();

    let visited_groups: Mutex<Vec<Projector>> = Mutex::new(Vec::new());
    let visited_sizes: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let visited_members: Mutex<Vec<Vec<Projector>>> = Mutex::new(Vec::new());

    let merged = partition
        .try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
            visited_groups.lock().unwrap().push(group_proj.clone());
            visited_sizes.lock().unwrap().push(group.len());
            visited_members
                .lock()
                .unwrap()
                .push(group.iter().map(|p| p.projector().clone()).collect());

            // Merge the group into one patch with the group's projector.
            let mut sum = group[0].data().clone();
            for patch in &group[1..] {
                sum = sum
                    .add(patch.data())
                    .map_err(PartitionedTreeTNError::from)?;
            }
            Ok(vec![SubDomainTreeTN::new(sum, group_proj.clone())?])
        })
        .unwrap();

    let visited_group_sizes = visited_sizes.into_inner().unwrap();
    assert_eq!(visited_group_sizes, vec![2, 3]);

    let expected_group_projectors = vec![
        Projector::from_pairs([(site_g.clone(), 0)]).unwrap(),
        Projector::from_pairs([(site_g.clone(), 1)]).unwrap(),
    ];
    assert_eq!(
        visited_groups.into_inner().unwrap(),
        expected_group_projectors
    );

    let output_projectors: Vec<Projector> = merged.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(output_projectors, expected_group_projectors);

    // Within each group, patches were visited in sorted projector order.
    let members = visited_members.into_inner().unwrap();
    assert_eq!(
        members[0],
        vec![
            Projector::from_pairs([(site_g.clone(), 0), (site_s.clone(), 0)]).unwrap(),
            Projector::from_pairs([(site_g.clone(), 0), (site_s.clone(), 1)]).unwrap(),
        ]
    );
    assert_eq!(
        members[1],
        vec![
            Projector::from_pairs([(site_g.clone(), 1), (site_s.clone(), 0)]).unwrap(),
            Projector::from_pairs([(site_g.clone(), 1), (site_s.clone(), 1)]).unwrap(),
            Projector::from_pairs([(site_g.clone(), 1), (site_s.clone(), 2)]).unwrap(),
        ]
    );

    // Merged dense values match the original partition.
    let merged_dense = merged.to_treetn().unwrap().to_dense().unwrap();
    let diff = merged_dense.sub(&before_dense).unwrap().maxabs().unwrap();
    assert!(diff < 1e-12, "merged dense mismatch: {diff}");
}

#[test]
fn try_map_patch_groups_empty_group_output_drops_group() {
    let site_g = DynIndex::new_dyn(2);
    let site_s = DynIndex::new_dyn(3);
    let pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)];
    let partition = make_grouped_complex_partition(&site_g, &site_s, &pairs);

    // Drop group 0, keep group 1's patches as-is.
    let filtered = partition
        .try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
            if group_proj.get(&site_g) == Some(0) {
                Ok(Vec::new())
            } else {
                Ok(group.iter().map(|&p| p.clone()).collect())
            }
        })
        .unwrap();
    assert_eq!(filtered.len(), 3);
    for (p, _) in filtered.iter() {
        assert_eq!(p.get(&site_g), Some(1));
    }

    // Dropping all groups produces an empty partition.
    let all_dropped = partition
        .try_map_patch_groups(&[site_g.clone()], |_, _| Ok(Vec::new()))
        .unwrap();
    assert!(all_dropped.is_empty());
}

#[test]
fn try_map_patch_groups_empty_group_by_yields_single_sorted_group() {
    let site_g = DynIndex::new_dyn(2);
    let site_s = DynIndex::new_dyn(3);
    let pairs = [(1, 2), (0, 1), (1, 0), (0, 0), (1, 1)];
    let partition = make_grouped_complex_partition(&site_g, &site_s, &pairs);
    let expected_order: Vec<Projector> = partition.iter().map(|(p, _)| p.clone()).collect();

    let calls: Mutex<Vec<(Projector, Vec<Projector>)>> = Mutex::new(Vec::new());
    let out = partition
        .try_map_patch_groups(&[], |group_proj, group| {
            calls.lock().unwrap().push((
                group_proj.clone(),
                group.iter().map(|p| p.projector().clone()).collect(),
            ));
            Ok(group.iter().map(|&p| p.clone()).collect())
        })
        .unwrap();

    let recorded = calls.into_inner().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, Projector::new());
    assert_eq!(recorded[0].1, expected_order);
    assert_eq!(out.len(), 5);
}

#[test]
fn try_map_patch_groups_deterministic_order() {
    let site_g = DynIndex::new_dyn(2);
    let site_s = DynIndex::new_dyn(3);
    let p_a = make_grouped_complex_partition(&site_g, &site_s, &[(0, 0), (0, 1), (1, 0), (1, 1)]);
    let p_b = make_grouped_complex_partition(&site_g, &site_s, &[(1, 1), (0, 1), (1, 0), (0, 0)]);

    let record = |p: &PartitionedTreeTN<usize>| -> Vec<(Projector, Vec<Projector>)> {
        let seen = Mutex::new(Vec::new());
        let _ = p
            .try_map_patch_groups(&[site_g.clone()], |gp, group| {
                seen.lock().unwrap().push((
                    gp.clone(),
                    group
                        .iter()
                        .map(|patch| patch.projector().clone())
                        .collect(),
                ));
                Ok(group.iter().map(|&patch| patch.clone()).collect())
            })
            .unwrap();
        seen.into_inner().unwrap()
    };

    assert_eq!(record(&p_a), record(&p_b));
}

#[test]
fn try_map_patch_groups_rejects_disagreeing_duplicate_and_malformed_outputs() {
    let site_g = DynIndex::new_dyn(2);
    let site_s = DynIndex::new_dyn(3);
    let pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)];
    let partition = make_grouped_complex_partition(&site_g, &site_s, &pairs);
    let before_projectors: Vec<Projector> = partition.iter().map(|(p, _)| p.clone()).collect();
    let before_dense = partition.to_treetn().unwrap().to_dense().unwrap();

    // 1. Returned patch projector disagrees with group_proj on group_by -> ProjectorMismatch.
    let disagree_err = partition
        .try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
            if group_proj.get(&site_g) == Some(0) {
                let wrong_proj = Projector::from_pairs([(site_g.clone(), 1), (site_s.clone(), 0)])?;
                Ok(vec![SubDomainTreeTN::new(
                    group[0].data().clone(),
                    wrong_proj,
                )?])
            } else {
                Ok(group.iter().map(|&p| p.clone()).collect())
            }
        })
        .unwrap_err();
    assert!(
        matches!(disagree_err, PartitionedTreeTNError::ProjectorMismatch),
        "expected ProjectorMismatch, got {disagree_err:?}"
    );

    // 2. Duplicate projector returned -> OverlappingProjectors (never a silent overwrite).
    let dup_err = partition
        .try_map_patch_groups(&[site_g.clone()], |_group_proj, group| {
            Ok(vec![group[0].clone(), group[0].clone()])
        })
        .unwrap_err();
    assert!(
        matches!(dup_err, PartitionedTreeTNError::OverlappingProjectors),
        "expected OverlappingProjectors, got {dup_err:?}"
    );

    // 3. Mismatched topology across groups -> TopologyMismatch.
    let topo_err = partition
        .try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
            if group_proj.get(&site_g) == Some(1) {
                let single = IdxTensor::from_dense(
                    vec![site_g.clone()],
                    vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                )
                .unwrap();
                let tree = TreeTN::from_tensors(vec![single], vec![0usize]).unwrap();
                Ok(vec![SubDomainTreeTN::new(tree, group_proj.clone())?])
            } else {
                Ok(group.iter().map(|&p| p.clone()).collect())
            }
        })
        .unwrap_err();
    assert!(
        matches!(topo_err, PartitionedTreeTNError::TopologyMismatch),
        "expected TopologyMismatch, got {topo_err:?}"
    );

    // 4. Mismatched site index across groups -> SiteIndexMismatch.
    let other_s = DynIndex::new_dyn(3);
    let site_err = partition
        .try_map_patch_groups(&[site_g.clone()], |group_proj, group| {
            if group_proj.get(&site_g) == Some(1) {
                let new_tree = group[0]
                    .data()
                    .replaceind(&site_s, &other_s)
                    .map_err(PartitionedTreeTNError::from)?;
                Ok(vec![SubDomainTreeTN::new(new_tree, group_proj.clone())?])
            } else {
                Ok(group.iter().map(|&p| p.clone()).collect())
            }
        })
        .unwrap_err();
    assert!(
        matches!(site_err, PartitionedTreeTNError::SiteIndexMismatch),
        "expected SiteIndexMismatch, got {site_err:?}"
    );

    // 5. Callback error propagates directly.
    let cb_err = partition
        .try_map_patch_groups(&[site_g.clone()], |_, _| {
            Err(PartitionedTreeTNError::InvalidCenter)
        })
        .unwrap_err();
    assert!(matches!(cb_err, PartitionedTreeTNError::InvalidCenter));

    // Input partition is completely untouched after all error cases.
    let after_projectors: Vec<Projector> = partition.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(after_projectors, before_projectors);
    let after_dense = partition.to_treetn().unwrap().to_dense().unwrap();
    let untouched_diff = after_dense.sub(&before_dense).unwrap().maxabs().unwrap();
    assert_eq!(untouched_diff, 0.0);
}

#[test]
fn try_map_patch_groups_empty_partition() {
    let empty: PartitionedTreeTN<usize> = PartitionedTreeTN::new();
    let site = DynIndex::new_dyn(2);
    let out = empty
        .try_map_patch_groups(&[site], |_, _| Err(PartitionedTreeTNError::InvalidCenter))
        .unwrap();
    assert!(out.is_empty());
}
