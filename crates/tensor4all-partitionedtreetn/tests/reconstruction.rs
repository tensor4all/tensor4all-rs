use num_complex::Complex64;
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, TensorElement};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, PartitionedTreeTNError, PatchSplitStrategy, Projector,
    SubDomainTreeTN, TreeTN,
};

fn diagonal<T: TensorElement + From<f64>>(weights: &[T]) -> SubDomainTreeTN {
    let n = weights.len();
    let site0 = DynIndex::new_dyn(n);
    let site1 = DynIndex::new_dyn(n);
    let bond = DynIndex::new_dyn(n);
    let left = (0..n * n)
        .map(|i| T::from(if i % (n + 1) == 0 { 1.0 } else { 0.0 }))
        .collect();
    let right = (0..n * n)
        .map(|i| {
            if i % (n + 1) == 0 {
                weights[i / n]
            } else {
                T::from(0.0)
            }
        })
        .collect();
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![site0, bond.clone()], left).unwrap(),
            IdxTensor::from_dense(vec![bond, site1], right).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    SubDomainTreeTN::from_treetn(tree).unwrap()
}

fn target(patch: &SubDomainTreeTN) -> ReconstructionTarget {
    ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomain(patch.clone()).unwrap())
        .unwrap()
}

fn result_dense(
    output: &ReconstructedTreeTN,
    zero_template: &TreeTN<IdxTensor, usize>,
) -> IdxTensor {
    let mut iter = output.regions().flat_map(|(_, terms)| terms.iter());
    let mut tree = if let Some(first) = iter.next() {
        first.data().clone()
    } else {
        zero_template.scale(AnyScalar::new_real(0.0)).unwrap()
    };
    for term in iter {
        tree = tree.add(term.data()).unwrap();
    }
    tree.to_dense().unwrap()
}

fn check_residual(original: &SubDomainTreeTN, output: &ReconstructedTreeTN, roundoff: f64) {
    let expected = original.data().clone().to_dense().unwrap();
    let actual = result_dense(output, original.data());
    let difference = actual.sub(&expected).unwrap();
    let residual = difference.norm().unwrap();
    assert!(
        residual <= output.report().error_bound + roundoff,
        "residual {residual}, bound {}",
        output.report().error_bound
    );
    assert!(output.report().error_bound <= output.report().absolute_tolerance);
    let projectors: Vec<_> = output.regions().map(|(p, _)| p.clone()).collect();
    assert!(Projector::are_disjoint(&projectors));
    for (region, terms) in output.regions() {
        assert!(terms.iter().all(|term| term.projector() == region));
    }
}

fn split_case<T: TensorElement + From<f64>>() {
    let original = diagonal(&[T::from(1.0), T::from(2.0), T::from(3.0)]);
    let prepared = target(&original);
    let result = reconstruct(
        &prepared,
        &0,
        ReconstructionTolerance {
            rtol: 1e-8,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.report().region_count, 3);
    assert_eq!(result.report().max_bond_dim, 1);
    assert_eq!(result.report().split_count, 1);
    assert!((prepared.reference_scale() - 14.0_f64.sqrt()).abs() < 1e-12);
    check_residual(&original, &result, 1e-11);
    assert_eq!(result.into_partition().unwrap().len(), 3);
}

#[test]
fn split_real() {
    split_case::<f64>();
}

#[test]
fn split_complex() {
    split_case::<Complex64>();
}

#[test]
fn rank_goal_and_region_limit_never_override_accuracy() {
    let original = diagonal(&[1.0, 2.0, 3.0]);
    for limit in [1, 2] {
        let result = reconstruct(
            &target(&original),
            &1,
            Default::default(),
            &ReconstructionOptions {
                target_bond_dim: Some(1),
                max_regions: limit,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.report().region_count, 1);
        assert_eq!(result.report().max_bond_dim, 3);
        check_residual(&original, &result, 1e-11);
    }
}

#[test]
fn sequential_patches_msb_at_right_end_of_reversed_bit_layout() {
    // T(r2, r1) = delta(r2, r1), with the MSB at the right-hand node.
    // This tests reconstruction geometry, not QFT application.
    let original = diagonal(&[1.0, 2.0]);
    let r2 = original
        .data()
        .site_space(&0)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();
    let r1 = original
        .data()
        .site_space(&1)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();
    for patch_order in [vec![r1.clone(), r2.clone()], vec![r2.clone(), r1.clone()]] {
        let output = reconstruct(
            &target(&original),
            &0,
            Default::default(),
            &ReconstructionOptions {
                target_bond_dim: Some(1),
                patch_order: patch_order.clone(),
                split_strategy: PatchSplitStrategy::Sequential,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.report().split_count, 1);
        assert_eq!(output.report().region_count, 2);
        assert_eq!(output.report().max_bond_dim, 1);
        for (projector, terms) in output.regions() {
            assert!(projector.is_projected_at(&patch_order[0]));
            assert!(!projector.is_projected_at(&patch_order[1]));
            for term in terms {
                assert_eq!(term.data().site_space(&0), original.data().site_space(&0));
                assert_eq!(term.data().site_space(&1), original.data().site_space(&1));
            }
        }
        check_residual(&original, &output, 1e-12);
    }
}

#[test]
fn sequential_does_not_skip_first_index_when_region_capacity_is_insufficient() {
    let first = DynIndex::new_dyn(3);
    let second = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![first.clone(), bond.clone()],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, second.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let original = SubDomainTreeTN::from_treetn(tree).unwrap();
    for split_strategy in [
        PatchSplitStrategy::Sequential,
        PatchSplitStrategy::ExactParameterGain,
    ] {
        let output = reconstruct(
            &target(&original),
            &0,
            Default::default(),
            &ReconstructionOptions {
                target_bond_dim: Some(1),
                patch_order: vec![first.clone(), second.clone()],
                split_strategy,
                max_regions: 2,
            },
        )
        .unwrap();
        let sequential = split_strategy == PatchSplitStrategy::Sequential;
        assert_eq!(output.report().split_count, usize::from(!sequential));
        assert_eq!(output.report().max_bond_dim, if sequential { 2 } else { 1 });
        for (projector, _) in output.regions() {
            assert!(!projector.is_projected_at(&first));
            assert_eq!(projector.is_projected_at(&second), !sequential);
        }
        check_residual(&original, &output, 1e-12);
    }
}

#[test]
fn global_reference_is_pinned_when_small_patch_is_dropped() {
    let site = DynIndex::new_dyn(2);
    let tiny = 1e-7;
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, tiny]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let original = SubDomainTreeTN::from_treetn(tree.clone()).unwrap();
    let mut partition = PartitionedTreeTN::from_subdomains(
        (0..2)
            .map(|i| {
                SubDomainTreeTN::new(
                    tree.clone(),
                    Projector::from_pairs([(site.clone(), i)]).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let prepared = ReconstructionTarget::from_partition(&partition).unwrap();
    // Later caller mutation cannot change the snapshotted target.
    partition = PartitionedTreeTN::new();
    assert_eq!(partition.len(), 0);
    let output = reconstruct(
        &prepared,
        &0,
        ReconstructionTolerance {
            rtol: 1e-5,
            atol: 0.0,
        },
        &Default::default(),
    )
    .unwrap();
    assert_eq!(output.report().term_count, 1);
    assert!((output.report().reference_scale - 1.0_f64.hypot(tiny)).abs() < 1e-14);
    assert!((output.report().error_bound - tiny).abs() < 1e-14);
    check_residual(&original, &output, 1e-14);
}

#[test]
fn no_merge_gain_preserves_superposition() {
    let original = diagonal(&[1.0, 1.0]);
    let site = original
        .data()
        .site_space(&0)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();
    let patches = (0..2)
        .map(|i| {
            original
                .project(&Projector::from_pairs([(site.clone(), i)]).unwrap())
                .unwrap()
                .unwrap()
        })
        .collect();
    let partition = PartitionedTreeTN::from_subdomains(patches).unwrap();
    let prepared = ReconstructionTarget::from_partition(&partition).unwrap();
    let output = reconstruct(&prepared, &0, Default::default(), &Default::default()).unwrap();
    assert_eq!(output.report().region_count, 1);
    assert_eq!(output.report().term_count, 2);
    assert_eq!(output.report().max_bond_dim, 1);
    assert_eq!(output.report().merge_count, 0);
    check_residual(&original, &output, 1e-12);
    assert!(output.into_partition().is_err());
}

#[test]
fn productive_merges_recover_constant_from_nonuniform_patches() {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let links: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(1)).collect();
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![sites[0].clone(), links[0].clone()], vec![1.0; 2]).unwrap(),
            IdxTensor::from_dense(
                vec![links[0].clone(), sites[1].clone(), links[1].clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![links[1].clone(), sites[2].clone()], vec![1.0; 2]).unwrap(),
        ],
        vec![0usize, 1, 2],
    )
    .unwrap();
    let original = SubDomainTreeTN::from_treetn(tree).unwrap();
    let supports = vec![
        vec![(sites[0].clone(), 0)],
        vec![(sites[0].clone(), 1), (sites[1].clone(), 0)],
        vec![(sites[0].clone(), 1), (sites[1].clone(), 1)],
    ];
    let partition = PartitionedTreeTN::from_subdomains(
        supports
            .into_iter()
            .map(|pairs| {
                original
                    .project(&Projector::from_pairs(pairs).unwrap())
                    .unwrap()
                    .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let prepared = ReconstructionTarget::from_partition(&partition).unwrap();
    let output = reconstruct(
        &prepared,
        &1,
        Default::default(),
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    // Deeper siblings merge before their coarser neighbor.
    assert_eq!(output.report().split_count, 0);
    assert_eq!(output.report().term_count, 1);
    assert_eq!(output.report().merge_count, 2);
    assert_eq!(output.report().reference_scale, prepared.reference_scale());
    check_residual(&original, &output, 1e-11);
}

#[test]
fn zero_tolerance_does_not_compress_or_force_rank() {
    let original = diagonal(&[1.0, 1e-10]);
    let output = reconstruct(
        &target(&original),
        &0,
        ReconstructionTolerance {
            rtol: 0.0,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            max_regions: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.report().error_bound, 0.0);
    assert_eq!(output.report().max_bond_dim, 2);
    check_residual(&original, &output, 0.0);
}

#[test]
fn absolute_tolerance_and_zero_targets() {
    let original = diagonal(&[1e-10, 0.0]);
    let output = reconstruct(
        &target(&original),
        &0,
        ReconstructionTolerance {
            rtol: 0.0,
            atol: 1e-8,
        },
        &Default::default(),
    )
    .unwrap();
    assert_eq!(output.report().term_count, 0);
    assert_eq!(output.report().error_bound, output.report().reference_scale);
    check_residual(&original, &output, 1e-20);
    let zero = diagonal(&[0.0, 0.0]);
    let output = reconstruct(&target(&zero), &0, Default::default(), &Default::default()).unwrap();
    assert_eq!(output.report().reference_scale, 0.0);
    assert_eq!(output.report().term_count, 0);
}

#[test]
fn product_target_has_factor_norm_and_correct_outer_product() {
    let left = diagonal(&[Complex64::new(1.0, 2.0), Complex64::new(0.0, 1.0)]);
    let right = diagonal(&[Complex64::new(3.0, 0.0), Complex64::new(0.0, 4.0)]);
    let prepared =
        ReconstructionTarget::from_tensor_products(vec![(left.clone(), right.clone())]).unwrap();
    assert!((prepared.reference_scale() - 150.0_f64.sqrt()).abs() < 1e-12);
    let output = reconstruct(
        &prepared,
        &0,
        Default::default(),
        &ReconstructionOptions {
            target_bond_dim: None,
            ..Default::default()
        },
    )
    .unwrap();
    let a = left.data().clone().to_dense().unwrap();
    let b = right.data().clone().to_dense().unwrap();
    let av = a.to_vec::<Complex64>().unwrap();
    let bv = b.to_vec::<Complex64>().unwrap();
    let indices = a.indices().iter().chain(b.indices()).cloned().collect();
    // Small independent dense oracle; first (left) index varies fastest.
    let data = bv
        .iter()
        .flat_map(|y| av.iter().map(move |x| x * y))
        .collect();
    let expected = IdxTensor::from_dense(indices, data).unwrap();
    let actual = result_dense(&output, left.data());
    let residual = actual.sub(&expected).unwrap().norm().unwrap();
    assert!(residual < 1e-10, "outer product residual: {residual}");
    assert_eq!(output.report().reference_scale, prepared.reference_scale());
}

#[test]
fn product_target_rejects_overlaps_shared_indices_and_mixed_dtypes() {
    let a = diagonal(&[1.0, 2.0]);
    let b = diagonal(&[3.0, 4.0]);
    assert!(matches!(
        ReconstructionTarget::from_tensor_products(vec![(a.clone(), a.clone())]),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        ReconstructionTarget::from_tensor_products(vec![(a.clone(), b.clone()), (a.clone(), b)]),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    let c = diagonal(&[Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)]);
    assert!(matches!(
        ReconstructionTarget::from_tensor_products(vec![(a, c)]),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
}

#[test]
fn validates_options_before_empty_and_zero_shortcuts() {
    let empty = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new()).unwrap();
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        for tolerance in [
            ReconstructionTolerance {
                rtol: value,
                atol: 0.0,
            },
            ReconstructionTolerance {
                rtol: 0.0,
                atol: value,
            },
        ] {
            assert!(reconstruct(&empty, &0, tolerance, &Default::default()).is_err());
        }
    }
    for options in [
        ReconstructionOptions {
            target_bond_dim: Some(0),
            ..Default::default()
        },
        ReconstructionOptions {
            max_regions: 0,
            ..Default::default()
        },
        ReconstructionOptions {
            patch_order: vec![DynIndex::new_dyn(2)],
            ..Default::default()
        },
    ] {
        assert!(reconstruct(&empty, &0, Default::default(), &options).is_err());
    }
    assert_eq!(
        reconstruct(&empty, &0, Default::default(), &Default::default())
            .unwrap()
            .report()
            .term_count,
        0
    );
}

#[test]
fn validates_center_split_identity_and_dimension() {
    let original = diagonal(&[0.0, 0.0]);
    let prepared = target(&original);
    let index = original.all_indices()[0].clone();
    assert!(matches!(
        reconstruct(&prepared, &9, Default::default(), &Default::default()),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
    for indices in [
        vec![index.clone(), index.clone()],
        vec![index.prime()],
        vec![DynIndex::new_dyn(2)],
    ] {
        assert!(reconstruct(
            &prepared,
            &0,
            Default::default(),
            &ReconstructionOptions {
                patch_order: indices,
                ..Default::default()
            }
        )
        .is_err());
    }
    let mut alias = index;
    alias.dim = 3;
    assert!(matches!(
        reconstruct(
            &prepared,
            &0,
            Default::default(),
            &ReconstructionOptions {
                patch_order: vec![alias],
                ..Default::default()
            }
        ),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
}

#[test]
fn long_product_state_stays_in_network_form() {
    let n = 48;
    let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    let links: Vec<_> = (1..n).map(|_| DynIndex::new_dyn(1)).collect();
    let tensors = (0..n)
        .map(|i| {
            let mut indices = Vec::new();
            if i > 0 {
                indices.push(links[i - 1].clone());
            }
            indices.push(sites[i].clone());
            if i < n - 1 {
                indices.push(links[i].clone());
            }
            IdxTensor::from_dense(indices, vec![1.0, 0.0]).unwrap()
        })
        .collect();
    let original =
        SubDomainTreeTN::from_treetn(TreeTN::from_tensors(tensors, (0..n).collect()).unwrap())
            .unwrap();
    let output = reconstruct(
        &target(&original),
        &0,
        Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(output.report().reference_scale, 1.0);
    assert_eq!(output.report().term_count, 1);
    assert_eq!(output.report().split_count, 0);
    let tree = output.into_partition().unwrap().to_treetn().unwrap();
    assert_eq!(
        tree.evaluate_point(&sites, &vec![0; n]).unwrap().real(),
        1.0
    );
    assert_eq!(
        tree.evaluate_point(&sites, &vec![1; n]).unwrap().real(),
        0.0
    );
}

#[test]
fn spectator_split_with_no_rank_gain_stops_and_keeps_full_index_identity() {
    let x = DynIndex::new_dyn(2);
    let spectator = x.prime();
    let y = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![x.clone(), spectator.clone(), bond.clone()],
                vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, y], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let original = SubDomainTreeTN::from_treetn(tree).unwrap();
    let prepared = target(&original);
    let no_gain = reconstruct(
        &prepared,
        &0,
        Default::default(),
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            patch_order: vec![spectator.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(no_gain.report().split_count, 0);
    assert_eq!(no_gain.report().max_bond_dim, 2);
    check_residual(&original, &no_gain, 1e-12);
    let gain = reconstruct(
        &prepared,
        &0,
        Default::default(),
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            patch_order: vec![x.clone()],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(gain.report().region_count, 2);
    for (projector, _) in gain.regions() {
        assert!(projector.is_projected_at(&x));
        assert!(!projector.is_projected_at(&spectator));
    }
    check_residual(&original, &gain, 1e-12);

    // Sequential cannot skip an unprofitable first index even though the
    // next index would reduce rank. Gain search may select that next index.
    for split_strategy in [
        PatchSplitStrategy::Sequential,
        PatchSplitStrategy::ExactParameterGain,
    ] {
        let output = reconstruct(
            &prepared,
            &0,
            Default::default(),
            &ReconstructionOptions {
                target_bond_dim: Some(1),
                patch_order: vec![spectator.clone(), x.clone()],
                split_strategy,
                ..Default::default()
            },
        )
        .unwrap();
        let sequential = split_strategy == PatchSplitStrategy::Sequential;
        assert_eq!(output.report().split_count, usize::from(!sequential));
        assert_eq!(output.report().max_bond_dim, if sequential { 2 } else { 1 });
        for (projector, _) in output.regions() {
            assert!(!projector.is_projected_at(&spectator));
            assert_eq!(projector.is_projected_at(&x), !sequential);
        }
        check_residual(&original, &output, 1e-12);
    }
}

#[test]
fn branched_tree_reconstruction_preserves_value_and_topology() {
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let mut tensors =
        vec![
            IdxTensor::from_dense(bonds.clone(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap(),
        ];
    for (bond, site) in bonds.iter().zip(&sites) {
        tensors.push(
            IdxTensor::from_dense(vec![bond.clone(), site.clone()], vec![1.0, 0.0, 0.0, 1.0])
                .unwrap(),
        );
    }
    let original =
        SubDomainTreeTN::from_treetn(TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap())
            .unwrap();
    let output = reconstruct(
        &target(&original),
        &0,
        Default::default(),
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            patch_order: vec![sites[1].clone()],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.report().region_count, 2);
    for (_, terms) in output.regions() {
        assert!(terms
            .iter()
            .all(|term| term.data().same_topology(original.data())));
    }
    check_residual(&original, &output, 1e-12);
}

#[test]
fn nested_splits_rebuild_original_target_and_keep_one_global_budget() {
    // Two independent entangled pairs on the same two named nodes.
    let left = diagonal(&[1.0, 2.0]);
    let right = diagonal(&[3.0, 4.0]);
    let prepared =
        ReconstructionTarget::from_tensor_products(vec![(left.clone(), right.clone())]).unwrap();
    let patch_order = vec![
        left.data()
            .site_space(&0)
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone(),
        right
            .data()
            .site_space(&0)
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone(),
    ];
    for split_strategy in [
        PatchSplitStrategy::Sequential,
        PatchSplitStrategy::ExactParameterGain,
    ] {
        let output = reconstruct(
            &prepared,
            &0,
            ReconstructionTolerance {
                rtol: 1e-8,
                atol: 0.0,
            },
            &ReconstructionOptions {
                target_bond_dim: Some(1),
                patch_order: patch_order.clone(),
                split_strategy,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.report().split_count, 3);
        assert_eq!(output.report().region_count, 4);
        assert_eq!(output.report().max_bond_dim, 1);
        assert_eq!(output.report().reference_scale, prepared.reference_scale());
        assert!(output.report().error_bound <= 1e-8 * prepared.reference_scale());
        let a = left.data().clone().to_dense().unwrap();
        let b = right.data().clone().to_dense().unwrap();
        let av = a.to_vec::<f64>().unwrap();
        let bv = b.to_vec::<f64>().unwrap();
        let expected = IdxTensor::from_dense(
            a.indices().iter().chain(b.indices()).cloned().collect(),
            bv.iter()
                .flat_map(|y| av.iter().map(move |x| x * y))
                .collect(),
        )
        .unwrap();
        let residual = result_dense(&output, left.data())
            .sub(&expected)
            .unwrap()
            .norm()
            .unwrap();
        assert!(residual < 1e-11, "nested split residual: {residual}");
        for (projector, _) in output.regions() {
            assert!(patch_order
                .iter()
                .all(|index| projector.is_projected_at(index)));
        }
    }
}

#[test]
fn disjoint_product_patches_use_sum_of_factor_norm_products() {
    let a = diagonal(&[1.0, 2.0]);
    let b = diagonal(&[3.0, 4.0]);
    let site = a
        .data()
        .site_space(&0)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();
    let pairs = (0..2)
        .map(|coordinate| {
            (
                a.project(&Projector::from_pairs([(site.clone(), coordinate)]).unwrap())
                    .unwrap()
                    .unwrap(),
                b.clone(),
            )
        })
        .collect();
    let prepared = ReconstructionTarget::from_tensor_products(pairs).unwrap();
    assert!((prepared.reference_scale() - 125.0_f64.sqrt()).abs() < 1e-12);
    let output = reconstruct(&prepared, &0, Default::default(), &Default::default()).unwrap();
    assert_eq!(output.report().reference_scale, prepared.reference_scale());
    assert_eq!(output.report().term_count, 2);
}

#[test]
fn nonfinite_target_norms_and_overflowing_allowances_are_rejected() {
    let site = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![f64::INFINITY, 1.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let partition =
        PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree).unwrap()).unwrap();
    assert!(ReconstructionTarget::from_partition(&partition).is_err());
    let original = diagonal(&[2.0, 3.0]);
    assert!(matches!(
        reconstruct(
            &target(&original),
            &0,
            ReconstructionTolerance {
                rtol: f64::MAX,
                atol: 0.0
            },
            &Default::default()
        ),
        Err(PartitionedTreeTNError::NonFiniteAdaptiveValue)
    ));
}

#[test]
fn measured_svd_residual_is_charged_against_original_norm() {
    let original = diagonal(&[1.0, 1e-6, 2e-6]);
    let prepared = target(&original);
    let output = reconstruct(
        &prepared,
        &0,
        ReconstructionTolerance {
            rtol: 1e-3,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.report().max_bond_dim, 1);
    assert_eq!(output.report().split_count, 0);
    assert!((output.report().error_bound - 5.0_f64.sqrt() * 1e-6).abs() < 1e-12);
    assert_eq!(output.report().reference_scale, prepared.reference_scale());
    check_residual(&original, &output, 1e-12);
}
