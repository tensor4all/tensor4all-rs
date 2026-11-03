use num_complex::Complex64;
use tensor4all_core::{contract_pair, DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_partitionedtreetn::{
    add_with_patching, PartitionedTreeTN, PartitionedTreeTNError, PatchingOptions, Projector,
    SubDomainTreeTN,
};
use tensor4all_treetn::{
    contraction::{ContractionMethod, ContractionOptions},
    TreeTN, TruncationOptions,
};

fn subdomain(site: &DynIndex, values: [f64; 2], coordinate: usize) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values.to_vec()).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

fn unprojected(site: &DynIndex, values: Vec<f64>) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(vec![site.clone()], values).unwrap();
    SubDomainTreeTN::from_treetn(TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap()).unwrap()
}

fn complex_subdomain(site: &DynIndex, coordinate: usize) -> SubDomainTreeTN {
    let tensor = IdxTensor::from_dense(
        vec![site.clone()],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0e12, -2.0e12)],
    )
    .unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(
        tree,
        Projector::from_pairs([(site.clone(), coordinate)]).unwrap(),
    )
    .unwrap()
}

#[test]
fn partition_replaces_exact_keys_and_rejects_overlaps_transactionally() {
    let site = DynIndex::new_dyn(2);
    let mut partition = PartitionedTreeTN::from_subdomains(vec![
        subdomain(&site, [1.0, 2.0], 0),
        subdomain(&site, [3.0, 4.0], 1),
    ])
    .unwrap();

    let replacement = subdomain(&site, [10.0, 20.0], 0);
    partition.insert(replacement).unwrap();
    assert_eq!(partition.len(), 2);
    let stored = partition
        .get(&Projector::from_pairs([(site.clone(), 0)]).unwrap())
        .unwrap();
    assert_eq!(
        stored
            .data()
            .tensor(stored.data().node_index(&0).unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![10.0, 0.0]
    );

    let before = partition.clone();
    let overlapping = unprojected(&site, vec![7.0, 8.0]);
    assert!(partition.insert(overlapping).is_err());
    assert_eq!(partition.len(), before.len());
    assert!(partition.contains(&Projector::from_pairs([(site.clone(), 0)]).unwrap()));
    assert!(partition.contains(&Projector::from_pairs([(site, 1)]).unwrap()));
}

#[test]
fn partition_norm_sum_and_deterministic_treetn_sum_use_eager_values() {
    let site = DynIndex::new_dyn(2);
    let partition = PartitionedTreeTN::from_subdomains(vec![
        subdomain(&site, [3.0, 1.0e12], 0),
        subdomain(&site, [1.0e12, 4.0], 1),
    ])
    .unwrap();

    assert!((partition.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);
    assert!((partition.norm().unwrap() - 5.0).abs() < 1.0e-12);

    let first = partition.to_treetn().unwrap();
    let second = partition.to_treetn().unwrap();
    let first_tensor = first.tensor(first.node_index(&0).unwrap()).unwrap();
    let second_tensor = second.tensor(second.node_index(&0).unwrap()).unwrap();
    assert_eq!(first_tensor.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(second_tensor.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);

    let empty = PartitionedTreeTN::<usize>::new();
    assert!(matches!(
        empty.to_treetn(),
        Err(PartitionedTreeTNError::Empty)
    ));
}

#[test]
fn partition_supports_complex_dtype_and_rejects_mixed_dtype_before_mutation() {
    let site = DynIndex::new_dyn(2);
    let complex = PartitionedTreeTN::from_subdomain(complex_subdomain(&site, 0)).unwrap();
    assert!((complex.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);

    let real = PartitionedTreeTN::from_subdomain(subdomain(&site, [2.0, 3.0], 1)).unwrap();
    assert!(matches!(
        PartitionedTreeTN::from_subdomains(vec![
            complex.values().next().unwrap().clone(),
            real.values().next().unwrap().clone(),
        ]),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));

    let before = complex.clone();
    let mut target = complex.clone();
    assert!(matches!(
        target.append(real.clone()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert_eq!(target.len(), before.len());

    assert!(matches!(
        complex.add(&real, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        complex.contract(&real, &0, ContractionOptions::default()),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
}

#[test]
fn partition_addition_allows_missing_keys_and_refines_overlapping_layouts() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [2.0, 100.0], 0)).unwrap();
    let right = PartitionedTreeTN::from_subdomain(subdomain(&site, [300.0, 4.0], 1)).unwrap();

    let sum = left.add(&right, &0, TruncationOptions::default()).unwrap();
    assert_eq!(sum.len(), 2);
    let dense = sum.to_treetn().unwrap();
    assert_eq!(
        dense
            .tensor(dense.node_index(&0).unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![2.0, 4.0]
    );

    // `{site=0}` and `{}` overlap without being equal. Refining onto the common
    // refinement keeps the whole unprojected patch instead of rejecting.
    let overlap = PartitionedTreeTN::from_subdomain(unprojected(&site, vec![1.0, 2.0])).unwrap();
    let refined_sum = left
        .add(&overlap, &0, TruncationOptions::default())
        .unwrap();
    assert_eq!(refined_sum.len(), 2);
    assert_eq!(
        refined_sum
            .to_treetn()
            .unwrap()
            .to_dense()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![3.0, 2.0]
    );
}

#[test]
fn partition_addition_validates_center_and_options_before_shortcuts() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right = PartitionedTreeTN::new();

    assert!(matches!(
        left.add(&right, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(matches!(
        left.add(
            &left,
            &99,
            TruncationOptions::default().with_max_bond_dim(0),
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        left.add(&left, &99, TruncationOptions::default()),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
}

#[test]
fn partition_append_rejects_cross_overlap_transactionally() {
    let site = DynIndex::new_dyn(2);
    let mut target = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let before = target.clone();
    let other = PartitionedTreeTN::from_subdomain(unprojected(&site, vec![9.0, 10.0])).unwrap();

    assert!(matches!(
        target.append(other),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    assert_eq!(target.len(), before.len());
    assert!(target.contains(&Projector::from_pairs([(site, 0)]).unwrap()));
}

fn shared_contract_patch(
    shared: &DynIndex,
    external: Option<&DynIndex>,
    shared_value: usize,
    external_value: Option<usize>,
    values: Vec<f64>,
) -> SubDomainTreeTN {
    let mut indices = vec![shared.clone()];
    let mut pairs = vec![(shared.clone(), shared_value)];
    if let Some(external) = external {
        indices.push(external.clone());
        pairs.push((external.clone(), external_value.unwrap_or(0)));
    }
    let tensor = IdxTensor::from_dense(indices, values).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    SubDomainTreeTN::new(tree, Projector::from_pairs(pairs).unwrap()).unwrap()
}

#[test]
fn contraction_prunes_contracted_projectors_and_retains_external_projectors() {
    let shared = DynIndex::new_dyn(2);
    let left_external = DynIndex::new_dyn(2);
    let right_external = DynIndex::new_dyn(2);
    let left = shared_contract_patch(
        &shared,
        Some(&left_external),
        0,
        Some(1),
        vec![0.0, 0.0, 0.0, 7.0],
    );
    let right = shared_contract_patch(
        &shared,
        Some(&right_external),
        0,
        Some(0),
        vec![5.0, 0.0, 0.0, 0.0],
    );

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(result.projector().get(&shared), None);
    assert_eq!(result.projector().get(&left_external), Some(1));
    assert_eq!(result.projector().get(&right_external), Some(0));
    assert!(result.all_indices().contains(&left_external));
    assert!(result.all_indices().contains(&right_external));
}

#[test]
fn partition_contraction_combines_duplicate_output_projectors_strictly() {
    let shared = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomains(vec![
        shared_contract_patch(&shared, None, 0, None, vec![2.0, 100.0]),
        shared_contract_patch(&shared, None, 1, None, vec![200.0, 3.0]),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        shared_contract_patch(&shared, None, 0, None, vec![5.0, 600.0]),
        shared_contract_patch(&shared, None, 1, None, vec![700.0, 7.0]),
    ])
    .unwrap();

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert_eq!(result.len(), 1);
    let dense = result.to_treetn().unwrap();
    let tensor = dense.tensor(dense.node_index(&0).unwrap()).unwrap();
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![31.0]);
}

#[test]
fn partition_contraction_rejects_topology_site_assignment_dtype_and_dense_options() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0]).unwrap()],
        vec![1usize],
    )
    .unwrap();
    let right = PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::new(
            right_tree,
            Projector::from_pairs([(site.clone(), 0)]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        left.contract(&right, &1, ContractionOptions::default()),
        Err(PartitionedTreeTNError::TopologyMismatch)
    ));

    assert!(matches!(
        left.contract(&left, &0, ContractionOptions::new(ContractionMethod::Naive),),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}

#[test]
fn partition_contract_duplicate_failure_does_not_mutate_inputs() {
    let site = DynIndex::new_dyn(2);
    let left = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let right = PartitionedTreeTN::from_subdomain(subdomain(&site, [3.0, 4.0], 1)).unwrap();
    let left_before = left.clone();
    let right_before = right.clone();
    let _ = left.contract(&right, &0, ContractionOptions::default());
    assert_eq!(left.len(), left_before.len());
    assert_eq!(right.len(), right_before.len());
}

fn single_node_mask(indices: Vec<DynIndex>, projector: Projector) -> SubDomainTreeTN {
    let dim: usize = indices.iter().map(|index| index.dim).product();
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(indices, vec![1.0_f64; dim]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    SubDomainTreeTN::new(tree, projector).unwrap()
}

#[test]
fn contraction_of_valid_disjoint_inputs_refines_overlapping_output_regions() {
    // `{a=0}` and `{b=0}` are distinct projector keys that still intersect in
    // the full (a, b) site space, so valid disjoint inputs can contract into
    // overlapping output regions. `contract` refines both operands on the
    // uncontracted indices their patches fix, so every output region fixes (a, b)
    // and any two are equal or disjoint.
    let shared = DynIndex::new_dyn(2);
    let left_external = DynIndex::new_dyn(2);
    let right_external = DynIndex::new_dyn(2);

    let left = PartitionedTreeTN::from_subdomains(vec![
        single_node_mask(
            vec![shared.clone(), left_external.clone()],
            Projector::from_pairs([(shared.clone(), 0)]).unwrap(),
        ),
        single_node_mask(
            vec![shared.clone(), left_external.clone()],
            Projector::from_pairs([(shared.clone(), 1), (left_external.clone(), 0)]).unwrap(),
        ),
        single_node_mask(
            vec![shared.clone(), left_external.clone()],
            Projector::from_pairs([(shared.clone(), 1), (left_external.clone(), 1)]).unwrap(),
        ),
    ])
    .unwrap();
    let right = PartitionedTreeTN::from_subdomains(vec![
        single_node_mask(
            vec![shared.clone(), right_external.clone()],
            Projector::from_pairs([(shared.clone(), 0), (right_external.clone(), 0)]).unwrap(),
        ),
        single_node_mask(
            vec![shared.clone(), right_external.clone()],
            Projector::from_pairs([(shared.clone(), 0), (right_external.clone(), 1)]).unwrap(),
        ),
        single_node_mask(
            vec![shared.clone(), right_external.clone()],
            Projector::from_pairs([(shared.clone(), 1)]).unwrap(),
        ),
    ])
    .unwrap();

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    // The documented workaround, done by hand: split every patch that leaves an
    // uncontracted index free into one sub-patch per value of that index. The
    // library must do the same work itself and match it.
    let left_by_hand = left
        .refine_to(&[
            Projector::from_pairs([(shared.clone(), 0), (left_external.clone(), 0)]).unwrap(),
            Projector::from_pairs([(shared.clone(), 0), (left_external.clone(), 1)]).unwrap(),
            Projector::from_pairs([(shared.clone(), 1)]).unwrap(),
        ])
        .unwrap();
    let right_by_hand = right
        .refine_to(&[
            Projector::from_pairs([(shared.clone(), 0), (right_external.clone(), 0)]).unwrap(),
            Projector::from_pairs([(shared.clone(), 0), (right_external.clone(), 1)]).unwrap(),
            Projector::from_pairs([(shared.clone(), 1), (right_external.clone(), 0)]).unwrap(),
            Projector::from_pairs([(shared.clone(), 1), (right_external.clone(), 1)]).unwrap(),
        ])
        .unwrap();
    let by_hand = left_by_hand
        .contract(&right_by_hand, &0, ContractionOptions::default())
        .unwrap();

    // Both operands are the all-ones tensor over their sites, so every output
    // entry is the sum over the two values of the contracted index.
    assert_eq!(result.len(), 4);
    assert!(result.projectors().all(|key| key.len() == 2));
    let dense = result.to_treetn().unwrap().to_dense().unwrap();
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![2.0, 2.0, 2.0, 2.0]);
    let difference = dense
        .sub(&by_hand.to_treetn().unwrap().to_dense().unwrap())
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(
        difference < 1e-12,
        "differs from the hand refinement by {difference}"
    );
}

#[test]
fn same_index_identity_with_mismatched_dimensions_is_rejected() {
    // `DynIndex` equality and hashing ignore the dimension, so two patches
    // with the same logical identity but different dims must be rejected at
    // the public boundary instead of silently replacing each other.
    let dim_two = DynIndex::new_dyn(2);
    let mut dim_three = dim_two.clone();
    dim_three.dim = 3;

    let make = |index: &DynIndex| {
        SubDomainTreeTN::new(
            TreeTN::from_tensors(
                vec![IdxTensor::from_dense(
                    vec![index.clone()],
                    (0..index.dim).map(|v| v as f64).collect(),
                )
                .unwrap()],
                vec![0usize],
            )
            .unwrap(),
            Projector::from_pairs([(index.clone(), 0)]).unwrap(),
        )
        .unwrap()
    };
    let partition = PartitionedTreeTN::from_subdomains(vec![make(&dim_two), make(&dim_three)]);
    assert!(matches!(
        partition,
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));

    // Strict subdomain addition routes through the same structural check.
    let left = make(&dim_two);
    let right = make(&dim_three);
    assert!(matches!(
        left.add(&right),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));

    // A single valid patch still works after the rejection above.
    let valid = PartitionedTreeTN::from_subdomain(make(&dim_two)).unwrap();
    assert_eq!(valid.len(), 1);

    // A projector alias sharing the full identity but with a different
    // dimension is rejected at the public constructor before `mask_index`.
    let canonical = DynIndex::new_dyn(2);
    let mut alias = canonical.clone();
    alias.dim = 3;
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![canonical.clone()], vec![1.0, 2.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    assert!(matches!(
        SubDomainTreeTN::new(tree, Projector::from_pairs([(alias.clone(), 1)]).unwrap()),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));

    // `project` with a dimension-mismatched alias is rejected the same way.
    let source = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![canonical.clone()], vec![1.0, 2.0]).unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        source.project(&Projector::from_pairs([(alias, 0)]).unwrap()),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
}

#[test]
fn adaptive_patch_order_rejects_dimension_mismatched_site_aliases() {
    // `patch_order` entries are matched by full identity; a same-identity
    // alias with a different dimension must be rejected with
    // `SiteIndexMismatch` before any split uses the aliased dimension.
    let site = DynIndex::new_dyn(2);
    let mut alias = site.clone();
    alias.dim = 3;
    let patch = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site], vec![1.0_f64, 2.0]).unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        add_with_patching(
            vec![patch],
            &0,
            &PatchingOptions {
                patch_order: vec![alias],
                ..PatchingOptions::default()
            },
        ),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
}

#[test]
fn add_with_patching_rejects_invalid_cutoff_and_zero_cap_transactionally() {
    let site = DynIndex::new_dyn(2);
    let patch = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site], vec![1.0_f64, 2.0]).unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();
    for cutoff in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(
            matches!(
                add_with_patching(
                    vec![patch.clone()],
                    &0,
                    &PatchingOptions {
                        cutoff,
                        ..PatchingOptions::default()
                    },
                ),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "add_with_patching must reject cutoff {cutoff}"
        );
    }
    assert!(matches!(
        add_with_patching(
            vec![patch],
            &0,
            &PatchingOptions {
                max_bond_dim: Some(0),
                ..PatchingOptions::default()
            },
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}

#[test]
fn non_finite_svd_truncation_thresholds_are_rejected_before_shortcuts() {
    let site = DynIndex::new_dyn(2);
    let partition = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    let mut subdomain = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0]).unwrap()],
            vec![0usize],
        )
        .unwrap(),
    )
    .unwrap();

    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let options = TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold));
        assert!(
            matches!(
                partition.add(&partition, &0, options),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "disjoint addition must reject threshold {threshold}"
        );
        assert!(
            matches!(
                partition.contract(
                    &partition,
                    &0,
                    ContractionOptions::default()
                        .with_svd_policy(SvdTruncationPolicy::new(threshold))
                ),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "pairwise contraction shortcut must reject threshold {threshold}"
        );
        assert!(
            matches!(
                subdomain.truncate(
                    &0,
                    TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold))
                ),
                Err(PartitionedTreeTNError::InvalidOptions { .. })
            ),
            "single-node truncate must reject threshold {threshold}"
        );
    }

    // `max_bond_dim == 0` is a separate invalid case on every validated path.
    assert!(matches!(
        partition.add(
            &partition,
            &0,
            TruncationOptions::default().with_max_bond_dim(0)
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        partition.contract(
            &partition,
            &0,
            ContractionOptions::default().with_max_bond_dim(0)
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    assert!(matches!(
        subdomain.truncate(&0, TruncationOptions::default().with_max_bond_dim(0)),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}

#[test]
fn empty_and_zero_node_subdomains_have_consistent_semantics() {
    // Zero-node TreeTNs are rejected with a typed `Empty`: a patch must carry
    // an actual topology. An empty partition is a distinct, valid zero object
    // whose norm is zero but whose algebra requires operands.
    let empty_tree = TreeTN::<IdxTensor, usize>::new();
    assert!(matches!(
        SubDomainTreeTN::from_treetn(empty_tree),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(matches!(
        SubDomainTreeTN::new(TreeTN::<IdxTensor, usize>::new(), Projector::new()),
        Err(PartitionedTreeTNError::Empty)
    ));

    let empty = PartitionedTreeTN::<usize>::new();
    assert!(empty.is_empty());
    assert_eq!(empty.norm_squared().unwrap(), 0.0);
    assert_eq!(empty.norm().unwrap(), 0.0);
    assert!(matches!(
        empty.to_treetn(),
        Err(PartitionedTreeTNError::Empty)
    ));

    let site = DynIndex::new_dyn(2);
    let nonempty = PartitionedTreeTN::from_subdomain(subdomain(&site, [1.0, 2.0], 0)).unwrap();
    assert!(matches!(
        empty.add(&nonempty, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(matches!(
        empty.contract(&nonempty, &0, ContractionOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
}

/// Star-shaped patch data for `to_treetn` layout tests: hub `0` carries the
/// two projected site indices `hub_sites` (stored as `[bond01, hub_sites[1],
/// bond02, hub_sites[0], bond03]`), leaves `1, 2, 3` carry `leaf_sites`.
/// Bonds are fresh for every patch, data is a deterministic function of
/// `patch`. Tensors are supplied leaves-first so the hub is not the first input.
fn star_patch_tensors(
    hub_sites: &[DynIndex; 2],
    leaf_sites: &[DynIndex; 3],
    patch: usize,
) -> (Vec<IdxTensor>, Vec<usize>) {
    let bonds: Vec<DynIndex> = (0..3).map(|k| DynIndex::new_dyn(2 + k % 2)).collect();
    let fill = |len: usize, salt: usize| -> Vec<f64> {
        (0..len)
            .map(|i| ((i * 7 + salt * 13 + patch * 29) as f64 * 0.377).sin())
            .collect()
    };
    let hub_legs = vec![
        bonds[0].clone(),
        hub_sites[1].clone(),
        bonds[1].clone(),
        hub_sites[0].clone(),
        bonds[2].clone(),
    ];
    let hub_len = 2 * 2 * 2 * 3 * 2;
    let mut tensors = Vec::new();
    let mut names = Vec::new();
    for (leaf, bond) in [(3usize, 2usize), (1, 0)] {
        let legs = vec![leaf_sites[leaf - 1].clone(), bonds[bond].clone()];
        tensors.push(IdxTensor::from_dense(legs, fill(2 * bonds[bond].dim(), leaf)).unwrap());
        names.push(leaf);
    }
    tensors.push(IdxTensor::from_dense(hub_legs, fill(hub_len, 0)).unwrap());
    names.push(0);
    let legs = vec![bonds[1].clone(), leaf_sites[1].clone()];
    tensors.push(IdxTensor::from_dense(legs, fill(2 * bonds[1].dim(), 2)).unwrap());
    names.push(2);
    (tensors, names)
}

/// Positional leg layout of a TreeTN with bond legs identified by neighbor.
fn positional_layout(tree: &TreeTN<IdxTensor, usize>) -> Vec<(usize, Vec<String>)> {
    tree.node_names()
        .into_iter()
        .map(|node| {
            let tensor = tree.tensor(tree.node_index(&node).unwrap()).unwrap();
            let site_space = tree.site_space(&node).unwrap();
            let legs = tensor
                .indices()
                .iter()
                .map(|index| {
                    if site_space.contains(index) {
                        return format!("site:{index:?}");
                    }
                    let neighbor = tree
                        .site_index_network()
                        .neighbors(&node)
                        .find(|neighbor| {
                            let edge = tree.edge_between(&node, neighbor).unwrap();
                            tree.bond_index(edge).unwrap() == index
                        })
                        .unwrap();
                    format!("bond:{neighbor}:{}", index.dim())
                })
                .collect();
            (node, legs)
        })
        .collect()
}

#[test]
fn to_treetn_layout_and_values_are_stable_for_branched_multi_patch_sums() {
    use tensor4all_core::TensorIndex;

    let hub_sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let leaf_sites = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    // Four patches, one per value of the two hub site indices.
    let patch_inputs: Vec<_> = (0..4)
        .map(|patch| {
            let projector = Projector::from_pairs([
                (hub_sites[0].clone(), patch / 2),
                (hub_sites[1].clone(), patch % 2),
            ])
            .unwrap();
            (
                star_patch_tensors(&hub_sites, &leaf_sites, patch),
                projector,
            )
        })
        .collect();
    let build = || {
        let subdomains = patch_inputs
            .iter()
            .map(|((tensors, names), projector)| {
                let tree = TreeTN::from_tensors(tensors.clone(), names.clone()).unwrap();
                SubDomainTreeTN::new(tree, projector.clone()).unwrap()
            })
            .collect();
        PartitionedTreeTN::from_subdomains(subdomains)
            .unwrap()
            .to_treetn()
            .unwrap()
    };
    let bits = |tensor: &IdxTensor| -> Vec<u64> {
        tensor
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect()
    };
    let fingerprint = |tree: &TreeTN<IdxTensor, usize>| {
        let dense = tree.to_dense().unwrap();
        let node_bits: Vec<Vec<u64>> = tree
            .node_names()
            .into_iter()
            .map(|node| bits(tree.tensor(tree.node_index(&node).unwrap()).unwrap()))
            .collect();
        (
            positional_layout(tree),
            node_bits,
            tree.external_indices(),
            dense.indices().to_vec(),
            bits(&dense),
        )
    };

    let reference_tree = build();
    let reference = fingerprint(&reference_tree);
    // The hub keeps the patches' site-leg order and lists its merged bonds by
    // neighbor name; with four patches each bond has dimension 4 * dim.
    let hub_layout = &reference.0.iter().find(|(node, _)| *node == 0).unwrap().1;
    assert_eq!(
        hub_layout,
        &vec![
            format!("site:{:?}", hub_sites[1]),
            format!("site:{:?}", hub_sites[0]),
            "bond:1:8".to_string(),
            "bond:2:12".to_string(),
            "bond:3:8".to_string(),
        ]
    );
    assert_eq!(
        reference.3,
        vec![
            hub_sites[1].clone(),
            hub_sites[0].clone(),
            leaf_sites[0].clone(),
            leaf_sites[1].clone(),
            leaf_sites[2].clone(),
        ]
    );
    for rebuild in 1..64 {
        assert_eq!(
            fingerprint(&build()),
            reference,
            "to_treetn rebuild {rebuild} differs from the first build"
        );
    }
}

// --- Different patch layouts: refinement, contract, and add (#1088) ---------

/// Deterministic pseudo-random value in [-1, 1) for a coordinate vector.
fn seeded(seed: u64, values: &[usize]) -> f64 {
    let state = values.iter().fold(seed, |state, &value| {
        (state ^ value as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(31)
    });
    (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Every coordinate vector of `indices`, first index fastest.
fn grid_points(indices: &[DynIndex]) -> Vec<Vec<usize>> {
    let total: usize = indices.iter().map(|index| index.dim).product();
    (0..total)
        .map(|mut position| {
            indices
                .iter()
                .map(|index| {
                    let value = position % index.dim;
                    position /= index.dim;
                    value
                })
                .collect()
        })
        .collect()
}

fn projector(pairs: &[(&DynIndex, usize)]) -> Projector {
    Projector::from_pairs(
        pairs
            .iter()
            .map(|(index, value)| ((*index).clone(), *value)),
    )
    .unwrap()
}

/// Compare projector lists as sets: `Projector` has no total order.
fn sorted(projectors: &[Projector]) -> Vec<Projector> {
    let key = |projector: &Projector| {
        let mut entries: Vec<(String, usize)> = projector
            .iter()
            .map(|(index, &value)| (format!("{index:?}"), value))
            .collect();
        entries.sort();
        entries
    };
    let mut projectors = projectors.to_vec();
    projectors.sort_by_key(key);
    projectors
}

/// Two-node chain `0 - 1`: node 0 carries `node0` then a bond, node 1 the bond
/// then `external`. The bond dimension is 3, so contraction is not trivial.
fn chain(node0: &[DynIndex], external: &DynIndex, seed: u64) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(3);
    let legs0: Vec<DynIndex> = node0.iter().cloned().chain([bond.clone()]).collect();
    let values0: Vec<f64> = grid_points(&legs0)
        .iter()
        .map(|coord| seeded(seed, coord))
        .collect();
    let legs1 = vec![bond.clone(), external.clone()];
    let values1: Vec<f64> = grid_points(&legs1)
        .iter()
        .map(|coord| seeded(seed.wrapping_add(7), coord))
        .collect();
    TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(legs0, values0).unwrap(),
            IdxTensor::from_dense(legs1, values1).unwrap(),
        ],
        vec![0usize, 1usize],
    )
    .unwrap()
}

/// One patch per projector, all sharing `tree`'s topology and site space.
fn layout(tree: &TreeTN<IdxTensor, usize>, projectors: &[Projector]) -> PartitionedTreeTN<usize> {
    PartitionedTreeTN::from_subdomains(
        projectors
            .iter()
            .map(|projector| SubDomainTreeTN::new(tree.clone(), projector.clone()).unwrap())
            .collect(),
    )
    .unwrap()
}

fn dense_of(partition: &PartitionedTreeTN<usize>) -> IdxTensor {
    partition.to_treetn().unwrap().to_dense().unwrap()
}

fn assert_close(actual: &IdxTensor, expected: &IdxTensor, label: &str) {
    assert_eq!(
        actual.indices().len(),
        expected.indices().len(),
        "{label}: rank"
    );
    let difference = actual.sub(expected).unwrap().maxabs().unwrap();
    assert!(difference < 1e-12, "{label}: max difference {difference}");
}

#[test]
fn refine_to_splits_patches_and_rejects_invalid_layouts() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 11);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let fine_projectors = [
        projector(&[(&k1, 0), (&k2, 0)]),
        projector(&[(&k1, 0), (&k2, 1)]),
        projector(&[(&k1, 1)]),
    ];

    let refined = coarse.refine_to(&fine_projectors).unwrap();
    assert_eq!(refined.len(), 3);
    assert_eq!(
        sorted(&refined.projectors().cloned().collect::<Vec<_>>()),
        sorted(&fine_projectors)
    );
    // Projection is an exact restriction: the refined patches sum to the same
    // dense tensor as the coarse patches.
    assert_close(&dense_of(&refined), &dense_of(&coarse), "refine_to values");

    // Refining onto the layout the partition already has changes nothing.
    let unchanged = coarse
        .refine_to(&[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])])
        .unwrap();
    assert_eq!(unchanged.len(), 2);
    assert_close(
        &dense_of(&unchanged),
        &dense_of(&coarse),
        "identity refinement",
    );

    // A layout projector that conflicts with every patch contributes nothing,
    // and a patch keeps only the sub-patch the layout actually names.
    let single = layout(&tree, &[projector(&[(&k1, 0)])]);
    assert!(single
        .refine_to(&[projector(&[(&k1, 1)])])
        .unwrap()
        .is_empty());
    let half = single
        .refine_to(&[projector(&[(&k1, 0), (&k2, 1)])])
        .unwrap();
    assert_eq!(half.len(), 1);

    // An overlapping layout is not a refinement: it must stay an error.
    assert!(matches!(
        coarse.refine_to(&[projector(&[(&k1, 0)]), Projector::new()]),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    assert!(matches!(
        coarse.refine_to(&[projector(&[(&k1, 0)]), projector(&[(&k1, 0)])]),
        Err(PartitionedTreeTNError::OverlappingProjectors)
    ));
    // An index the partition's tree does not carry is a caller error.
    let foreign = DynIndex::new_dyn(2);
    assert!(matches!(
        coarse.refine_to(&[projector(&[(&k1, 0), (&foreign, 0)])]),
        Err(PartitionedTreeTNError::ProjectorIndexNotFound { .. })
    ));
}

#[test]
fn common_refinement_matches_the_documented_layout() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 13);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let fine_projectors = [
        projector(&[(&k1, 0), (&k2, 0)]),
        projector(&[(&k1, 0), (&k2, 1)]),
        projector(&[(&k1, 1)]),
    ];
    let fine = layout(&tree, &fine_projectors);

    // The issue's example: for `{k1=0},{k1=1}` and the layout above, the second
    // layout is the common refinement.
    let atoms = PartitionedTreeTN::common_refinement(&[&coarse, &fine]).unwrap();
    assert_eq!(sorted(&atoms), sorted(&fine_projectors));

    // Operand order does not matter.
    assert_eq!(
        atoms,
        PartitionedTreeTN::common_refinement(&[&fine, &coarse]).unwrap()
    );
    // A single operand is refined onto its own layout.
    assert_eq!(
        sorted(&PartitionedTreeTN::common_refinement(&[&coarse]).unwrap()),
        sorted(&[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])])
    );
    // Both operands share one refined key set, so add can sum patch by patch.
    let left = coarse.refine_to(&atoms).unwrap();
    let right = fine.refine_to(&atoms).unwrap();
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 3);
    assert_close(&dense_of(&left), &dense_of(&coarse), "coarse coverage");
    assert_close(&dense_of(&right), &dense_of(&fine), "fine coverage");
}

#[test]
fn common_refinement_covers_each_operand_remainder() {
    // Independent per-patch splitting is not a common refinement: `{b=0,c=0}`
    // and `{}` would give `{b=0,c=0}` plus sub-regions of it. Every atom must
    // stay disjoint and both operands must be exactly covered.
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(2);
    let tree = chain(&[a.clone(), b.clone()], &c, 17);
    let left = layout(
        &tree,
        &[
            projector(&[(&b, 0), (&c, 0)]),
            projector(&[(&a, 0), (&b, 1)]),
        ],
    );
    let right = layout(&tree, &[Projector::new()]);

    let atoms = PartitionedTreeTN::common_refinement(&[&left, &right]).unwrap();
    assert!(Projector::are_disjoint(&atoms));
    // Every patch is a union of atoms that lie inside it: each atom is inside
    // one patch or disjoint from all of them.
    for patch in left.projectors().chain(right.projectors()) {
        for atom in &atoms {
            assert!(
                atom.is_subset_of(patch) || !atom.is_compatible_with(patch),
                "atom {atom:?} partially overlaps patch {patch:?}"
            );
        }
    }

    // Both operands keep every value, and the sum covers both.
    let left_refined = left.refine_to(&atoms).unwrap();
    let right_refined = right.refine_to(&atoms).unwrap();
    assert_close(&dense_of(&left_refined), &dense_of(&left), "left remainder");
    assert_close(
        &dense_of(&right_refined),
        &dense_of(&right),
        "right remainder",
    );

    // An empty operand contributes no atoms of its own.
    let empty = PartitionedTreeTN::<usize>::new();
    assert_eq!(
        sorted(&PartitionedTreeTN::common_refinement(&[&empty, &left]).unwrap()),
        sorted(&left.projectors().cloned().collect::<Vec<_>>())
    );
}

#[test]
fn common_refinement_covers_three_operands() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 71);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let fine = layout(
        &tree,
        &[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1)]),
        ],
    );
    let full = layout(
        &tree,
        &[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1), (&k2, 0)]),
            projector(&[(&k1, 1), (&k2, 1)]),
        ],
    );

    let atoms = PartitionedTreeTN::common_refinement(&[&coarse, &fine, &full]).unwrap();
    assert!(Projector::are_disjoint(&atoms));
    // `full` already refines both others, so it is the coarsest common refinement.
    assert_eq!(
        sorted(&atoms),
        sorted(&full.projectors().cloned().collect::<Vec<_>>())
    );
    // The atom set is a property of the layouts, not of the operand order.
    for order in [
        [&coarse, &full, &fine],
        [&fine, &coarse, &full],
        [&fine, &full, &coarse],
        [&full, &coarse, &fine],
        [&full, &fine, &coarse],
    ] {
        assert_eq!(atoms, PartitionedTreeTN::common_refinement(&order).unwrap());
    }
    for operand in [&coarse, &fine, &full] {
        let refined = operand.refine_to(&atoms).unwrap();
        assert_close(
            &dense_of(&refined),
            &dense_of(operand),
            "reference coverage",
        );
    }
}

#[test]
fn add_refines_layouts_that_only_differ_on_a_dimension_one_index() {
    // `{x=0}` with `x` of dimension one restricts `{}` without having an
    // alternative coordinate, so a refinement that only counts cells misses it.
    let x = DynIndex::new_dyn(1);
    let y = DynIndex::new_dyn(2);
    let tree = chain(std::slice::from_ref(&x), &y, 83);
    let unprojected = layout(&tree, &[Projector::new()]);
    let projected = layout(&tree, &[projector(&[(&x, 0)])]);

    let atoms = PartitionedTreeTN::common_refinement(&[&unprojected, &projected]).unwrap();
    assert_eq!(atoms, vec![projector(&[(&x, 0)])]);
    let sum = unprojected
        .add(&projected, &0, TruncationOptions::default())
        .unwrap();
    assert_eq!(sum.len(), 1);
    let expected = dense_of(&unprojected).add(&dense_of(&projected)).unwrap();
    assert_close(&dense_of(&sum), &expected, "dimension-one layout");
}

#[test]
fn common_refinement_rejects_identity_matching_dimension_mismatches() {
    // `DynIndex` equality excludes the dimension, so two operands can project an
    // identity-matching index with different dimensions. Enumerating complements
    // would then depend on the operand order, so this must be rejected.
    let wide = DynIndex::new_dyn(2);
    let mut narrow = wide.clone();
    narrow.dim = 1;
    let y = DynIndex::new_dyn(2);
    let z = DynIndex::new_dyn(2);
    let unprojected = layout(
        &chain(&[wide.clone(), y.clone()], &z, 89),
        &[Projector::new()],
    );
    let narrow_patch = layout(
        &chain(&[narrow.clone(), y.clone()], &z, 97),
        &[projector(&[(&narrow, 0)])],
    );
    let wide_patch = layout(
        &chain(&[wide.clone(), y], &z, 101),
        &[projector(&[(&wide, 0)])],
    );

    assert!(matches!(
        PartitionedTreeTN::common_refinement(&[&unprojected, &narrow_patch, &wide_patch]),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
    assert!(matches!(
        PartitionedTreeTN::common_refinement(&[&wide_patch, &unprojected, &narrow_patch]),
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    ));
}

#[test]
fn common_refinement_deduplicates_repeated_layouts() {
    // A region contributed by two later operands must not be appended twice.
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let tree = chain(std::slice::from_ref(&x), &y, 73);
    let first = layout(&tree, &[projector(&[(&x, 0)])]);
    let second = layout(&tree, &[projector(&[(&x, 1)])]);

    let atoms = PartitionedTreeTN::common_refinement(&[&first, &second, &second]).unwrap();
    assert!(Projector::are_disjoint(&atoms));
    assert_eq!(
        sorted(&atoms),
        sorted(&[projector(&[(&x, 0)]), projector(&[(&x, 1)])])
    );
}

#[test]
fn common_refinement_is_order_independent() {
    // A first-operand-seeded construction gives different atoms depending on
    // which operand seeds the cells, which would make `add` order dependent.
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let z = DynIndex::new_dyn(2);
    let tree = chain(&[x.clone(), y.clone()], &z, 79);
    let unprojected = layout(&tree, &[Projector::new()]);
    let split = layout(
        &tree,
        &[
            projector(&[(&x, 0), (&y, 0)]),
            projector(&[(&y, 1), (&z, 0)]),
        ],
    );

    let forwards = PartitionedTreeTN::common_refinement(&[&unprojected, &split]).unwrap();
    let backwards = PartitionedTreeTN::common_refinement(&[&split, &unprojected]).unwrap();
    assert_eq!(forwards, backwards);
    assert!(Projector::are_disjoint(&forwards));
    for operand in [&unprojected, &split] {
        let refined = operand.refine_to(&forwards).unwrap();
        assert_close(&dense_of(&refined), &dense_of(operand), "coverage");
    }

    // Equal atom sets mean equal keys and values in both add orders.
    let sum = unprojected
        .add(&split, &0, TruncationOptions::default())
        .unwrap();
    let swapped = split
        .add(&unprojected, &0, TruncationOptions::default())
        .unwrap();
    assert_eq!(
        sorted(&sum.projectors().cloned().collect::<Vec<_>>()),
        sorted(&swapped.projectors().cloned().collect::<Vec<_>>())
    );
    let expected = dense_of(&unprojected).add(&dense_of(&split)).unwrap();
    assert_close(&dense_of(&sum), &expected, "unprojected add");
    assert_close(&dense_of(&swapped), &expected, "swapped unprojected add");
}

#[test]
fn contract_refines_when_one_operand_carries_the_output_index() {
    // The issue's contract example: `left` patches `(k1, a)` coarsely, `right`
    // patches `(k1, k2, a)` finely, and only `right` carries `k2`, which
    // survives the contraction. `left` cannot be refined on `k2`.
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let left_tree = chain(std::slice::from_ref(&k1), &a, 19);
    let right_tree = chain(&[k1.clone(), k2.clone()], &a, 23);
    let left = layout(
        &left_tree,
        &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])],
    );
    let right = layout(
        &right_tree,
        &[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1)]),
        ],
    );

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert_eq!(result.len(), 2);
    let expected = contract_pair(&dense_of(&left), &dense_of(&right)).unwrap();
    assert_close(&dense_of(&result), &expected, "contract refinement");

    // The caller-side workaround: refine `right` on `k2` by hand, then contract.
    let by_hand = right
        .refine_to(&[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1), (&k2, 0)]),
            projector(&[(&k1, 1), (&k2, 1)]),
        ])
        .unwrap();
    let worked_around = left
        .contract(&by_hand, &0, ContractionOptions::default())
        .unwrap();
    assert_close(
        &dense_of(&result),
        &dense_of(&worked_around),
        "contract by hand",
    );
}

#[test]
fn contract_refines_when_the_left_operand_carries_the_output_index() {
    // Mirror of the issue's case: the fine layout is now on the left.
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let left_tree = chain(&[k1.clone(), k2.clone()], &a, 53);
    let right_tree = chain(std::slice::from_ref(&k1), &a, 59);
    let left = layout(
        &left_tree,
        &[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1)]),
        ],
    );
    let right = layout(
        &right_tree,
        &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])],
    );

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert_eq!(result.len(), 2);
    let expected = contract_pair(&dense_of(&left), &dense_of(&right)).unwrap();
    assert_close(
        &dense_of(&result),
        &expected,
        "mirrored contract refinement",
    );
}

#[test]
fn contract_keeps_equal_layouts_unrefined() {
    // Equal layouts produce equal or disjoint output keys already, so no
    // refinement runs and the coarse output key `{}` survives.
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let left_tree = chain(std::slice::from_ref(&k1), &a, 29);
    let right_tree = chain(&[k1.clone(), k2.clone()], &a, 31);
    let left = layout(
        &left_tree,
        &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])],
    );
    let right = layout(
        &right_tree,
        &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])],
    );

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains(&Projector::new()));
    let expected = contract_pair(&dense_of(&left), &dense_of(&right)).unwrap();
    assert_close(&dense_of(&result), &expected, "contract equal layouts");
}

#[test]
fn contract_returns_an_empty_partition_for_disjoint_layouts() {
    // No patch pair intersects, so there is nothing to refine and nothing to
    // contract; this must stay an empty result rather than an error.
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let left_tree = chain(std::slice::from_ref(&k1), &a, 61);
    let right_tree = chain(&[k1.clone(), k2], &a, 67);
    let left = layout(&left_tree, &[projector(&[(&k1, 0)])]);
    let right = layout(&right_tree, &[projector(&[(&k1, 1)])]);

    let result = left
        .contract(&right, &0, ContractionOptions::default())
        .unwrap();
    assert!(result.is_empty());
}

#[test]
fn add_refines_overlapping_unequal_layouts_and_commutes() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 37);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let fine = layout(
        &tree,
        &[
            projector(&[(&k1, 0), (&k2, 0)]),
            projector(&[(&k1, 0), (&k2, 1)]),
            projector(&[(&k1, 1)]),
        ],
    );

    let sum = coarse.add(&fine, &0, TruncationOptions::default()).unwrap();
    let expected = dense_of(&coarse).add(&dense_of(&fine)).unwrap();
    assert_close(&dense_of(&sum), &expected, "add refined");

    let swapped = fine.add(&coarse, &0, TruncationOptions::default()).unwrap();
    assert_eq!(
        sorted(&sum.projectors().cloned().collect::<Vec<_>>()),
        sorted(&swapped.projectors().cloned().collect::<Vec<_>>())
    );
    assert_close(&dense_of(&swapped), &expected, "add commutativity");
}

#[test]
fn add_keeps_the_part_the_other_operand_does_not_cover() {
    // `missing` has no `{k1=0, k2=1}` patch. Refining by intersections alone
    // would drop `coarse`'s part on it, which is silently wrong; the common
    // refinement keeps it.
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 41);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let missing = layout(
        &tree,
        &[projector(&[(&k1, 0), (&k2, 0)]), projector(&[(&k1, 1)])],
    );

    let sum = coarse
        .add(&missing, &0, TruncationOptions::default())
        .unwrap();
    assert_eq!(sum.len(), 3);
    assert!(sum.contains(&projector(&[(&k1, 0), (&k2, 1)])));
    let expected = dense_of(&coarse).add(&dense_of(&missing)).unwrap();
    assert_close(&dense_of(&sum), &expected, "add remainder");
}

#[test]
fn add_with_equal_layouts_is_unchanged() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 43);
    let coarsest = [projector(&[(&k1, 0)]), projector(&[(&k1, 1)])];
    let left = layout(&tree, &coarsest);
    let right = layout(&tree, &coarsest);

    let sum = left.add(&right, &0, TruncationOptions::default()).unwrap();
    assert_eq!(
        sorted(&sum.projectors().cloned().collect::<Vec<_>>()),
        sorted(&coarsest)
    );
    let expected = dense_of(&left).add(&dense_of(&right)).unwrap();
    assert_close(&dense_of(&sum), &expected, "add equal layouts");
}

#[test]
fn contract_and_add_reject_empty_operands() {
    let k1 = DynIndex::new_dyn(2);
    let k2 = DynIndex::new_dyn(2);
    let a = DynIndex::new_dyn(2);
    let tree = chain(&[k1.clone(), k2.clone()], &a, 47);
    let coarse = layout(&tree, &[projector(&[(&k1, 0)]), projector(&[(&k1, 1)])]);
    let empty = PartitionedTreeTN::<usize>::new();

    assert!(matches!(
        coarse.add(&empty, &0, TruncationOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(matches!(
        coarse.contract(&empty, &0, ContractionOptions::default()),
        Err(PartitionedTreeTNError::Empty)
    ));
    assert!(PartitionedTreeTN::<usize>::common_refinement(&[])
        .unwrap()
        .is_empty());
}
