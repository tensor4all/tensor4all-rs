//! Tests of merge_sibling_patches: patches from outside the driver, with given bounds.

use super::*;

// ---------------------------------------------------------------------------
// merge_sibling_patches: patches from outside the driver, with given bounds.
// ---------------------------------------------------------------------------

/// The projector fixing the first `fixed.len()` sites of `s` to `fixed`.
fn projector(s: &[DynIndex], fixed: &[usize]) -> Projector {
    Projector::from_pairs(s.iter().cloned().zip(fixed.iter().copied())).unwrap()
}

/// `network` restricted to the patch `fixed`.
fn patch(
    s: &[DynIndex],
    fixed: &[usize],
    network: TreeTN<IdxTensor, usize>,
) -> SubDomainTreeTN<usize> {
    SubDomainTreeTN::from_treetn(network)
        .unwrap()
        .project(&projector(s, fixed))
        .unwrap()
        .unwrap()
}

/// The rank-two network `A + eps * B`.
fn rank_two(s: &[DynIndex], eps: f64) -> TreeTN<IdxTensor, usize> {
    let mut small = B;
    small[0] = [eps * B[0][0], eps * B[0][1]];
    SubDomainTreeTN::from_treetn(product(s, A))
        .unwrap()
        .add(&SubDomainTreeTN::from_treetn(product(s, small)).unwrap())
        .unwrap()
        .into_data()
}

fn certified(s: &[DynIndex], fixed: &[usize], rms: f64) -> PatchErrorBound {
    PatchErrorBound::new(projector(s, fixed), rms).with_certified(true)
}

#[test]
fn external_patches_merge_recursively_back_into_one_patch() {
    let s = sites();
    let keys: [&[usize]; 4] = [&[0, 0], &[0, 1], &[1, 0], &[1, 1]];
    let partition = PartitionedTreeTN::from_subdomains(
        keys.iter().map(|k| patch(&s, k, product(&s, A))).collect(),
    )
    .unwrap();
    let bounds = keys.iter().map(|k| certified(&s, k, 0.0)).collect();
    let merged = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap();
    // Two groups at depth two, then their merged regions at depth one.
    assert_eq!(merged.merge.merges.len(), 3);
    assert_eq!(merged.partition.len(), 1);
    assert!(merged.partition.get(&Projector::new()).is_some());
    assert_eq!(merged.truncated_leaves, 0);
    let record = merged.merge.merges.last().unwrap();
    assert_eq!(
        merged.patch_errors,
        vec![certified(
            &s,
            &[],
            record.members_rms + record.truncation_rms
        )]
    );
    assert_dense(
        &merged.partition,
        &dense(&s, |p| product_value(A, p)),
        1e-12,
    );
    let GlobalL2Error::Certified { rms_error, .. } = merged.merge.error.global else {
        panic!("certified bounds and computed truncations stay certified");
    };
    assert!(rms_error <= 1e-8);
}

#[test]
fn leaf_slack_share_truncates_patches_before_merging() {
    let s = sites();
    let eps = 1e-4;
    let tau = 1e-2;
    // The sibling's bound exceeds tau, so the group never merges and the
    // rank-two patch shows the leaf truncation alone.
    let build = || {
        PartitionedTreeTN::from_subdomains(vec![
            patch(&s, &[0], rank_two(&s, eps)),
            patch(&s, &[1], product(&s, A)),
        ])
        .unwrap()
    };
    let input_rms = 0.1 * tau;
    let bounds = || {
        vec![
            certified(&s, &[0], input_rms),
            certified(&s, &[1], 2.0 * tau),
        ]
    };
    let bond_of_first = |result: &SiblingPatchMergeResult<usize>| {
        result
            .partition
            .get(&projector(&s, &[0]))
            .unwrap()
            .max_bond_dim()
    };

    let kept = merge_sibling_patches(
        build(),
        bounds(),
        &s,
        tau,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap();
    assert_eq!(kept.truncated_leaves, 0);
    assert_eq!(kept.merge.rejected_by_status, 1);
    assert_eq!(bond_of_first(&kept), 2);
    assert_eq!(kept.patch_errors, bounds());

    let options = SiblingPatchMergeOptions::new(2).with_leaf_slack_share(0.5);
    let truncated = merge_sibling_patches(build(), bounds(), &s, tau, &0, &options).unwrap();
    assert_eq!(truncated.truncated_leaves, 1);
    assert_eq!(bond_of_first(&truncated), 1);
    // The truncation stays within half of the patch's allowance.
    let mut small = B;
    small[0] = [eps * B[0][0], eps * B[0][1]];
    let exact = dense(&s, |p| {
        if p[0] == 0 {
            product_value(A, p) + product_value(small, p)
        } else {
            0.0
        }
    });
    let first = PartitionedTreeTN::from_subdomains(vec![truncated
        .partition
        .get(&projector(&s, &[0]))
        .unwrap()
        .clone()])
    .unwrap();
    let actual = first.to_treetn().unwrap().to_dense().unwrap();
    let rms = actual.sub(&exact).unwrap().norm().unwrap() / 8.0_f64.sqrt();
    assert!(rms > 0.0 && rms <= 0.5 * tau, "rms error {rms:e}");
    assert_eq!(truncated.patch_errors[0].projector, projector(&s, &[0]));
    assert!(truncated.patch_errors[0].certified);
    let accounted = truncated.patch_errors[0].rms;
    assert!(
        (accounted - (input_rms + rms)).abs() < 1e-12,
        "accounted {accounted:e}, input {input_rms:e}, measured truncation {rms:e}"
    );
    assert_eq!(truncated.patch_errors[1], bounds()[1]);
}

#[test]
fn an_external_bound_above_tau_blocks_its_group() {
    let s = sites();
    let partition = PartitionedTreeTN::from_subdomains(vec![
        patch(&s, &[0], product(&s, A)),
        patch(&s, &[1], product(&s, A)),
    ])
    .unwrap();
    let bounds = vec![certified(&s, &[0], 2e-8), certified(&s, &[1], 0.0)];
    let merged = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2).with_leaf_slack_share(1.0),
    )
    .unwrap();
    assert!(merged.merge.merges.is_empty());
    assert_eq!(merged.merge.rejected_by_status, 1);
    assert_eq!(merged.partition.len(), 2);
    assert_eq!(
        merged.patch_errors,
        vec![certified(&s, &[0], 2e-8), certified(&s, &[1], 0.0)]
    );
}

#[test]
fn an_external_zero_patch_merges_with_its_sibling() {
    let s = sites();
    let partition =
        PartitionedTreeTN::from_subdomains(vec![patch(&s, &[0], product(&s, A))]).unwrap();
    let bounds = vec![certified(&s, &[0], 0.0), certified(&s, &[1], 0.0)];
    let merged = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap();
    assert_eq!(merged.merge.merges.len(), 1);
    assert_eq!(merged.partition.len(), 1);
    assert_dense(
        &merged.partition,
        &dense(&s, |p| if p[0] == 0 { product_value(A, p) } else { 0.0 }),
        1e-12,
    );
}

#[test]
fn bounds_that_leave_a_region_uncovered_are_rejected() {
    // The right half's zero patch is missing, so half of the domain would
    // silently vanish from the result while the report stays certified.
    let s = sites();
    let partition =
        PartitionedTreeTN::from_subdomains(vec![patch(&s, &[0], product(&s, A))]).unwrap();
    let error = merge_sibling_patches(
        partition,
        vec![certified(&s, &[0], 0.0)],
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    assert!(
        matches!(error, PatchedInterpolationError::InvalidInput { .. }),
        "{error}"
    );
}

#[test]
fn a_patch_that_is_also_a_declared_zero_parent_is_rejected() {
    // `[0]` is a real patch and its two children are declared zero. Merging
    // them would replace the patch's network by the zero region.
    let s = sites();
    let partition = PartitionedTreeTN::from_subdomains(vec![
        patch(&s, &[0], product(&s, A)),
        patch(&s, &[1], product(&s, A)),
    ])
    .unwrap();
    let bounds = vec![
        certified(&s, &[0], 0.0),
        certified(&s, &[0, 0], 0.0),
        certified(&s, &[0, 1], 0.0),
        certified(&s, &[1], 0.0),
    ];
    let error = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    assert!(
        matches!(error, PatchedInterpolationError::InvalidInput { .. }),
        "{error}"
    );
}

#[test]
fn a_split_order_index_with_the_wrong_dimension_is_rejected() {
    // Index equality ignores the dimension, so a dim-1 alias of a binary site
    // would make the cut look complete while half of the domain stays missing.
    let s = sites();
    let partition =
        PartitionedTreeTN::from_subdomains(vec![patch(&s, &[0], product(&s, A))]).unwrap();
    let mut order = s.clone();
    order[0].dim = 1;
    let error = merge_sibling_patches(
        partition,
        vec![certified(&s, &[0], 0.0)],
        &order,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    assert!(
        matches!(error, PatchedInterpolationError::InvalidInput { .. }),
        "{error}"
    );
}

#[test]
fn a_patch_bound_with_the_wrong_dimension_is_rejected() {
    // A dim-4 alias of a binary site would weigh this half at N/4 points instead
    // of N/2 and keep the merged error certified.
    let s = sites();
    let partition = PartitionedTreeTN::from_subdomains(vec![
        patch(&s, &[0], product(&s, A)),
        patch(&s, &[1], product(&s, A)),
    ])
    .unwrap();
    let mut alias = s[0].clone();
    alias.dim = 4;
    let bounds = vec![
        PatchErrorBound::new(Projector::from_pairs([(alias, 0)]).unwrap(), 0.0)
            .with_certified(true),
        certified(&s, &[1], 0.0),
    ];
    let error = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    assert!(
        matches!(error, PatchedInterpolationError::InvalidInput { .. }),
        "{error}"
    );
}

/// A rank-one chain network over `sites`: `prod_i factor[s_i]`.
fn chain(sites: &[DynIndex], factor: [f64; 2]) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<DynIndex> = (0..sites.len() - 1).map(|_| DynIndex::new_dyn(1)).collect();
    let tensors = sites
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            if node + 1 < sites.len() {
                indices.push(bonds[node].clone());
            }
            IdxTensor::from_dense(indices, factor.to_vec()).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
}

#[test]
fn a_domain_size_f64_cannot_represent_is_rejected() {
    // Two patches over 1025 binary sites: the domain product is 2^1025, which
    // overflows to inf, so every point count divided from it was NaN while both
    // bounds stayed within tolerance and the result reported Certified.
    let s: Vec<DynIndex> = (0..1025).map(|_| DynIndex::new_dyn(2)).collect();
    let partition = PartitionedTreeTN::from_subdomains(vec![
        patch(&s, &[0], chain(&s, A[0])),
        patch(&s, &[1], chain(&s, A[1])),
    ])
    .unwrap();
    let bounds = vec![certified(&s, &[0], 0.0), certified(&s, &[1], 0.0)];
    let error = merge_sibling_patches(
        partition,
        bounds,
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    let PatchedInterpolationError::InvalidInput { message } = error else {
        panic!("the domain product is rejected as input: {error}");
    };
    assert!(message.contains("is inf"), "{message}");
}

#[test]
fn an_empty_domain_is_rejected() {
    // A dim-0 site makes the domain product 0, so every weight is 0/0 and the
    // empty domain was reported as Certified.
    let site = DynIndex::new_dyn(0);
    let network = TreeTN::from_tensors(
        vec![IdxTensor::from_diag(vec![site.clone()], Vec::<f64>::new()).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let partition =
        PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::from_treetn(network).unwrap()])
            .unwrap();
    let order = [site];
    let error = merge_sibling_patches(
        partition,
        vec![certified(&order, &[], 0.0)],
        &order,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap_err();
    let PatchedInterpolationError::InvalidInput { message } = error else {
        panic!("the empty domain is rejected as input: {error}");
    };
    assert!(message.contains("is 0"), "{message}");
}

#[test]
fn invalid_external_inputs_fail_before_any_merge() {
    let s = sites();
    let build = || {
        PartitionedTreeTN::from_subdomains(vec![
            patch(&s, &[0], product(&s, A)),
            patch(&s, &[1], product(&s, A)),
        ])
        .unwrap()
    };
    let bounds = || vec![certified(&s, &[0], 0.0), certified(&s, &[1], 0.0)];
    let options = SiblingPatchMergeOptions::new(2);
    let invalid_input =
        |result: Result<SiblingPatchMergeResult<usize>, PatchedInterpolationError>| {
            matches!(result, Err(PatchedInterpolationError::InvalidInput { .. }))
        };
    assert!(invalid_input(merge_sibling_patches(
        build(),
        bounds(),
        &s,
        -1.0,
        &0,
        &options
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        bounds(),
        &s,
        f64::NAN,
        &0,
        &options
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        bounds(),
        &s,
        1e-8,
        &0,
        &SiblingPatchMergeOptions::new(0)
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        bounds(),
        &s,
        1e-8,
        &0,
        &options.with_leaf_slack_share(1.5)
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        vec![],
        &s,
        1e-8,
        &0,
        &options
    )));
    assert!(invalid_input(merge_sibling_patches(
        PartitionedTreeTN::new(),
        bounds(),
        &s,
        1e-8,
        &0,
        &options
    )));
    // A patch without a bound, a negative bound, a duplicate, a non-prefix projector.
    assert!(invalid_input(merge_sibling_patches(
        build(),
        vec![certified(&s, &[0], 0.0)],
        &s,
        1e-8,
        &0,
        &options
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        vec![certified(&s, &[0], -1.0), certified(&s, &[1], 0.0)],
        &s,
        1e-8,
        &0,
        &options
    )));
    assert!(invalid_input(merge_sibling_patches(
        build(),
        vec![
            certified(&s, &[0], 0.0),
            certified(&s, &[0], 0.0),
            certified(&s, &[1], 0.0)
        ],
        &s,
        1e-8,
        &0,
        &options
    )));
    let reversed: Vec<DynIndex> = s.iter().rev().cloned().collect();
    assert!(invalid_input(merge_sibling_patches(
        build(),
        bounds(),
        &reversed,
        1e-8,
        &0,
        &options
    )));
    assert!(matches!(
        merge_sibling_patches(build(), bounds(), &s, 1e-8, &99, &options),
        Err(PatchedInterpolationError::Partition {
            source: PartitionedTreeTNError::InvalidCenter
        })
    ));
}

#[test]
fn merged_bounds_keep_the_terms_rms_and_certification() {
    let s = sites();
    for certifications in [[true, true], [true, false], [false, false]] {
        let partition = PartitionedTreeTN::from_subdomains(vec![
            patch(&s, &[0], product(&s, A)),
            patch(&s, &[1], product(&s, A)),
        ])
        .unwrap();
        let bounds = vec![
            certified(&s, &[0], 1e-5).with_certified(certifications[0]),
            certified(&s, &[1], 2e-5).with_certified(certifications[1]),
        ];
        let merged = merge_sibling_patches(
            partition,
            bounds,
            &s,
            1e-3,
            &0,
            &SiblingPatchMergeOptions::new(2),
        )
        .unwrap();
        assert_eq!(merged.merge.merges.len(), 1);
        let record = &merged.merge.merges[0];
        assert_eq!(
            merged.patch_errors,
            vec![PatchErrorBound::new(
                record.region.clone(),
                record.members_rms + record.truncation_rms,
            )
            .with_certified(certifications.iter().all(|&value| value))]
        );
    }
}

#[test]
fn absent_zero_bounds_form_a_reusable_canonical_cut() {
    let s = sites();
    let tau = 1e-3;
    // Keep both original and merged zero regions; their certification follows
    // all input bounds, including bounds of networks in another region.
    for merge_zeros in [false, true] {
        for certifications in [[true, true], [false, true], [true, false]] {
            let partition =
                PartitionedTreeTN::from_subdomains(vec![patch(&s, &[0], product(&s, A))]).unwrap();
            let zero_keys: Vec<&[usize]> = if merge_zeros {
                vec![&[1, 0], &[1, 1]]
            } else {
                vec![&[1]]
            };
            let mut bounds: Vec<_> = zero_keys
                .iter()
                .map(|key| certified(&s, key, 1e-5).with_certified(certifications[1]))
                .collect();
            // Deliberately reverse the cut's order on input.
            bounds.push(certified(&s, &[0], 2.0 * tau).with_certified(certifications[0]));
            let result = merge_sibling_patches(
                partition,
                bounds,
                &s,
                tau,
                &0,
                &SiblingPatchMergeOptions::new(2),
            )
            .unwrap();
            assert_eq!(result.partition.len(), 1);
            assert_eq!(result.merge.merges.len(), usize::from(merge_zeros));
            assert_eq!(result.patch_errors.len(), 2);
            assert_eq!(result.patch_errors[0].projector, projector(&s, &[0]));
            assert_eq!(result.patch_errors[0].rms, 2.0 * tau);
            assert_eq!(result.patch_errors[0].certified, certifications[0]);
            let zero = &result.patch_errors[1];
            assert_eq!(zero.projector, projector(&s, &[1]));
            assert!(result.partition.get(&zero.projector).is_none());
            assert!((zero.rms - 1e-5).abs() < 1e-20);
            assert_eq!(zero.certified, certifications.iter().all(|&value| value));
            let expected = result.patch_errors.clone();
            let chained = merge_sibling_patches(
                result.partition,
                result.patch_errors,
                &s,
                tau,
                &0,
                &SiblingPatchMergeOptions::new(2),
            )
            .unwrap();
            assert_eq!(chained.patch_errors, expected);
            assert_eq!(chained.merge.rejected_by_status, 1);
        }
    }
}

#[test]
fn chaining_accounted_bounds_unblocks_a_tighter_budget() {
    let mut s = sites();
    // Distinct physical sites may share an ID while differing in prime level.
    s[1] = s[0].prime();
    let tau_1 = 1e-2;
    let tau_2 = 1e-3;
    // Stage 1 merges the two left quarters. The whole domain has rank two,
    // so the strict cap of two keeps the left and right halves separate.
    let partition = PartitionedTreeTN::from_subdomains(vec![
        patch(&s, &[0, 0], product(&s, A)),
        patch(&s, &[0, 1], product(&s, A)),
        patch(&s, &[1], product(&s, B)),
    ])
    .unwrap();
    let reference = partition.to_treetn().unwrap().to_dense().unwrap();
    let bounds = [&[0, 0][..], &[0, 1][..], &[1][..]]
        .iter()
        .map(|key| certified(&s, key, 1e-5))
        .collect();
    let stage_1 = merge_sibling_patches(
        partition,
        bounds,
        &s,
        tau_1,
        &0,
        &SiblingPatchMergeOptions::new(2),
    )
    .unwrap();
    assert_eq!(stage_1.merge.merges.len(), 1);
    assert_eq!(stage_1.merge.merges[0].region, projector(&s, &[0]));
    assert_eq!(stage_1.merge.rejected_by_cap, 1);
    assert_eq!(stage_1.partition.len(), 2);
    assert_eq!(stage_1.patch_errors.len(), 2);
    let left_record = &stage_1.merge.merges[0];
    assert_eq!(
        stage_1.patch_errors[0],
        certified(
            &s,
            &[0],
            left_record.members_rms + left_record.truncation_rms
        )
    );
    assert_eq!(stage_1.patch_errors[1], certified(&s, &[1], 1e-5));
    let members: Vec<_> = stage_1
        .patch_errors
        .iter()
        .map(|bound| bound.projector.clone())
        .collect();
    assert_eq!(members, vec![projector(&s, &[0]), projector(&s, &[1])]);
    let conservative = stage_1
        .patch_errors
        .iter()
        .map(|bound| PatchErrorBound::new(bound.projector.clone(), tau_1).with_certified(true))
        .collect();
    let options = SiblingPatchMergeOptions::new(3);
    let blocked = merge_sibling_patches(
        stage_1.partition.clone(),
        conservative,
        &s,
        tau_2,
        &0,
        &options,
    )
    .unwrap();
    // There is exactly one sibling group, the root. Assigning tau_1 to its
    // members blocks that group by status and yields no merge record.
    assert_eq!(blocked.partition.len(), 2);
    assert_eq!(blocked.merge.rejected_by_status, 1);
    assert!(blocked.merge.merges.is_empty());
    let chained = merge_sibling_patches(
        stage_1.partition,
        stage_1.patch_errors,
        &s,
        tau_2,
        &0,
        &options,
    )
    .unwrap();
    assert_eq!(chained.partition.len(), 1);
    assert_eq!(chained.merge.rejected_by_status, 0);
    assert_eq!(chained.merge.merges.len(), 1);
    let record = &chained.merge.merges[0];
    assert_eq!(record.region, Projector::new());
    assert_eq!(record.members, members);
    assert_eq!(
        chained.patch_errors[0].rms,
        record.members_rms + record.truncation_rms
    );
    let actual = chained.partition.to_treetn().unwrap().to_dense().unwrap();
    let rms = actual.sub(&reference).unwrap().norm().unwrap() / 4.0;
    assert!(
        rms <= tau_2,
        "chained RMS residual {rms:e} exceeds {tau_2:e}"
    );
}
