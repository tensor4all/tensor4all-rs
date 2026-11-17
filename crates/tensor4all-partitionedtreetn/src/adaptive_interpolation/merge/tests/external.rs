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
    let bounds = || vec![certified(&s, &[0], 0.0), certified(&s, &[1], 2.0 * tau)];
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
