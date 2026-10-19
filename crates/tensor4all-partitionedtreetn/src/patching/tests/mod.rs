use super::*;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

#[test]
fn adaptive_budgets_are_volume_proportional() {
    // One two-site node; projected sites contribute volume 1 and
    // unprojected sites contribute their dimension. Patch A fixes s0
    // (volume 1 * 3 = 3), patch B fixes s1 (volume 2 * 1 = 2).
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(3);
    let make = |projector: Projector| {
        SubDomainTreeTN::new(
            TreeTN::from_tensors(
                vec![
                    IdxTensor::from_dense(vec![site0.clone(), site1.clone()], vec![1.0_f64; 6])
                        .unwrap(),
                ],
                vec![0usize],
            )
            .unwrap(),
            projector,
        )
        .unwrap()
    };
    let a = make(Projector::from_pairs([(site0.clone(), 0)]).unwrap());
    let b = make(Projector::from_pairs([(site1.clone(), 1)]).unwrap());

    let budgets = assign_volume_budgets(vec![a, b], 0.1, 25.0, 5).unwrap();
    let a_budget = budgets[0].budget_squared().unwrap();
    let b_budget = budgets[1].budget_squared().unwrap();

    // local_cutoff_p = cutoff * ||F||^2 * volume_p / total_volume.
    assert!((a_budget - 0.1 * 25.0 * 3.0 / 5.0).abs() < 1.0e-12);
    assert!((b_budget - 0.1 * 25.0 * 2.0 / 5.0).abs() < 1.0e-12);
    assert!((a_budget / b_budget - 1.5).abs() < 1.0e-12);
}

#[test]
fn local_cutoff_budget_of_zero_leaves_only_the_hard_cap() {
    let site = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 0.5]).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();

    let budgets = assign_volume_budgets(vec![subdomain], 0.0, 1.25, 2).unwrap();
    assert_eq!(budgets[0].budget_squared(), Some(0.0));
}

#[test]
fn checked_volume_and_parameter_count_arithmetic_is_fallible() {
    let volume = checked_product([Ok(usize::MAX), Ok(2)], || {
        PartitionedTreeTNError::VolumeOverflow
    });
    assert!(matches!(
        volume,
        Err(PartitionedTreeTNError::VolumeOverflow)
    ));

    let parameters = checked_product([Ok(usize::MAX), Ok(2)], || {
        PartitionedTreeTNError::LogicalParameterCountOverflow
    });
    assert!(matches!(
        parameters,
        Err(PartitionedTreeTNError::LogicalParameterCountOverflow)
    ));
}

#[test]
fn logical_parameter_count_uses_structured_logical_dimensions() {
    let site = DynIndex::new_dyn(3);
    let auxiliary = DynIndex::new_dyn(3);
    let tensor = IdxTensor::from_diag(vec![site, auxiliary], vec![2.0, 3.0, 5.0]).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();

    assert_eq!(logical_parameter_count(&subdomain).unwrap(), 9);
}

#[test]
fn projection_removes_exact_zero_bond_space_without_error_budget() {
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_bond(2).unwrap();
    let left =
        IdxTensor::from_dense(vec![site0.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, site1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).unwrap(),
    )
    .unwrap();
    let projector = Projector::from_pairs([(site0.clone(), 0)]).unwrap();
    let projected = subdomain.project(&projector).unwrap().unwrap();
    let compressed = project_if_present(&subdomain, &site0, 0, &0)
        .unwrap()
        .unwrap();

    assert_eq!(projected.max_bond_dim(), 2);
    assert_eq!(compressed.max_bond_dim(), 1);
    assert!(
        (compressed.norm_squared().unwrap() - projected.norm_squared().unwrap()).abs() < 1.0e-12
    );
}

#[test]
fn adaptive_add_projects_original_addends_before_retrying() {
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let product_patch = |left_values: Vec<f64>, right_values: Vec<f64>| {
        let bond = DynIndex::new_bond(1).unwrap();
        let left = IdxTensor::from_dense(vec![site0.clone(), bond.clone()], left_values).unwrap();
        let right = IdxTensor::from_dense(vec![bond, site1.clone()], right_values).unwrap();
        SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).unwrap(),
        )
        .unwrap()
    };
    let contributions = vec![
        product_patch(vec![1.0, 0.0], vec![1.0, 0.0]),
        product_patch(vec![0.0, 1.0], vec![0.0, 1.0]),
    ];
    let contraction = ContractionOptions::default().with_max_bond_dim(1);
    let patching = PatchingOptions {
        cutoff: 0.0,
        allocation: BudgetAllocation::Volume,
        max_bond_dim: Some(1),
        patch_order: vec![site0.clone()],
        split_strategy: PatchSplitStrategy::Sequential,
    };

    let result = add_project_first(contributions, &0, &contraction, &patching).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|patch| patch.max_bond_dim() == 1));
    assert!(result
        .iter()
        .all(|patch| patch.projector().get(&site0).is_some()));
    assert!(
        (result
            .iter()
            .map(|patch| patch.norm_squared().unwrap())
            .sum::<f64>()
            - 2.0)
            .abs()
            < 1.0e-12
    );
}

#[test]
fn exact_parameter_gain_can_choose_a_later_candidate() {
    let site0 = DynIndex::new_dyn(4);
    let site1 = DynIndex::new_dyn(2);
    let site2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);
    let left =
        IdxTensor::from_dense(vec![site0.clone(), bond01.clone()], vec![1.0_f64; 8]).unwrap();
    let mut center_values = vec![0.0_f64; 8];
    for channel in 0..2 {
        center_values[channel + 2 * channel + 4 * channel] = 1.0;
    }
    let center =
        IdxTensor::from_dense(vec![bond01, site1.clone(), bond12.clone()], center_values).unwrap();
    let right = IdxTensor::from_dense(vec![bond12, site2], vec![1.0_f64; 4]).unwrap();
    let tree = TreeTN::from_tensors(vec![left, center, right], vec![0usize, 1, 2]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(tree)
        .unwrap()
        .with_budget_squared(1.0e-20)
        .unwrap();
    let options = PatchingOptions {
        cutoff: 1.0e-24,
        allocation: BudgetAllocation::Volume,
        max_bond_dim: Some(1),
        patch_order: vec![site0.clone(), site1.clone()],
        split_strategy: PatchSplitStrategy::ExactParameterGain,
    };

    assert_eq!(
        choose_split_index(&subdomain, &1, &options).unwrap(),
        Some(site1)
    );
    let sequential = PatchingOptions {
        split_strategy: PatchSplitStrategy::Sequential,
        ..options
    };
    assert_eq!(
        choose_split_index(&subdomain, &1, &sequential).unwrap(),
        Some(site0)
    );
}

/// A contraction group of two pairs `(L, R_q)`, `q = 0, 1`, where
/// `L = delta(x, c2)` and `R_q` is `delta(c2, y)` restricted to `c1 = q`,
/// times unit vectors on the remaining sites. Each contribution is
/// `delta(x, y)` (rank two across every edge between `x` and `y`), both
/// share the empty output projector, and fixing `x` makes them rank one.
struct SaturatedGroup {
    pairs: Vec<(SubDomainTreeTN, SubDomainTreeTN)>,
    center: usize,
    x: DynIndex,
    c1: DynIndex,
    /// Dense value of the whole group sum.
    expected: IdxTensor,
}

fn delta_values(dims: &[usize], equal: (usize, usize)) -> Vec<f64> {
    // Column-major: the first index varies fastest.
    let len: usize = dims.iter().product();
    (0..len)
        .map(|mut flat| {
            let mut coords = Vec::with_capacity(dims.len());
            for &dim in dims {
                coords.push(flat % dim);
                flat /= dim;
            }
            if coords[equal.0] == coords[equal.1] {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn right_patches(right: &TreeTN<IdxTensor, usize>, c1: &DynIndex) -> Vec<SubDomainTreeTN> {
    (0..2)
        .map(|q| {
            SubDomainTreeTN::new(
                right.clone(),
                Projector::from_pairs([(c1.clone(), q)]).unwrap(),
            )
            .unwrap()
        })
        .collect()
}

/// Chain `0 - 1`: `x` at node 0 and `y` at node 1.
fn chain_saturated_group() -> SaturatedGroup {
    let x = DynIndex::new_dyn(2);
    let e = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let c1 = DynIndex::new_dyn(2);
    let c2 = DynIndex::new_dyn(2);
    let left_bond = DynIndex::new_bond(2).unwrap();
    let right_bond = DynIndex::new_bond(1).unwrap();
    let left = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![x.clone(), left_bond.clone()],
                delta_values(&[2, 2], (0, 1)),
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![left_bond, c1.clone(), c2.clone()],
                delta_values(&[2, 2, 2], (0, 2)),
            )
            .unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let right = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![e.clone(), right_bond.clone()], vec![1.0, 0.0]).unwrap(),
            IdxTensor::from_dense(
                vec![right_bond, c1.clone(), c2, y.clone()],
                delta_values(&[1, 2, 2, 2], (2, 3)),
            )
            .unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let left = SubDomainTreeTN::from_treetn(left).unwrap();
    let pairs = right_patches(&right, &c1)
        .into_iter()
        .map(|right| (left.clone(), right))
        .collect();
    // 2 * u(e) * delta(x, y) over [x, e, y] with u = (1, 0).
    let expected_values = delta_values(&[2, 2, 2], (0, 2))
        .into_iter()
        .enumerate()
        .map(|(flat, value)| {
            if (flat / 2) % 2 == 0 {
                2.0 * value
            } else {
                0.0
            }
        })
        .collect();
    SaturatedGroup {
        pairs,
        center: 0,
        x: x.clone(),
        c1,
        expected: IdxTensor::from_dense(vec![x, e, y], expected_values).unwrap(),
    }
}

/// Star with center 0 and leaves 1 and 2: `x` at leaf 1, `y` at leaf 2,
/// so the rank-two correlation passes through the branching center.
fn star_saturated_group() -> SaturatedGroup {
    let x = DynIndex::new_dyn(2);
    let g = DynIndex::new_dyn(2);
    let h = DynIndex::new_dyn(2);
    let e = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let c1 = DynIndex::new_dyn(2);
    let c2 = DynIndex::new_dyn(2);
    let a1 = DynIndex::new_bond(2).unwrap();
    let a2 = DynIndex::new_bond(2).unwrap();
    let r1 = DynIndex::new_bond(1).unwrap();
    let r2 = DynIndex::new_bond(1).unwrap();
    // Left: v(g) * delta(x, c2) with v = (1, 1).
    let left = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![a1.clone(), a2.clone(), g.clone()],
                delta_values(&[2, 2, 2], (0, 1)),
            )
            .unwrap(),
            IdxTensor::from_dense(vec![x.clone(), a1], delta_values(&[2, 2], (0, 1))).unwrap(),
            IdxTensor::from_dense(
                vec![a2, c1.clone(), c2.clone()],
                delta_values(&[2, 2, 2], (0, 2)),
            )
            .unwrap(),
        ],
        vec![0usize, 1, 2],
    )
    .unwrap();
    // Right: u(h) * u(e) * delta(c2, y) with u = (1, 0).
    let right = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![r1.clone(), r2.clone(), e.clone()], vec![1.0, 0.0]).unwrap(),
            IdxTensor::from_dense(vec![h.clone(), r1], vec![1.0, 0.0]).unwrap(),
            IdxTensor::from_dense(
                vec![r2, c1.clone(), c2, y.clone()],
                delta_values(&[1, 2, 2, 2], (2, 3)),
            )
            .unwrap(),
        ],
        vec![0usize, 1, 2],
    )
    .unwrap();
    let left = SubDomainTreeTN::from_treetn(left).unwrap();
    let pairs = right_patches(&right, &c1)
        .into_iter()
        .map(|right| (left.clone(), right))
        .collect();
    // 2 * v(g) * u(h) * u(e) * delta(x, y) over [x, g, h, e, y].
    let expected_values = delta_values(&[2, 2, 2, 2, 2], (0, 4))
        .into_iter()
        .enumerate()
        .map(|(flat, value)| {
            let h_zero = (flat / 4) % 2 == 0;
            let e_zero = (flat / 8) % 2 == 0;
            if h_zero && e_zero {
                2.0 * value
            } else {
                0.0
            }
        })
        .collect();
    SaturatedGroup {
        pairs,
        center: 0,
        x: x.clone(),
        c1,
        expected: IdxTensor::from_dense(vec![x, g, h, e, y], expected_values).unwrap(),
    }
}

/// Run one group with cap two and return the patches and the number of
/// exact group sums that were built and then discarded by a split.
fn run_group(
    group: &SaturatedGroup,
    split_strategy: PatchSplitStrategy,
    patch_order: Vec<DynIndex>,
) -> (Vec<SubDomainTreeTN>, usize) {
    DISCARDED_GROUP_PROBES.with(|count| count.set(0));
    let contraction =
        ContractionOptions::default().with_svd_policy(SvdTruncationPolicy::new(1.0e-12));
    let patching = PatchingOptions {
        cutoff: 0.0,
        allocation: BudgetAllocation::Volume,
        max_bond_dim: Some(2),
        patch_order,
        split_strategy,
    };
    let result = contract_group_project_first(
        group.pairs.clone(),
        None,
        &group.center,
        &contraction,
        &patching,
    )
    .unwrap();
    (result, DISCARDED_GROUP_PROBES.with(|count| count.get()))
}

fn dense_error(result: Vec<SubDomainTreeTN>, expected: &IdxTensor) -> f64 {
    PartitionedTreeTN::from_subdomains(result)
        .unwrap()
        .to_treetn()
        .unwrap()
        .to_dense()
        .unwrap()
        .sub(expected)
        .unwrap()
        .maxabs()
        .unwrap()
}

fn assert_split_on_x(result: &[SubDomainTreeTN], x: &DynIndex) {
    assert_eq!(result.len(), 2);
    let mut coordinates: Vec<_> = result
        .iter()
        .map(|patch| patch.projector().get(x).unwrap())
        .collect();
    coordinates.sort_unstable();
    assert_eq!(coordinates, vec![0, 1]);
    assert!(result.iter().all(|patch| patch.max_bond_dim() == 1));
}

#[test]
fn sequential_split_skips_group_sum_when_a_contribution_saturates() {
    for group in [chain_saturated_group(), star_saturated_group()] {
        let (result, discarded) = run_group(
            &group,
            PatchSplitStrategy::Sequential,
            vec![group.x.clone()],
        );

        assert_eq!(discarded, 0);
        assert_split_on_x(&result, &group.x);
        let error = dense_error(result, &group.expected);
        assert!(error < 1.0e-12, "max abs error {error}");
    }
}

#[test]
fn exact_parameter_gain_still_builds_the_group_probe() {
    for group in [chain_saturated_group(), star_saturated_group()] {
        let (result, discarded) = run_group(
            &group,
            PatchSplitStrategy::ExactParameterGain,
            vec![group.x.clone()],
        );

        assert_eq!(discarded, 1);
        assert_split_on_x(&result, &group.x);
        let error = dense_error(result, &group.expected);
        assert!(error < 1.0e-12, "max abs error {error}");
    }
}

#[test]
fn sequential_without_split_index_returns_the_group_sum() {
    // `c1` is contracted away, so no output index is available to split.
    for group in [chain_saturated_group(), star_saturated_group()] {
        let (result, discarded) = run_group(
            &group,
            PatchSplitStrategy::Sequential,
            vec![group.c1.clone()],
        );

        assert_eq!(discarded, 0);
        assert_eq!(result.len(), 1);
        assert!(result[0].projector().is_empty());
        assert_eq!(result[0].max_bond_dim(), 2);
        let error = dense_error(result, &group.expected);
        assert!(error < 1.0e-12, "max abs error {error}");
    }
}

#[test]
fn thresholds_select_the_documented_svd_policies() {
    use tensor4all_core::{SingularValueMeasure, ThresholdScale, TruncationRule};

    let absolute = PatchThreshold::AbsoluteSquared(0.25).policy();
    assert_eq!(
        absolute,
        SvdTruncationPolicy::new(0.25)
            .with_absolute()
            .with_squared_values()
            .with_discarded_tail_sum()
    );
    assert_eq!(absolute.threshold, 0.25);
    assert_eq!(absolute.scale, ThresholdScale::Absolute);

    let relative = PatchThreshold::RelativeSquared(0.25).policy();
    assert_eq!(
        relative,
        SvdTruncationPolicy::new(0.25)
            .with_relative()
            .with_squared_values()
            .with_discarded_tail_sum()
    );
    assert_eq!(relative.scale, ThresholdScale::Relative);
    assert_eq!(relative.measure, SingularValueMeasure::SquaredValue);
    assert_eq!(relative.rule, TruncationRule::DiscardedTailSum);
}

#[test]
fn norm_truncation_clears_inherited_absolute_budgets() {
    let site = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 10.0]).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
    let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();
    let budgets = assign_volume_budgets(vec![subdomain], 0.1, 101.0, 2).unwrap();
    let inherited = budgets[0].budget_squared().unwrap();
    assert!((inherited - 10.1).abs() < 1.0e-12);

    let partition = PartitionedTreeTN::from_subdomains(budgets).unwrap();
    let norm = truncate_adaptive(&partition, &0, 0.1, BudgetAllocation::Norm, None).unwrap();
    assert_eq!(norm.values().next().unwrap().budget_squared(), None);
    assert_eq!(norm.len(), 1);

    let volume = truncate_adaptive(&partition, &0, 0.1, BudgetAllocation::Volume, None).unwrap();
    assert!(volume.values().next().unwrap().budget_squared().is_some());
}
