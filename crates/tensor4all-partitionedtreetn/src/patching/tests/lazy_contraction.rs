use super::*;
use crate::partitioned_tree_tn::CONTRACTED_PAIRS;
use tensor4all_core::{FactorizeAlg, IndexLike, TensorContractionLike};

/// Add a dimension-one scalar leaf so the existing analytic fixture takes
/// the general-tree schedule (its original three-node star is a chain).
fn degree_three_group() -> SaturatedGroup {
    fn extend(patch: SubDomainTreeTN) -> SubDomainTreeTN {
        let link = DynIndex::new_dyn(1);
        let one = IdxTensor::from_dense(vec![link.clone()], vec![1.]).unwrap();
        let names = patch.data().node_names();
        let mut tensors: Vec<_> = names
            .iter()
            .map(|name| {
                let tensor = patch
                    .data()
                    .tensor(patch.data().node_index(name).unwrap())
                    .unwrap();
                if *name == 0 {
                    tensor.outer_product(&one).unwrap()
                } else {
                    tensor.clone()
                }
            })
            .collect();
        tensors.push(one);
        let mut names = names;
        names.push(3);
        SubDomainTreeTN::new(
            TreeTN::from_tensors(tensors, names).unwrap(),
            patch.projector().clone(),
        )
        .unwrap()
    }
    let mut group = star_saturated_group();
    group.pairs = group
        .pairs
        .into_iter()
        .map(|(a, b)| (extend(a), extend(b)))
        .collect();
    group
}

fn partitions(group: &SaturatedGroup) -> (PartitionedTreeTN, PartitionedTreeTN) {
    let left = PartitionedTreeTN::from_subdomains(vec![group.pairs[0].0.clone()]).unwrap();
    let right =
        PartitionedTreeTN::from_subdomains(group.pairs.iter().map(|(_, b)| b.clone()).collect())
            .unwrap();
    (left, right)
}

fn options(
    order: Vec<DynIndex>,
    strategy: PatchSplitStrategy,
    cap: Option<usize>,
) -> PatchingOptions {
    PatchingOptions {
        cutoff: 1e-24,
        allocation: BudgetAllocation::Volume,
        max_bond_dim: cap,
        patch_order: order,
        split_strategy: strategy,
        sum_engine: SumEngine::Binary,
    }
}

fn measure<R>(run: impl FnOnce() -> R) -> (R, Vec<(usize, usize, usize)>) {
    GROUP_PAIR_PROBES.with(|probes| probes.borrow_mut().clear());
    let result = run();
    (
        result,
        GROUP_PAIR_PROBES.with(|probes| probes.borrow().clone()),
    )
}

fn assert_partition(partition: &PartitionedTreeTN, expected: &IdxTensor) {
    let actual = partition.to_treetn().unwrap().to_dense().unwrap();
    let error = actual.sub(expected).unwrap().maxabs().unwrap();
    assert!(error < 1e-10, "max-absolute residual {error:e}");
}

fn collect_plan<'a>(
    left: &'a PartitionedTreeTN,
    right: &'a PartitionedTreeTN,
    center: &usize,
    wanted: Option<&HashSet<Projector>>,
) -> Result<Vec<crate::partitioned_tree_tn::PlannedPair<'a, usize>>> {
    let mut plan = Vec::new();
    left.for_each_planned_pair(right, center, wanted, |pair| {
        plan.push(pair);
        Ok(())
    })?;
    Ok(plan)
}

#[test]
fn planning_preserves_selection_and_order_without_contracting_any_pair() {
    let group = degree_three_group();
    let (left, right) = partitions(&group);
    CONTRACTED_PAIRS.with(|count| count.set(0));
    let planned = collect_plan(&left, &right, &0, None).unwrap();
    assert_eq!(CONTRACTED_PAIRS.with(|count| count.get()), 0);
    assert_eq!(planned.len(), 2);
    assert_eq!(
        planned
            .iter()
            .map(|(_, b, _)| b.projector().get(&group.c1).unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(planned.iter().all(|(_, _, output)| output.is_empty()));
    let none = HashSet::new();
    assert!(collect_plan(&left, &right, &0, Some(&none))
        .unwrap()
        .is_empty());
    assert!(matches!(
        collect_plan(&left, &right, &99, Some(&none)),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
    let reversed = PartitionedTreeTN::from_subdomains(
        group.pairs.iter().rev().map(|(_, b)| b.clone()).collect(),
    )
    .unwrap();
    let order = collect_plan(&left, &reversed, &0, None).unwrap();
    assert_eq!(
        order
            .iter()
            .map(|(_, b, _)| b.projector().get(&group.c1).unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let mut visited = 0;
    let failed = left.for_each_planned_pair(&right, &0, None, |_| {
        visited += 1;
        Err(PartitionedTreeTNError::Empty)
    });
    assert!(matches!(failed, Err(PartitionedTreeTNError::Empty)));
    assert_eq!(visited, 1);
    assert_eq!(CONTRACTED_PAIRS.with(|count| count.get()), 0);
}

#[test]
fn public_sequential_entry_stops_after_the_first_saturated_pair() {
    for group in [chain_saturated_group(), degree_three_group()] {
        let (left, right) = partitions(&group);
        let reversed_right = PartitionedTreeTN::from_subdomains(
            group.pairs.iter().rev().map(|(_, b)| b.clone()).collect(),
        )
        .unwrap();
        let opts = options(
            vec![group.x.clone()],
            PatchSplitStrategy::Sequential,
            Some(2),
        );
        for method in [
            ContractionOptions::zipup(),
            ContractionOptions::fit().with_max_bond_dim(2),
            ContractionOptions::src().with_max_bond_dim(2),
            ContractionOptions::zipup()
                .with_max_bond_dim(2)
                .with_factorize_alg(FactorizeAlg::RSVD),
        ] {
            let (result, probes) = measure(|| contract_adaptive(&left, &right, &0, &method, &opts));
            let result = result.unwrap();
            assert_eq!(probes[0], (2, 1, 1));
            assert_split_on_x(&result.values().cloned().collect::<Vec<_>>(), &group.x);
            assert_partition(&result, &group.expected);
            let wanted = Projector::new();
            let (selected, selected_probes) = measure(|| {
                contract_adaptive_selected(
                    &left,
                    &reversed_right,
                    &0,
                    &method,
                    &opts,
                    &[wanted.clone(), wanted],
                )
            });
            let selected = selected.unwrap();
            assert_eq!(selected_probes[0], (2, 1, 1));
            assert_partition(&selected, &group.expected);
        }
    }
}

#[test]
fn gain_no_candidate_and_no_cap_paths_complete_every_pair() {
    let group = degree_three_group();
    let contraction = ContractionOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(1e-12));
    for (strategy, order, cap) in [
        (
            PatchSplitStrategy::ExactParameterGain,
            vec![group.x.clone()],
            Some(2),
        ),
        (
            PatchSplitStrategy::Sequential,
            vec![group.c1.clone()],
            Some(2),
        ),
        (PatchSplitStrategy::Sequential, vec![group.x.clone()], None),
        (
            PatchSplitStrategy::Sequential,
            vec![group.x.clone()],
            Some(3),
        ),
    ] {
        let opts = options(order, strategy, cap);
        let (result, probes) =
            measure(|| contract_group_project_first(group.pairs.clone(), &0, &contraction, &opts));
        let result = result.unwrap();
        assert_eq!(probes[0], (2, 2, if cap == Some(3) { 2 } else { 0 }));
        let error = dense_error(result, &group.expected);
        assert!(error < 1e-10, "max-absolute residual {error:e}");
    }
}

#[test]
fn implicit_order_uses_the_first_completed_contributions_site_layout() {
    let group = degree_three_group();
    let opts = options(vec![], PatchSplitStrategy::Sequential, Some(2));
    let (result, probes) = measure(|| {
        contract_group_project_first(
            group.pairs.clone(),
            &0,
            &ContractionOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(1e-12)),
            &opts,
        )
    });
    let result = result.unwrap();
    assert_eq!(probes[0], (2, 1, 0));
    // Root site g is first in the ordinary node/axis order; it is not x,
    // which was deliberately first in the dense oracle's axis order.
    let g = &group.expected.indices()[1];
    assert!(result.iter().all(|patch| patch.is_projected_at(g)));
    let error = dense_error(result, &group.expected);
    assert!(error < 1e-10, "max-absolute residual {error:e}");
}

#[test]
fn a_completed_unsaturated_pair_does_not_hide_a_later_saturated_pair() {
    let mut group = degree_three_group();
    let right = &group.pairs[0].1;
    let names = right.data().node_names();
    let tensors = names
        .iter()
        .map(|name| {
            let original = right
                .data()
                .tensor(right.data().node_index(name).unwrap())
                .unwrap();
            IdxTensor::from_dense(
                original.indices().to_vec(),
                vec![1.; original.indices().iter().map(IndexLike::dim).product()],
            )
            .unwrap()
        })
        .collect();
    group.pairs[0].1 = SubDomainTreeTN::new(
        TreeTN::from_tensors(tensors, names).unwrap(),
        right.projector().clone(),
    )
    .unwrap();
    // Distinct contracted-coordinate keys retain the operand identity that
    // the projection cache uses. The first left patch is locally rank one;
    // making only the first right patch separable would leave stored rank two.
    let left = &group.pairs[0].0;
    let names = left.data().node_names();
    let tensors = names
        .iter()
        .map(|name| {
            let original = left
                .data()
                .tensor(left.data().node_index(name).unwrap())
                .unwrap();
            IdxTensor::from_dense(
                original.indices().to_vec(),
                vec![1.; original.indices().iter().map(IndexLike::dim).product()],
            )
            .unwrap()
        })
        .collect();
    group.pairs[0].0 = SubDomainTreeTN::new(
        TreeTN::from_tensors(tensors, names).unwrap(),
        Projector::from_pairs([(group.c1.clone(), 0)]).unwrap(),
    )
    .unwrap();
    group.pairs[1].0 = group.pairs[1]
        .0
        .project(&Projector::from_pairs([(group.c1.clone(), 1)]).unwrap())
        .unwrap()
        .unwrap();
    let indices = group.expected.indices().to_vec();
    let values = (0..32)
        .map(|flat| {
            let x = flat % 2;
            let h = (flat / 4) % 2;
            let e = (flat / 8) % 2;
            let y = (flat / 16) % 2;
            8. + f64::from(h == 0 && e == 0 && x == y)
        })
        .collect();
    let expected = IdxTensor::from_dense(indices, values).unwrap();
    let opts = options(
        vec![group.x.clone()],
        PatchSplitStrategy::Sequential,
        Some(2),
    );
    let (result, probes) = measure(|| {
        contract_group_project_first(
            group.pairs.clone(),
            &0,
            &ContractionOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(1e-12)),
            &opts,
        )
    });
    let result = result.unwrap();
    assert_eq!(probes[0], (2, 2, 2));
    let error = dense_error(result, &expected);
    assert!(error < 1e-10, "max-absolute residual {error:e}");
}

#[test]
fn edgeless_cap_one_contributions_still_split() {
    let x = DynIndex::new_dyn(2);
    let k = x.prime();
    let left = SubDomainTreeTN::from_treetn(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![x.clone(), k.clone()], vec![1., 2., 4., 5.]).unwrap()],
            vec![0],
        )
        .unwrap(),
    )
    .unwrap();
    let right_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![k.clone()], vec![3., -1.]).unwrap()],
        vec![0],
    )
    .unwrap();
    let left = PartitionedTreeTN::from_subdomains(vec![left]).unwrap();
    let right = PartitionedTreeTN::from_subdomains(right_patches(&right_tree, &k)).unwrap();
    let opts = options(vec![x.clone()], PatchSplitStrategy::Sequential, Some(1));
    let (result, probes) =
        measure(|| contract_adaptive(&left, &right, &0, &ContractionOptions::zipup(), &opts));
    let result = result.unwrap();
    assert_eq!(probes[0], (2, 1, 1));
    assert_eq!(result.len(), 2);
    for coordinate in 0..2 {
        let key = Projector::from_pairs([(x.clone(), coordinate)]).unwrap();
        assert!(result.get(&key).is_some());
    }
    assert_partition(
        &result,
        &IdxTensor::from_dense(vec![x], vec![-1., 1.]).unwrap(),
    );
}

#[test]
fn selected_descendants_do_not_select_an_unmatched_ancestor_group() {
    let group = degree_three_group();
    let (left, right) = partitions(&group);
    let opts = options(
        vec![group.x.clone()],
        PatchSplitStrategy::Sequential,
        Some(2),
    );
    let descendant = Projector::from_pairs([(group.x.clone(), 0)]).unwrap();
    let (result, probes) = measure(|| {
        contract_adaptive_selected(
            &left,
            &right,
            &0,
            &ContractionOptions::zipup(),
            &opts,
            &[descendant],
        )
    });
    assert!(result.unwrap().is_empty());
    assert!(probes.is_empty());
}

#[test]
fn subdomain_outcome_retains_budget_and_rejects_conflicting_regions() {
    let group = degree_three_group();
    let (left, right) = &group.pairs[0];
    let left = left.clone().with_budget_squared(0.25).unwrap();
    let result = left
        .contract_with_outcome(
            right,
            &0,
            ContractionOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(1e-12)),
            std::num::NonZeroUsize::new(3).unwrap(),
        )
        .unwrap()
        .unwrap();
    let SubDomainContractionOutcome::Completed(result) = result else {
        panic!("below cap three")
    };
    assert_eq!(result.budget_squared(), Some(0.25));
    let conflict = left
        .project(&Projector::from_pairs([(group.c1.clone(), 1)]).unwrap())
        .unwrap()
        .unwrap();
    assert!(conflict
        .contract_with_outcome(
            right,
            &0,
            ContractionOptions::zipup(),
            std::num::NonZeroUsize::new(1).unwrap()
        )
        .unwrap()
        .is_none());
    assert!(matches!(
        conflict.contract_with_outcome(
            right,
            &0,
            ContractionOptions::zipup().with_max_bond_dim(0),
            std::num::NonZeroUsize::new(1).unwrap()
        ),
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}
