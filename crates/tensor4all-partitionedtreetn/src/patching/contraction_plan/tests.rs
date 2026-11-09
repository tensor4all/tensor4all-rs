use super::*;
use crate::partitioned_tree_tn::CONTRACTED_PAIRS;
use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

fn nested_outputs() -> (PartitionedTreeTN, PartitionedTreeTN, DynIndex, DynIndex) {
    let k = DynIndex::new_dyn(3);
    let x = DynIndex::new_dyn(3);
    let left = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![k.clone(), x.clone()], vec![1.0; 9]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let right = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![k.clone()], vec![2.0, 3.0, 5.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let patch = |tree: &TreeTN<IdxTensor, usize>, entries| {
        SubDomainTreeTN::new(tree.clone(), Projector::from_pairs(entries).unwrap())
            .unwrap()
            .with_budget_squared(0.25)
            .unwrap()
    };
    (
        PartitionedTreeTN::from_subdomains(vec![
            patch(&left, vec![(k.clone(), 0)]),
            patch(&left, vec![(k.clone(), 1), (x.clone(), 0)]),
            patch(&left, vec![(k.clone(), 2)]),
        ])
        .unwrap(),
        PartitionedTreeTN::from_subdomain(
            SubDomainTreeTN::from_treetn(right)
                .unwrap()
                .with_budget_squared(0.5)
                .unwrap(),
        )
        .unwrap(),
        k,
        x,
    )
}

#[test]
fn grid_limit_precedes_numerical_attempts_and_preserves_original_pair_order() {
    let (left, right, k, x) = nested_outputs();
    CONTRACTED_PAIRS.with(|count| count.set(0));
    assert!(matches!(
        plan_groups(&left, &right, &0, None, 2),
        Err(PartitionedTreeTNError::RefinementTooLarge { atoms: 3, limit: 2 })
    ));
    assert_eq!(CONTRACTED_PAIRS.with(|count| count.get()), 0);
    let groups = plan_groups(&left, &right, &0, None, 3).unwrap();
    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![3, 2, 2]
    );
    assert_eq!(
        groups[0]
            .iter()
            .map(|(a, _)| a.projector().get(&k))
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)]
    );
    for (coordinate, pairs) in groups.iter().enumerate() {
        for (a, b) in pairs {
            assert_eq!(a.projector().get(&x), Some(coordinate));
            assert_eq!(b.projector().get(&x), None);
            assert_eq!(a.budget_squared(), Some(0.25));
            assert_eq!(b.budget_squared(), Some(0.5));
        }
    }
    assert_eq!(CONTRACTED_PAIRS.with(|count| count.get()), 0);
    // Selecting the original empty key admits k=0,2, even where k=1's
    // x=0 output overlaps it. With no selected overlap, no bound applies.
    let wanted = HashSet::from([Projector::new()]);
    let selected = plan_groups(&left, &right, &0, Some(&wanted), 0).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].len(), 2);
    assert_eq!(selected[0][0].0.projector().get(&k), Some(0));
    assert_eq!(selected[0][1].0.projector().get(&k), Some(2));
}

#[test]
fn nested_output_regions_are_summed_on_disjoint_cells() {
    let (left, right, _, x) = nested_outputs();
    let options = PatchingOptions {
        cutoff: 0.0,
        max_bond_dim: None,
        ..PatchingOptions::default()
    };
    let full =
        contract_adaptive(&left, &right, &0, &ContractionOptions::zipup(), &options).unwrap();
    let expected = IdxTensor::from_dense(vec![x], vec![10.0, 7.0, 7.0]).unwrap();
    assert_eq!(full.len(), 3);
    let actual = full.to_treetn().unwrap().to_dense().unwrap();
    assert!(actual.sub(&expected).unwrap().maxabs().unwrap() < 1e-12);
}
