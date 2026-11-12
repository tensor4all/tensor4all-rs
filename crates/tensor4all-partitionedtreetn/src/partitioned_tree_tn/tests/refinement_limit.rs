//! The ceiling on the automatic refinement of `add` and `contract`, at an
//! injected small limit so the boundary case needs only a few atoms.

use super::super::*;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

fn single_node(indices: &[DynIndex]) -> TreeTN<IdxTensor, usize> {
    let len = indices.iter().map(|index| index.dim).product();
    let values: Vec<f64> = (0..len).map(|value| 1.0 + value as f64).collect();
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(indices.to_vec(), values).unwrap()],
        vec![0usize],
    )
    .unwrap()
}

fn partition(
    tree: &TreeTN<IdxTensor, usize>,
    projectors: &[Vec<(&DynIndex, usize)>],
) -> PartitionedTreeTN {
    PartitionedTreeTN::from_subdomains(
        projectors
            .iter()
            .map(|pairs| {
                let projector = Projector::from_pairs(
                    pairs
                        .iter()
                        .map(|(index, value)| ((*index).clone(), *value)),
                )
                .unwrap();
                SubDomainTreeTN::new(tree.clone(), projector).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn add_refines_at_the_limit_and_rejects_one_atom_more() {
    // `{a=0}` against `{}` refines onto one atom per coordinate of `a`.
    let a = DynIndex::new_dyn(3);
    let tree = single_node(std::slice::from_ref(&a));
    let unprojected = partition(&tree, &[vec![]]);
    let projected = partition(&tree, &[vec![(&a, 0)]]);

    let sum = unprojected.add_with(&projected, None, 3).unwrap();
    assert_eq!(sum.len(), 3);
    assert!(matches!(
        unprojected.add_with(&projected, None, 2),
        Err(PartitionedTreeTNError::RefinementTooLarge { atoms: 3, limit: 2 })
    ));
}

#[test]
fn contract_refines_at_the_limit_and_rejects_one_atom_more() {
    // `k` is contracted. The right patches fix the uncontracted `x` and `y`,
    // whose outputs `{x=0}` and `{y=0}` overlap, so the refinement grid has
    // 2 * 2 atoms.
    let k = DynIndex::new_dyn(2);
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let left = partition(
        &single_node(std::slice::from_ref(&k)),
        &[vec![(&k, 0)], vec![(&k, 1)]],
    );
    let right = partition(
        &single_node(&[k.clone(), x.clone(), y.clone()]),
        &[vec![(&k, 0), (&x, 0)], vec![(&k, 1), (&y, 0)]],
    );

    let (refined_left, refined_right) = left
        .refined_for_contract(&right, &0, 4)
        .unwrap()
        .expect("overlapping outputs are refined");
    assert_eq!(refined_left.len(), 2);
    assert_eq!(refined_right.len(), 4);
    assert!(matches!(
        left.refined_for_contract(&right, &0, 3),
        Err(PartitionedTreeTNError::RefinementTooLarge { atoms: 4, limit: 3 })
    ));
}

#[test]
fn bounded_output_grid_handles_empty_dimension_one_and_saturating_counts() {
    assert_eq!(bounded_value_grid(&[], 1).unwrap(), vec![Projector::new()]);
    assert!(matches!(
        bounded_value_grid(&[], 0),
        Err(PartitionedTreeTNError::RefinementTooLarge { atoms: 1, limit: 0 })
    ));
    let one = DynIndex::new_dyn(1);
    assert_eq!(
        bounded_value_grid(std::slice::from_ref(&one), 1).unwrap(),
        vec![Projector::from_pairs([(one, 0)]).unwrap()]
    );
    let huge = DynIndex::new_dyn(usize::MAX);
    let two = DynIndex::new_dyn(2);
    assert!(matches!(
        bounded_value_grid(&[huge, two], AUTOMATIC_REFINEMENT_LIMIT),
        Err(PartitionedTreeTNError::RefinementTooLarge {
            atoms: usize::MAX,
            limit: AUTOMATIC_REFINEMENT_LIMIT
        })
    ));
}
