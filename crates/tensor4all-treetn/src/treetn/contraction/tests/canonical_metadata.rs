//! Canonical certificates on the general-tree schedule (#1290).

use super::*;
use crate::treetn::contraction::{
    GlobalLeaves, IdentityNodes, LeafOps, SessionLeaves, ZipupTopologyMode,
};

/// A four-node GHZ state with amplitudes 1 and 8, and a bond-one local
/// operator. Nodes outside `outputs` have their site summed out.
fn star(outputs: &[usize]) -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let op_bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(1)).collect();
    let mut hub_indices = vec![sites[0].clone()];
    hub_indices.extend(bonds.iter().cloned());
    let mut hub = vec![0.0; 16];
    hub[0] = 1.0;
    hub[15] = 1.0;
    let mut states = vec![IdxTensor::from_dense(hub_indices, hub).unwrap()];
    let mut operators = Vec::new();
    for node in 0..4 {
        if node != 0 {
            states.push(
                IdxTensor::from_dense(
                    vec![sites[node].clone(), bonds[node - 1].clone()],
                    vec![1.0, 0.0, 0.0, 2.0],
                )
                .unwrap(),
            );
        }
        let mut indices = vec![sites[node].clone()];
        let data = if outputs.contains(&node) {
            indices.push(DynIndex::new_dyn(2));
            vec![1.0, 0.0, 0.0, 1.0]
        } else {
            vec![1.0, 1.0]
        };
        if node == 0 {
            indices.extend(op_bonds.iter().cloned());
        } else {
            indices.push(op_bonds[node - 1].clone());
        }
        operators.push(IdxTensor::from_dense(indices, data).unwrap());
    }
    (
        TreeTN::from_tensors(operators, vec![0, 1, 2, 3]).unwrap(),
        TreeTN::from_tensors(states, vec![0, 1, 2, 3]).unwrap(),
    )
}

fn assert_certificate(result: &TreeTN<IdxTensor, usize>, center: usize, form: CanonicalForm) {
    assert_eq!(result.canonical_region(), &[center].into_iter().collect());
    assert_eq!(result.canonical_form(), Some(form));
    result.validate_ortho_consistency().unwrap();
    // Check the complete map too: validation alone does not reject stale keys
    // for pruned bonds that are no longer in the graph.
    assert_eq!(result.ortho_towards.len(), result.edge_count());
    for (child, parent) in result.edges_to_canonicalize_by_names(&center).unwrap() {
        let edge = result.edge_between(&child, &parent).unwrap();
        assert_eq!(result.ortho_towards_node(edge), Some(&parent));
    }
}

#[test]
fn truncating_svd_tree_zipup_certifies_every_center_on_both_leaf_routes() {
    let context = cpu_context();
    for scoped in [false, true] {
        let (mut operator, mut state) = star(&[0, 1, 2, 3]);
        let expected = operator.contract_naive(&state).unwrap();
        if scoped {
            operator.adopt_into_in(&context).unwrap();
            state.adopt_into_in(&context).unwrap();
        }
        let session = SessionLeaves(context.clone());
        let leaves: &dyn LeafOps<IdxTensor> = if scoped { &session } else { &GlobalLeaves };
        for center in 0..4 {
            let result = operator
                .contract_zipup_impl(
                    &state,
                    &center,
                    CanonicalForm::Unitary,
                    Some(SvdTruncationPolicy::new(1e-12)),
                    Some(1),
                    ZipupTopologyMode::PruneScalarSubtrees,
                    &IdentityNodes::new(),
                    leaves,
                )
                .unwrap();
            assert_certificate(&result, center, CanonicalForm::Unitary);
            assert_eq!(result.link_dims(), vec![1; 3]);
            let dense = result.contract_to_tensor_with_leaves(leaves).unwrap();
            let dense =
                IdxTensor::from_storage(dense.indices().to_vec(), dense.to_storage().unwrap())
                    .unwrap();
            let error = dense.sub(&expected).unwrap().maxabs().unwrap();
            assert!(
                (error - 1.0).abs() < 1e-12,
                "discarded GHZ amplitude: {error:e}"
            );
            let center_tensor = result.tensor(result.node_index(&center).unwrap()).unwrap();
            let center_tensor = IdxTensor::from_storage(
                center_tensor.indices().to_vec(),
                center_tensor.to_storage().unwrap(),
            )
            .unwrap();
            let center_norm = center_tensor.norm().unwrap();
            assert!(
                (center_norm - 8.0).abs() < 1e-12,
                "retained GHZ norm: {center_norm:e}"
            );
        }
    }
}

#[test]
fn lu_and_ci_tree_zipup_follow_canonicalize_direction_conventions() {
    let (operator, state) = star(&[0, 1, 2, 3]);
    let expected = operator.contract_naive(&state).unwrap();
    for form in [CanonicalForm::LU, CanonicalForm::CI] {
        for center in 0..4 {
            let result = operator
                .contract_zipup_with(&state, &center, form, None, None)
                .unwrap();
            assert_certificate(&result, center, form);
            let error = result
                .to_dense()
                .unwrap()
                .sub(&expected)
                .unwrap()
                .maxabs()
                .unwrap();
            assert!(
                error < 1e-12,
                "{form:?} GHZ reconstruction error: {error:e}"
            );
        }
    }
}

#[test]
fn tree_zipup_certifies_only_surviving_edges_or_preserves_scalar_nodes() {
    let context = cpu_context();
    for outputs in [vec![0, 2, 3], vec![]] {
        for scoped in [false, true] {
            let (mut operator, mut state) = star(&outputs);
            let expected = operator.contract_naive(&state).unwrap();
            if scoped {
                operator.adopt_into_in(&context).unwrap();
                state.adopt_into_in(&context).unwrap();
            }
            let session = SessionLeaves(context.clone());
            let leaves: &dyn LeafOps<IdxTensor> = if scoped { &session } else { &GlobalLeaves };
            for mode in [
                ZipupTopologyMode::PruneScalarSubtrees,
                ZipupTopologyMode::PreserveInputTopology,
            ] {
                let result = operator
                    .contract_zipup_impl(
                        &state,
                        &0,
                        CanonicalForm::Unitary,
                        Some(SvdTruncationPolicy::new(0.0)),
                        None,
                        mode,
                        &IdentityNodes::new(),
                        leaves,
                    )
                    .unwrap();
                assert_certificate(&result, 0, CanonicalForm::Unitary);
                if mode == ZipupTopologyMode::PruneScalarSubtrees {
                    assert_eq!(result.node_count(), if outputs.is_empty() { 1 } else { 3 });
                    assert!(result.node_index(&1).is_none());
                } else {
                    assert_eq!(result.node_count(), 4);
                    let edge = result.edge_between(&1, &0).unwrap();
                    assert_eq!(result.bond_index(edge).unwrap().dim(), 1);
                }
                let dense = result.contract_to_tensor_with_leaves(leaves).unwrap();
                let dense =
                    IdxTensor::from_storage(dense.indices().to_vec(), dense.to_storage().unwrap())
                        .unwrap();
                let error = dense.sub(&expected).unwrap().maxabs().unwrap();
                assert!(
                    error < 1e-12,
                    "scalar-subtree reconstruction error: {error:e}"
                );
            }
        }
    }
}
