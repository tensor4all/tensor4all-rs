#![cfg(feature = "tenferro-cuda")]

//! Component-wise transfer for the TreeTN CUDA route (#553 / #859 F5).
//!
//! The slice is upload-only staged assembly: a caller can bring a network resident one component at a time, but a
//! partly resident network is not contractible. These tests pin the selection rules, the structural preservation
//! and the refusal, so that "component-wise" means measured behaviour rather than a name.

use std::sync::Arc;

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CudaTreeTNError, TreeTN};

fn dense(indices: Vec<DynIndex>, seed: usize) -> IdxTensor {
    let elements = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .expect("element count");
    IdxTensor::from_dense(
        indices,
        (0..elements)
            .map(|offset| ((seed + offset) % 19) as f64 / 19.0 + 0.25)
            .collect(),
    )
    .expect("dense tensor")
}

/// A four-node chain, including a bond index that shares its dimension with a site index but differs in prime
/// level, so structural preservation is tested on identity rather than on dimensions.
fn chain() -> TreeTN<IdxTensor, usize> {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let mut bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
    bonds[1] = bonds[1].prime();
    let mut tensors = Vec::new();
    for (node, site) in sites.iter().enumerate() {
        let mut indices = Vec::new();
        if let Some(left) = node.checked_sub(1).and_then(|index| bonds.get(index)) {
            indices.push(left.clone());
        }
        indices.push(site.clone());
        if let Some(right) = bonds.get(node) {
            indices.push(right.clone());
        }
        tensors.push(dense(indices, node * 31 + 1));
    }
    TreeTN::from_tensors(tensors, (0..4).collect()).expect("chain")
}

fn structure(tn: &TreeTN<IdxTensor, usize>) -> Vec<(usize, Vec<(i64, usize)>)> {
    // Per node: (prime level, dimension) for every index in order, so an index that only differs in prime level
    // is visible.
    let mut names = tn.node_names();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let node = tn.node_index(&name).expect("node");
            let tensor = tn.tensor(node).expect("tensor");
            let dims = tensor
                .indices()
                .iter()
                .map(|index| (index.plev, index.dim()))
                .collect();
            (name, dims)
        })
        .collect()
}

fn tensor_at(tn: &TreeTN<IdxTensor, usize>, name: usize) -> IdxTensor {
    let node = tn.node_index(&name).expect("node");
    tn.tensor(node).expect("tensor").clone()
}

fn context() -> Arc<CudaExecutionContext> {
    Arc::new(CudaExecutionContext::new().expect("CUDA ordinal 0 must be available"))
}

#[test]
fn a_partial_upload_transfers_only_the_selected_nodes() {
    let host = chain();
    let cuda = context();
    let before = structure(&host);

    // One component at a time, asserting readiness after each step through the existing validator.
    let step_one = host.upload_nodes_cuda(&[0], &cuda).expect("upload node 0");
    assert!(tensor_at(&step_one, 0)
        .validate_cuda_residency(&cuda)
        .is_ok());
    for node in 1..4 {
        assert!(
            tensor_at(&step_one, node)
                .validate_cuda_residency(&cuda)
                .is_err(),
            "node {node} must not become resident by selecting node 0"
        );
    }
    assert_eq!(structure(&step_one), before, "structure must be preserved");

    let step_two = step_one
        .upload_nodes_cuda(&[1, 2, 3], &cuda)
        .expect("upload the rest");
    for node in 0..4 {
        assert!(tensor_at(&step_two, node)
            .validate_cuda_residency(&cuda)
            .is_ok());
    }
    assert_eq!(structure(&step_two), before);

    // The whole-network entry agrees, and both contract to the same host result.
    let whole = host.upload_cuda(&cuda).expect("whole upload");
    assert_eq!(structure(&whole), before);
    let step_values = step_two
        .contract_to_tensor_cuda(&cuda)
        .expect("contract")
        .download(&cuda)
        .expect("download")
        .to_vec::<f64>()
        .expect("values");
    let whole_values = whole
        .contract_to_tensor_cuda(&cuda)
        .expect("contract")
        .download(&cuda)
        .expect("download")
        .to_vec::<f64>()
        .expect("values");
    let host_values = host
        .contract_to_tensor()
        .expect("host")
        .to_vec::<f64>()
        .expect("values");
    assert_eq!(step_values, whole_values);
    for (value, reference) in step_values.iter().zip(&host_values) {
        assert!((value - reference).abs() <= 1e-10, "{value} != {reference}");
    }
}

#[test]
fn selection_rules_are_resolved_before_any_transfer() {
    let host = chain();
    let cuda = context();

    // Empty selection: a clone, nothing transferred.
    let empty = host.upload_nodes_cuda(&[], &cuda).expect("empty selection");
    assert_eq!(structure(&empty), structure(&host));
    assert!(tensor_at(&empty, 0).validate_cuda_residency(&cuda).is_err());

    // Unknown name: typed, and nothing transferred.
    let error = host
        .upload_nodes_cuda(&[9], &cuda)
        .expect_err("an unknown name must be refused");
    assert!(
        matches!(error, CudaTreeTNError::UnknownNode { .. }),
        "{error}"
    );
    assert!(tensor_at(&host, 0).validate_cuda_residency(&cuda).is_err());

    // Duplicates collapse; selecting an already-resident node is a no-op rather than a re-upload.
    let once = host.upload_nodes_cuda(&[0], &cuda).expect("first");
    let _twice = once
        .upload_nodes_cuda(&[0, 0], &cuda)
        .expect("duplicate selection");
    assert!(
        once.contract_to_tensor_cuda(&cuda).is_err(),
        "a partly resident network is not contractible"
    );

    // Resident in this context already: selecting it again is a no-op, not a second transfer - and the values are
    // still the same.
    let again = once
        .upload_nodes_cuda(&[0], &cuda)
        .expect("re-selecting a resident node");
    assert_eq!(structure(&again), structure(&once));
}

#[test]
fn a_failed_component_leaves_the_callers_values_and_graph_unchanged() {
    let host = chain();
    let cuda = context();
    let before = structure(&host);

    // Node 0 uploads, node 9 in the same call does not exist: the call must not publish anything.
    let error = host
        .upload_nodes_cuda(&[0, 9], &cuda)
        .expect_err("a partially invalid selection must be refused");
    assert!(
        matches!(error, CudaTreeTNError::UnknownNode { .. }),
        "{error}"
    );
    assert_eq!(
        structure(&host),
        before,
        "the source graph must be unchanged"
    );
    // The documented side effect: a source tensor can gain a lazily initialised CPU eager cache. Its eager-context
    // residency is therefore reported through the existing validator rather than asserted here.
}
