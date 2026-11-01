//! #1078: the context-scoped shortening accepts a tree it did not build.
//!
//! A tree loaded from disk or materialised by TCI carries no eager runtime of
//! its own, so the context-scoped entries adopt it before the sweeps. A tree
//! whose tensors are bound to a *different* context stays an error: re-binding
//! those would copy their storage.

use std::sync::Arc;

use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, IndexLike};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetn::{TreeTN, TruncationOptions};

const SITES: usize = 4;

fn cpu_context(workers: usize) -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(workers).expect("CPU backend"),
    )))
}

fn chain(host_storage: bool, context: Option<&ExecutionContext>) -> TreeTN<IdxTensor, usize> {
    let sites: Vec<DynIndex> = (0..SITES).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..SITES - 1).map(|_| DynIndex::new_dyn(2)).collect();
    let mut tensors = Vec::with_capacity(SITES);
    for site in 0..SITES {
        let mut indices = vec![sites[site].clone()];
        if site > 0 {
            indices.push(bonds[site - 1].clone());
        }
        if site + 1 < SITES {
            indices.push(bonds[site].clone());
        }
        let len: usize = indices.iter().map(|index| index.dim()).product();
        let data = (0..len).map(|entry| 1.0 + 0.25 * entry as f64).collect();
        tensors.push(
            match (host_storage, context) {
                (true, _) => IdxTensor::from_dense_compact(indices, data),
                (false, Some(context)) => IdxTensor::from_dense_in(context, indices, data),
                (false, None) => IdxTensor::from_dense(indices, data),
            }
            .expect("site tensor"),
        );
    }
    TreeTN::from_tensors(tensors, (0..SITES).collect()).expect("tree")
}

fn assert_same_values(left: &TreeTN<IdxTensor, usize>, right: &TreeTN<IdxTensor, usize>) {
    let left = left.to_dense().expect("dense left");
    let right = right.to_dense().expect("dense right");
    assert!(
        left.isapprox(&right, 1e-12, 1e-12).expect("compare"),
        "the two dense readouts disagree"
    );
}

fn options(max_bond_dim: usize) -> TruncationOptions {
    TruncationOptions::default().with_max_bond_dim(max_bond_dim)
}

#[test]
fn truncate_in_adopts_host_storage_and_matches_the_context_free_result() {
    let context = cpu_context(2);
    let tree = chain(true, None);

    let in_context = tree
        .clone()
        .truncate_in([0], options(1), &context)
        .expect("context-scoped truncation of a host-storage tree");
    let plain = tree
        .truncate([0], options(1))
        .expect("context-free truncation");

    assert_eq!(in_context.link_dims(), vec![1; SITES - 1]);
    assert_eq!(in_context.link_dims(), plain.link_dims());
    assert_same_values(&in_context, &plain);
}

#[test]
fn adopt_into_in_binds_every_node_tensor_and_keeps_the_values() {
    let context = cpu_context(2);
    let mut tree = chain(true, None);
    let before = tree.clone();

    tree.adopt_into_in(&context).expect("adoption");

    for node in tree.node_indices() {
        tree.tensor(node)
            .expect("node tensor")
            .validate_context(&context)
            .expect("adopted tensor belongs to the context");
    }
    assert_same_values(&tree, &before);
}

#[test]
fn truncate_in_rejects_a_tree_bound_to_a_different_context() {
    let first = cpu_context(2);
    let second = cpu_context(2);
    let tree = chain(false, Some(&first));

    let error = tree
        .truncate_in([0], options(1), &second)
        .expect_err("a tensor bound to another context cannot be adopted without a copy");

    assert!(
        format!("{error:#}").contains("different execution context"),
        "unexpected error: {error:#}"
    );
}
