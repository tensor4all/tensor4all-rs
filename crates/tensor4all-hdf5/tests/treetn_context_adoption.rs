//! #1078: a tree read back from disk shortens in a caller-owned context.
//!
//! `load_treetn` publishes its node tensors through `IdxTensor::from_dense`,
//! i.e. bound to the process-global implicit runtime, so this is the "loaded
//! from disk" case of the acceptance set: the context-scoped shortening has to
//! adopt the loaded tree instead of refusing it.

use std::sync::Arc;

use tensor4all_core::index::DynIndex;
use tensor4all_core::{ExecutionContext, IdxTensor, IndexLike};
use tensor4all_hdf5::{load_treetn, save_treetn};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetn::{TreeTN, TruncationOptions};

const SITES: usize = 4;

fn chain() -> TreeTN<IdxTensor, usize> {
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
        tensors.push(IdxTensor::from_dense(indices, data).expect("site tensor"));
    }
    TreeTN::from_tensors(tensors, (0..SITES).collect()).expect("tree")
}

fn options(max_bond_dim: usize) -> TruncationOptions {
    TruncationOptions::default().with_max_bond_dim(max_bond_dim)
}

#[test]
fn a_tree_loaded_from_disk_shortens_in_a_caller_context() {
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(2).expect("CPU backend"),
    )));
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("treetn.h5");
    let path = path.to_str().expect("utf-8 path");
    save_treetn(path, "tn", &chain()).expect("save");
    let loaded = load_treetn::<usize>(path, "tn").expect("load");

    let in_context = loaded
        .clone()
        .truncate_in([0], options(1), &context)
        .expect("context-scoped truncation of a loaded tree");
    let plain = loaded
        .truncate([0], options(1))
        .expect("context-free truncation");

    assert_eq!(in_context.link_dims(), vec![1; SITES - 1]);
    assert_eq!(in_context.link_dims(), plain.link_dims());
    let in_context_dense = in_context.to_dense().expect("dense");
    let plain_dense = plain.to_dense().expect("dense");
    assert!(
        in_context_dense
            .isapprox(&plain_dense, 1e-12, 1e-12)
            .expect("compare"),
        "the two dense readouts disagree"
    );
}
