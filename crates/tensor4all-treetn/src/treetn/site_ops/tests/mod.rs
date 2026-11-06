//! Unit tests for [`fix_and_remove_site`](super::fix_and_remove_site).

use super::*;

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{DynIndex, IdxTensor};

fn dense(indices: Vec<DynIndex>, values: &[f64]) -> IdxTensor {
    IdxTensor::from_dense(indices, values.to_vec()).unwrap()
}

/// Two-node chain `[site_a, bond] - [bond, site_b]` with a non-trivial site
/// vector on the second node.
fn two_pole_chain<V>(names: [V; 2]) -> (TreeTN<IdxTensor, V>, DynIndex, DynIndex)
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);
    let node_a = dense(vec![site_a.clone(), bond.clone()], &[1.0, 0.0, 0.0, 1.0]);
    let node_b = dense(vec![bond, site_b.clone()], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tree = TreeTN::<IdxTensor, V>::from_tensors(vec![node_a, node_b], names.to_vec()).unwrap();
    (tree, site_a, site_b)
}

fn fixes_and_removes_last_site<V>(names: [V; 2])
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (tree, site_a, _site_b) = two_pole_chain(names.clone());
    let reduced = fix_and_remove_site(tree, &names[1], 2).unwrap();

    assert_eq!(reduced.node_names(), vec![names[0].clone()]);
    let dense = reduced.contract_to_tensor().unwrap();
    assert_eq!(dense.indices(), &[site_a]);
    // Column-major [site_a, bond]: bond = 0 and 1 entries of site_b = 2.
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![5.0, 6.0]);
}

fn rejects_multi_site_node<V>(names: [V; 2])
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let left_site = DynIndex::new_dyn(2);
    let extra_site = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    // The first node carries two site legs: the single-site contract is violated.
    let node_a = dense(vec![left_site, extra_site, bond.clone()], &[0.5; 2 * 3 * 2]);
    let node_b = dense(vec![bond, right_site], &[1.0, 2.0, 3.0, 4.0]);
    let tree = TreeTN::<IdxTensor, V>::from_tensors(vec![node_a, node_b], names.to_vec()).unwrap();

    let err = fix_and_remove_site(tree, &names[0], 0).unwrap_err();

    assert!(
        err.to_string()
            .contains("must carry exactly one site index, got 2"),
        "unexpected error: {err}"
    );
}

fn rejects_site_free_node<V>(names: [V; 2])
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let bond = DynIndex::new_dyn(3);
    let site = DynIndex::new_dyn(2);
    // The first node carries only the bond leg, so it has no site index to fix.
    let node_a = dense(vec![bond.clone()], &[1.0, 2.0, 3.0]);
    let node_b = dense(vec![bond, site], &[0.0; 6]);
    let tree = TreeTN::<IdxTensor, V>::from_tensors(vec![node_a, node_b], names.to_vec()).unwrap();

    let err = fix_and_remove_site(tree, &names[0], 0).unwrap_err();

    assert!(
        err.to_string()
            .contains("must carry exactly one site index, got 0"),
        "unexpected error: {err}"
    );
}

#[test]
fn fixes_and_removes_sites() {
    fixes_and_removes_last_site(["a", "b"]);
}

#[test]
fn rejects_multi_site_nodes() {
    rejects_multi_site_node(["a", "b"]);
}

#[test]
fn rejects_site_free_nodes() {
    rejects_site_free_node(["a", "b"]);
}
