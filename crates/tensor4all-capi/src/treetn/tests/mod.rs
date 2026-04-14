use super::*;
use crate::index::{t4a_index_new, t4a_index_release};
use crate::tensor::{
    t4a_tensor_copy_dense_f64, t4a_tensor_new_dense_f64, t4a_tensor_rank, t4a_tensor_release,
};

fn new_index(dim: usize) -> *mut t4a_index {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_index_new(dim, std::ptr::null(), 0, &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn new_tensor(indices: &[*const t4a_index], data: &[f64]) -> *mut t4a_tensor {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            indices.len(),
            indices.as_ptr(),
            data.as_ptr(),
            data.len(),
            &mut out
        ),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn read_dense_f64_tensor(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut data = vec![0.0; len];
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, data.as_mut_ptr(), data.len(), &mut len),
        T4A_SUCCESS
    );
    data
}

fn new_treetn(tensors: &[*const t4a_tensor]) -> *mut t4a_treetn {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_new(tensors.as_ptr(), tensors.len(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn read_dense_f64_treetn(treetn: *const t4a_treetn) -> Vec<f64> {
    let mut dense = std::ptr::null_mut();
    assert_eq!(t4a_treetn_to_dense(treetn, &mut dense), T4A_SUCCESS);
    let values = read_dense_f64_tensor(dense);
    t4a_tensor_release(dense);
    values
}

fn make_two_site_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let s0 = new_index(2);
    let bond = new_index(2);
    let s1 = new_index(2);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    // t0(s0, bond) = [[1, 0], [0, 1]]
    let t0 = new_tensor(
        &[s0 as *const t4a_index, bond as *const t4a_index],
        &[1.0, 0.0, 0.0, 1.0],
    );
    // t1(bond, s1) = [[1, 3], [2, 4]]
    let t1 = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    (tt, vec![t0, t1], vec![s0, bond, bond_clone, s1])
}

fn cleanup(tt: *mut t4a_treetn, tensors: Vec<*mut t4a_tensor>, indices: Vec<*mut t4a_index>) {
    t4a_treetn_release(tt);
    for tensor in tensors {
        t4a_tensor_release(tensor);
    }
    for index in indices {
        t4a_index_release(index);
    }
}

#[test]
fn test_treetn_topology_queries() {
    let (tt, tensors, indices) = make_two_site_treetn();

    let mut n_vertices = 0usize;
    assert_eq!(t4a_treetn_num_vertices(tt, &mut n_vertices), T4A_SUCCESS);
    assert_eq!(n_vertices, 2);

    let mut neigh_len = 0usize;
    assert_eq!(
        t4a_treetn_neighbors(tt, 0, std::ptr::null_mut(), 0, &mut neigh_len),
        T4A_SUCCESS
    );
    assert_eq!(neigh_len, 1);
    let mut neighbors = vec![0usize; neigh_len];
    assert_eq!(
        t4a_treetn_neighbors(
            tt,
            0,
            neighbors.as_mut_ptr(),
            neighbors.len(),
            &mut neigh_len
        ),
        T4A_SUCCESS
    );
    assert_eq!(neighbors, vec![1]);

    let mut siteind_len = 0usize;
    assert_eq!(
        t4a_treetn_siteinds(tt, 0, std::ptr::null_mut(), 0, &mut siteind_len),
        T4A_SUCCESS
    );
    assert_eq!(siteind_len, 1);
    let mut siteinds = vec![std::ptr::null_mut(); siteind_len];
    assert_eq!(
        t4a_treetn_siteinds(
            tt,
            0,
            siteinds.as_mut_ptr(),
            siteinds.len(),
            &mut siteind_len
        ),
        T4A_SUCCESS
    );
    let mut site_dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(siteinds[0], &mut site_dim),
        T4A_SUCCESS
    );
    assert_eq!(site_dim, 2);
    t4a_index_release(siteinds[0]);

    let mut link = std::ptr::null_mut();
    assert_eq!(t4a_treetn_linkind(tt, 0, 1, &mut link), T4A_SUCCESS);
    let mut link_dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(link, &mut link_dim),
        T4A_SUCCESS
    );
    assert_eq!(link_dim, 2);
    t4a_index_release(link);

    let mut tensor = std::ptr::null_mut();
    assert_eq!(t4a_treetn_tensor(tt, 0, &mut tensor), T4A_SUCCESS);
    let mut rank = 0usize;
    assert_eq!(t4a_tensor_rank(tensor, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);
    t4a_tensor_release(tensor);

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_set_tensor_and_to_dense() {
    let (tt, mut tensors, mut indices) = make_two_site_treetn();

    let bond = indices[1];
    let s1 = indices[3];
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );
    let replacement = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[10.0, 20.0, 30.0, 40.0],
    );
    assert_eq!(t4a_treetn_set_tensor(tt, 1, replacement), T4A_SUCCESS);
    assert_eq!(read_dense_f64_treetn(tt), vec![10.0, 20.0, 30.0, 40.0]);

    tensors.push(replacement);
    indices.push(bond_clone);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_orthogonalize_preserves_dense_tensor() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let before = read_dense_f64_treetn(tt);
    assert_eq!(
        t4a_treetn_orthogonalize(tt, 0, t4a_canonical_form::LU),
        T4A_SUCCESS
    );
    let after = read_dense_f64_treetn(tt);
    assert_eq!(before, after);
    cleanup(tt, tensors, indices);
}

fn make_truncatable_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
    let s0 = new_index(2);
    let bond = new_index(4);
    let s1 = new_index(2);
    let mut bond_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(bond, &mut bond_clone),
        T4A_SUCCESS
    );

    let t0 = new_tensor(
        &[s0 as *const t4a_index, bond as *const t4a_index],
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
    let t1 = new_tensor(
        &[bond_clone as *const t4a_index, s1 as *const t4a_index],
        &[1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0],
    );
    let tt = new_treetn(&[t0 as *const t4a_tensor, t1 as *const t4a_tensor]);
    (tt, vec![t0, t1], vec![s0, bond, bond_clone, s1])
}

fn link_dim(tt: *const t4a_treetn) -> usize {
    let mut link = std::ptr::null_mut();
    assert_eq!(t4a_treetn_linkind(tt, 0, 1, &mut link), T4A_SUCCESS);
    let mut dim = 0usize;
    assert_eq!(crate::index::t4a_index_dim(link, &mut dim), T4A_SUCCESS);
    t4a_index_release(link);
    dim
}

#[test]
fn test_treetn_truncate_with_rtol_and_cutoff() {
    let (tt_rtol, tensors_rtol, indices_rtol) = make_truncatable_treetn();
    assert_eq!(link_dim(tt_rtol), 4);
    assert_eq!(t4a_treetn_truncate(tt_rtol, 1e-12, 0.0, 1), T4A_SUCCESS);
    assert_eq!(link_dim(tt_rtol), 1);
    cleanup(tt_rtol, tensors_rtol, indices_rtol);

    let (tt_cutoff, tensors_cutoff, indices_cutoff) = make_truncatable_treetn();
    assert_eq!(link_dim(tt_cutoff), 4);
    assert_eq!(t4a_treetn_truncate(tt_cutoff, 0.0, 1e-24, 1), T4A_SUCCESS);
    assert_eq!(link_dim(tt_cutoff), 1);
    cleanup(tt_cutoff, tensors_cutoff, indices_cutoff);
}

#[test]
fn test_treetn_evaluate_multiple_points() {
    let (tt, tensors, indices) = make_two_site_treetn();
    let s0 = indices[0];
    let s1 = indices[3];
    let index_ptrs = [s0 as *const t4a_index, s1 as *const t4a_index];
    let values = [
        0usize, 0usize, // point 0
        1usize, 0usize, // point 1
        1usize, 1usize, // point 2
    ];
    let mut out_re = [0.0; 3];
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            index_ptrs.as_ptr(),
            2,
            values.as_ptr(),
            3,
            out_re.as_mut_ptr(),
            std::ptr::null_mut()
        ),
        T4A_SUCCESS
    );
    assert_eq!(out_re, [1.0, 2.0, 4.0]);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_contract_and_to_dense() {
    let in_idx = new_index(2);
    let out_idx = new_index(2);
    let mut in_clone = std::ptr::null_mut();
    assert_eq!(
        crate::index::t4a_index_clone(in_idx, &mut in_clone),
        T4A_SUCCESS
    );

    let op_tensor = new_tensor(
        &[out_idx as *const t4a_index, in_idx as *const t4a_index],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let state_tensor = new_tensor(&[in_clone as *const t4a_index], &[5.0, 7.0]);
    let op = new_treetn(&[op_tensor as *const t4a_tensor]);
    let state = new_treetn(&[state_tensor as *const t4a_tensor]);

    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_contract(
            op,
            state,
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(result), vec![26.0, 38.0]);

    t4a_treetn_release(result);
    t4a_treetn_release(state);
    t4a_treetn_release(op);
    t4a_tensor_release(state_tensor);
    t4a_tensor_release(op_tensor);
    t4a_index_release(in_clone);
    t4a_index_release(out_idx);
    t4a_index_release(in_idx);
}
