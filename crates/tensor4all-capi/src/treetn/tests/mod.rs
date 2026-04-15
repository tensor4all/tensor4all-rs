use super::*;
use crate::index::{t4a_index_new, t4a_index_release};
use crate::tensor::{
    t4a_tensor_copy_dense_f64, t4a_tensor_new_dense_c64, t4a_tensor_new_dense_f64, t4a_tensor_rank,
    t4a_tensor_release,
};

fn last_error() -> String {
    let mut len = 0usize;
    assert_eq!(
        crate::t4a_last_error_message(std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        crate::t4a_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    std::ffi::CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

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

fn read_siteinds(tt: *const t4a_treetn, vertex: usize) -> Vec<*mut t4a_index> {
    let mut len = 0usize;
    assert_eq!(
        t4a_treetn_siteinds(tt, vertex, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut indices = vec![std::ptr::null_mut(); len];
    assert_eq!(
        t4a_treetn_siteinds(tt, vertex, indices.as_mut_ptr(), indices.len(), &mut len),
        T4A_SUCCESS
    );
    indices
}

fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (slot, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - e).abs();
        assert!(
            err <= tol,
            "value mismatch at slot {slot}: actual={a}, expected={e}, err={err}, tol={tol}"
        );
    }
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
fn test_treetn_clone_inner_norm_and_assignment() {
    let (tt, tensors, indices) = make_two_site_treetn();

    assert_eq!(t4a_treetn_is_assigned(tt), 1);
    assert_eq!(t4a_treetn_is_assigned(std::ptr::null()), 0);

    let mut clone = std::ptr::null_mut();
    assert_eq!(t4a_treetn_clone(tt, &mut clone), T4A_SUCCESS);
    assert!(!clone.is_null());

    let mut inner_re = 0.0;
    let mut inner_im = 0.0;
    assert_eq!(
        t4a_treetn_inner(tt, clone, &mut inner_re, &mut inner_im),
        T4A_SUCCESS
    );
    assert!((inner_re - 30.0).abs() < 1e-12);
    assert_eq!(inner_im, 0.0);

    let mut norm = 0.0;
    assert_eq!(t4a_treetn_norm(clone, &mut norm), T4A_SUCCESS);
    assert!((norm * norm - inner_re).abs() < 1e-12);

    t4a_treetn_release(clone);
    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_query_helpers_reject_invalid_requests() {
    let (tt, tensors, indices) = make_two_site_treetn();

    assert_eq!(
        t4a_treetn_num_vertices(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_n is null"));

    let mut neigh_len = 0usize;
    let mut neighbors: [usize; 0] = [];
    assert_eq!(
        t4a_treetn_neighbors(tt, 0, neighbors.as_mut_ptr(), 0, &mut neigh_len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(neigh_len, 1);

    assert_eq!(
        t4a_treetn_neighbors(tt, 2, std::ptr::null_mut(), 0, &mut neigh_len),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("vertex 2 does not exist"));

    let mut siteind_len = 0usize;
    let mut siteinds: [*mut t4a_index; 0] = [];
    assert_eq!(
        t4a_treetn_siteinds(tt, 0, siteinds.as_mut_ptr(), 0, &mut siteind_len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(siteind_len, 1);

    let mut tensor = std::ptr::null_mut();
    assert_eq!(t4a_treetn_tensor(tt, 9, &mut tensor), T4A_INVALID_ARGUMENT);
    assert!(tensor.is_null());

    let mut link = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_linkind(tt, 0, 0, &mut link),
        T4A_INVALID_ARGUMENT
    );
    assert!(link.is_null());

    cleanup(tt, tensors, indices);
}

#[test]
fn test_treetn_creation_and_scalar_api_validation_errors() {
    let mut tt = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 0, &mut tt),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("n_tensors must be greater than zero"));

    assert_eq!(
        t4a_treetn_new(std::ptr::null(), 1, &mut tt),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("tensors is null"));

    let (tt, tensors, indices) = make_two_site_treetn();
    assert_eq!(
        t4a_treetn_set_tensor(tt, 9, tensors[0] as *const t4a_tensor),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("vertex 9 does not exist"));

    assert_eq!(
        t4a_treetn_set_tensor(tt, 0, std::ptr::null()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("tensor is null"));

    let mut inner_re = 0.0;
    assert_eq!(
        t4a_treetn_inner(tt, tt, &mut inner_re, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_re or out_im is null"));

    assert_eq!(t4a_treetn_norm(tt, std::ptr::null_mut()), T4A_NULL_POINTER);
    assert!(last_error().contains("out_norm is null"));

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
fn test_treetn_evaluate_complex_requires_out_im_buffer() {
    let s = new_index(2);
    let index_ptrs = [s as *const t4a_index];
    let interleaved = [1.0, 2.0, 0.0, 0.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(1, index_ptrs.as_ptr(), interleaved.as_ptr(), 2, &mut tensor),
        T4A_SUCCESS
    );

    let tt = new_treetn(&[tensor as *const t4a_tensor]);
    let values = [0usize];
    let mut out_re = 0.0;
    assert_eq!(
        t4a_treetn_evaluate(
            tt,
            index_ptrs.as_ptr(),
            1,
            values.as_ptr(),
            1,
            &mut out_re,
            std::ptr::null_mut()
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("out_im is required"));

    t4a_treetn_release(tt);
    t4a_tensor_release(tensor);
    t4a_index_release(s);
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

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
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

#[test]
fn test_treetn_apply_operator_chain_identity_single_site() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.5, -2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [0usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64_treetn(result), vec![1.5, -2.0]);

    let siteinds = read_siteinds(result, 0);
    assert_eq!(siteinds.len(), 1);
    let mut dim = 0usize;
    assert_eq!(
        crate::index::t4a_index_dim(siteinds[0], &mut dim),
        T4A_SUCCESS
    );
    assert_eq!(dim, 2);
    t4a_index_release(siteinds[0]);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}

#[test]
fn test_treetn_apply_operator_chain_partial_operator() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[2.0, 0.0, 0.0, 5.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [1usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];

    let mut result: *mut t4a_treetn = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_SUCCESS
    );
    let dense = read_dense_f64_treetn(result);
    assert_vec_close(&dense, &[2.0, 4.0, 15.0, 20.0], 1e-10);

    t4a_treetn_release(result);
    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_rejects_out_of_range_mapped_node() {
    let (state_tt, state_tensors, mut state_indices) = make_two_site_treetn();

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [2usize];
    let input_indices = [internal_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    let err = last_error();
    assert!(
        err.contains("mapped node position 2"),
        "unexpected error: {err}"
    );
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_tensor_release(op);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    cleanup(state_tt, state_tensors, std::mem::take(&mut state_indices));
}

#[test]
fn test_treetn_apply_operator_chain_rejects_empty_mapping() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.0, 2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes: [usize; 0] = [];
    let input_indices: [*const t4a_index; 0] = [];
    let output_indices: [*const t4a_index; 0] = [];
    let true_output_indices: [*const t4a_index; 0] = [];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("mapped node positions must not be empty"));
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}

#[test]
fn test_treetn_apply_operator_chain_rejects_unknown_internal_input_index() {
    let state_site = new_index(2);
    let state = new_tensor(&[state_site as *const t4a_index], &[1.0, 2.0]);
    let state_tt = new_treetn(&[state as *const t4a_tensor]);

    let internal_in = new_index(2);
    let internal_out = new_index(2);
    let wrong_in = new_index(2);
    let target_out = new_index(2);
    let op = new_tensor(
        &[
            internal_out as *const t4a_index,
            internal_in as *const t4a_index,
        ],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let op_tt = new_treetn(&[op as *const t4a_tensor]);

    let mapped_nodes = [0usize];
    let input_indices = [wrong_in as *const t4a_index];
    let output_indices = [internal_out as *const t4a_index];
    let true_output_indices = [target_out as *const t4a_index];
    let mut result = std::ptr::null_mut();
    assert_eq!(
        t4a_treetn_apply_operator_chain(
            op_tt,
            state_tt,
            mapped_nodes.as_ptr(),
            mapped_nodes.len(),
            input_indices.as_ptr(),
            output_indices.as_ptr(),
            true_output_indices.as_ptr(),
            t4a_contract_method::Naive,
            0.0,
            0.0,
            0,
            &mut result
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("provided internal input index"));
    assert!(result.is_null());

    t4a_treetn_release(op_tt);
    t4a_treetn_release(state_tt);
    t4a_tensor_release(op);
    t4a_tensor_release(state);
    t4a_index_release(target_out);
    t4a_index_release(wrong_in);
    t4a_index_release(internal_out);
    t4a_index_release(internal_in);
    t4a_index_release(state_site);
}
