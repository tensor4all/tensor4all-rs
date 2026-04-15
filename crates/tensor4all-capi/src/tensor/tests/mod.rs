use super::*;
use crate::index::{t4a_index_new, t4a_index_release};

fn new_index(dim: usize) -> *mut t4a_index {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_index_new(dim, std::ptr::null(), 0, &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn new_tensor_f64(indices: &[*const t4a_index], data: &[f64]) -> *mut t4a_tensor {
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

fn read_dense_f64(tensor: *const t4a_tensor) -> Vec<f64> {
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

#[test]
fn test_tensor_dense_f64_query_then_fill() {
    let i = new_index(2);
    let j = new_index(3);
    let tensor = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut rank = 0usize;
    assert_eq!(t4a_tensor_rank(tensor, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);

    let mut dims_len = 0usize;
    assert_eq!(
        t4a_tensor_dims(tensor, std::ptr::null_mut(), 0, &mut dims_len),
        T4A_SUCCESS
    );
    assert_eq!(dims_len, 2);
    let mut dims = vec![0usize; dims_len];
    assert_eq!(
        t4a_tensor_dims(tensor, dims.as_mut_ptr(), dims.len(), &mut dims_len),
        T4A_SUCCESS
    );
    assert_eq!(dims, vec![2, 3]);

    let mut indices_len = 0usize;
    assert_eq!(
        t4a_tensor_indices(tensor, std::ptr::null_mut(), 0, &mut indices_len),
        T4A_SUCCESS
    );
    let mut indices = vec![std::ptr::null_mut(); indices_len];
    assert_eq!(
        t4a_tensor_indices(
            tensor,
            indices.as_mut_ptr(),
            indices.len(),
            &mut indices_len
        ),
        T4A_SUCCESS
    );
    assert_eq!(indices_len, 2);

    let mut copied_dims = Vec::new();
    for index in &indices {
        let mut dim = 0usize;
        assert_eq!(crate::index::t4a_index_dim(*index, &mut dim), T4A_SUCCESS);
        copied_dims.push(dim);
        t4a_index_release(*index);
    }
    assert_eq!(copied_dims, vec![2, 3]);

    let mut kind = t4a_scalar_kind::C64;
    assert_eq!(t4a_tensor_scalar_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_scalar_kind::F64);
    assert_eq!(read_dense_f64(tensor), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_dense_c64_roundtrip() {
    let i = new_index(2);
    let j = new_index(2);
    let mut tensor = std::ptr::null_mut();
    let interleaved = [1.0, 0.5, 2.0, -1.5, 3.0, 2.5, 4.0, -3.5];
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    assert_eq!(
        t4a_tensor_new_dense_c64(2, index_ptrs.as_ptr(), interleaved.as_ptr(), 4, &mut tensor),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());

    let mut kind = t4a_scalar_kind::F64;
    assert_eq!(t4a_tensor_scalar_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_scalar_kind::C64);

    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_dense_c64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    assert_eq!(len, 4);
    let mut out = vec![0.0; 2 * len];
    assert_eq!(
        t4a_tensor_copy_dense_c64(tensor, out.as_mut_ptr(), len, &mut len),
        T4A_SUCCESS
    );
    assert_eq!(out, interleaved);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_contract_matches_matrix_multiplication() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);

    let mut j_clone = std::ptr::null_mut();
    assert_eq!(crate::index::t4a_index_clone(j, &mut j_clone), T4A_SUCCESS);

    // A = [ [1, 3, 5], [2, 4, 6] ] in column-major [2, 3]
    let a = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // B = [ [7, 10], [8, 11], [9, 12] ] in column-major [3, 2]
    let mut b = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            2,
            [j_clone as *const t4a_index, k as *const t4a_index].as_ptr(),
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0].as_ptr(),
            6,
            &mut b
        ),
        T4A_SUCCESS
    );

    let mut c = std::ptr::null_mut();
    assert_eq!(t4a_tensor_contract(a, b, &mut c), T4A_SUCCESS);
    assert_eq!(read_dense_f64(c), vec![76.0, 100.0, 103.0, 136.0]);

    t4a_tensor_release(c);
    t4a_tensor_release(b);
    t4a_tensor_release(a);
    t4a_index_release(j_clone);
    t4a_index_release(k);
    t4a_index_release(j);
    t4a_index_release(i);
}
