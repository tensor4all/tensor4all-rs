use super::*;
use crate::index::*;

#[test]
fn test_tensor_lifecycle() {
    // Create indices
    let i = t4a_index_new(2);
    let j = t4a_index_new(3);
    assert!(!i.is_null());
    assert!(!j.is_null());

    // Create tensor
    let index_ptrs = [i as *const _, j as *const _];
    let dims = [2_usize, 3_usize];
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let tensor = t4a_tensor_new_dense_f64(2, index_ptrs.as_ptr(), dims.as_ptr(), data.as_ptr(), 6);
    assert!(!tensor.is_null());

    // Test is_assigned
    assert_eq!(t4a_tensor_is_assigned(tensor as *const _), 1);

    // Test clone
    let cloned = t4a_tensor_clone(tensor as *const _);
    assert!(!cloned.is_null());

    // Clean up
    t4a_tensor_release(cloned);
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_accessors() {
    // Create indices
    let i = t4a_index_new(2);
    let j = t4a_index_new(3);

    // Create tensor
    let index_ptrs = [i as *const _, j as *const _];
    let dims = [2_usize, 3_usize];
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let tensor = t4a_tensor_new_dense_f64(2, index_ptrs.as_ptr(), dims.as_ptr(), data.as_ptr(), 6);

    // Get rank
    let mut rank: usize = 0;
    assert_eq!(
        t4a_tensor_get_rank(tensor as *const _, &mut rank),
        T4A_SUCCESS
    );
    assert_eq!(rank, 2);

    // Get dims
    let mut out_dims = [0_usize; 2];
    assert_eq!(
        t4a_tensor_get_dims(tensor as *const _, out_dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(out_dims, [2, 3]);

    // Get storage kind
    let mut kind = t4a_storage_kind::DenseC64;
    assert_eq!(
        t4a_tensor_get_storage_kind(tensor as *const _, &mut kind),
        T4A_SUCCESS
    );
    assert_eq!(kind, t4a_storage_kind::DenseF64);

    // Get data
    let mut out_len: usize = 0;
    assert_eq!(
        t4a_tensor_get_data_f64(tensor as *const _, ptr::null_mut(), 0, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 6);

    let mut out_data = [0.0; 6];
    assert_eq!(
        t4a_tensor_get_data_f64(tensor as *const _, out_data.as_mut_ptr(), 6, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_data, data);

    // Get indices
    let mut out_indices: [*mut t4a_index; 2] = [ptr::null_mut(); 2];
    assert_eq!(
        t4a_tensor_get_indices(tensor as *const _, out_indices.as_mut_ptr(), 2),
        T4A_SUCCESS
    );

    // Verify indices have correct dimensions
    let mut dim0: usize = 0;
    let mut dim1: usize = 0;
    crate::index::t4a_index_dim(out_indices[0] as *const _, &mut dim0);
    crate::index::t4a_index_dim(out_indices[1] as *const _, &mut dim1);
    assert_eq!(dim0, 2);
    assert_eq!(dim1, 3);

    // Clean up indices
    for idx in &out_indices {
        t4a_index_release(*idx);
    }

    // Clean up
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_c64() {
    // Create indices
    let i = t4a_index_new(2);
    let j = t4a_index_new(2);

    // Create tensor
    let index_ptrs = [i as *const _, j as *const _];
    let dims = [2_usize, 2_usize];
    let data_interleaved = [1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5];

    let tensor = t4a_tensor_new_dense_c64(
        2,
        index_ptrs.as_ptr(),
        dims.as_ptr(),
        data_interleaved.as_ptr(),
        4,
    );
    assert!(!tensor.is_null());

    // Get storage kind
    let mut kind = t4a_storage_kind::DenseF64;
    assert_eq!(
        t4a_tensor_get_storage_kind(tensor as *const _, &mut kind),
        T4A_SUCCESS
    );
    assert_eq!(kind, t4a_storage_kind::DenseC64);

    // Get data
    let mut out_len: usize = 0;
    assert_eq!(
        t4a_tensor_get_data_c64(tensor as *const _, ptr::null_mut(), 0, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 4);

    let mut out_data = [0.0; 8];
    assert_eq!(
        t4a_tensor_get_data_c64(tensor as *const _, out_data.as_mut_ptr(), 4, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 4);
    assert_eq!(out_data, data_interleaved);

    // Clean up
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_onehot() {
    let i = t4a_index_new(3);
    let j = t4a_index_new(4);
    assert!(!i.is_null());
    assert!(!j.is_null());

    let index_ptrs = [i as *const _, j as *const _];
    let vals = [1_usize, 2_usize];

    let tensor = t4a_tensor_onehot(2, index_ptrs.as_ptr(), vals.as_ptr(), 2);
    assert!(!tensor.is_null());

    // Verify rank
    let mut rank = 0_usize;
    assert_eq!(
        t4a_tensor_get_rank(tensor as *const _, &mut rank),
        T4A_SUCCESS
    );
    assert_eq!(rank, 2);

    // Verify dims
    let mut dims = [0_usize; 2];
    assert_eq!(
        t4a_tensor_get_dims(tensor as *const _, dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(dims, [3, 4]);

    // Verify data: position (1,2) in 3x4 = offset 7 in column-major order
    let mut out_len = 0_usize;
    let mut data = [0.0_f64; 12];
    assert_eq!(
        t4a_tensor_get_data_f64(tensor as *const _, data.as_mut_ptr(), 12, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 12);
    let mut expected = [0.0_f64; 12];
    expected[7] = 1.0;
    assert_eq!(data, expected);

    // Clean up
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_onehot_null_guards() {
    // Null index_ptrs
    let vals = [0_usize];
    let result = t4a_tensor_onehot(1, ptr::null(), vals.as_ptr(), 1);
    assert!(result.is_null());

    // Null vals
    let i = t4a_index_new(3);
    let index_ptrs = [i as *const _];
    let result = t4a_tensor_onehot(1, index_ptrs.as_ptr(), ptr::null(), 1);
    assert!(result.is_null());

    // Mismatched rank and vals_len
    let result = t4a_tensor_onehot(1, index_ptrs.as_ptr(), vals.as_ptr(), 2);
    assert!(result.is_null());

    t4a_index_release(i);
}

#[test]
fn test_tensor_onehot_empty() {
    // rank=0 should return a scalar tensor
    let tensor = t4a_tensor_onehot(0, ptr::null(), ptr::null(), 0);
    assert!(!tensor.is_null());

    let mut rank = 99_usize;
    assert_eq!(
        t4a_tensor_get_rank(tensor as *const _, &mut rank),
        T4A_SUCCESS
    );
    assert_eq!(rank, 0);

    t4a_tensor_release(tensor);
}

#[test]
fn test_tensor_new_dense_f64_dims_mismatch_rejected() {
    let i = t4a_index_new(2);
    let j = t4a_index_new(3);
    assert!(!i.is_null());
    assert!(!j.is_null());

    let index_ptrs = [i as *const _, j as *const _];
    // Mismatch: actual dims should be [2,3]
    let dims = [2_usize, 4_usize];
    let data = [0.0_f64; 8];

    let tensor = t4a_tensor_new_dense_f64(2, index_ptrs.as_ptr(), dims.as_ptr(), data.as_ptr(), 8);
    assert!(tensor.is_null());

    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_contract_basic() {
    use std::ffi::CString;

    let tags_i = CString::new("i").unwrap();
    let tags_j = CString::new("j").unwrap();
    let tags_k = CString::new("k").unwrap();

    let idx_i = t4a_index_new_with_tags(2, tags_i.as_ptr());
    let idx_j = t4a_index_new_with_tags(3, tags_j.as_ptr());
    let idx_k = t4a_index_new_with_tags(4, tags_k.as_ptr());
    assert!(!idx_i.is_null());
    assert!(!idx_j.is_null());
    assert!(!idx_k.is_null());

    let idx_j2 = t4a_index_clone(idx_j);
    assert!(!idx_j2.is_null());

    let data_a = [1.0_f64; 6];
    let inds_a = [idx_i as *const _, idx_j as *const _];
    let dims_a = [2_usize, 3_usize];
    let tensor_a = t4a_tensor_new_dense_f64(
        2,
        inds_a.as_ptr(),
        dims_a.as_ptr(),
        data_a.as_ptr(),
        data_a.len(),
    );
    assert!(!tensor_a.is_null());

    let data_b = [1.0_f64; 12];
    let inds_b = [idx_j2 as *const _, idx_k as *const _];
    let dims_b = [3_usize, 4_usize];
    let tensor_b = t4a_tensor_new_dense_f64(
        2,
        inds_b.as_ptr(),
        dims_b.as_ptr(),
        data_b.as_ptr(),
        data_b.len(),
    );
    assert!(!tensor_b.is_null());

    let mut tensor_c: *mut crate::types::t4a_tensor = ptr::null_mut();
    assert_eq!(
        t4a_tensor_contract(tensor_a, tensor_b, &mut tensor_c),
        T4A_SUCCESS
    );
    assert!(!tensor_c.is_null());

    let mut rank: usize = 0;
    assert_eq!(t4a_tensor_get_rank(tensor_c, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);

    let mut dims = [0_usize; 2];
    assert_eq!(
        t4a_tensor_get_dims(tensor_c, dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(dims, [2, 4]);

    let mut out_len = 0_usize;
    assert_eq!(
        t4a_tensor_get_data_f64(tensor_c, ptr::null_mut(), 0, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 8);

    let mut data_c = [0.0_f64; 8];
    assert_eq!(
        t4a_tensor_get_data_f64(tensor_c, data_c.as_mut_ptr(), data_c.len(), &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 8);
    for value in data_c {
        assert!((value - 3.0).abs() < 1e-12);
    }

    t4a_tensor_release(tensor_a);
    t4a_tensor_release(tensor_b);
    t4a_tensor_release(tensor_c);
    t4a_index_release(idx_i);
    t4a_index_release(idx_j);
    t4a_index_release(idx_j2);
    t4a_index_release(idx_k);
}

#[test]
fn test_tensor_contract_null_checks() {
    let mut out: *mut crate::types::t4a_tensor = ptr::null_mut();

    assert_eq!(
        t4a_tensor_contract(ptr::null(), ptr::null(), &mut out),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_tensor_contract(ptr::null(), ptr::null(), ptr::null_mut()),
        T4A_NULL_POINTER
    );
}

#[test]
fn test_tensor_contract_no_common_indices() {
    let idx_i = t4a_index_new(2);
    let idx_j = t4a_index_new(3);
    assert!(!idx_i.is_null());
    assert!(!idx_j.is_null());

    let data_a = [1.0_f64, 2.0_f64];
    let inds_a = [idx_i as *const _];
    let dims_a = [2_usize];
    let tensor_a = t4a_tensor_new_dense_f64(
        1,
        inds_a.as_ptr(),
        dims_a.as_ptr(),
        data_a.as_ptr(),
        data_a.len(),
    );
    assert!(!tensor_a.is_null());

    let data_b = [1.0_f64, 1.0_f64, 1.0_f64];
    let inds_b = [idx_j as *const _];
    let dims_b = [3_usize];
    let tensor_b = t4a_tensor_new_dense_f64(
        1,
        inds_b.as_ptr(),
        dims_b.as_ptr(),
        data_b.as_ptr(),
        data_b.len(),
    );
    assert!(!tensor_b.is_null());

    let mut tensor_c: *mut crate::types::t4a_tensor = ptr::null_mut();
    assert_eq!(
        t4a_tensor_contract(tensor_a, tensor_b, &mut tensor_c),
        T4A_SUCCESS
    );
    assert!(!tensor_c.is_null());

    let mut rank: usize = 0;
    assert_eq!(t4a_tensor_get_rank(tensor_c, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);

    let mut dims = [0_usize; 2];
    assert_eq!(
        t4a_tensor_get_dims(tensor_c, dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(dims, [2, 3]);

    let mut out_len = 0_usize;
    assert_eq!(
        t4a_tensor_get_data_f64(tensor_c, ptr::null_mut(), 0, &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(out_len, 6);

    let mut data_c = [0.0_f64; 6];
    assert_eq!(
        t4a_tensor_get_data_f64(tensor_c, data_c.as_mut_ptr(), data_c.len(), &mut out_len),
        T4A_SUCCESS
    );
    assert_eq!(data_c, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

    t4a_tensor_release(tensor_a);
    t4a_tensor_release(tensor_b);
    t4a_tensor_release(tensor_c);
    t4a_index_release(idx_i);
    t4a_index_release(idx_j);
}
