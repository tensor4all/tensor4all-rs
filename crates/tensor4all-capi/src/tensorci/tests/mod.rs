use super::*;

// Simple test callback that returns sum of indices
#[allow(dead_code)]
extern "C" fn sum_callback(
    indices: *const i64,
    n_indices: libc::size_t,
    result: *mut f64,
    _user_data: *mut c_void,
) -> i32 {
    unsafe {
        let sum: i64 = (0..n_indices).map(|i| *indices.add(i)).sum();
        *result = sum as f64;
    }
    0
}

#[test]
fn test_tci2_new() {
    let dims: [libc::size_t; 3] = [2, 3, 4];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 3);
    assert!(!tci.is_null());

    let mut len: libc::size_t = 0;
    let status = t4a_tci2_f64_len(tci, &mut len);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(len, 3);

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_accessors() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 2);
    assert!(!tci.is_null());

    let mut rank = usize::MAX;
    assert_eq!(t4a_tci2_f64_rank(tci, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 0);

    let mut link_dims = [usize::MAX; 1];
    assert_eq!(
        t4a_tci2_f64_link_dims(tci, link_dims.as_mut_ptr(), link_dims.len()),
        T4A_SUCCESS
    );
    assert_eq!(link_dims, [0]);

    let mut max_sample = -1.0;
    assert_eq!(
        t4a_tci2_f64_max_sample_value(tci, &mut max_sample),
        T4A_SUCCESS
    );
    assert_eq!(max_sample, 0.0);

    let mut max_bond_error = -1.0;
    assert_eq!(
        t4a_tci2_f64_max_bond_error(tci, &mut max_bond_error),
        T4A_SUCCESS
    );
    assert_eq!(max_bond_error, 0.0);

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_sweep_requires_real_work_before_reporting_success() {
    extern "C" fn counting_callback(
        _indices: *const i64,
        _n_indices: libc::size_t,
        result: *mut f64,
        user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            *(user_data as *mut usize) += 1;
            *result = 1.0;
        }
        0
    }

    let dims: [libc::size_t; 2] = [2, 3];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 2);
    assert!(!tci.is_null());

    let mut call_count = 0usize;
    let mut sweep_error = -1.0;
    let status = t4a_tci2_f64_sweep(
        tci,
        counting_callback,
        &mut call_count as *mut usize as *mut c_void,
        1e-8,
        8,
        1,
        &mut sweep_error,
    );

    assert!(
        status != T4A_SUCCESS || call_count > 0,
        "sweep returned success without invoking the evaluation callback"
    );

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_add_global_pivots_validates_shape_and_nulls() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 2);
    assert!(!tci.is_null());

    let pivot: [libc::size_t; 1] = [0];
    assert_eq!(
        t4a_tci2_f64_add_global_pivots(tci, pivot.as_ptr(), 1, 1),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_tci2_f64_add_global_pivots(std::ptr::null_mut(), pivot.as_ptr(), 1, 1),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_tci2_f64_add_global_pivots(tci, std::ptr::null(), 1, 2),
        T4A_NULL_POINTER
    );

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_add_global_pivots_updates_rank_and_link_dims() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 2);
    assert!(!tci.is_null());

    let pivot: [libc::size_t; 2] = [1, 2];
    assert_eq!(
        t4a_tci2_f64_add_global_pivots(tci, pivot.as_ptr(), 1, 2),
        T4A_SUCCESS
    );

    let mut rank = 0usize;
    assert_eq!(t4a_tci2_f64_rank(tci, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 1);

    let mut link_dims = [0usize; 1];
    assert_eq!(
        t4a_tci2_f64_link_dims(tci, link_dims.as_mut_ptr(), link_dims.len()),
        T4A_SUCCESS
    );
    assert_eq!(link_dims, [1]);

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_accessors_validate_pointers_and_buffers() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 2);
    assert!(!tci.is_null());

    let mut out_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_len(std::ptr::null(), &mut out_len),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_tci2_f64_rank(tci, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );

    assert_eq!(
        t4a_tci2_f64_link_dims(tci, std::ptr::null_mut(), 0),
        T4A_NULL_POINTER
    );
    let mut empty_dims: [libc::size_t; 0] = [];
    assert_eq!(
        t4a_tci2_f64_link_dims(tci, empty_dims.as_mut_ptr(), empty_dims.len()),
        T4A_INVALID_ARGUMENT
    );

    assert_eq!(
        t4a_tci2_f64_max_sample_value(tci, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_tci2_f64_max_bond_error(tci, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert!(t4a_tci2_f64_to_tensor_train(std::ptr::null()).is_null());
    assert!(t4a_tci2_f64_to_tensor_train(tci).is_null());
    assert_eq!(
        t4a_crossinterpolate2_f64(
            dims.as_ptr(),
            2,
            std::ptr::null(),
            0,
            sum_callback,
            std::ptr::null_mut(),
            1e-8,
            8,
            4,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        T4A_NULL_POINTER
    );

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_crossinterpolate2_constant() {
    // Constant function: f(i,j) = 1.0
    extern "C" fn const_callback(
        _indices: *const i64,
        _n_indices: libc::size_t,
        result: *mut f64,
        _user_data: *mut c_void,
    ) -> i32 {
        unsafe { *result = 1.0 };
        0
    }

    let dims: [libc::size_t; 2] = [3, 4];
    let mut tci: *mut t4a_tci2_f64 = std::ptr::null_mut();
    let mut final_error: f64 = 0.0;

    let status = t4a_crossinterpolate2_f64(
        dims.as_ptr(),
        2,
        std::ptr::null(), // default initial pivot
        0,
        const_callback,
        std::ptr::null_mut(),
        1e-10,
        100,
        20,
        &mut tci,
        &mut final_error,
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(!tci.is_null());

    // Check rank (should be 1 for constant function)
    let mut rank: libc::size_t = 0;
    t4a_tci2_f64_rank(tci, &mut rank);
    assert_eq!(rank, 1);

    // Convert to TT and verify sum
    let tt = t4a_tci2_f64_to_tensor_train(tci);
    assert!(!tt.is_null());

    let mut sum: f64 = 0.0;
    crate::simplett::t4a_simplett_f64_sum(tt, &mut sum);
    assert!((sum - 12.0).abs() < 1e-8); // 3 * 4 = 12

    crate::simplett::t4a_simplett_f64_release(tt);
    t4a_tci2_f64_release(tci);
}

#[test]
fn test_crossinterpolate2_with_explicit_initial_pivots() {
    extern "C" fn product_callback(
        indices: *const i64,
        n_indices: libc::size_t,
        result: *mut f64,
        _user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            let product = (0..n_indices)
                .map(|i| *indices.add(i) as f64 + 1.0)
                .product::<f64>();
            *result = product;
        }
        0
    }

    let dims: [libc::size_t; 2] = [2, 2];
    let initial_pivots: [libc::size_t; 2] = [1, 1];
    let mut tci: *mut t4a_tci2_f64 = std::ptr::null_mut();

    let status = t4a_crossinterpolate2_f64(
        dims.as_ptr(),
        2,
        initial_pivots.as_ptr(),
        1,
        product_callback,
        std::ptr::null_mut(),
        1e-10,
        0,
        10,
        &mut tci,
        std::ptr::null_mut(),
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(!tci.is_null());

    let mut rank = 0usize;
    assert_eq!(t4a_tci2_f64_rank(tci, &mut rank), T4A_SUCCESS);
    assert!(rank >= 1);

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_new_rejects_too_few_sites() {
    let dims: [libc::size_t; 1] = [2];
    let tci = t4a_tci2_f64_new(dims.as_ptr(), 1);
    assert!(tci.is_null());
}
