use super::*;
use num_complex::Complex64;

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

fn complex_product(indices: &[usize]) -> Complex64 {
    indices.iter().fold(Complex64::new(1.0, 0.0), |acc, &idx| {
        acc * Complex64::new(idx as f64 + 1.0, 2.0 * idx as f64 + 1.0)
    })
}

extern "C" fn complex_product_callback(
    indices: *const i64,
    n_indices: libc::size_t,
    result_re: *mut f64,
    result_im: *mut f64,
    _user_data: *mut c_void,
) -> i32 {
    let indices: Vec<usize> = (0..n_indices)
        .map(|i| unsafe { *indices.add(i) as usize })
        .collect();
    let value = complex_product(&indices);
    unsafe {
        *result_re = value.re;
        *result_im = value.im;
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

    // First add a pivot so the TCI has non-empty index sets
    let pivots: [libc::size_t; 2] = [0, 0];
    t4a_tci2_f64_add_global_pivots(tci, pivots.as_ptr(), 1, 2);

    let mut call_count = 0usize;
    let status = t4a_tci2_f64_sweep2site(
        tci,
        counting_callback,
        &mut call_count as *mut usize as *mut c_void,
        1,    // forward
        1e-8, // tolerance
        8,    // max_bonddim
        0,    // pivot_search: Full
        1,    // strictly_nested
    );

    assert!(
        status != T4A_SUCCESS || call_count > 0,
        "sweep2site returned success without invoking the evaluation callback"
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

/// Minimal test case for fill_site_tensors C-API bug.
///
/// Scenario: crossinterpolate2 → fill_site_tensors → to_tensor_train → evaluate
/// Expected: values remain correct after fill_site_tensors
#[test]
fn test_fill_site_tensors_preserves_values() {
    // f(i, j) = (1 + i) * (1 + j)
    extern "C" fn product_fn(
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

    let dims: [libc::size_t; 2] = [3, 3];
    let mut tci: *mut t4a_tci2_f64 = std::ptr::null_mut();
    let mut final_error: f64 = 0.0;

    // Step 1: crossinterpolate2
    let status = t4a_crossinterpolate2_f64(
        dims.as_ptr(),
        2,
        std::ptr::null(),
        0,
        product_fn,
        std::ptr::null_mut(),
        1e-10,
        0,
        20,
        &mut tci,
        &mut final_error,
    );
    assert_eq!(status, T4A_SUCCESS);

    // Step 2: Verify values BEFORE fill_site_tensors
    let tt_before = t4a_tci2_f64_to_tensor_train(tci);
    assert!(!tt_before.is_null());

    let mut val_before = 0.0f64;
    let indices_00: [libc::size_t; 2] = [0, 0];
    crate::simplett::t4a_simplett_f64_evaluate(tt_before, indices_00.as_ptr(), 2, &mut val_before);
    eprintln!("Before fill: tt(0,0) = {val_before}");
    assert!(
        (val_before - 1.0).abs() < 1e-8,
        "Before fill: expected 1.0, got {val_before}"
    );
    crate::simplett::t4a_simplett_f64_release(tt_before);

    // Step 3: fill_site_tensors
    let status = t4a_tci2_f64_fill_site_tensors(tci, product_fn, std::ptr::null_mut());
    assert_eq!(status, T4A_SUCCESS);

    // Step 4: Verify values AFTER fill_site_tensors
    let tt_after = t4a_tci2_f64_to_tensor_train(tci);
    assert!(!tt_after.is_null());

    let mut val_after = 0.0f64;
    crate::simplett::t4a_simplett_f64_evaluate(tt_after, indices_00.as_ptr(), 2, &mut val_after);
    eprintln!("After fill: tt(0,0) = {val_after}");
    assert!(
        (val_after - 1.0).abs() < 1e-8,
        "After fill_site_tensors: expected 1.0, got {val_after}"
    );

    crate::simplett::t4a_simplett_f64_release(tt_after);
    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_low_level_sweeps_and_index_set_accessors() {
    extern "C" fn product_fn(
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

    let dims: [libc::size_t; 2] = [2, 3];
    let mut tci: *mut t4a_tci2_f64 = std::ptr::null_mut();
    let mut final_error: f64 = 0.0;

    assert_eq!(
        t4a_crossinterpolate2_f64(
            dims.as_ptr(),
            2,
            std::ptr::null(),
            0,
            product_fn,
            std::ptr::null_mut(),
            1e-10,
            0,
            10,
            &mut tci,
            &mut final_error,
        ),
        T4A_SUCCESS
    );
    assert!(!tci.is_null());

    let mut pivot_error = -1.0f64;
    assert_eq!(t4a_tci2_f64_pivot_error(tci, &mut pivot_error), T4A_SUCCESS);
    assert!(pivot_error >= 0.0);

    assert_eq!(
        t4a_tci2_f64_sweep2site(tci, product_fn, std::ptr::null_mut(), 1, 1e-10, 0, 0, 0,),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tci2_f64_sweep1site(tci, product_fn, std::ptr::null_mut(), 1, 1e-14, 0.0, 0, 1),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tci2_f64_make_canonical(tci, product_fn, std::ptr::null_mut(), 1e-14, 0.0, 0),
        T4A_SUCCESS
    );

    let mut i_n = 0usize;
    let mut i_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_i_set_size(tci, 1, &mut i_n, &mut i_len),
        T4A_SUCCESS
    );
    assert!(i_n > 0);
    assert_eq!(i_len, 1);

    let mut i_buf = vec![usize::MAX; i_n * i_len];
    let mut out_i_n = 0usize;
    let mut out_i_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_get_i_set(
            tci,
            1,
            i_buf.as_mut_ptr(),
            i_buf.len(),
            &mut out_i_n,
            &mut out_i_len,
        ),
        T4A_SUCCESS
    );
    assert_eq!((out_i_n, out_i_len), (i_n, i_len));
    assert!(i_buf.iter().all(|&idx| idx < dims[0]));

    let mut i_small_n = 0usize;
    let mut i_small_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_get_i_set(
            tci,
            1,
            i_buf.as_mut_ptr(),
            i_buf.len().saturating_sub(1),
            &mut i_small_n,
            &mut i_small_len,
        ),
        T4A_INVALID_ARGUMENT
    );

    let mut j_n = 0usize;
    let mut j_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_j_set_size(tci, 0, &mut j_n, &mut j_len),
        T4A_SUCCESS
    );
    assert!(j_n > 0);
    assert_eq!(j_len, 1);

    let mut j_buf = vec![usize::MAX; j_n * j_len];
    let mut out_j_n = 0usize;
    let mut out_j_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_get_j_set(
            tci,
            0,
            j_buf.as_mut_ptr(),
            j_buf.len(),
            &mut out_j_n,
            &mut out_j_len,
        ),
        T4A_SUCCESS
    );
    assert_eq!((out_j_n, out_j_len), (j_n, j_len));
    assert!(j_buf.iter().all(|&idx| idx < dims[1]));

    let mut j_small_n = 0usize;
    let mut j_small_len = 0usize;
    assert_eq!(
        t4a_tci2_f64_get_j_set(
            tci,
            0,
            j_buf.as_mut_ptr(),
            j_buf.len().saturating_sub(1),
            &mut j_small_n,
            &mut j_small_len,
        ),
        T4A_INVALID_ARGUMENT
    );

    t4a_tci2_f64_release(tci);
}

#[test]
fn test_tci2_c64_low_level_sweeps_and_index_set_accessors() {
    let dims: [libc::size_t; 3] = [2, 2, 2];
    let tci = t4a_tci2_c64_new(dims.as_ptr(), 3);
    assert!(!tci.is_null());

    let pivot: [libc::size_t; 3] = [0, 0, 0];
    assert_eq!(
        t4a_tci2_c64_add_global_pivots(tci, pivot.as_ptr(), 1, 3),
        T4A_SUCCESS
    );

    assert_eq!(
        t4a_tci2_c64_sweep2site(
            tci,
            complex_product_callback,
            std::ptr::null_mut(),
            1,
            1e-12,
            0,
            0,
            0,
        ),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tci2_c64_sweep1site(
            tci,
            complex_product_callback,
            std::ptr::null_mut(),
            1,
            1e-14,
            0.0,
            0,
            1,
        ),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tci2_c64_fill_site_tensors(tci, complex_product_callback, std::ptr::null_mut()),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tci2_c64_make_canonical(
            tci,
            complex_product_callback,
            std::ptr::null_mut(),
            1e-14,
            0.0,
            0,
        ),
        T4A_SUCCESS
    );

    let mut max_sample = 0.0;
    assert_eq!(
        t4a_tci2_c64_max_sample_value(tci, &mut max_sample),
        T4A_SUCCESS
    );
    assert!(max_sample > 0.0);

    let mut max_bond_error = -1.0;
    assert_eq!(
        t4a_tci2_c64_max_bond_error(tci, &mut max_bond_error),
        T4A_SUCCESS
    );
    assert!(max_bond_error >= 0.0);

    let mut pivot_error = -1.0;
    assert_eq!(t4a_tci2_c64_pivot_error(tci, &mut pivot_error), T4A_SUCCESS);
    assert!(pivot_error >= 0.0);

    let mut i_n = 0usize;
    let mut i_len = 0usize;
    assert_eq!(
        t4a_tci2_c64_i_set_size(tci, 1, &mut i_n, &mut i_len),
        T4A_SUCCESS
    );
    assert!(i_n > 0);
    assert_eq!(i_len, 1);

    let mut i_buf = vec![usize::MAX; i_n * i_len];
    let mut out_i_n = 0usize;
    let mut out_i_len = 0usize;
    assert_eq!(
        t4a_tci2_c64_get_i_set(
            tci,
            1,
            i_buf.as_mut_ptr(),
            i_buf.len(),
            &mut out_i_n,
            &mut out_i_len,
        ),
        T4A_SUCCESS
    );
    assert_eq!((out_i_n, out_i_len), (i_n, i_len));

    let mut j_n = 0usize;
    let mut j_len = 0usize;
    assert_eq!(
        t4a_tci2_c64_j_set_size(tci, 1, &mut j_n, &mut j_len),
        T4A_SUCCESS
    );
    assert!(j_n > 0);
    assert_eq!(j_len, 1);

    let mut j_buf = vec![usize::MAX; j_n * j_len];
    let mut out_j_n = 0usize;
    let mut out_j_len = 0usize;
    assert_eq!(
        t4a_tci2_c64_get_j_set(
            tci,
            1,
            j_buf.as_mut_ptr(),
            j_buf.len(),
            &mut out_j_n,
            &mut out_j_len,
        ),
        T4A_SUCCESS
    );
    assert_eq!((out_j_n, out_j_len), (j_n, j_len));

    let tt = t4a_tci2_c64_to_tensor_train(tci);
    assert!(!tt.is_null());

    let indices: [libc::size_t; 3] = [1, 0, 1];
    let (mut out_re, mut out_im) = (0.0, 0.0);
    assert_eq!(
        crate::t4a_simplett_c64_evaluate(
            tt,
            indices.as_ptr(),
            indices.len(),
            &mut out_re,
            &mut out_im,
        ),
        T4A_SUCCESS
    );
    assert!(
        (Complex64::new(out_re, out_im) - complex_product(&[1, 0, 1])).norm() < 1e-12,
        "expected {}, got {}",
        complex_product(&[1, 0, 1]),
        Complex64::new(out_re, out_im)
    );

    crate::t4a_simplett_c64_release(tt);
    t4a_tci2_c64_release(tci);
}

#[test]
fn test_crossinterpolate2_c64_product() {
    let dims: [libc::size_t; 4] = [2, 2, 2, 2];
    let initial_pivots: [libc::size_t; 4] = [0, 0, 0, 0];
    let mut tci: *mut t4a_tci2_c64 = std::ptr::null_mut();
    let mut final_error = 0.0;

    assert_eq!(
        t4a_crossinterpolate2_c64(
            dims.as_ptr(),
            4,
            initial_pivots.as_ptr(),
            1,
            complex_product_callback,
            std::ptr::null_mut(),
            1e-12,
            0,
            10,
            &mut tci,
            &mut final_error,
        ),
        T4A_SUCCESS
    );
    assert!(!tci.is_null());
    assert!(final_error < 1e-8, "final_error = {final_error}");

    let mut rank = 0usize;
    assert_eq!(t4a_tci2_c64_rank(tci, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 1);

    let tt = t4a_tci2_c64_to_tensor_train(tci);
    assert!(!tt.is_null());

    let indices: [libc::size_t; 4] = [1, 1, 1, 1];
    let (mut out_re, mut out_im) = (0.0, 0.0);
    assert_eq!(
        crate::t4a_simplett_c64_evaluate(
            tt,
            indices.as_ptr(),
            indices.len(),
            &mut out_re,
            &mut out_im,
        ),
        T4A_SUCCESS
    );
    assert!(
        (Complex64::new(out_re, out_im) - complex_product(&[1, 1, 1, 1])).norm() < 1e-12,
        "expected {}, got {}",
        complex_product(&[1, 1, 1, 1]),
        Complex64::new(out_re, out_im)
    );

    crate::t4a_simplett_c64_release(tt);
    t4a_tci2_c64_release(tci);
}
