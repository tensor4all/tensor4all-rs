use super::*;
use num_complex::Complex64;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};

#[test]
fn test_simplett_private_wrapper_roundtrip() {
    let mut wrapper = t4a_simplett_f64::new(TensorTrain::<f64>::constant(&[2], 3.0));
    assert_eq!(wrapper.inner().len(), 1);
    assert!((wrapper.inner().sum() - 6.0).abs() < 1e-10);

    wrapper.inner_mut().scale(2.0);
    assert!((wrapper.inner().sum() - 12.0).abs() < 1e-10);

    let inner = wrapper.into_inner();
    assert_eq!(inner.len(), 1);
    assert!((inner.sum() - 12.0).abs() < 1e-10);
}

#[test]
fn test_simplett_lifecycle_edge_cases() {
    t4a_simplett_f64_release(std::ptr::null_mut());
    assert!(t4a_simplett_f64_clone(std::ptr::null()).is_null());

    let empty_const = t4a_simplett_f64_constant(std::ptr::null(), 0, 2.0);
    assert!(!empty_const.is_null());
    let empty_zero = t4a_simplett_f64_zeros(std::ptr::null(), 0);
    assert!(!empty_zero.is_null());

    let mut len = usize::MAX;
    assert_eq!(t4a_simplett_f64_len(empty_const, &mut len), T4A_SUCCESS);
    assert_eq!(len, 0);
    assert_eq!(t4a_simplett_f64_len(empty_zero, &mut len), T4A_SUCCESS);
    assert_eq!(len, 0);

    let mut sum = 1.0;
    assert_eq!(t4a_simplett_f64_sum(empty_const, &mut sum), T4A_SUCCESS);
    assert_eq!(sum, 0.0);

    let mut norm = 1.0;
    assert_eq!(t4a_simplett_f64_norm(empty_zero, &mut norm), T4A_SUCCESS);
    assert_eq!(norm, 0.0);

    t4a_simplett_f64_release(empty_const);
    t4a_simplett_f64_release(empty_zero);
}

#[test]
fn test_simplett_zeros_accessors_and_site_tensor() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_zeros(dims.as_ptr(), 2);
    assert!(!tt.is_null());

    let mut site_dims = [0usize; 2];
    let status = t4a_simplett_f64_site_dims(tt, site_dims.as_mut_ptr(), site_dims.len());
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(site_dims, dims);

    let mut link_dims = [usize::MAX; 1];
    let status = t4a_simplett_f64_link_dims(tt, link_dims.as_mut_ptr(), link_dims.len());
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(link_dims, [1]);

    let mut rank = 0usize;
    let status = t4a_simplett_f64_rank(tt, &mut rank);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(rank, 1);

    let mut sum = 1.0;
    let status = t4a_simplett_f64_sum(tt, &mut sum);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(sum, 0.0);

    let mut norm = 1.0;
    let status = t4a_simplett_f64_norm(tt, &mut norm);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(norm, 0.0);

    let mut tensor_data = [1.0; 3];
    let (mut left_dim, mut site_dim, mut right_dim) = (0usize, 0usize, 0usize);
    let status = t4a_simplett_f64_site_tensor(
        tt,
        1,
        tensor_data.as_mut_ptr(),
        tensor_data.len(),
        &mut left_dim,
        &mut site_dim,
        &mut right_dim,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!((left_dim, site_dim, right_dim), (1, 3, 1));
    assert_eq!(tensor_data, [0.0; 3]);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_constant() {
    let dims: [libc::size_t; 3] = [2, 3, 4];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 3, 1.5);
    assert!(!tt.is_null());

    let mut len: libc::size_t = 0;
    let status = t4a_simplett_f64_len(tt, &mut len);
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(len, 3);

    let mut sum: f64 = 0.0;
    let status = t4a_simplett_f64_sum(tt, &mut sum);
    assert_eq!(status, T4A_SUCCESS);
    assert!((sum - 1.5 * 24.0).abs() < 1e-10);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_evaluate() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);
    assert!(!tt.is_null());

    let indices: [libc::size_t; 2] = [0, 1];
    let mut value: f64 = 0.0;
    let status = t4a_simplett_f64_evaluate(tt, indices.as_ptr(), 2, &mut value);
    assert_eq!(status, T4A_SUCCESS);
    assert!((value - 2.0).abs() < 1e-10);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_site_tensor_uses_column_major_linearization() {
    let t0 = tensor3_from_data(vec![0.0, 1.0, 2.0, 3.0], 1, 2, 2);
    let t1 = tensor3_from_data(vec![10.0, 11.0, 12.0, 13.0], 2, 2, 1);
    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let handle = Box::into_raw(Box::new(t4a_simplett_f64::new(tt)));

    let mut out_data = [0.0_f64; 4];
    let mut out_left = 0_usize;
    let mut out_site = 0_usize;
    let mut out_right = 0_usize;

    let status = t4a_simplett_f64_site_tensor(
        handle,
        0,
        out_data.as_mut_ptr(),
        out_data.len(),
        &mut out_left,
        &mut out_site,
        &mut out_right,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!((out_left, out_site, out_right), (1, 2, 2));
    assert_eq!(out_data, [0.0, 1.0, 2.0, 3.0]);

    t4a_simplett_f64_release(handle);
}

#[test]
fn test_simplett_clone_preserves_observable_values() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);
    let cloned = t4a_simplett_f64_clone(tt);
    assert!(!cloned.is_null());

    let mut original_sum = 0.0;
    let mut cloned_sum = 0.0;
    assert_eq!(t4a_simplett_f64_sum(tt, &mut original_sum), T4A_SUCCESS);
    assert_eq!(t4a_simplett_f64_sum(cloned, &mut cloned_sum), T4A_SUCCESS);
    assert_eq!(original_sum, cloned_sum);

    let mut site_tensor = [0.0; 3];
    let status = t4a_simplett_f64_site_tensor(
        cloned,
        1,
        site_tensor.as_mut_ptr(),
        site_tensor.len(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!(site_tensor, [2.0; 3]);

    t4a_simplett_f64_release(tt);
    t4a_simplett_f64_release(cloned);
}

#[test]
fn test_simplett_constant_norm_and_first_site_tensor() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);
    assert!(!tt.is_null());

    let mut norm = 0.0;
    assert_eq!(t4a_simplett_f64_norm(tt, &mut norm), T4A_SUCCESS);
    assert!((norm - f64::sqrt(24.0)).abs() < 1e-10);

    let mut tensor_data = [0.0; 2];
    let (mut left_dim, mut site_dim, mut right_dim) = (0usize, 0usize, 0usize);
    let status = t4a_simplett_f64_site_tensor(
        tt,
        0,
        tensor_data.as_mut_ptr(),
        tensor_data.len(),
        &mut left_dim,
        &mut site_dim,
        &mut right_dim,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert_eq!((left_dim, site_dim, right_dim), (1, 2, 1));
    assert_eq!(tensor_data, [1.0, 1.0]);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_accessors_validate_pointers_and_buffer_sizes() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 1.0);

    let mut out_len = 0usize;
    assert_eq!(
        t4a_simplett_f64_len(std::ptr::null(), &mut out_len),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_len(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_sum(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_norm(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_rank(tt, std::ptr::null_mut()),
        T4A_NULL_POINTER
    );

    let mut small_dims = [0usize; 1];
    assert_eq!(
        t4a_simplett_f64_site_dims(tt, small_dims.as_mut_ptr(), small_dims.len()),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_simplett_f64_site_dims(tt, std::ptr::null_mut(), 0),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_link_dims(tt, std::ptr::null_mut(), 0),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_link_dims(tt, small_dims.as_mut_ptr(), small_dims.len()),
        T4A_SUCCESS
    );

    let bad_indices: [libc::size_t; 2] = [2, 0];
    let mut out_value = 0.0;
    assert_eq!(
        t4a_simplett_f64_evaluate(tt, bad_indices.as_ptr(), bad_indices.len(), &mut out_value),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_simplett_f64_evaluate(tt, std::ptr::null(), bad_indices.len(), &mut out_value),
        T4A_NULL_POINTER
    );

    let mut tensor_buf = [0.0; 1];
    assert_eq!(
        t4a_simplett_f64_site_tensor(
            tt,
            2,
            tensor_buf.as_mut_ptr(),
            tensor_buf.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_simplett_f64_site_tensor(
            tt,
            1,
            tensor_buf.as_mut_ptr(),
            tensor_buf.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        T4A_INVALID_ARGUMENT
    );
    assert_eq!(
        t4a_simplett_f64_site_tensor(
            tt,
            0,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        T4A_NULL_POINTER
    );

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_c64_constant_accessors_and_site_tensor() {
    let dims: [libc::size_t; 2] = [2, 3];
    let value = Complex64::new(2.0, 3.0);
    let tt = t4a_simplett_c64_constant(dims.as_ptr(), 2, value.re, value.im);
    assert!(!tt.is_null());

    let mut len = 0usize;
    assert_eq!(t4a_simplett_c64_len(tt, &mut len), T4A_SUCCESS);
    assert_eq!(len, 2);

    let mut site_dims = [0usize; 2];
    assert_eq!(
        t4a_simplett_c64_site_dims(tt, site_dims.as_mut_ptr(), site_dims.len()),
        T4A_SUCCESS
    );
    assert_eq!(site_dims, dims);

    let mut link_dims = [usize::MAX; 1];
    assert_eq!(
        t4a_simplett_c64_link_dims(tt, link_dims.as_mut_ptr(), link_dims.len()),
        T4A_SUCCESS
    );
    assert_eq!(link_dims, [1]);

    let mut rank = 0usize;
    assert_eq!(t4a_simplett_c64_rank(tt, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 1);

    let indices: [libc::size_t; 2] = [1, 2];
    let (mut eval_re, mut eval_im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_evaluate(
            tt,
            indices.as_ptr(),
            indices.len(),
            &mut eval_re,
            &mut eval_im
        ),
        T4A_SUCCESS
    );
    assert_eq!(Complex64::new(eval_re, eval_im), value);

    let (mut sum_re, mut sum_im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_sum(tt, &mut sum_re, &mut sum_im),
        T4A_SUCCESS
    );
    assert_eq!(
        Complex64::new(sum_re, sum_im),
        value * Complex64::new(6.0, 0.0)
    );

    let mut norm = 0.0;
    assert_eq!(t4a_simplett_c64_norm(tt, &mut norm), T4A_SUCCESS);
    assert!((norm - f64::sqrt(78.0)).abs() < 1e-12);

    let mut tensor_data = [0.0; 6];
    let (mut left_dim, mut site_dim, mut right_dim) = (0usize, 0usize, 0usize);
    assert_eq!(
        t4a_simplett_c64_site_tensor(
            tt,
            1,
            tensor_data.as_mut_ptr(),
            6,
            &mut left_dim,
            &mut site_dim,
            &mut right_dim,
        ),
        T4A_SUCCESS
    );
    assert_eq!((left_dim, site_dim, right_dim), (1, 3, 1));
    assert_eq!(tensor_data, [2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);

    t4a_simplett_c64_release(tt);
}

#[test]
fn test_simplett_c64_compress_and_partial_sum() {
    let dims: [libc::size_t; 2] = [2, 3];
    let value = Complex64::new(2.0, 3.0);
    let tt = t4a_simplett_c64_constant(dims.as_ptr(), 2, value.re, value.im);
    assert!(!tt.is_null());

    assert_eq!(t4a_simplett_c64_compress(tt, 2, 1e-12, 0), T4A_SUCCESS);

    let dims_to_sum: [libc::size_t; 1] = [1];
    let mut partial: *mut t4a_simplett_c64 = std::ptr::null_mut();
    assert_eq!(
        t4a_simplett_c64_partial_sum(tt, dims_to_sum.as_ptr(), dims_to_sum.len(), &mut partial),
        T4A_SUCCESS
    );
    assert!(!partial.is_null());

    let idx0: [libc::size_t; 1] = [0];
    let (mut part_re, mut part_im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_evaluate(
            partial,
            idx0.as_ptr(),
            idx0.len(),
            &mut part_re,
            &mut part_im
        ),
        T4A_SUCCESS
    );
    assert!((Complex64::new(part_re, part_im) - value * Complex64::new(3.0, 0.0)).norm() < 1e-12);

    let idx1: [libc::size_t; 1] = [1];
    assert_eq!(
        t4a_simplett_c64_evaluate(
            partial,
            idx1.as_ptr(),
            idx1.len(),
            &mut part_re,
            &mut part_im
        ),
        T4A_SUCCESS
    );
    assert!((Complex64::new(part_re, part_im) - value * Complex64::new(3.0, 0.0)).norm() < 1e-12);

    t4a_simplett_c64_release(partial);
    t4a_simplett_c64_release(tt);
}

// ============================================================================
// Category 1: from_site_tensors tests
// ============================================================================

#[test]
fn test_simplett_f64_from_site_tensors() {
    // Build a 2-site TT with known data
    // Site 0: shape (1, 2, 2), data = [0, 1, 2, 3] column-major
    // Site 1: shape (2, 2, 1), data = [10, 11, 12, 13] column-major
    let left_dims: [libc::size_t; 2] = [1, 2];
    let site_dims: [libc::size_t; 2] = [2, 2];
    let right_dims: [libc::size_t; 2] = [2, 1];
    let data: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];

    let mut tt: *mut t4a_simplett_f64 = std::ptr::null_mut();
    let status = t4a_simplett_f64_from_site_tensors(
        2,
        left_dims.as_ptr(),
        site_dims.as_ptr(),
        right_dims.as_ptr(),
        data.as_ptr(),
        data.len(),
        &mut tt,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!tt.is_null());

    // Check accessors
    let mut len = 0usize;
    assert_eq!(t4a_simplett_f64_len(tt, &mut len), T4A_SUCCESS);
    assert_eq!(len, 2);

    let mut out_site_dims = [0usize; 2];
    assert_eq!(
        t4a_simplett_f64_site_dims(tt, out_site_dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(out_site_dims, [2, 2]);

    let mut out_link_dims = [0usize; 1];
    assert_eq!(
        t4a_simplett_f64_link_dims(tt, out_link_dims.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert_eq!(out_link_dims, [2]);

    // Verify site tensor data roundtrip
    let mut out_data = [0.0f64; 4];
    let (mut ld, mut sd, mut rd) = (0usize, 0usize, 0usize);
    assert_eq!(
        t4a_simplett_f64_site_tensor(tt, 0, out_data.as_mut_ptr(), 4, &mut ld, &mut sd, &mut rd),
        T4A_SUCCESS
    );
    assert_eq!((ld, sd, rd), (1, 2, 2));
    assert_eq!(out_data, [0.0, 1.0, 2.0, 3.0]);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_f64_from_site_tensors_data_length_mismatch() {
    let left_dims: [libc::size_t; 1] = [1];
    let site_dims: [libc::size_t; 1] = [2];
    let right_dims: [libc::size_t; 1] = [1];
    let data: [f64; 3] = [1.0, 2.0, 3.0]; // too many for 1*2*1=2

    let mut tt: *mut t4a_simplett_f64 = std::ptr::null_mut();
    let status = t4a_simplett_f64_from_site_tensors(
        1,
        left_dims.as_ptr(),
        site_dims.as_ptr(),
        right_dims.as_ptr(),
        data.as_ptr(),
        data.len(),
        &mut tt,
    );
    assert_ne!(status, T4A_SUCCESS);
}

#[test]
fn test_simplett_c64_from_site_tensors() {
    // 1-site TT: shape (1, 2, 1), complex data [1+2i, 3+4i]
    let left_dims: [libc::size_t; 1] = [1];
    let site_dims: [libc::size_t; 1] = [2];
    let right_dims: [libc::size_t; 1] = [1];
    // interleaved: [re, im, re, im]
    let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

    let mut tt: *mut t4a_simplett_c64 = std::ptr::null_mut();
    let status = t4a_simplett_c64_from_site_tensors(
        1,
        left_dims.as_ptr(),
        site_dims.as_ptr(),
        right_dims.as_ptr(),
        data.as_ptr(),
        data.len(),
        &mut tt,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!tt.is_null());

    // Evaluate at index 0 -> 1+2i
    let indices: [libc::size_t; 1] = [0];
    let (mut re, mut im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_evaluate(tt, indices.as_ptr(), 1, &mut re, &mut im),
        T4A_SUCCESS
    );
    assert!((re - 1.0).abs() < 1e-12);
    assert!((im - 2.0).abs() < 1e-12);

    // Evaluate at index 1 -> 3+4i
    let indices: [libc::size_t; 1] = [1];
    assert_eq!(
        t4a_simplett_c64_evaluate(tt, indices.as_ptr(), 1, &mut re, &mut im),
        T4A_SUCCESS
    );
    assert!((re - 3.0).abs() < 1e-12);
    assert!((im - 4.0).abs() < 1e-12);

    t4a_simplett_c64_release(tt);
}

// ============================================================================
// Category 1: add tests
// ============================================================================

#[test]
fn test_simplett_f64_add() {
    let dims: [libc::size_t; 2] = [2, 3];
    let a = t4a_simplett_f64_constant(dims.as_ptr(), 2, 1.0);
    let b = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);

    let mut result: *mut t4a_simplett_f64 = std::ptr::null_mut();
    assert_eq!(t4a_simplett_f64_add(a, b, &mut result), T4A_SUCCESS);
    assert!(!result.is_null());

    let mut sum = 0.0;
    assert_eq!(t4a_simplett_f64_sum(result, &mut sum), T4A_SUCCESS);
    // (1 + 2) * 2 * 3 = 18
    assert!((sum - 18.0).abs() < 1e-10);

    t4a_simplett_f64_release(a);
    t4a_simplett_f64_release(b);
    t4a_simplett_f64_release(result);
}

#[test]
fn test_simplett_c64_add() {
    let dims: [libc::size_t; 2] = [2, 3];
    let a = t4a_simplett_c64_constant(dims.as_ptr(), 2, 1.0, 1.0);
    let b = t4a_simplett_c64_constant(dims.as_ptr(), 2, 2.0, -1.0);

    let mut result: *mut t4a_simplett_c64 = std::ptr::null_mut();
    assert_eq!(t4a_simplett_c64_add(a, b, &mut result), T4A_SUCCESS);
    assert!(!result.is_null());

    let (mut sum_re, mut sum_im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_sum(result, &mut sum_re, &mut sum_im),
        T4A_SUCCESS
    );
    // (1+i + 2-i) * 6 = 3*6 + 0i = 18
    assert!((sum_re - 18.0).abs() < 1e-10);
    assert!(sum_im.abs() < 1e-10);

    t4a_simplett_c64_release(a);
    t4a_simplett_c64_release(b);
    t4a_simplett_c64_release(result);
}

// ============================================================================
// Category 1: scale tests
// ============================================================================

#[test]
fn test_simplett_f64_scale() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);

    assert_eq!(t4a_simplett_f64_scale(tt, 3.0), T4A_SUCCESS);

    let mut sum = 0.0;
    assert_eq!(t4a_simplett_f64_sum(tt, &mut sum), T4A_SUCCESS);
    // 2 * 3 * 6 = 36
    assert!((sum - 36.0).abs() < 1e-10);

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_c64_scale() {
    let dims: [libc::size_t; 1] = [2];
    let tt = t4a_simplett_c64_constant(dims.as_ptr(), 1, 1.0, 0.0);

    // Scale by i: (0 + 1i)
    assert_eq!(t4a_simplett_c64_scale(tt, 0.0, 1.0), T4A_SUCCESS);

    let (mut sum_re, mut sum_im) = (0.0, 0.0);
    assert_eq!(
        t4a_simplett_c64_sum(tt, &mut sum_re, &mut sum_im),
        T4A_SUCCESS
    );
    // Each element was 1+0i, now 0+1i, sum over 2 = 0+2i
    assert!(sum_re.abs() < 1e-12);
    assert!((sum_im - 2.0).abs() < 1e-12);

    t4a_simplett_c64_release(tt);
}

// ============================================================================
// Category 1: dot tests
// ============================================================================

#[test]
fn test_simplett_f64_dot() {
    let dims: [libc::size_t; 2] = [2, 3];
    let a = t4a_simplett_f64_constant(dims.as_ptr(), 2, 2.0);
    let b = t4a_simplett_f64_constant(dims.as_ptr(), 2, 3.0);

    let mut value = 0.0;
    assert_eq!(t4a_simplett_f64_dot(a, b, &mut value), T4A_SUCCESS);
    // dot = sum(a[i]*b[i]) = 2*3*6 = 36
    assert!((value - 36.0).abs() < 1e-10);

    t4a_simplett_f64_release(a);
    t4a_simplett_f64_release(b);
}

#[test]
fn test_simplett_c64_dot() {
    let dims: [libc::size_t; 1] = [2];
    let a = t4a_simplett_c64_constant(dims.as_ptr(), 1, 1.0, 1.0); // 1+i
    let b = t4a_simplett_c64_constant(dims.as_ptr(), 1, 1.0, -1.0); // 1-i

    let (mut re, mut im) = (0.0, 0.0);
    assert_eq!(t4a_simplett_c64_dot(a, b, &mut re, &mut im), T4A_SUCCESS);
    // dot = sum((1+i)*(1-i)) = sum(1-i+i-i^2) = sum(2) = 2*2 = 4
    assert!((re - 4.0).abs() < 1e-10);
    assert!(im.abs() < 1e-10);

    t4a_simplett_c64_release(a);
    t4a_simplett_c64_release(b);
}

// ============================================================================
// Category 1: reverse tests
// ============================================================================

#[test]
fn test_simplett_f64_reverse() {
    // Build a non-symmetric TT to verify reversal
    let t0 = tensor3_from_data(vec![1.0, 2.0], 1, 2, 1);
    let t1 = tensor3_from_data(vec![10.0, 20.0, 30.0], 1, 3, 1);
    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let handle = Box::into_raw(Box::new(t4a_simplett_f64::new(tt)));

    let mut reversed: *mut t4a_simplett_f64 = std::ptr::null_mut();
    assert_eq!(t4a_simplett_f64_reverse(handle, &mut reversed), T4A_SUCCESS);
    assert!(!reversed.is_null());

    // Original: site_dims = [2, 3]
    // Reversed: site_dims = [3, 2]
    let mut out_dims = [0usize; 2];
    assert_eq!(
        t4a_simplett_f64_site_dims(reversed, out_dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(out_dims, [3, 2]);

    // Verify: reversed[s1, s0] == original[s0, s1]
    let original_indices: [libc::size_t; 2] = [1, 2];
    let reversed_indices: [libc::size_t; 2] = [2, 1];
    let mut orig_val = 0.0;
    let mut rev_val = 0.0;
    assert_eq!(
        t4a_simplett_f64_evaluate(handle, original_indices.as_ptr(), 2, &mut orig_val),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_simplett_f64_evaluate(reversed, reversed_indices.as_ptr(), 2, &mut rev_val),
        T4A_SUCCESS
    );
    assert!((orig_val - rev_val).abs() < 1e-12);

    t4a_simplett_f64_release(handle);
    t4a_simplett_f64_release(reversed);
}

#[test]
fn test_simplett_c64_reverse() {
    let dims: [libc::size_t; 2] = [2, 3];
    let handle = t4a_simplett_c64_constant(dims.as_ptr(), 2, 2.0, 3.0);

    let mut reversed: *mut t4a_simplett_c64 = std::ptr::null_mut();
    assert_eq!(t4a_simplett_c64_reverse(handle, &mut reversed), T4A_SUCCESS);

    let mut out_dims = [0usize; 2];
    assert_eq!(
        t4a_simplett_c64_site_dims(reversed, out_dims.as_mut_ptr(), 2),
        T4A_SUCCESS
    );
    assert_eq!(out_dims, [3, 2]);

    t4a_simplett_c64_release(handle);
    t4a_simplett_c64_release(reversed);
}

// ============================================================================
// Category 1: fulltensor tests
// ============================================================================

#[test]
fn test_simplett_f64_fulltensor() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 5.0);

    // Query length
    let mut data_len = 0usize;
    assert_eq!(
        t4a_simplett_f64_fulltensor(tt, std::ptr::null_mut(), 0, &mut data_len),
        T4A_SUCCESS
    );
    assert_eq!(data_len, 6); // 2*3 = 6

    // Fill
    let mut data = vec![0.0; data_len];
    assert_eq!(
        t4a_simplett_f64_fulltensor(tt, data.as_mut_ptr(), data_len, &mut data_len),
        T4A_SUCCESS
    );
    for &v in &data {
        assert!((v - 5.0).abs() < 1e-12);
    }

    t4a_simplett_f64_release(tt);
}

#[test]
fn test_simplett_c64_fulltensor() {
    let dims: [libc::size_t; 1] = [3];
    let tt = t4a_simplett_c64_constant(dims.as_ptr(), 1, 1.0, 2.0);

    // Query length
    let mut data_len = 0usize;
    assert_eq!(
        t4a_simplett_c64_fulltensor(tt, std::ptr::null_mut(), 0, &mut data_len),
        T4A_SUCCESS
    );
    assert_eq!(data_len, 6); // 3 elements * 2 doubles each

    // Fill
    let mut data = vec![0.0; data_len];
    assert_eq!(
        t4a_simplett_c64_fulltensor(tt, data.as_mut_ptr(), data_len, &mut data_len),
        T4A_SUCCESS
    );
    // Should be [1, 2, 1, 2, 1, 2]
    for chunk in data.chunks_exact(2) {
        assert!((chunk[0] - 1.0).abs() < 1e-12);
        assert!((chunk[1] - 2.0).abs() < 1e-12);
    }

    t4a_simplett_c64_release(tt);
}

#[test]
fn test_simplett_f64_fulltensor_buffer_too_small() {
    let dims: [libc::size_t; 2] = [2, 3];
    let tt = t4a_simplett_f64_constant(dims.as_ptr(), 2, 1.0);

    let mut data_len = 0usize;
    let mut data = [0.0; 3]; // too small for 6 elements
    assert_eq!(
        t4a_simplett_f64_fulltensor(tt, data.as_mut_ptr(), data.len(), &mut data_len),
        T4A_BUFFER_TOO_SMALL
    );

    t4a_simplett_f64_release(tt);
}

// ============================================================================
// Category 1: null pointer guard tests
// ============================================================================

#[test]
fn test_simplett_new_functions_null_guards() {
    assert_eq!(
        t4a_simplett_f64_add(std::ptr::null(), std::ptr::null(), std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_scale(std::ptr::null_mut(), 1.0),
        T4A_NULL_POINTER
    );
    let mut val = 0.0;
    assert_eq!(
        t4a_simplett_f64_dot(std::ptr::null(), std::ptr::null(), &mut val),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_simplett_f64_reverse(std::ptr::null(), std::ptr::null_mut()),
        T4A_NULL_POINTER
    );
    let mut data_len = 0usize;
    assert_eq!(
        t4a_simplett_f64_fulltensor(std::ptr::null(), std::ptr::null_mut(), 0, &mut data_len),
        T4A_NULL_POINTER
    );
}
