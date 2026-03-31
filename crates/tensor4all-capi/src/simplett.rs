//! C API for tensor4all-simplett TensorTrain
//!
//! This provides a simpler TensorTrain interface designed for TCI operations.
//! The tensors are stored as flat arrays with explicit dimensions.

use crate::types::{t4a_simplett_c64, t4a_simplett_f64};
use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER,
    T4A_SUCCESS,
};
use num_complex::Complex64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

// ============================================================================
// Lifecycle functions
// ============================================================================

/// Release a SimpleTT tensor train handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_release(ptr: *mut t4a_simplett_f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Clone a SimpleTT tensor train.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_clone(ptr: *const t4a_simplett_f64) -> *mut t4a_simplett_f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        Box::into_raw(Box::new(t4a_simplett_f64::new(tt.inner().clone())))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Check if the SimpleTT f64 handle is assigned (non-null and dereferenceable).
///
/// # Returns
/// 1 if valid, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_is_assigned(ptr: *const t4a_simplett_f64) -> i32 {
    if ptr.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*ptr;
        1
    }));

    match result {
        Ok(v) => v,
        Err(panic) => {
            let msg = crate::panic_message(&*panic);
            crate::set_last_error(&msg);
            0
        }
    }
}

// ============================================================================
// Constructors
// ============================================================================

/// Create a constant tensor train.
///
/// # Arguments
/// * `site_dims` - Array of site dimensions
/// * `n_sites` - Number of sites
/// * `value` - Constant value for all elements
///
/// # Returns
/// A new tensor train handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_constant(
    site_dims: *const libc::size_t,
    n_sites: libc::size_t,
    value: libc::c_double,
) -> *mut t4a_simplett_f64 {
    if site_dims.is_null() && n_sites > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites).map(|i| unsafe { *site_dims.add(i) }).collect();
        let tt = TensorTrain::<f64>::constant(&dims, value);
        Box::into_raw(Box::new(t4a_simplett_f64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a zero tensor train.
///
/// # Arguments
/// * `site_dims` - Array of site dimensions
/// * `n_sites` - Number of sites
///
/// # Returns
/// A new tensor train handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_zeros(
    site_dims: *const libc::size_t,
    n_sites: libc::size_t,
) -> *mut t4a_simplett_f64 {
    if site_dims.is_null() && n_sites > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites).map(|i| unsafe { *site_dims.add(i) }).collect();
        let tt = TensorTrain::<f64>::zeros(&dims);
        Box::into_raw(Box::new(t4a_simplett_f64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the number of sites.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_len(
    ptr: *const t4a_simplett_f64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_len = tt.inner().len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the site dimensions.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_dims` - Output buffer for site dimensions
/// * `buf_len` - Buffer length
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_site_dims(
    ptr: *const t4a_simplett_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().site_dims();
        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        for (i, d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = *d };
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the link (bond) dimensions.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_dims` - Output buffer for link dimensions (length = n_sites - 1)
/// * `buf_len` - Buffer length
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_link_dims(
    ptr: *const t4a_simplett_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().link_dims();
        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        for (i, d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = *d };
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum bond dimension (rank).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_rank(
    ptr: *const t4a_simplett_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_rank = tt.inner().rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Evaluate the tensor train at a given multi-index.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `indices` - Array of indices (one per site)
/// * `n_indices` - Number of indices
/// * `out_value` - Output value
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_evaluate(
    ptr: *const t4a_simplett_f64,
    indices: *const libc::size_t,
    n_indices: libc::size_t,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let idx: Vec<usize> = (0..n_indices).map(|i| unsafe { *indices.add(i) }).collect();

        match tt.inner().evaluate(&idx) {
            Ok(val) => {
                unsafe { *out_value = val };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the sum over all indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_sum(
    ptr: *const t4a_simplett_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_value = tt.inner().sum() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Compute the Frobenius norm.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_norm(
    ptr: *const t4a_simplett_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_value = tt.inner().norm() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get site tensor data at a specific site.
///
/// The tensor has shape (left_dim, site_dim, right_dim) in column-major order.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `site` - Site index (0-based)
/// * `out_data` - Output buffer for tensor data
/// * `buf_len` - Buffer length
/// * `out_left_dim` - Output left bond dimension
/// * `out_site_dim` - Output site dimension
/// * `out_right_dim` - Output right bond dimension
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_site_tensor(
    ptr: *const t4a_simplett_f64,
    site: libc::size_t,
    out_data: *mut libc::c_double,
    buf_len: libc::size_t,
    out_left_dim: *mut libc::size_t,
    out_site_dim: *mut libc::size_t,
    out_right_dim: *mut libc::size_t,
) -> StatusCode {
    use tensor4all_simplett::Tensor3Ops;

    if ptr.is_null() || out_data.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        if site >= tt.inner().len() {
            return T4A_INVALID_ARGUMENT;
        }

        let tensor = tt.inner().site_tensor(site);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();
        let total_size = left_dim * site_dim * right_dim;

        if buf_len < total_size {
            return T4A_INVALID_ARGUMENT;
        }

        // Copy data in column-major order: left axis varies fastest.
        let mut idx = 0;
        for r in 0..right_dim {
            for s in 0..site_dim {
                for l in 0..left_dim {
                    unsafe { *out_data.add(idx) = *tensor.get3(l, s, r) };
                    idx += 1;
                }
            }
        }

        if !out_left_dim.is_null() {
            unsafe { *out_left_dim = left_dim };
        }
        if !out_site_dim.is_null() {
            unsafe { *out_site_dim = site_dim };
        }
        if !out_right_dim.is_null() {
            unsafe { *out_right_dim = right_dim };
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Compression
// ============================================================================

/// Compress a tensor train in-place.
///
/// # Arguments
/// * `ptr` - SimpleTT handle (mutable)
/// * `method` - 0=LU, 1=CI, 2=SVD
/// * `tolerance` - Relative tolerance for truncation
/// * `max_bonddim` - Maximum bond dimension (0 for unlimited)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_compress(
    ptr: *mut t4a_simplett_f64,
    method: libc::c_int,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        use tensor4all_simplett::compression::{CompressionMethod, CompressionOptions};

        let comp_method = match method {
            0 => CompressionMethod::LU,
            1 => CompressionMethod::CI,
            2 => CompressionMethod::SVD,
            _ => {
                return crate::err_status(
                    format!("Invalid compression method: {method}. Use 0=LU, 1=CI, 2=SVD"),
                    T4A_INVALID_ARGUMENT,
                );
            }
        };

        let options = CompressionOptions {
            method: comp_method,
            tolerance,
            max_bond_dim: if max_bonddim == 0 {
                usize::MAX
            } else {
                max_bonddim
            },
            ..Default::default()
        };

        match tt.inner_mut().compress(&options) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Partial sum
// ============================================================================

/// Sum over selected dimensions, returning a new TensorTrain.
///
/// # Arguments
/// * `ptr` - SimpleTT handle (const)
/// * `dims` - Array of 0-indexed dimensions to sum over
/// * `n_dims` - Number of dimensions to sum
/// * `out` - Output: new SimpleTT handle
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_partial_sum(
    ptr: *const t4a_simplett_f64,
    dims: *const libc::size_t,
    n_dims: libc::size_t,
    out: *mut *mut t4a_simplett_f64,
) -> StatusCode {
    if ptr.is_null() || out.is_null() || (dims.is_null() && n_dims > 0) {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dim_vec: Vec<usize> = (0..n_dims).map(|i| unsafe { *dims.add(i) }).collect();

        match tt.inner().partial_sum(&dim_vec) {
            Ok(result_tt) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_simplett_f64::new(result_tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Complex64 lifecycle functions
// ============================================================================

/// Release a SimpleTT tensor train handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_release(ptr: *mut t4a_simplett_c64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Clone a SimpleTT tensor train.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_clone(ptr: *const t4a_simplett_c64) -> *mut t4a_simplett_c64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        Box::into_raw(Box::new(tt.clone()))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Check if the SimpleTT c64 handle is assigned (non-null and dereferenceable).
///
/// # Returns
/// 1 if valid, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_is_assigned(ptr: *const t4a_simplett_c64) -> i32 {
    if ptr.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = &*ptr;
        1
    }));

    match result {
        Ok(v) => v,
        Err(panic) => {
            let msg = crate::panic_message(&*panic);
            crate::set_last_error(&msg);
            0
        }
    }
}

// ============================================================================
// Complex64 constructors
// ============================================================================

/// Create a complex constant tensor train.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_constant(
    site_dims: *const libc::size_t,
    n_sites: libc::size_t,
    value_re: libc::c_double,
    value_im: libc::c_double,
) -> *mut t4a_simplett_c64 {
    if site_dims.is_null() && n_sites > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites).map(|i| unsafe { *site_dims.add(i) }).collect();
        let tt = TensorTrain::<Complex64>::constant(&dims, Complex64::new(value_re, value_im));
        Box::into_raw(Box::new(t4a_simplett_c64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

/// Create a zero complex tensor train.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_zeros(
    site_dims: *const libc::size_t,
    n_sites: libc::size_t,
) -> *mut t4a_simplett_c64 {
    if site_dims.is_null() && n_sites > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites).map(|i| unsafe { *site_dims.add(i) }).collect();
        let tt = TensorTrain::<Complex64>::zeros(&dims);
        Box::into_raw(Box::new(t4a_simplett_c64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// Complex64 accessors
// ============================================================================

/// Get the number of sites.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_len(
    ptr: *const t4a_simplett_c64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_len = tt.inner().len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the site dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_site_dims(
    ptr: *const t4a_simplett_c64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().site_dims();
        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        for (i, d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = *d };
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the link (bond) dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_link_dims(
    ptr: *const t4a_simplett_c64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().link_dims();
        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        for (i, d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = *d };
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum bond dimension (rank).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_rank(
    ptr: *const t4a_simplett_c64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_rank = tt.inner().rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Evaluate the tensor train at a given multi-index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_evaluate(
    ptr: *const t4a_simplett_c64,
    indices: *const libc::size_t,
    n_indices: libc::size_t,
    out_value_re: *mut libc::c_double,
    out_value_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_value_re.is_null() || out_value_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let idx: Vec<usize> = (0..n_indices).map(|i| unsafe { *indices.add(i) }).collect();

        match tt.inner().evaluate(&idx) {
            Ok(val) => {
                unsafe {
                    *out_value_re = val.re;
                    *out_value_im = val.im;
                }
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the sum over all indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_sum(
    ptr: *const t4a_simplett_c64,
    out_value_re: *mut libc::c_double,
    out_value_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value_re.is_null() || out_value_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let value = tt.inner().sum();
        unsafe {
            *out_value_re = value.re;
            *out_value_im = value.im;
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Compute the Frobenius norm.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_norm(
    ptr: *const t4a_simplett_c64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_value = tt.inner().norm() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get site tensor data at a specific site as interleaved complex values.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_site_tensor(
    ptr: *const t4a_simplett_c64,
    site: libc::size_t,
    out_data: *mut libc::c_double,
    buf_len: libc::size_t,
    out_left_dim: *mut libc::size_t,
    out_site_dim: *mut libc::size_t,
    out_right_dim: *mut libc::size_t,
) -> StatusCode {
    use tensor4all_simplett::Tensor3Ops;

    if ptr.is_null() || out_data.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        if site >= tt.inner().len() {
            return T4A_INVALID_ARGUMENT;
        }

        let tensor = tt.inner().site_tensor(site);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();
        let total_size = left_dim * site_dim * right_dim;

        if buf_len < 2 * total_size {
            return T4A_INVALID_ARGUMENT;
        }

        let mut idx = 0;
        for r in 0..right_dim {
            for s in 0..site_dim {
                for l in 0..left_dim {
                    let value = *tensor.get3(l, s, r);
                    unsafe {
                        *out_data.add(2 * idx) = value.re;
                        *out_data.add(2 * idx + 1) = value.im;
                    }
                    idx += 1;
                }
            }
        }

        if !out_left_dim.is_null() {
            unsafe { *out_left_dim = left_dim };
        }
        if !out_site_dim.is_null() {
            unsafe { *out_site_dim = site_dim };
        }
        if !out_right_dim.is_null() {
            unsafe { *out_right_dim = right_dim };
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Complex64 compression
// ============================================================================

/// Compress a complex tensor train in-place.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_compress(
    ptr: *mut t4a_simplett_c64,
    method: libc::c_int,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        use tensor4all_simplett::compression::{CompressionMethod, CompressionOptions};

        let comp_method = match method {
            0 => CompressionMethod::LU,
            1 => CompressionMethod::CI,
            2 => CompressionMethod::SVD,
            _ => {
                return crate::err_status(
                    format!("Invalid compression method: {method}. Use 0=LU, 1=CI, 2=SVD"),
                    T4A_INVALID_ARGUMENT,
                );
            }
        };

        let options = CompressionOptions {
            method: comp_method,
            tolerance,
            max_bond_dim: if max_bonddim == 0 {
                usize::MAX
            } else {
                max_bonddim
            },
            ..Default::default()
        };

        match tt.inner_mut().compress(&options) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Complex64 partial sum
// ============================================================================

/// Sum over selected dimensions, returning a new complex TensorTrain.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_partial_sum(
    ptr: *const t4a_simplett_c64,
    dims: *const libc::size_t,
    n_dims: libc::size_t,
    out: *mut *mut t4a_simplett_c64,
) -> StatusCode {
    if ptr.is_null() || out.is_null() || (dims.is_null() && n_dims > 0) {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dim_vec: Vec<usize> = (0..n_dims).map(|i| unsafe { *dims.add(i) }).collect();

        match tt.inner().partial_sum(&dim_vec) {
            Ok(result_tt) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_simplett_c64::new(result_tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// from_site_tensors — f64
// ============================================================================

/// Construct a SimpleTT from concatenated site tensor data (column-major).
///
/// Each tensor has shape (left_dim[i], site_dim[i], right_dim[i]).
/// `data` contains all tensors concatenated in order.
///
/// # Arguments
/// * `n_sites` - Number of sites (tensors)
/// * `left_dims` - Array of left bond dimensions [n_sites]
/// * `site_dims` - Array of site dimensions [n_sites]
/// * `right_dims` - Array of right bond dimensions [n_sites]
/// * `data` - Concatenated column-major tensor data
/// * `data_len` - Total length of data
/// * `out_ptr` - Output: new SimpleTT handle
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_from_site_tensors(
    n_sites: libc::size_t,
    left_dims: *const libc::size_t,
    site_dims: *const libc::size_t,
    right_dims: *const libc::size_t,
    data: *const libc::c_double,
    data_len: libc::size_t,
    out_ptr: *mut *mut t4a_simplett_f64,
) -> StatusCode {
    if out_ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    if n_sites > 0
        && (left_dims.is_null() || site_dims.is_null() || right_dims.is_null() || data.is_null())
    {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_simplett::tensor3_from_data;

        let mut tensors = Vec::with_capacity(n_sites);
        let mut offset = 0usize;

        for i in 0..n_sites {
            let ld = unsafe { *left_dims.add(i) };
            let sd = unsafe { *site_dims.add(i) };
            let rd = unsafe { *right_dims.add(i) };
            let size = ld * sd * rd;

            if offset + size > data_len {
                return crate::err_status(
                    format!(
                        "Data buffer too small: need {} elements for site {i}, but only {} remain",
                        size,
                        data_len - offset
                    ),
                    T4A_INVALID_ARGUMENT,
                );
            }

            let slice = unsafe { std::slice::from_raw_parts(data.add(offset), size) };
            let tensor = tensor3_from_data(slice.to_vec(), ld, sd, rd);
            tensors.push(tensor);
            offset += size;
        }

        if offset != data_len {
            return crate::err_status(
                format!("Data length mismatch: used {offset} elements but data_len is {data_len}"),
                T4A_INVALID_ARGUMENT,
            );
        }

        match TensorTrain::new(tensors) {
            Ok(tt) => {
                unsafe { *out_ptr = Box::into_raw(Box::new(t4a_simplett_f64::new(tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// from_site_tensors — c64
// ============================================================================

/// Construct a complex SimpleTT from concatenated site tensor data.
///
/// Data is interleaved [re, im, re, im, ...] in column-major order per tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_from_site_tensors(
    n_sites: libc::size_t,
    left_dims: *const libc::size_t,
    site_dims: *const libc::size_t,
    right_dims: *const libc::size_t,
    data: *const libc::c_double,
    data_len: libc::size_t,
    out_ptr: *mut *mut t4a_simplett_c64,
) -> StatusCode {
    if out_ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    if n_sites > 0
        && (left_dims.is_null() || site_dims.is_null() || right_dims.is_null() || data.is_null())
    {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_simplett::tensor3_from_data;

        let mut tensors = Vec::with_capacity(n_sites);
        let mut offset = 0usize;

        for i in 0..n_sites {
            let ld = unsafe { *left_dims.add(i) };
            let sd = unsafe { *site_dims.add(i) };
            let rd = unsafe { *right_dims.add(i) };
            let n_elements = ld * sd * rd;
            let n_doubles = 2 * n_elements;

            if offset + n_doubles > data_len {
                return crate::err_status(
                    format!(
                        "Data buffer too small: need {} doubles for site {i}, but only {} remain",
                        n_doubles,
                        data_len - offset
                    ),
                    T4A_INVALID_ARGUMENT,
                );
            }

            let slice = unsafe { std::slice::from_raw_parts(data.add(offset), n_doubles) };
            let complex_data: Vec<Complex64> = slice
                .chunks_exact(2)
                .map(|c| Complex64::new(c[0], c[1]))
                .collect();
            let tensor = tensor3_from_data(complex_data, ld, sd, rd);
            tensors.push(tensor);
            offset += n_doubles;
        }

        if offset != data_len {
            return crate::err_status(
                format!("Data length mismatch: used {offset} doubles but data_len is {data_len}"),
                T4A_INVALID_ARGUMENT,
            );
        }

        match TensorTrain::new(tensors) {
            Ok(tt) => {
                unsafe { *out_ptr = Box::into_raw(Box::new(t4a_simplett_c64::new(tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// add — f64
// ============================================================================

/// Add two tensor trains, returning a new tensor train.
///
/// The two tensor trains must have the same site dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_add(
    a: *const t4a_simplett_f64,
    b: *const t4a_simplett_f64,
    out: *mut *mut t4a_simplett_f64,
) -> StatusCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };

        match tt_a.inner().add(tt_b.inner()) {
            Ok(result_tt) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_simplett_f64::new(result_tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// add — c64
// ============================================================================

/// Add two complex tensor trains, returning a new tensor train.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_add(
    a: *const t4a_simplett_c64,
    b: *const t4a_simplett_c64,
    out: *mut *mut t4a_simplett_c64,
) -> StatusCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };

        match tt_a.inner().add(tt_b.inner()) {
            Ok(result_tt) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_simplett_c64::new(result_tt))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// scale — f64
// ============================================================================

/// Scale a tensor train in-place by a scalar factor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_scale(
    ptr: *mut t4a_simplett_f64,
    factor: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        tt.inner_mut().scale(factor);
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// scale — c64
// ============================================================================

/// Scale a complex tensor train in-place by a complex scalar factor.
///
/// The factor is given as (re, im) pair.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_scale(
    ptr: *mut t4a_simplett_c64,
    factor_re: libc::c_double,
    factor_im: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        tt.inner_mut().scale(Complex64::new(factor_re, factor_im));
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// dot — f64
// ============================================================================

/// Compute the inner product (dot product) of two tensor trains.
///
/// Returns sum over all indices i of a[i] * b[i].
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_dot(
    a: *const t4a_simplett_f64,
    b: *const t4a_simplett_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if a.is_null() || b.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };

        match tt_a.inner().dot(tt_b.inner()) {
            Ok(val) => {
                unsafe { *out_value = val };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// dot — c64
// ============================================================================

/// Compute the inner product (dot product) of two complex tensor trains.
///
/// Returns (re, im) via out params.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_dot(
    a: *const t4a_simplett_c64,
    b: *const t4a_simplett_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if a.is_null() || b.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };

        match tt_a.inner().dot(tt_b.inner()) {
            Ok(val) => {
                unsafe {
                    *out_re = val.re;
                    *out_im = val.im;
                }
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// reverse — f64
// ============================================================================

/// Reverse the tensor train (swap left and right), returning a new handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_reverse(
    ptr: *const t4a_simplett_f64,
    out: *mut *mut t4a_simplett_f64,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let reversed = tt.inner().reverse();
        unsafe { *out = Box::into_raw(Box::new(t4a_simplett_f64::new(reversed))) };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// reverse — c64
// ============================================================================

/// Reverse the complex tensor train (swap left and right), returning a new handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_reverse(
    ptr: *const t4a_simplett_c64,
    out: *mut *mut t4a_simplett_c64,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let reversed = tt.inner().reverse();
        unsafe { *out = Box::into_raw(Box::new(t4a_simplett_c64::new(reversed))) };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// fulltensor — f64
// ============================================================================

/// Expand the tensor train to a dense tensor.
///
/// Returns data in column-major order.
/// Caller knows dimensions via site_dims(); only the data buffer is needed.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_data` - Output buffer for data (NULL to query length)
/// * `buf_len` - Buffer length
/// * `out_data_len` - Actual data length
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_f64_fulltensor(
    ptr: *const t4a_simplett_f64,
    out_data: *mut libc::c_double,
    buf_len: libc::size_t,
    out_data_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_data_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let (data, _shape) = tt.inner().fulltensor();

        unsafe { *out_data_len = data.len() };

        if out_data.is_null() {
            // Query mode
            return T4A_SUCCESS;
        }

        if buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out_data, data.len());
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// fulltensor — c64
// ============================================================================

/// Expand the complex tensor train to a dense tensor.
///
/// Returns interleaved [re, im, ...] data in column-major order.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_data` - Output buffer for interleaved complex data (NULL to query length)
/// * `buf_len` - Buffer length (in doubles, i.e. 2 * n_elements)
/// * `out_data_len` - Actual data length in doubles (2 * n_elements)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_simplett_c64_fulltensor(
    ptr: *const t4a_simplett_c64,
    out_data: *mut libc::c_double,
    buf_len: libc::size_t,
    out_data_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_data_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let (data, _shape) = tt.inner().fulltensor();
        let n_doubles = 2 * data.len();

        unsafe { *out_data_len = n_doubles };

        if out_data.is_null() {
            // Query mode
            return T4A_SUCCESS;
        }

        if buf_len < n_doubles {
            return T4A_BUFFER_TOO_SMALL;
        }

        for (i, val) in data.iter().enumerate() {
            unsafe {
                *out_data.add(2 * i) = val.re;
                *out_data.add(2 * i + 1) = val.im;
            }
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

#[cfg(test)]
mod tests;
