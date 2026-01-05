//! C API for TensorTrain
//!
//! Provides functions for creating, manipulating, and accessing tensor trains.
//!
//! ## Naming convention
//! - `t4a_tt_f64_*` - Functions for TensorTrain<f64>
//! - `t4a_tt_c64_*` - Functions for TensorTrain<Complex64>

use std::panic::catch_unwind;
use std::ptr;

use num_complex::Complex64;
use tensor4all_core_linalg::default_svd_rtol;
use tensor4all_tensortrain::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, TensorTrain,
};

use crate::types::{t4a_tt_c64, t4a_tt_f64};
use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER,
    T4A_SUCCESS,
};

// Generate lifecycle functions for t4a_tt_f64
impl_opaque_type_common!(tt_f64);

// Generate lifecycle functions for t4a_tt_c64
impl_opaque_type_common!(tt_c64);

// ============================================================================
// Constructors - f64
// ============================================================================

/// Create a new tensor train representing the zero function (f64)
///
/// # Arguments
/// - `site_dims`: Array of site dimensions
/// - `num_sites`: Number of sites (length of site_dims array)
///
/// # Returns
/// - Pointer to new t4a_tt_f64 on success
/// - NULL on error
///
/// # Safety
/// - `site_dims` must be a valid pointer to an array of length `num_sites`
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_new_zeros(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_tt_f64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        let tt = TensorTrain::<f64>::zeros(&dims);
        Box::into_raw(Box::new(t4a_tt_f64::new(tt)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create a new tensor train representing a constant function (f64)
///
/// # Arguments
/// - `site_dims`: Array of site dimensions
/// - `num_sites`: Number of sites (length of site_dims array)
/// - `value`: The constant value
///
/// # Returns
/// - Pointer to new t4a_tt_f64 on success
/// - NULL on error
///
/// # Safety
/// - `site_dims` must be a valid pointer to an array of length `num_sites`
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_new_constant(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
    value: libc::c_double,
) -> *mut t4a_tt_f64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        let tt = TensorTrain::<f64>::constant(&dims, value);
        Box::into_raw(Box::new(t4a_tt_f64::new(tt)))
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Constructors - Complex64
// ============================================================================

/// Create a new tensor train representing the zero function (Complex64)
///
/// # Arguments
/// - `site_dims`: Array of site dimensions
/// - `num_sites`: Number of sites (length of site_dims array)
///
/// # Returns
/// - Pointer to new t4a_tt_c64 on success
/// - NULL on error
///
/// # Safety
/// - `site_dims` must be a valid pointer to an array of length `num_sites`
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_new_zeros(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_tt_c64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        let tt = TensorTrain::<Complex64>::zeros(&dims);
        Box::into_raw(Box::new(t4a_tt_c64::new(tt)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create a new tensor train representing a constant function (Complex64)
///
/// # Arguments
/// - `site_dims`: Array of site dimensions
/// - `num_sites`: Number of sites (length of site_dims array)
/// - `value_re`: Real part of the constant value
/// - `value_im`: Imaginary part of the constant value
///
/// # Returns
/// - Pointer to new t4a_tt_c64 on success
/// - NULL on error
///
/// # Safety
/// - `site_dims` must be a valid pointer to an array of length `num_sites`
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_new_constant(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
    value_re: libc::c_double,
    value_im: libc::c_double,
) -> *mut t4a_tt_c64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        let value = Complex64::new(value_re, value_im);
        let tt = TensorTrain::<Complex64>::constant(&dims, value);
        Box::into_raw(Box::new(t4a_tt_c64::new(tt)))
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Accessors - f64
// ============================================================================

/// Get the number of sites in a tensor train (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_len` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_len(
    ptr: *const t4a_tt_f64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_len = tt.inner().len() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the site dimensions of a tensor train (f64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `out_dims`: Buffer to write dimensions (must have length >= num_sites)
/// - `buf_len`: Length of the buffer
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_dims` must be a valid pointer to a buffer of at least `num_sites` elements
#[no_mangle]
pub extern "C" fn t4a_tt_f64_site_dims(
    ptr: *const t4a_tt_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().site_dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &dim) in dims.iter().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the link dimensions (bond dimensions) of a tensor train (f64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `out_dims`: Buffer to write dimensions (must have length >= num_sites - 1)
/// - `buf_len`: Length of the buffer
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_dims` must be a valid pointer to a buffer of at least `num_sites - 1` elements
#[no_mangle]
pub extern "C" fn t4a_tt_f64_link_dims(
    ptr: *const t4a_tt_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().link_dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &dim) in dims.iter().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the maximum bond dimension (rank) of a tensor train (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_rank` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_rank(
    ptr: *const t4a_tt_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_rank = tt.inner().rank() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Evaluate a tensor train at a given index set (f64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `indices`: Array of indices (length = num_sites)
/// - `num_indices`: Number of indices (must equal num_sites)
/// - `out_value`: Output: evaluated value
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `indices` must be a valid pointer to an array of length `num_indices`
/// - `out_value` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_evaluate(
    ptr: *const t4a_tt_f64,
    indices: *const libc::size_t,
    num_indices: libc::size_t,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let idx: Vec<usize> = unsafe { std::slice::from_raw_parts(indices, num_indices).to_vec() };

        match tt.inner().evaluate(&idx) {
            Ok(value) => {
                unsafe { *out_value = value };
                T4A_SUCCESS
            }
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the sum of all elements in a tensor train (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_sum` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_sum(
    ptr: *const t4a_tt_f64,
    out_sum: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_sum.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_sum = tt.inner().sum() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the Frobenius norm of a tensor train (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_norm` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_norm(
    ptr: *const t4a_tt_f64,
    out_norm: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_norm.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_norm = tt.inner().norm() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the logarithm of the Frobenius norm of a tensor train (f64)
///
/// This is more numerically stable than computing norm and then taking log.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_log_norm` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_log_norm(
    ptr: *const t4a_tt_f64,
    out_log_norm: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_log_norm.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_log_norm = tt.inner().log_norm() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Modifiers - f64
// ============================================================================

/// Scale a tensor train by a factor, returning a new object (f64)
///
/// The original tensor train is unchanged.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_scaled(
    ptr: *const t4a_tt_f64,
    factor: libc::c_double,
) -> *mut t4a_tt_f64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let scaled = tt.inner().scaled(factor);
        Box::into_raw(Box::new(t4a_tt_f64::new(scaled)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Scale a tensor train by a factor in place (f64)
///
/// Modifies the tensor train in place.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
#[no_mangle]
pub extern "C" fn t4a_tt_f64_scale_inplace(
    ptr: *mut t4a_tt_f64,
    factor: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        tt.inner_mut().scale(factor);
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Convert a tensor train to a full tensor (f64)
///
/// Returns the full tensor data in row-major order along with the shape.
/// Warning: This can be very large for high-dimensional tensors!
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `out_data`: Buffer to write data (if NULL, only out_len is written)
/// - `data_buf_len`: Length of the data buffer
/// - `out_len`: Output: required data buffer length
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - `out_data` can be NULL (to query required length)
/// - `out_len` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_fulltensor(
    ptr: *const t4a_tt_f64,
    out_data: *mut libc::c_double,
    data_buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let (data, _shape) = tt.inner().fulltensor();

        unsafe { *out_len = data.len() };

        if out_data.is_null() {
            return T4A_SUCCESS;
        }

        if data_buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), out_data, data.len());
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Accessors - Complex64
// ============================================================================

/// Get the number of sites in a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_len` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_c64_len(
    ptr: *const t4a_tt_c64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_len = tt.inner().len() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the site dimensions of a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_dims` must be a valid pointer to a buffer of at least `num_sites` elements
#[no_mangle]
pub extern "C" fn t4a_tt_c64_site_dims(
    ptr: *const t4a_tt_c64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().site_dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &dim) in dims.iter().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the link dimensions (bond dimensions) of a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_dims` must be a valid pointer to a buffer of at least `num_sites - 1` elements
#[no_mangle]
pub extern "C" fn t4a_tt_c64_link_dims(
    ptr: *const t4a_tt_c64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().link_dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &dim) in dims.iter().enumerate() {
                *out_dims.add(i) = dim;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the maximum bond dimension (rank) of a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_rank` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_c64_rank(
    ptr: *const t4a_tt_c64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_rank = tt.inner().rank() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Evaluate a tensor train at a given index set (Complex64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `indices`: Array of indices (length = num_sites)
/// - `num_indices`: Number of indices (must equal num_sites)
/// - `out_re`: Output: real part of evaluated value
/// - `out_im`: Output: imaginary part of evaluated value
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `indices` must be a valid pointer to an array of length `num_indices`
/// - `out_re` and `out_im` must be valid pointers
#[no_mangle]
pub extern "C" fn t4a_tt_c64_evaluate(
    ptr: *const t4a_tt_c64,
    indices: *const libc::size_t,
    num_indices: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let idx: Vec<usize> = unsafe { std::slice::from_raw_parts(indices, num_indices).to_vec() };

        match tt.inner().evaluate(&idx) {
            Ok(value) => {
                unsafe {
                    *out_re = value.re;
                    *out_im = value.im;
                };
                T4A_SUCCESS
            }
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the sum of all elements in a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_re` and `out_im` must be valid pointers
#[no_mangle]
pub extern "C" fn t4a_tt_c64_sum(
    ptr: *const t4a_tt_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let sum = tt.inner().sum();
        unsafe {
            *out_re = sum.re;
            *out_im = sum.im;
        };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the Frobenius norm of a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_norm` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_c64_norm(
    ptr: *const t4a_tt_c64,
    out_norm: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_norm.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_norm = tt.inner().norm() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the logarithm of the Frobenius norm of a tensor train (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_log_norm` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_c64_log_norm(
    ptr: *const t4a_tt_c64,
    out_log_norm: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_log_norm.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_log_norm = tt.inner().log_norm() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Modifiers - Complex64
// ============================================================================

/// Scale a tensor train by a factor, returning a new object (Complex64)
///
/// The original tensor train is unchanged.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_scaled(
    ptr: *const t4a_tt_c64,
    factor_re: libc::c_double,
    factor_im: libc::c_double,
) -> *mut t4a_tt_c64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let factor = Complex64::new(factor_re, factor_im);
        let scaled = tt.inner().scaled(factor);
        Box::into_raw(Box::new(t4a_tt_c64::new(scaled)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Scale a tensor train by a factor in place (Complex64)
///
/// Modifies the tensor train in place.
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
#[no_mangle]
pub extern "C" fn t4a_tt_c64_scale_inplace(
    ptr: *mut t4a_tt_c64,
    factor_re: libc::c_double,
    factor_im: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        let factor = Complex64::new(factor_re, factor_im);
        tt.inner_mut().scale(factor);
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Convert a tensor train to a full tensor (Complex64)
///
/// Returns the full tensor data in row-major order.
/// Warning: This can be very large for high-dimensional tensors!
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `out_re`: Buffer to write real parts (if NULL, only out_len is written)
/// - `out_im`: Buffer to write imaginary parts (if NULL, only out_len is written)
/// - `data_buf_len`: Length of the data buffers
/// - `out_len`: Output: required data buffer length
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - `out_re` and `out_im` can be NULL (to query required length)
/// - `out_len` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_c64_fulltensor(
    ptr: *const t4a_tt_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
    data_buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let (data, _shape) = tt.inner().fulltensor();

        unsafe { *out_len = data.len() };

        if out_re.is_null() || out_im.is_null() {
            return T4A_SUCCESS;
        }

        if data_buf_len < data.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, z) in data.iter().enumerate() {
                *out_re.add(i) = z.re;
                *out_im.add(i) = z.im;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Arithmetic Operations - f64
// ============================================================================

/// Add two tensor trains element-wise (f64)
///
/// The resulting bond dimension is the sum of the input bond dimensions.
/// Use compress() afterward to reduce the bond dimension.
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_add(
    a: *const t4a_tt_f64,
    b: *const t4a_tt_f64,
) -> *mut t4a_tt_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().add(tt_b.inner()) {
            Ok(sum) => Box::into_raw(Box::new(t4a_tt_f64::new(sum))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Subtract two tensor trains element-wise (f64)
///
/// Computes: a - b
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_sub(
    a: *const t4a_tt_f64,
    b: *const t4a_tt_f64,
) -> *mut t4a_tt_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().sub(tt_b.inner()) {
            Ok(diff) => Box::into_raw(Box::new(t4a_tt_f64::new(diff))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Negate a tensor train (multiply by -1) (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_negate(ptr: *const t4a_tt_f64) -> *mut t4a_tt_f64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let negated = tt.inner().negate();
        Box::into_raw(Box::new(t4a_tt_f64::new(negated)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Reverse a tensor train (swap left and right) (f64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_reverse(ptr: *const t4a_tt_f64) -> *mut t4a_tt_f64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let reversed = tt.inner().reverse();
        Box::into_raw(Box::new(t4a_tt_f64::new(reversed)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Compute the Hadamard (element-wise) product of two tensor trains (f64)
///
/// For each index i: result[i] = a[i] * b[i]
///
/// The resulting bond dimension is the product of the input bond dimensions.
/// Use compress() afterward to reduce the bond dimension.
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_hadamard(
    a: *const t4a_tt_f64,
    b: *const t4a_tt_f64,
) -> *mut t4a_tt_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().hadamard(tt_b.inner()) {
            Ok(prod) => Box::into_raw(Box::new(t4a_tt_f64::new(prod))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Compute the inner product (dot product) of two tensor trains (f64)
///
/// Returns: sum over all indices i of a[i] * b[i]
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_f64
/// - `out_dot` must be a valid pointer
#[no_mangle]
pub extern "C" fn t4a_tt_f64_dot(
    a: *const t4a_tt_f64,
    b: *const t4a_tt_f64,
    out_dot: *mut libc::c_double,
) -> StatusCode {
    if a.is_null() || b.is_null() || out_dot.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().dot(tt_b.inner()) {
            Ok(dot) => {
                unsafe { *out_dot = dot };
                T4A_SUCCESS
            }
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compress a tensor train in-place (f64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `tolerance`: Relative tolerance for truncation (e.g., 1e-12)
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
#[no_mangle]
pub extern "C" fn t4a_tt_f64_compress(
    ptr: *mut t4a_tt_f64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        let options = CompressionOptions {
            method: CompressionMethod::LU,
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            normalize_error: true,
        };
        match tt.inner_mut().compress(&options) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Create a compressed copy of a tensor train (f64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `tolerance`: Relative tolerance for truncation (e.g., 1e-12)
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_f64
/// - Caller owns the returned tensor train and must call t4a_tt_f64_release
#[no_mangle]
pub extern "C" fn t4a_tt_f64_compressed(
    ptr: *const t4a_tt_f64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_tt_f64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let options = CompressionOptions {
            method: CompressionMethod::LU,
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            normalize_error: true,
        };
        match tt.inner().compressed(&options) {
            Ok(compressed) => Box::into_raw(Box::new(t4a_tt_f64::new(compressed))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Arithmetic Operations - Complex64
// ============================================================================

/// Add two tensor trains element-wise (Complex64)
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_add(
    a: *const t4a_tt_c64,
    b: *const t4a_tt_c64,
) -> *mut t4a_tt_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().add(tt_b.inner()) {
            Ok(sum) => Box::into_raw(Box::new(t4a_tt_c64::new(sum))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Subtract two tensor trains element-wise (Complex64)
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_sub(
    a: *const t4a_tt_c64,
    b: *const t4a_tt_c64,
) -> *mut t4a_tt_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().sub(tt_b.inner()) {
            Ok(diff) => Box::into_raw(Box::new(t4a_tt_c64::new(diff))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Negate a tensor train (multiply by -1) (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_negate(ptr: *const t4a_tt_c64) -> *mut t4a_tt_c64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let negated = tt.inner().negate();
        Box::into_raw(Box::new(t4a_tt_c64::new(negated)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Reverse a tensor train (swap left and right) (Complex64)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_reverse(ptr: *const t4a_tt_c64) -> *mut t4a_tt_c64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let reversed = tt.inner().reverse();
        Box::into_raw(Box::new(t4a_tt_c64::new(reversed)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Compute the Hadamard (element-wise) product of two tensor trains (Complex64)
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_hadamard(
    a: *const t4a_tt_c64,
    b: *const t4a_tt_c64,
) -> *mut t4a_tt_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().hadamard(tt_b.inner()) {
            Ok(prod) => Box::into_raw(Box::new(t4a_tt_c64::new(prod))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Compute the inner product (dot product) of two tensor trains (Complex64)
///
/// Returns: sum over all indices i of a[i] * b[i]
///
/// # Safety
/// - `a` and `b` must be valid pointers to t4a_tt_c64
/// - `out_re` and `out_im` must be valid pointers
#[no_mangle]
pub extern "C" fn t4a_tt_c64_dot(
    a: *const t4a_tt_c64,
    b: *const t4a_tt_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if a.is_null() || b.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt_a = unsafe { &*a };
        let tt_b = unsafe { &*b };
        match tt_a.inner().dot(tt_b.inner()) {
            Ok(dot) => {
                unsafe {
                    *out_re = dot.re;
                    *out_im = dot.im;
                };
                T4A_SUCCESS
            }
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compress a tensor train in-place (Complex64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `tolerance`: Relative tolerance for truncation (e.g., 1e-12)
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
#[no_mangle]
pub extern "C" fn t4a_tt_c64_compress(
    ptr: *mut t4a_tt_c64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        let options = CompressionOptions {
            method: CompressionMethod::LU,
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            normalize_error: true,
        };
        match tt.inner_mut().compress(&options) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Create a compressed copy of a tensor train (Complex64)
///
/// # Arguments
/// - `ptr`: Tensor train handle
/// - `tolerance`: Relative tolerance for truncation (e.g., 1e-12)
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
///
/// # Safety
/// - `ptr` must be a valid pointer to a t4a_tt_c64
/// - Caller owns the returned tensor train and must call t4a_tt_c64_release
#[no_mangle]
pub extern "C" fn t4a_tt_c64_compressed(
    ptr: *const t4a_tt_c64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_tt_c64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let options = CompressionOptions {
            method: CompressionMethod::LU,
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            normalize_error: true,
        };
        match tt.inner().compressed(&options) {
            Ok(compressed) => Box::into_raw(Box::new(t4a_tt_c64::new(compressed))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Configuration
// ============================================================================

/// Get the default relative tolerance (rtol) for SVD truncation
///
/// This is the default value used by tensor train compression when no tolerance
/// is explicitly specified. The rtol represents the relative Frobenius error:
///   ||A - A_approx||_F / ||A||_F <= rtol
///
/// # Relation to ITensors.jl cutoff
/// ITensors.jl uses `cutoff` (squared relative error). The conversion is:
///   rtol = sqrt(cutoff)
///   cutoff = rtol^2
///
/// # Returns
/// - The default rtol value (currently 1e-12)
#[no_mangle]
pub extern "C" fn t4a_get_default_svd_rtol() -> libc::c_double {
    default_svd_rtol()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_f64_lifecycle() {
        let dims = [2_usize, 3, 2];
        let tt = t4a_tt_f64_new_zeros(dims.as_ptr(), 3);
        assert!(!tt.is_null());

        // Test is_assigned
        assert_eq!(t4a_tt_f64_is_assigned(tt as *const _), 1);

        // Test clone
        let cloned = t4a_tt_f64_clone(tt as *const _);
        assert!(!cloned.is_null());

        // Clean up
        t4a_tt_f64_release(cloned);
        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_constant() {
        let dims = [2_usize, 2];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 5.0);
        assert!(!tt.is_null());

        // Get len
        let mut len: usize = 0;
        assert_eq!(t4a_tt_f64_len(tt as *const _, &mut len), T4A_SUCCESS);
        assert_eq!(len, 2);

        // Get site_dims
        let mut out_dims = [0_usize; 2];
        assert_eq!(
            t4a_tt_f64_site_dims(tt as *const _, out_dims.as_mut_ptr(), 2),
            T4A_SUCCESS
        );
        assert_eq!(out_dims, [2, 2]);

        // Get sum (should be 5.0 * 2 * 2 = 20.0)
        let mut sum: f64 = 0.0;
        assert_eq!(t4a_tt_f64_sum(tt as *const _, &mut sum), T4A_SUCCESS);
        assert!((sum - 20.0).abs() < 1e-10);

        // Evaluate
        let indices = [0_usize, 1];
        let mut value: f64 = 0.0;
        assert_eq!(
            t4a_tt_f64_evaluate(tt as *const _, indices.as_ptr(), 2, &mut value),
            T4A_SUCCESS
        );
        assert!((value - 5.0).abs() < 1e-10);

        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_scale() {
        let dims = [2_usize, 2];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 1.0);

        // Scale immutable
        let scaled = t4a_tt_f64_scaled(tt as *const _, 3.0);
        assert!(!scaled.is_null());

        let mut sum: f64 = 0.0;
        t4a_tt_f64_sum(scaled as *const _, &mut sum);
        assert!((sum - 12.0).abs() < 1e-10);

        // Original unchanged
        t4a_tt_f64_sum(tt as *const _, &mut sum);
        assert!((sum - 4.0).abs() < 1e-10);

        // Scale in place
        t4a_tt_f64_scale_inplace(tt, 2.0);
        t4a_tt_f64_sum(tt as *const _, &mut sum);
        assert!((sum - 8.0).abs() < 1e-10);

        t4a_tt_f64_release(scaled);
        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_fulltensor() {
        let dims = [2_usize, 3];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 5.0);

        // Query length
        let mut len: usize = 0;
        assert_eq!(
            t4a_tt_f64_fulltensor(tt as *const _, ptr::null_mut(), 0, &mut len),
            T4A_SUCCESS
        );
        assert_eq!(len, 6);

        // Get data
        let mut data = [0.0; 6];
        assert_eq!(
            t4a_tt_f64_fulltensor(tt as *const _, data.as_mut_ptr(), 6, &mut len),
            T4A_SUCCESS
        );

        // All elements should be 5.0
        for val in &data {
            assert!((val - 5.0).abs() < 1e-10);
        }

        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_c64_lifecycle() {
        let dims = [2_usize, 3, 2];
        let tt = t4a_tt_c64_new_zeros(dims.as_ptr(), 3);
        assert!(!tt.is_null());

        assert_eq!(t4a_tt_c64_is_assigned(tt as *const _), 1);

        let cloned = t4a_tt_c64_clone(tt as *const _);
        assert!(!cloned.is_null());

        t4a_tt_c64_release(cloned);
        t4a_tt_c64_release(tt);
    }

    #[test]
    fn test_tt_c64_constant() {
        let dims = [2_usize, 2];
        let tt = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 3.0, 4.0);
        assert!(!tt.is_null());

        // Evaluate
        let indices = [0_usize, 1];
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        assert_eq!(
            t4a_tt_c64_evaluate(tt as *const _, indices.as_ptr(), 2, &mut re, &mut im),
            T4A_SUCCESS
        );
        assert!((re - 3.0).abs() < 1e-10);
        assert!((im - 4.0).abs() < 1e-10);

        t4a_tt_c64_release(tt);
    }

    #[test]
    fn test_tt_f64_norm() {
        let dims = [2_usize, 3];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 2.0);

        let mut norm: f64 = 0.0;
        assert_eq!(t4a_tt_f64_norm(tt as *const _, &mut norm), T4A_SUCCESS);
        // norm = sqrt(sum of squares) = sqrt(6 * 4) = sqrt(24)
        assert!((norm - 24.0_f64.sqrt()).abs() < 1e-10);

        let mut log_norm: f64 = 0.0;
        assert_eq!(
            t4a_tt_f64_log_norm(tt as *const _, &mut log_norm),
            T4A_SUCCESS
        );
        assert!((log_norm - norm.ln()).abs() < 1e-10);

        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_add() {
        let dims = [2_usize, 3];
        let tt1 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 1.0);
        let tt2 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 2.0);

        let result = t4a_tt_f64_add(tt1 as *const _, tt2 as *const _);
        assert!(!result.is_null());

        // Sum should be (1.0 + 2.0) * 2 * 3 = 18.0
        let mut sum: f64 = 0.0;
        t4a_tt_f64_sum(result as *const _, &mut sum);
        assert!((sum - 18.0).abs() < 1e-10);

        t4a_tt_f64_release(result);
        t4a_tt_f64_release(tt2);
        t4a_tt_f64_release(tt1);
    }

    #[test]
    fn test_tt_f64_sub() {
        let dims = [2_usize, 3];
        let tt1 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 5.0);
        let tt2 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 2.0);

        let result = t4a_tt_f64_sub(tt1 as *const _, tt2 as *const _);
        assert!(!result.is_null());

        // Sum should be (5.0 - 2.0) * 2 * 3 = 18.0
        let mut sum: f64 = 0.0;
        t4a_tt_f64_sum(result as *const _, &mut sum);
        assert!((sum - 18.0).abs() < 1e-10);

        t4a_tt_f64_release(result);
        t4a_tt_f64_release(tt2);
        t4a_tt_f64_release(tt1);
    }

    #[test]
    fn test_tt_f64_negate() {
        let dims = [2_usize, 2];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 3.0);

        let neg = t4a_tt_f64_negate(tt as *const _);
        assert!(!neg.is_null());

        let mut sum1: f64 = 0.0;
        let mut sum2: f64 = 0.0;
        t4a_tt_f64_sum(tt as *const _, &mut sum1);
        t4a_tt_f64_sum(neg as *const _, &mut sum2);
        assert!((sum1 + sum2).abs() < 1e-10);

        t4a_tt_f64_release(neg);
        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_reverse() {
        let dims = [2_usize, 3];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 5.0);

        let rev = t4a_tt_f64_reverse(tt as *const _);
        assert!(!rev.is_null());

        // Reversed site dims should be [3, 2]
        let mut out_dims = [0_usize; 2];
        t4a_tt_f64_site_dims(rev as *const _, out_dims.as_mut_ptr(), 2);
        assert_eq!(out_dims, [3, 2]);

        // Sum should be preserved
        let mut sum1: f64 = 0.0;
        let mut sum2: f64 = 0.0;
        t4a_tt_f64_sum(tt as *const _, &mut sum1);
        t4a_tt_f64_sum(rev as *const _, &mut sum2);
        assert!((sum1 - sum2).abs() < 1e-10);

        t4a_tt_f64_release(rev);
        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_hadamard() {
        let dims = [2_usize, 3];
        let tt1 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 2.0);
        let tt2 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 3.0);

        let result = t4a_tt_f64_hadamard(tt1 as *const _, tt2 as *const _);
        assert!(!result.is_null());

        // Sum should be 2.0 * 3.0 * 2 * 3 = 36.0
        let mut sum: f64 = 0.0;
        t4a_tt_f64_sum(result as *const _, &mut sum);
        assert!((sum - 36.0).abs() < 1e-10);

        t4a_tt_f64_release(result);
        t4a_tt_f64_release(tt2);
        t4a_tt_f64_release(tt1);
    }

    #[test]
    fn test_tt_f64_dot() {
        let dims = [2_usize, 3];
        let tt1 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 2.0);
        let tt2 = t4a_tt_f64_new_constant(dims.as_ptr(), 2, 3.0);

        let mut dot: f64 = 0.0;
        let status = t4a_tt_f64_dot(tt1 as *const _, tt2 as *const _, &mut dot);
        assert_eq!(status, T4A_SUCCESS);

        // dot = 2.0 * 3.0 * 2 * 3 = 36.0
        assert!((dot - 36.0).abs() < 1e-10);

        t4a_tt_f64_release(tt2);
        t4a_tt_f64_release(tt1);
    }

    #[test]
    fn test_tt_f64_compress() {
        let dims = [2_usize, 3, 2];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 3, 1.0);

        let mut sum_before: f64 = 0.0;
        t4a_tt_f64_sum(tt as *const _, &mut sum_before);

        // Compress with default settings
        let status = t4a_tt_f64_compress(tt, 1e-12, 0);
        assert_eq!(status, T4A_SUCCESS);

        let mut sum_after: f64 = 0.0;
        t4a_tt_f64_sum(tt as *const _, &mut sum_after);
        assert!((sum_before - sum_after).abs() < 1e-10);

        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_f64_compressed() {
        let dims = [2_usize, 3, 2];
        let tt = t4a_tt_f64_new_constant(dims.as_ptr(), 3, 1.0);

        let compressed = t4a_tt_f64_compressed(tt as *const _, 1e-12, 0);
        assert!(!compressed.is_null());

        let mut sum1: f64 = 0.0;
        let mut sum2: f64 = 0.0;
        t4a_tt_f64_sum(tt as *const _, &mut sum1);
        t4a_tt_f64_sum(compressed as *const _, &mut sum2);
        assert!((sum1 - sum2).abs() < 1e-10);

        t4a_tt_f64_release(compressed);
        t4a_tt_f64_release(tt);
    }

    #[test]
    fn test_tt_c64_add() {
        let dims = [2_usize, 2];
        let tt1 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 1.0, 0.0);
        let tt2 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 0.0, 1.0);

        let result = t4a_tt_c64_add(tt1 as *const _, tt2 as *const _);
        assert!(!result.is_null());

        // Sum should be (1+i) * 4 = 4 + 4i
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        t4a_tt_c64_sum(result as *const _, &mut re, &mut im);
        assert!((re - 4.0).abs() < 1e-10);
        assert!((im - 4.0).abs() < 1e-10);

        t4a_tt_c64_release(result);
        t4a_tt_c64_release(tt2);
        t4a_tt_c64_release(tt1);
    }

    #[test]
    fn test_tt_c64_hadamard() {
        let dims = [2_usize, 2];
        let tt1 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 2.0, 0.0);
        let tt2 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 0.0, 3.0);

        let result = t4a_tt_c64_hadamard(tt1 as *const _, tt2 as *const _);
        assert!(!result.is_null());

        // Each element is 2 * 3i = 6i, sum = 6i * 4 = 24i
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        t4a_tt_c64_sum(result as *const _, &mut re, &mut im);
        assert!((re).abs() < 1e-10);
        assert!((im - 24.0).abs() < 1e-10);

        t4a_tt_c64_release(result);
        t4a_tt_c64_release(tt2);
        t4a_tt_c64_release(tt1);
    }

    #[test]
    fn test_tt_c64_dot() {
        let dims = [2_usize, 2];
        let tt1 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 2.0, 0.0);
        let tt2 = t4a_tt_c64_new_constant(dims.as_ptr(), 2, 3.0, 0.0);

        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        let status = t4a_tt_c64_dot(tt1 as *const _, tt2 as *const _, &mut re, &mut im);
        assert_eq!(status, T4A_SUCCESS);

        // dot = 2 * 3 * 4 = 24
        assert!((re - 24.0).abs() < 1e-10);
        assert!((im).abs() < 1e-10);

        t4a_tt_c64_release(tt2);
        t4a_tt_c64_release(tt1);
    }
}
