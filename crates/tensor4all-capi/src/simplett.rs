//! C API for tensor4all-simplett TensorTrain
//!
//! This provides a simpler TensorTrain interface designed for TCI operations.
//! The tensors are stored as flat arrays with explicit dimensions.

use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

// ============================================================================
// Opaque handle type
// ============================================================================

/// Opaque handle for a SimpleTT `TensorTrain<f64>`
#[repr(C)]
pub struct t4a_simplett_f64 {
    _private: *const c_void,
}

#[allow(dead_code)]
impl t4a_simplett_f64 {
    pub(crate) fn new(tt: TensorTrain<f64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tt)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &TensorTrain<f64> {
        unsafe { &*(self._private as *const TensorTrain<f64>) }
    }

    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut TensorTrain<f64> {
        unsafe { &mut *(self._private as *mut TensorTrain<f64>) }
    }

    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> TensorTrain<f64> {
        let ptr = self._private as *mut TensorTrain<f64>;
        std::mem::forget(self);
        unsafe { *Box::from_raw(ptr) }
    }
}

impl Drop for t4a_simplett_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut TensorTrain<f64>);
            }
        }
    }
}

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

#[cfg(test)]
mod tests;
