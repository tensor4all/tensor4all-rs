//! C API for tensor4all-simplett TensorTrain
//!
//! This provides a simpler TensorTrain interface designed for TCI operations.
//! The tensors are stored as flat arrays with explicit dimensions.

use crate::{StatusCode, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
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

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_simplett::{tensor3_from_data, TensorTrain};

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
}
