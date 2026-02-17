//! C API for QuanticsTCI (Quantics Tensor Cross Interpolation)
//!
//! This provides C-compatible interface for interpolating functions using
//! quantics tensor cross interpolation. It wraps `tensor4all-quanticstci`
//! which combines TCI with quantics grid representations.

use crate::types::{t4a_qgrid_disc, t4a_qtci_f64, t4a_unfolding_scheme};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_quanticstci::{
    quanticscrossinterpolate, quanticscrossinterpolate_discrete, QtciOptions,
};

// ============================================================================
// Callback types
// ============================================================================

/// Callback function type for evaluating a function at continuous coordinates.
///
/// # Arguments
/// * `coords` - Array of f64 coordinates (one per dimension)
/// * `ndims` - Number of dimensions
/// * `result` - Pointer to store the result value
/// * `user_data` - User data passed to the callback
///
/// # Returns
/// * 0 on success
/// * Non-zero on error
pub type QtciEvalCallbackF64 = extern "C" fn(
    coords: *const libc::c_double,
    ndims: libc::size_t,
    result: *mut libc::c_double,
    user_data: *mut c_void,
) -> i32;

/// Callback function type for evaluating a function at discrete (integer) indices.
///
/// # Arguments
/// * `indices` - Array of i64 indices (one per dimension, 1-indexed)
/// * `ndims` - Number of dimensions
/// * `result` - Pointer to store the result value
/// * `user_data` - User data passed to the callback
///
/// # Returns
/// * 0 on success
/// * Non-zero on error
pub type QtciEvalCallbackI64 = extern "C" fn(
    indices: *const i64,
    ndims: libc::size_t,
    result: *mut libc::c_double,
    user_data: *mut c_void,
) -> i32;

// ============================================================================
// Lifecycle functions
// ============================================================================

/// Release a QuanticsTCI handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_release(ptr: *mut t4a_qtci_f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ============================================================================
// High-level interpolation functions
// ============================================================================

/// Perform quantics cross interpolation on a continuous domain with a DiscretizedGrid.
///
/// # Arguments
/// * `grid` - Discretized grid describing the function domain
/// * `eval_fn` - Callback for function evaluation (takes f64 coordinates)
/// * `user_data` - User data passed to callback
/// * `tolerance` - Convergence tolerance
/// * `max_bonddim` - Maximum bond dimension (0 = unlimited)
/// * `max_iter` - Maximum number of iterations
/// * `unfoldingscheme` - Unfolding scheme for quantics representation
/// * `out_qtci` - Output: QTCI handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_f64(
    grid: *const t4a_qgrid_disc,
    eval_fn: QtciEvalCallbackF64,
    user_data: *mut c_void,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    out_qtci: *mut *mut t4a_qtci_f64,
) -> StatusCode {
    if grid.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let grid_ref = unsafe { &*grid };
        let inner_grid = grid_ref.inner();

        // Create evaluation function wrapper
        let f = move |coords: &[f64]| -> f64 {
            let mut result: f64 = 0.0;
            let status = eval_fn(coords.as_ptr(), coords.len(), &mut result, user_data);
            if status != 0 {
                f64::NAN
            } else {
                result
            }
        };

        // Build options
        // Note: The unfolding scheme is already encoded in the grid's structure,
        // so we don't need to pass it separately to the options.
        let options = QtciOptions::default()
            .with_tolerance(tolerance)
            .with_maxiter(max_iter);

        let options = if max_bonddim > 0 {
            options.with_maxbonddim(max_bonddim)
        } else {
            options
        };

        // Run interpolation
        match quanticscrossinterpolate(inner_grid, f, None, options) {
            Ok((qtci, _ranks, _errors)) => {
                unsafe { *out_qtci = Box::into_raw(Box::new(t4a_qtci_f64::new(qtci))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Perform quantics cross interpolation on a discrete integer domain.
///
/// # Arguments
/// * `sizes` - Array of grid sizes per dimension (must be powers of 2)
/// * `ndims` - Number of dimensions
/// * `eval_fn` - Callback for function evaluation (takes i64 indices, 1-indexed)
/// * `user_data` - User data passed to callback
/// * `tolerance` - Convergence tolerance
/// * `max_bonddim` - Maximum bond dimension (0 = unlimited)
/// * `max_iter` - Maximum number of iterations
/// * `unfoldingscheme` - Unfolding scheme for quantics representation
/// * `out_qtci` - Output: QTCI handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_discrete_f64(
    sizes: *const libc::size_t,
    ndims: libc::size_t,
    eval_fn: QtciEvalCallbackI64,
    user_data: *mut c_void,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    unfoldingscheme: t4a_unfolding_scheme,
    out_qtci: *mut *mut t4a_qtci_f64,
) -> StatusCode {
    if sizes.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }
    if ndims == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let size_slice: Vec<usize> = unsafe { std::slice::from_raw_parts(sizes, ndims) }.to_vec();

        // Create evaluation function wrapper
        let f = move |indices: &[i64]| -> f64 {
            let mut result: f64 = 0.0;
            let status = eval_fn(indices.as_ptr(), indices.len(), &mut result, user_data);
            if status != 0 {
                f64::NAN
            } else {
                result
            }
        };

        // Build options
        let scheme: quanticsgrids::UnfoldingScheme = unfoldingscheme.into();
        let options = QtciOptions::default()
            .with_tolerance(tolerance)
            .with_maxiter(max_iter)
            .with_unfoldingscheme(scheme);

        let options = if max_bonddim > 0 {
            options.with_maxbonddim(max_bonddim)
        } else {
            options
        };

        // Run interpolation
        match quanticscrossinterpolate_discrete(&size_slice, f, None, options) {
            Ok((qtci, _ranks, _errors)) => {
                unsafe { *out_qtci = Box::into_raw(Box::new(t4a_qtci_f64::new(qtci))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the maximum bond dimension (rank).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_rank(
    ptr: *const t4a_qtci_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        unsafe { *out_rank = qtci.inner().rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the link (bond) dimensions.
///
/// # Arguments
/// * `ptr` - QTCI handle
/// * `out_dims` - Output buffer for link dimensions (length = n_sites - 1)
/// * `buf_len` - Buffer length
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_link_dims(
    ptr: *const t4a_qtci_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        let dims = qtci.inner().link_dims();
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

// ============================================================================
// Operations
// ============================================================================

/// Evaluate the QTCI at grid indices.
///
/// # Arguments
/// * `ptr` - QTCI handle
/// * `indices` - Grid indices (1-indexed, one per original dimension)
/// * `n_indices` - Number of indices (= number of dimensions)
/// * `out_value` - Output value
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_evaluate(
    ptr: *const t4a_qtci_f64,
    indices: *const i64,
    n_indices: libc::size_t,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        let idx: Vec<i64> = unsafe { std::slice::from_raw_parts(indices, n_indices) }.to_vec();

        match qtci.inner().evaluate(&idx) {
            Ok(val) => {
                unsafe { *out_value = val };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the factorized sum over all grid points.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_sum(
    ptr: *const t4a_qtci_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        match qtci.inner().sum() {
            Ok(val) => {
                unsafe { *out_value = val };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the integral over the continuous domain.
///
/// This is the sum multiplied by the grid step sizes.
/// Only meaningful for QTCI constructed with a DiscretizedGrid.
/// For discrete grids, this returns the plain sum.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_integral(
    ptr: *const t4a_qtci_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        match qtci.inner().integral() {
            Ok(val) => {
                unsafe { *out_value = val };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Convert the QTCI to a SimpleTT tensor train.
///
/// # Arguments
/// * `ptr` - QTCI handle
///
/// # Returns
/// A new SimpleTT handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_to_tensor_train(
    ptr: *const t4a_qtci_f64,
) -> *mut crate::simplett::t4a_simplett_f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        match qtci.inner().tensor_train() {
            Ok(tt) => Box::into_raw(Box::new(crate::simplett::t4a_simplett_f64::new(tt))),
            Err(e) => crate::err_null(e),
        }
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Discrete callback: f(i, j) = i + j (1-indexed)
    extern "C" fn discrete_sum_callback(
        indices: *const i64,
        ndims: libc::size_t,
        result: *mut f64,
        _user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            let sum: i64 = (0..ndims).map(|i| *indices.add(i)).sum();
            *result = sum as f64;
        }
        0
    }

    // Continuous callback: f(x, y) = x + y
    extern "C" fn continuous_sum_callback(
        coords: *const libc::c_double,
        ndims: libc::size_t,
        result: *mut libc::c_double,
        _user_data: *mut c_void,
    ) -> i32 {
        unsafe {
            let sum: f64 = (0..ndims).map(|i| *coords.add(i)).sum();
            *result = sum;
        }
        0
    }

    #[test]
    fn test_discrete_qtci() {
        let sizes: [libc::size_t; 2] = [4, 4];
        let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

        let status = t4a_quanticscrossinterpolate_discrete_f64(
            sizes.as_ptr(),
            2,
            discrete_sum_callback,
            std::ptr::null_mut(),
            1e-10,
            0, // unlimited
            100,
            t4a_unfolding_scheme::Fused,
            &mut qtci,
        );

        assert_eq!(status, T4A_SUCCESS);
        assert!(!qtci.is_null());

        // Check rank
        let mut rank: libc::size_t = 0;
        let status = t4a_qtci_f64_rank(qtci, &mut rank);
        assert_eq!(status, T4A_SUCCESS);
        assert!(rank > 0);

        // Evaluate at (3, 4) -> should be 7
        let indices: [i64; 2] = [3, 4];
        let mut value: f64 = 0.0;
        let status = t4a_qtci_f64_evaluate(qtci, indices.as_ptr(), 2, &mut value);
        assert_eq!(status, T4A_SUCCESS);
        assert!((value - 7.0).abs() < 1e-6, "Expected 7.0, got {}", value);

        // Evaluate at (1, 1) -> should be 2
        let indices: [i64; 2] = [1, 1];
        let status = t4a_qtci_f64_evaluate(qtci, indices.as_ptr(), 2, &mut value);
        assert_eq!(status, T4A_SUCCESS);
        assert!((value - 2.0).abs() < 1e-6, "Expected 2.0, got {}", value);

        // Sum should return a finite positive value
        let mut sum: f64 = 0.0;
        let status = t4a_qtci_f64_sum(qtci, &mut sum);
        assert_eq!(status, T4A_SUCCESS);
        assert!(sum.is_finite());
        assert!(sum > 0.0);

        t4a_qtci_f64_release(qtci);
    }

    #[test]
    fn test_continuous_qtci() {
        // Create a 1D discretized grid via the C API
        let rs: [libc::size_t; 1] = [4]; // 2^4 = 16 grid points
        let lower: [f64; 1] = [0.0];
        let upper: [f64; 1] = [1.0];
        let mut grid: *mut t4a_qgrid_disc = std::ptr::null_mut();

        let status = crate::quanticsgrids::t4a_qgrid_disc_new(
            1,
            rs.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            t4a_unfolding_scheme::Fused,
            &mut grid,
        );
        assert_eq!(status, T4A_SUCCESS);

        let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
        let status = t4a_quanticscrossinterpolate_f64(
            grid,
            continuous_sum_callback,
            std::ptr::null_mut(),
            1e-10,
            0,
            100,
            &mut qtci,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!qtci.is_null());

        // Check rank
        let mut rank: libc::size_t = 0;
        t4a_qtci_f64_rank(qtci, &mut rank);
        assert!(rank > 0);

        // Integral of f(x) = x from 0 to 1 should be ~0.5
        let mut integral: f64 = 0.0;
        let status = t4a_qtci_f64_integral(qtci, &mut integral);
        assert_eq!(status, T4A_SUCCESS);
        // Discretization introduces some error
        assert!(
            (integral - 0.5).abs() < 0.1,
            "Expected ~0.5, got {}",
            integral
        );

        t4a_qtci_f64_release(qtci);
        crate::quanticsgrids::t4a_qgrid_disc_release(grid);
    }

    #[test]
    fn test_qtci_to_tensor_train() {
        let sizes: [libc::size_t; 2] = [4, 4];
        let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

        let status = t4a_quanticscrossinterpolate_discrete_f64(
            sizes.as_ptr(),
            2,
            discrete_sum_callback,
            std::ptr::null_mut(),
            1e-10,
            0,
            100,
            t4a_unfolding_scheme::Fused,
            &mut qtci,
        );
        assert_eq!(status, T4A_SUCCESS);

        let tt = t4a_qtci_f64_to_tensor_train(qtci);
        assert!(!tt.is_null());

        // Check that the tensor train has correct number of sites
        let mut len: libc::size_t = 0;
        crate::simplett::t4a_simplett_f64_len(tt, &mut len);
        assert!(len > 0);

        crate::simplett::t4a_simplett_f64_release(tt);
        t4a_qtci_f64_release(qtci);
    }

    #[test]
    fn test_null_pointer_guards() {
        let mut rank: libc::size_t = 0;
        assert_eq!(
            t4a_qtci_f64_rank(std::ptr::null(), &mut rank),
            T4A_NULL_POINTER
        );

        let mut value: f64 = 0.0;
        assert_eq!(
            t4a_qtci_f64_sum(std::ptr::null(), &mut value),
            T4A_NULL_POINTER
        );

        assert_eq!(
            t4a_qtci_f64_integral(std::ptr::null(), &mut value),
            T4A_NULL_POINTER
        );

        let indices: [i64; 2] = [1, 1];
        assert_eq!(
            t4a_qtci_f64_evaluate(std::ptr::null(), indices.as_ptr(), 2, &mut value),
            T4A_NULL_POINTER
        );

        let tt = t4a_qtci_f64_to_tensor_train(std::ptr::null());
        assert!(tt.is_null());

        // Release null should not crash
        t4a_qtci_f64_release(std::ptr::null_mut());

        // Constructor null checks
        let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
        assert_eq!(
            t4a_quanticscrossinterpolate_discrete_f64(
                std::ptr::null(),
                2,
                discrete_sum_callback,
                std::ptr::null_mut(),
                1e-10,
                0,
                100,
                t4a_unfolding_scheme::Fused,
                &mut qtci,
            ),
            T4A_NULL_POINTER
        );
    }
}
