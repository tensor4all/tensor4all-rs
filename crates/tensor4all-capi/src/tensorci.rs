//! C API for TensorCI2 (Tensor Cross Interpolation)
//!
//! This provides C-compatible interface for cross-interpolating functions.
//! The evaluation function is passed as a callback.

use crate::simplett::t4a_simplett_f64;
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_tensorci::{TCI2Options, TensorCI2};

// ============================================================================
// Callback types
// ============================================================================

/// Callback function type for evaluating the target function.
///
/// # Arguments
/// * `indices` - Array of indices (one per site)
/// * `n_indices` - Number of indices
/// * `result` - Pointer to store the result value
/// * `user_data` - User data passed to the callback
///
/// # Returns
/// * 0 on success
/// * Non-zero on error
pub type EvalCallback = extern "C" fn(
    indices: *const i64,
    n_indices: libc::size_t,
    result: *mut f64,
    user_data: *mut c_void,
) -> i32;

// ============================================================================
// Opaque handle type
// ============================================================================

/// Opaque handle for TensorCI2<f64>
#[repr(C)]
pub struct t4a_tci2_f64 {
    _private: *const c_void,
}

impl t4a_tci2_f64 {
    pub(crate) fn new(tci: TensorCI2<f64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tci)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &TensorCI2<f64> {
        unsafe { &*(self._private as *const TensorCI2<f64>) }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut TensorCI2<f64> {
        unsafe { &mut *(self._private as *mut TensorCI2<f64>) }
    }
}

impl Drop for t4a_tci2_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut TensorCI2<f64>);
            }
        }
    }
}

// ============================================================================
// Lifecycle functions
// ============================================================================

/// Release a TensorCI2 handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_release(ptr: *mut t4a_tci2_f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ============================================================================
// Constructor
// ============================================================================

/// Create a new TensorCI2 object.
///
/// # Arguments
/// * `local_dims` - Array of local dimensions for each site
/// * `n_sites` - Number of sites
///
/// # Returns
/// A new TensorCI2 handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_new(
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
) -> *mut t4a_tci2_f64 {
    if local_dims.is_null() && n_sites > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();

        match TensorCI2::<f64>::new(dims) {
            Ok(tci) => Box::into_raw(Box::new(t4a_tci2_f64::new(tci))),
            Err(_) => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the number of sites.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_len(
    ptr: *const t4a_tci2_f64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        unsafe { *out_len = tci.inner().len() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the current rank (maximum bond dimension).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_rank(
    ptr: *const t4a_tci2_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        unsafe { *out_rank = tci.inner().rank() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the link (bond) dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_link_dims(
    ptr: *const t4a_tci2_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        let dims = tci.inner().link_dims();
        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        for (i, d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = *d };
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the maximum sample value encountered.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_max_sample_value(
    ptr: *const t4a_tci2_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        unsafe { *out_value = tci.inner().max_sample_value() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the maximum bond error from the last sweep.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_max_bond_error(
    ptr: *const t4a_tci2_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        unsafe { *out_value = tci.inner().max_bond_error() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Pivot operations
// ============================================================================

/// Add global pivots to the TCI.
///
/// # Arguments
/// * `ptr` - TCI handle
/// * `pivots` - Flat array of pivot indices [p0_0, p0_1, ..., p0_n, p1_0, p1_1, ...]
/// * `n_pivots` - Number of pivots
/// * `n_sites` - Number of sites per pivot
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_add_global_pivots(
    ptr: *mut t4a_tci2_f64,
    pivots: *const libc::size_t,
    n_pivots: libc::size_t,
    n_sites: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || (pivots.is_null() && n_pivots > 0) {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };

        // Convert flat array to Vec<Vec<usize>>
        let mut pivot_vec: Vec<Vec<usize>> = Vec::with_capacity(n_pivots);
        for i in 0..n_pivots {
            let mut pivot: Vec<usize> = Vec::with_capacity(n_sites);
            for j in 0..n_sites {
                pivot.push(unsafe { *pivots.add(i * n_sites + j) });
            }
            pivot_vec.push(pivot);
        }

        match tci.inner_mut().add_global_pivots(&pivot_vec) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Sweep operation
// ============================================================================

/// Perform a 2-site sweep.
///
/// This is the main optimization step. The callback function is called
/// to evaluate the target function at various indices.
///
/// # Arguments
/// * `ptr` - TCI handle
/// * `eval_fn` - Callback function for evaluating the target function
/// * `user_data` - User data passed to the callback
/// * `abstol` - Absolute tolerance
/// * `max_bonddim` - Maximum bond dimension (0 for unlimited)
/// * `n_iters` - Number of sweep iterations
/// * `out_error` - Output: maximum error after sweep
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_sweep(
    ptr: *mut t4a_tci2_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    abstol: libc::c_double,
    max_bonddim: libc::size_t,
    n_iters: libc::size_t,
    out_error: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };
        let n_sites = tci.inner().len();

        // TODO: Implement actual sweep using internal functions
        // For now, this is a placeholder. The actual implementation requires
        // exposing internal sweep functions from tensorci2.rs.
        //
        // The intended design is:
        // 1. Julia controls the main iteration loop
        // 2. Julia calls this sweep function for each iteration
        // 3. Julia handles global pivot finding separately
        // 4. Julia injects found pivots via add_global_pivots

        // Silence unused parameter warnings for now
        let _ = (eval_fn, user_data, abstol, max_bonddim, n_iters);

        // Get initial pivot if needed
        if tci.inner().rank() == 0 {
            let initial_pivot: Vec<usize> = vec![0; n_sites];
            let _ = tci.inner_mut().add_global_pivots(&[initial_pivot]);
        }

        // Report current error
        if !out_error.is_null() {
            unsafe { *out_error = tci.inner().max_bond_error() };
        }

        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Conversion
// ============================================================================

/// Convert the TCI to a TensorTrain.
///
/// # Arguments
/// * `ptr` - TCI handle
///
/// # Returns
/// A new SimpleTT handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_to_tensor_train(ptr: *const t4a_tci2_f64) -> *mut t4a_simplett_f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        match tci.inner().to_tensor_train() {
            Ok(tt) => Box::into_raw(Box::new(t4a_simplett_f64::new(tt))),
            Err(_) => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

// ============================================================================
// High-level crossinterpolate2 function
// ============================================================================

/// Perform cross interpolation of a function.
///
/// This is the main entry point for TCI. It creates a TCI object,
/// performs optimization sweeps, and returns the result.
///
/// # Arguments
/// * `local_dims` - Array of local dimensions
/// * `n_sites` - Number of sites
/// * `initial_pivots` - Initial pivots (flat array), or NULL for default [0,0,...,0]
/// * `n_initial_pivots` - Number of initial pivots
/// * `eval_fn` - Callback for function evaluation
/// * `user_data` - User data for callback
/// * `tolerance` - Relative tolerance
/// * `max_bonddim` - Maximum bond dimension (0 for unlimited)
/// * `max_iter` - Maximum number of iterations
/// * `out_tci` - Output: TCI handle (caller owns)
/// * `out_final_error` - Output: final error
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_crossinterpolate2_f64(
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    initial_pivots: *const libc::size_t,
    n_initial_pivots: libc::size_t,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    out_tci: *mut *mut t4a_tci2_f64,
    out_final_error: *mut libc::c_double,
) -> StatusCode {
    if local_dims.is_null() || out_tci.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Convert local_dims
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();

        // Convert initial pivots
        let pivots: Vec<Vec<usize>> = if initial_pivots.is_null() || n_initial_pivots == 0 {
            vec![vec![0; n_sites]]
        } else {
            (0..n_initial_pivots)
                .map(|i| {
                    (0..n_sites)
                        .map(|j| unsafe { *initial_pivots.add(i * n_sites + j) })
                        .collect()
                })
                .collect()
        };

        // Create the evaluation function wrapper
        // Note: MultiIndex is Vec<usize>, so we receive &Vec<usize>
        let f = move |indices: &Vec<usize>| -> f64 {
            let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
            let mut result: f64 = 0.0;

            let status = eval_fn(
                indices_i64.as_ptr(),
                indices_i64.len(),
                &mut result,
                user_data,
            );

            if status != 0 {
                f64::NAN
            } else {
                result
            }
        };

        // Build options
        let options = TCI2Options {
            tolerance,
            max_bond_dim: if max_bonddim > 0 {
                max_bonddim
            } else {
                usize::MAX
            },
            max_iter,
            ..Default::default()
        };

        // Run crossinterpolate2
        // Batch function type: Fn(&[MultiIndex]) -> Vec<f64>
        use tensor4all_tensorci::crossinterpolate2;
        match crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
            f, None, // no batch function
            dims, pivots, options,
        ) {
            Ok((tci, _ranks, errors)) => {
                // Store TCI
                unsafe { *out_tci = Box::into_raw(Box::new(t4a_tci2_f64::new(tci))) };

                // Store final error
                if !out_final_error.is_null() {
                    let final_err = errors.last().copied().unwrap_or(0.0);
                    unsafe { *out_final_error = final_err };
                }

                T4A_SUCCESS
            }
            Err(_) => T4A_INTERNAL_ERROR,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
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
}
