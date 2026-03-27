//! C API for TensorCI2 (Tensor Cross Interpolation)
//!
//! This provides C-compatible interface for cross-interpolating functions.
//! The evaluation function is passed as a callback.

use crate::simplett::t4a_simplett_f64;
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_simplett::AbstractTensorTrain;
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

/// Opaque handle for `TensorCI2<f64>`
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
            Err(e) => crate::err_null(e),
        }
    }));

    crate::unwrap_catch_ptr(result)
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

    crate::unwrap_catch(result)
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

    crate::unwrap_catch(result)
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

    crate::unwrap_catch(result)
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

    crate::unwrap_catch(result)
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

    crate::unwrap_catch(result)
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
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Low-level sweep operations
// ============================================================================

/// Helper: create a Rust closure from an EvalCallback.
fn make_eval_closure(eval_fn: EvalCallback, user_data: *mut c_void) -> impl Fn(&Vec<usize>) -> f64 {
    move |indices: &Vec<usize>| -> f64 {
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
    }
}

/// Perform one 2-site sweep (forward or backward).
///
/// Rust manages Iset_history internally for non-strictly-nested mode.
///
/// # Arguments
/// * `ptr` - TCI handle (mutable)
/// * `eval_fn` - Callback for function evaluation
/// * `user_data` - User data for callback
/// * `forward` - 1 for forward sweep, 0 for backward
/// * `tolerance` - Tolerance for LU truncation
/// * `max_bonddim` - Maximum bond dimension (0 for unlimited)
/// * `pivot_search` - 0=Full, 1=Rook
/// * `strictly_nested` - 1=strictly nested, 0=use history
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_sweep2site(
    ptr: *mut t4a_tci2_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    forward: libc::c_int,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    _pivot_search: libc::c_int,
    _strictly_nested: libc::c_int,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };
        let f = make_eval_closure(eval_fn, user_data);
        let max_bd = if max_bonddim == 0 {
            usize::MAX
        } else {
            max_bonddim
        };

        let options = TCI2Options {
            tolerance,
            max_bond_dim: max_bd,
            ..Default::default()
        };

        match tci
            .inner_mut()
            .sweep2site::<_, fn(&[Vec<usize>]) -> Vec<f64>>(&f, &None, forward != 0, &options)
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Perform one 1-site sweep for cleanup / canonicalization.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_sweep1site(
    ptr: *mut t4a_tci2_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    forward: libc::c_int,
    rel_tol: libc::c_double,
    abs_tol: libc::c_double,
    max_bonddim: libc::size_t,
    update_tensors: libc::c_int,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };
        let f = make_eval_closure(eval_fn, user_data);
        let max_bd = if max_bonddim == 0 {
            usize::MAX
        } else {
            max_bonddim
        };

        match tci.inner_mut().sweep1site(
            &f,
            forward != 0,
            rel_tol,
            abs_tol,
            max_bd,
            update_tensors != 0,
        ) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Fill all site tensors from function evaluations.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_fill_site_tensors(
    ptr: *mut t4a_tci2_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };
        let f = make_eval_closure(eval_fn, user_data);
        match tci.inner_mut().fill_site_tensors(&f) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Make TCI canonical (3 one-site sweeps).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_make_canonical(
    ptr: *mut t4a_tci2_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    rel_tol: libc::c_double,
    abs_tol: libc::c_double,
    max_bonddim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &mut *ptr };
        let f = make_eval_closure(eval_fn, user_data);
        let max_bd = if max_bonddim == 0 {
            usize::MAX
        } else {
            max_bonddim
        };
        match tci.inner_mut().make_canonical(&f, rel_tol, abs_tol, max_bd) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Get pivot error (max bond error from last sweep).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_pivot_error(
    ptr: *const t4a_tci2_f64,
    out_error: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_error.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        unsafe { *out_error = tci.inner().max_bond_error() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// I-set / J-set access
// ============================================================================

/// Query I-set size at site p.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_i_set_size(
    ptr: *const t4a_tci2_f64,
    site: libc::size_t,
    out_n_indices: *mut libc::size_t,
    out_index_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_n_indices.is_null() || out_index_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        let i_set = tci.inner().i_set(site);
        let n = i_set.len();
        let idx_len = if n > 0 { i_set[0].len() } else { 0 };
        unsafe {
            *out_n_indices = n;
            *out_index_len = idx_len;
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy I-set at site p to caller-provided flat buffer.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_get_i_set(
    ptr: *const t4a_tci2_f64,
    site: libc::size_t,
    out_buf: *mut libc::size_t,
    buf_capacity: libc::size_t,
    out_n_indices: *mut libc::size_t,
    out_index_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_buf.is_null() || out_n_indices.is_null() || out_index_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        let i_set = tci.inner().i_set(site);
        let n = i_set.len();
        let idx_len = if n > 0 { i_set[0].len() } else { 0 };
        let total = n * idx_len;

        if total > buf_capacity {
            return crate::err_status(
                format!("Buffer too small: need {total}, got {buf_capacity}"),
                T4A_INVALID_ARGUMENT,
            );
        }

        unsafe {
            *out_n_indices = n;
            *out_index_len = idx_len;
        }

        let mut offset = 0;
        for multi_idx in i_set {
            for &idx in multi_idx {
                unsafe { *out_buf.add(offset) = idx };
                offset += 1;
            }
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Query J-set size at site p.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_j_set_size(
    ptr: *const t4a_tci2_f64,
    site: libc::size_t,
    out_n_indices: *mut libc::size_t,
    out_index_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_n_indices.is_null() || out_index_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        let j_set = tci.inner().j_set(site);
        let n = j_set.len();
        let idx_len = if n > 0 { j_set[0].len() } else { 0 };
        unsafe {
            *out_n_indices = n;
            *out_index_len = idx_len;
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy J-set at site p to caller-provided flat buffer.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tci2_f64_get_j_set(
    ptr: *const t4a_tci2_f64,
    site: libc::size_t,
    out_buf: *mut libc::size_t,
    buf_capacity: libc::size_t,
    out_n_indices: *mut libc::size_t,
    out_index_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_buf.is_null() || out_n_indices.is_null() || out_index_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tci = unsafe { &*ptr };
        let j_set = tci.inner().j_set(site);
        let n = j_set.len();
        let idx_len = if n > 0 { j_set[0].len() } else { 0 };
        let total = n * idx_len;

        if total > buf_capacity {
            return crate::err_status(
                format!("Buffer too small: need {total}, got {buf_capacity}"),
                T4A_INVALID_ARGUMENT,
            );
        }

        unsafe {
            *out_n_indices = n;
            *out_index_len = idx_len;
        }

        let mut offset = 0;
        for multi_idx in j_set {
            for &idx in multi_idx {
                unsafe { *out_buf.add(offset) = idx };
                offset += 1;
            }
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
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
            Err(e) => crate::err_null(e),
        }
    }));

    crate::unwrap_catch_ptr(result)
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
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// opt_first_pivot
// ============================================================================

/// Find optimal initial pivot by greedy local search.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_opt_first_pivot_f64(
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    first_pivot: *const libc::size_t,
    max_sweep: libc::size_t,
    out_pivot: *mut libc::size_t,
) -> StatusCode {
    if local_dims.is_null() || first_pivot.is_null() || out_pivot.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();
        let pivot: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *first_pivot.add(i) })
            .collect();
        let f = make_eval_closure(eval_fn, user_data);

        let optimized =
            tensor4all_tensorci::opt_first_pivot::<f64, _>(&f, &dims, &pivot, max_sweep);

        for (i, &idx) in optimized.iter().enumerate() {
            unsafe { *out_pivot.add(i) = idx };
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// estimate_true_error
// ============================================================================

/// Estimate true interpolation error.
///
/// Returns pivots and errors sorted by descending error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_estimate_true_error_f64(
    tt_ptr: *const t4a_simplett_f64,
    eval_fn: EvalCallback,
    user_data: *mut c_void,
    nsearch: libc::size_t,
    out_pivots: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_results: *mut libc::size_t,
    pivot_buf_capacity: libc::size_t,
) -> StatusCode {
    if tt_ptr.is_null() || out_pivots.is_null() || out_errors.is_null() || out_n_results.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*tt_ptr };
        let f = make_eval_closure(eval_fn, user_data);
        let mut rng = rand::rng();

        let results =
            tensor4all_tensorci::estimate_true_error(tt.inner(), &f, nsearch, None, &mut rng);

        let n_sites = tt.inner().len();
        let n_results = results.len();

        // Check buffer capacity
        if n_results * n_sites > pivot_buf_capacity {
            return crate::err_status(
                format!(
                    "Pivot buffer too small: need {}, got {}",
                    n_results * n_sites,
                    pivot_buf_capacity
                ),
                T4A_INVALID_ARGUMENT,
            );
        }

        unsafe { *out_n_results = n_results };

        for (i, (pivot, error)) in results.iter().enumerate() {
            for (j, &idx) in pivot.iter().enumerate() {
                unsafe { *out_pivots.add(i * n_sites + j) = idx };
            }
            unsafe { *out_errors.add(i) = *error };
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

#[cfg(test)]
mod tests;
