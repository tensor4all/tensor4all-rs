//! C API for QuanticsTCI (Quantics Tensor Cross Interpolation)
//!
//! This provides C-compatible interface for interpolating functions using
//! quantics tensor cross interpolation. It wraps `tensor4all-quanticstci`
//! which combines TCI with quantics grid representations.

use crate::types::{
    t4a_qgrid_disc, t4a_qtci_c64, t4a_qtci_f64, t4a_qtci_options, t4a_unfolding_scheme,
};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use num_complex::Complex64;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_quanticstci::{
    quanticscrossinterpolate, quanticscrossinterpolate_discrete, QtciOptions,
};

// ============================================================================
// Callback types — f64
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
// Callback types — c64
// ============================================================================

/// Callback function type for evaluating a complex function at continuous coordinates.
///
/// The result buffer receives an interleaved [re, im] pair (2 doubles).
pub type QtciEvalCallbackC64 = extern "C" fn(
    coords: *const libc::c_double,
    ndims: libc::size_t,
    result: *mut libc::c_double, // [re, im]
    user_data: *mut c_void,
) -> i32;

/// Callback function type for evaluating a complex function at discrete (integer) indices.
///
/// The result buffer receives an interleaved [re, im] pair (2 doubles).
pub type QtciEvalCallbackI64C64 = extern "C" fn(
    indices: *const i64,
    ndims: libc::size_t,
    result: *mut libc::c_double, // [re, im]
    user_data: *mut c_void,
) -> i32;

// ============================================================================
// QtciOptions lifecycle
// ============================================================================

/// Create a new QtciOptions handle with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_default() -> *mut t4a_qtci_options {
    let result = catch_unwind(AssertUnwindSafe(|| {
        Box::into_raw(Box::new(t4a_qtci_options::new(QtciOptions::default())))
    }));
    crate::unwrap_catch_ptr(result)
}

/// Release a QtciOptions handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_release(ptr: *mut t4a_qtci_options) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Clone a QtciOptions handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_clone(ptr: *const t4a_qtci_options) -> *mut t4a_qtci_options {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &*ptr };
        Box::into_raw(Box::new(opts.clone()))
    }));
    crate::unwrap_catch_ptr(result)
}

/// Check if the QtciOptions handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_is_assigned(ptr: *const t4a_qtci_options) -> i32 {
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
// QtciOptions setters
// ============================================================================

/// Set the convergence tolerance.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_tolerance(
    ptr: *mut t4a_qtci_options,
    tolerance: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().tolerance = tolerance;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set the maximum bond dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_maxbonddim(
    ptr: *mut t4a_qtci_options,
    dim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().maxbonddim = if dim == 0 { None } else { Some(dim) };
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set the maximum number of iterations.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_maxiter(
    ptr: *mut t4a_qtci_options,
    iter: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().maxiter = iter;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set the number of random initial pivots.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_nrandominitpivot(
    ptr: *mut t4a_qtci_options,
    n: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().nrandominitpivot = n;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set the unfolding scheme.
///
/// * 0 = Fused
/// * 1 = Interleaved
/// * 2 = Grouped
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_unfoldingscheme(
    ptr: *mut t4a_qtci_options,
    scheme: t4a_unfolding_scheme,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().unfoldingscheme = scheme.into();
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set whether to normalize error by max sample value.
///
/// * 0 = false
/// * Non-zero = true
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_normalize_error(
    ptr: *mut t4a_qtci_options,
    flag: libc::c_int,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().normalize_error = flag != 0;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set the verbosity level (0 = silent).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_verbosity(
    ptr: *mut t4a_qtci_options,
    level: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().verbosity = level;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set number of global pivots to search per iteration.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_nsearchglobalpivot(
    ptr: *mut t4a_qtci_options,
    n: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().nsearchglobalpivot = n;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Set number of random searches for global pivots.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_options_set_nsearch(
    ptr: *mut t4a_qtci_options,
    n: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let opts = unsafe { &mut *ptr };
        opts.inner_mut().nsearch = n;
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

// ============================================================================
// f64 lifecycle functions
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

/// Clone a QuanticsTCI f64 handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_clone(ptr: *const t4a_qtci_f64) -> *mut t4a_qtci_f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        Box::into_raw(Box::new(t4a_qtci_f64::new(qtci.inner().clone())))
    }));
    crate::unwrap_catch_ptr(result)
}

/// Check if the QuanticsTCI handle is assigned (non-null and dereferenceable).
///
/// # Returns
/// 1 if valid, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_is_assigned(ptr: *const t4a_qtci_f64) -> i32 {
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
// c64 lifecycle functions
// ============================================================================

/// Release a QuanticsTCI c64 handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_release(ptr: *mut t4a_qtci_c64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Clone a QuanticsTCI c64 handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_clone(ptr: *const t4a_qtci_c64) -> *mut t4a_qtci_c64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        Box::into_raw(Box::new(t4a_qtci_c64::new(qtci.inner().clone())))
    }));
    crate::unwrap_catch_ptr(result)
}

/// Check if the QuanticsTCI c64 handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_is_assigned(ptr: *const t4a_qtci_c64) -> i32 {
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
// Helper: build QtciOptions from C API params
// ============================================================================

/// Build a QtciOptions from the C API parameters.
/// If `options` is null, returns defaults with overrides from the legacy params.
fn build_options(
    options: *const t4a_qtci_options,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
) -> QtciOptions {
    if !options.is_null() {
        let opts = unsafe { &*options };
        opts.inner().clone()
    } else {
        let mut opts = QtciOptions::default()
            .with_tolerance(tolerance)
            .with_maxiter(max_iter);
        if max_bonddim > 0 {
            opts = opts.with_maxbonddim(max_bonddim);
        }
        opts
    }
}

/// Parse flat initial pivots from C API into Vec<Vec<i64>>.
fn parse_initial_pivots(
    initial_pivots: *const i64,
    n_pivots: libc::size_t,
    ndims: usize,
) -> Option<Vec<Vec<i64>>> {
    if initial_pivots.is_null() || n_pivots == 0 {
        return None;
    }
    let flat = unsafe { std::slice::from_raw_parts(initial_pivots, n_pivots * ndims) };
    let pivots: Vec<Vec<i64>> = flat
        .chunks_exact(ndims)
        .map(|chunk| chunk.to_vec())
        .collect();
    Some(pivots)
}

/// Write convergence info to caller buffers.
fn write_convergence_info(
    ranks: &[usize],
    errors: &[f64],
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) {
    let n_iters = ranks.len();
    if !out_n_iters.is_null() {
        unsafe { *out_n_iters = n_iters };
    }
    if !out_ranks.is_null() {
        for (i, &r) in ranks.iter().enumerate() {
            unsafe { *out_ranks.add(i) = r };
        }
    }
    if !out_errors.is_null() {
        for (i, &e) in errors.iter().enumerate() {
            unsafe { *out_errors.add(i) = e };
        }
    }
}

// ============================================================================
// High-level interpolation functions — f64
// ============================================================================

/// Perform quantics cross interpolation on a continuous domain with a DiscretizedGrid.
///
/// # Arguments
/// * `grid` - Discretized grid describing the function domain
/// * `eval_fn` - Callback for function evaluation (takes f64 coordinates)
/// * `user_data` - User data passed to callback
/// * `options` - QtciOptions handle (NULL = use legacy tolerance/max_bonddim/max_iter)
/// * `tolerance` - Convergence tolerance (ignored if options is non-NULL)
/// * `max_bonddim` - Maximum bond dimension, 0 = unlimited (ignored if options is non-NULL)
/// * `max_iter` - Maximum iterations (ignored if options is non-NULL)
/// * `initial_pivots` - Flat initial pivot grid indices (n_pivots x ndims, 1-indexed), or NULL
/// * `n_pivots` - Number of initial pivots (0 if none)
/// * `out_qtci` - Output: QTCI handle (caller owns)
/// * `out_ranks` - Output buffer for ranks per iteration (NULL to skip)
/// * `out_errors` - Output buffer for errors per iteration (NULL to skip)
/// * `out_n_iters` - Output: actual number of iterations (NULL to skip)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_f64(
    grid: *const t4a_qgrid_disc,
    eval_fn: QtciEvalCallbackF64,
    user_data: *mut c_void,
    options: *const t4a_qtci_options,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    initial_pivots: *const i64,
    n_pivots: libc::size_t,
    out_qtci: *mut *mut t4a_qtci_f64,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let grid_ref = unsafe { &*grid };
        let inner_grid = grid_ref.inner();

        let f = move |coords: &[f64]| -> f64 {
            let mut result: f64 = 0.0;
            let status = eval_fn(coords.as_ptr(), coords.len(), &mut result, user_data);
            if status != 0 {
                f64::NAN
            } else {
                result
            }
        };

        let opts = build_options(options, tolerance, max_bonddim, max_iter);
        let ndims = inner_grid.ndims();
        let pivots = parse_initial_pivots(initial_pivots, n_pivots, ndims);

        match quanticscrossinterpolate(inner_grid, f, pivots, opts) {
            Ok((qtci, ranks, errors)) => {
                write_convergence_info(&ranks, &errors, out_ranks, out_errors, out_n_iters);
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
/// * `options` - QtciOptions handle (NULL = use legacy tolerance/max_bonddim/max_iter)
/// * `tolerance` - Convergence tolerance (ignored if options is non-NULL)
/// * `max_bonddim` - Maximum bond dimension, 0 = unlimited (ignored if options is non-NULL)
/// * `max_iter` - Maximum iterations (ignored if options is non-NULL)
/// * `unfoldingscheme` - Unfolding scheme (ignored if options is non-NULL)
/// * `initial_pivots` - Flat initial pivot grid indices (n_pivots x ndims, 1-indexed), or NULL
/// * `n_pivots` - Number of initial pivots (0 if none)
/// * `out_qtci` - Output: QTCI handle (caller owns)
/// * `out_ranks` - Output buffer for ranks per iteration (NULL to skip)
/// * `out_errors` - Output buffer for errors per iteration (NULL to skip)
/// * `out_n_iters` - Output: actual number of iterations (NULL to skip)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_discrete_f64(
    sizes: *const libc::size_t,
    ndims: libc::size_t,
    eval_fn: QtciEvalCallbackI64,
    user_data: *mut c_void,
    options: *const t4a_qtci_options,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    unfoldingscheme: t4a_unfolding_scheme,
    initial_pivots: *const i64,
    n_pivots: libc::size_t,
    out_qtci: *mut *mut t4a_qtci_f64,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if sizes.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }
    if ndims == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let size_slice: Vec<usize> = unsafe { std::slice::from_raw_parts(sizes, ndims) }.to_vec();

        let f = move |indices: &[i64]| -> f64 {
            let mut result: f64 = 0.0;
            let status = eval_fn(indices.as_ptr(), indices.len(), &mut result, user_data);
            if status != 0 {
                f64::NAN
            } else {
                result
            }
        };

        let opts = if !options.is_null() {
            let o = unsafe { &*options };
            o.inner().clone()
        } else {
            let scheme: quanticsgrids::UnfoldingScheme = unfoldingscheme.into();
            let mut o = QtciOptions::default()
                .with_tolerance(tolerance)
                .with_maxiter(max_iter)
                .with_unfoldingscheme(scheme);
            if max_bonddim > 0 {
                o = o.with_maxbonddim(max_bonddim);
            }
            o
        };

        let pivots = parse_initial_pivots(initial_pivots, n_pivots, ndims);

        match quanticscrossinterpolate_discrete(&size_slice, f, pivots, opts) {
            Ok((qtci, ranks, errors)) => {
                write_convergence_info(&ranks, &errors, out_ranks, out_errors, out_n_iters);
                unsafe { *out_qtci = Box::into_raw(Box::new(t4a_qtci_f64::new(qtci))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// High-level interpolation functions — c64
// ============================================================================

/// Perform quantics cross interpolation (c64) on a continuous domain.
///
/// The callback writes interleaved [re, im] to the result buffer.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_c64(
    grid: *const t4a_qgrid_disc,
    eval_fn: QtciEvalCallbackC64,
    user_data: *mut c_void,
    options: *const t4a_qtci_options,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    initial_pivots: *const i64,
    n_pivots: libc::size_t,
    out_qtci: *mut *mut t4a_qtci_c64,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let grid_ref = unsafe { &*grid };
        let inner_grid = grid_ref.inner();

        let f = move |coords: &[f64]| -> Complex64 {
            let mut result_buf: [f64; 2] = [0.0, 0.0];
            let status = eval_fn(
                coords.as_ptr(),
                coords.len(),
                result_buf.as_mut_ptr(),
                user_data,
            );
            if status != 0 {
                Complex64::new(f64::NAN, f64::NAN)
            } else {
                Complex64::new(result_buf[0], result_buf[1])
            }
        };

        let opts = build_options(options, tolerance, max_bonddim, max_iter);
        let ndims = inner_grid.ndims();
        let pivots = parse_initial_pivots(initial_pivots, n_pivots, ndims);

        match quanticscrossinterpolate(inner_grid, f, pivots, opts) {
            Ok((qtci, ranks, errors)) => {
                write_convergence_info(&ranks, &errors, out_ranks, out_errors, out_n_iters);
                unsafe { *out_qtci = Box::into_raw(Box::new(t4a_qtci_c64::new(qtci))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Perform quantics cross interpolation (c64) on a discrete integer domain.
///
/// The callback writes interleaved [re, im] to the result buffer.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_quanticscrossinterpolate_discrete_c64(
    sizes: *const libc::size_t,
    ndims: libc::size_t,
    eval_fn: QtciEvalCallbackI64C64,
    user_data: *mut c_void,
    options: *const t4a_qtci_options,
    tolerance: libc::c_double,
    max_bonddim: libc::size_t,
    max_iter: libc::size_t,
    unfoldingscheme: t4a_unfolding_scheme,
    initial_pivots: *const i64,
    n_pivots: libc::size_t,
    out_qtci: *mut *mut t4a_qtci_c64,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if sizes.is_null() || out_qtci.is_null() {
        return T4A_NULL_POINTER;
    }
    if ndims == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let size_slice: Vec<usize> = unsafe { std::slice::from_raw_parts(sizes, ndims) }.to_vec();

        let f = move |indices: &[i64]| -> Complex64 {
            let mut result_buf: [f64; 2] = [0.0, 0.0];
            let status = eval_fn(
                indices.as_ptr(),
                indices.len(),
                result_buf.as_mut_ptr(),
                user_data,
            );
            if status != 0 {
                Complex64::new(f64::NAN, f64::NAN)
            } else {
                Complex64::new(result_buf[0], result_buf[1])
            }
        };

        let opts = if !options.is_null() {
            let o = unsafe { &*options };
            o.inner().clone()
        } else {
            let scheme: quanticsgrids::UnfoldingScheme = unfoldingscheme.into();
            let mut o = QtciOptions::default()
                .with_tolerance(tolerance)
                .with_maxiter(max_iter)
                .with_unfoldingscheme(scheme);
            if max_bonddim > 0 {
                o = o.with_maxbonddim(max_bonddim);
            }
            o
        };

        let pivots = parse_initial_pivots(initial_pivots, n_pivots, ndims);

        match quanticscrossinterpolate_discrete(&size_slice, f, pivots, opts) {
            Ok((qtci, ranks, errors)) => {
                write_convergence_info(&ranks, &errors, out_ranks, out_errors, out_n_iters);
                unsafe { *out_qtci = Box::into_raw(Box::new(t4a_qtci_c64::new(qtci))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// f64 accessors
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

/// Evaluate the QTCI at grid indices.
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
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_to_tensor_train(
    ptr: *const t4a_qtci_f64,
) -> *mut crate::types::t4a_simplett_f64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        let tt = qtci.inner().tensor_train();
        Box::into_raw(Box::new(crate::types::t4a_simplett_f64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// c64 accessors
// ============================================================================

/// Get the maximum bond dimension (rank) — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_rank(
    ptr: *const t4a_qtci_c64,
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

/// Get the link (bond) dimensions — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_link_dims(
    ptr: *const t4a_qtci_c64,
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

/// Evaluate the QTCI at grid indices — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_evaluate(
    ptr: *const t4a_qtci_c64,
    indices: *const i64,
    n_indices: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        let idx: Vec<i64> = unsafe { std::slice::from_raw_parts(indices, n_indices) }.to_vec();

        match qtci.inner().evaluate(&idx) {
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

/// Compute the factorized sum — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_sum(
    ptr: *const t4a_qtci_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        match qtci.inner().sum() {
            Ok(val) => {
                unsafe {
                    *out_re = val.re;
                    *out_im = val.im;
                }
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the integral — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_integral(
    ptr: *const t4a_qtci_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        match qtci.inner().integral() {
            Ok(val) => {
                unsafe {
                    *out_re = val.re;
                    *out_im = val.im;
                }
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Convert the QTCI to a SimpleTT tensor train — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_to_tensor_train(
    ptr: *const t4a_qtci_c64,
) -> *mut crate::types::t4a_simplett_c64 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        let tt = qtci.inner().tensor_train();
        Box::into_raw(Box::new(crate::types::t4a_simplett_c64::new(tt)))
    }));

    crate::unwrap_catch_ptr(result)
}

// ============================================================================
// Category 4: TreeTCI2 state accessors
// ============================================================================

/// Get the maximum bond error from the TreeTCI2 state — f64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_max_bond_error(
    ptr: *const t4a_qtci_f64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        unsafe { *out_value = qtci.inner().tci().max_bond_error() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum rank from the TreeTCI2 state — f64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_f64_max_rank(
    ptr: *const t4a_qtci_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        unsafe { *out_rank = qtci.inner().tci().max_rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum bond error from the TreeTCI2 state — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_max_bond_error(
    ptr: *const t4a_qtci_c64,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        unsafe { *out_value = qtci.inner().tci().max_bond_error() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum rank from the TreeTCI2 state — c64.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtci_c64_max_rank(
    ptr: *const t4a_qtci_c64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let qtci = unsafe { &*ptr };
        unsafe { *out_rank = qtci.inner().tci().max_rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
