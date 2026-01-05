//! C API for MPO (Matrix Product Operator) contraction
//!
//! Provides functions for creating, manipulating, and contracting MPOs.
//!
//! ## Naming convention
//! - `t4a_mpo_f64_*` - Functions for MPO<f64>
//! - `t4a_mpo_c64_*` - Functions for MPO<Complex64>

use std::panic::catch_unwind;
use std::ptr;

use num_complex::Complex64;
use tensor4all_core_common::ContractionAlgorithm;
use tensor4all_mpocontraction::{
    contract, contract_fit, contract_naive, contract_zipup, ContractionOptions, FactorizeMethod,
    FitOptions, MPO,
};

use crate::{
    StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER,
    T4A_SUCCESS,
};

// ============================================================================
// Opaque type definitions
// ============================================================================

/// Opaque type for MPO<f64>
#[repr(C)]
pub struct t4a_mpo_f64 {
    _private: *const std::ffi::c_void,
}

impl t4a_mpo_f64 {
    pub(crate) fn new(mpo: MPO<f64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(mpo)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn inner(&self) -> &MPO<f64> {
        unsafe { &*(self._private as *const MPO<f64>) }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut MPO<f64> {
        unsafe { &mut *(self._private as *mut MPO<f64>) }
    }
}

impl Drop for t4a_mpo_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                drop(Box::from_raw(self._private as *mut MPO<f64>));
            }
        }
    }
}

impl Clone for t4a_mpo_f64 {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

/// Opaque type for MPO<Complex64>
#[repr(C)]
pub struct t4a_mpo_c64 {
    _private: *const std::ffi::c_void,
}

impl t4a_mpo_c64 {
    pub(crate) fn new(mpo: MPO<Complex64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(mpo)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn inner(&self) -> &MPO<Complex64> {
        unsafe { &*(self._private as *const MPO<Complex64>) }
    }

    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut MPO<Complex64> {
        unsafe { &mut *(self._private as *mut MPO<Complex64>) }
    }
}

impl Drop for t4a_mpo_c64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                drop(Box::from_raw(self._private as *mut MPO<Complex64>));
            }
        }
    }
}

impl Clone for t4a_mpo_c64 {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

// Generate lifecycle functions
impl_opaque_type_common!(mpo_f64);
impl_opaque_type_common!(mpo_c64);

// ============================================================================
// Constructors - f64
// ============================================================================

/// Create a new MPO representing the zero operator (f64)
///
/// # Arguments
/// - `site_dims_1`: Array of first site dimensions
/// - `site_dims_2`: Array of second site dimensions
/// - `num_sites`: Number of sites
///
/// # Returns
/// - Pointer to new t4a_mpo_f64 on success
/// - NULL on error
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_new_zeros(
    site_dims_1: *const libc::size_t,
    site_dims_2: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_mpo_f64 {
    if site_dims_1.is_null() || site_dims_2.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims1: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_1, num_sites).to_vec() };
        let dims2: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_2, num_sites).to_vec() };
        let site_dims: Vec<(usize, usize)> = dims1.into_iter().zip(dims2).collect();
        let mpo = MPO::<f64>::zeros(&site_dims);
        Box::into_raw(Box::new(t4a_mpo_f64::new(mpo)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create a new MPO representing a constant operator (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_new_constant(
    site_dims_1: *const libc::size_t,
    site_dims_2: *const libc::size_t,
    num_sites: libc::size_t,
    value: libc::c_double,
) -> *mut t4a_mpo_f64 {
    if site_dims_1.is_null() || site_dims_2.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims1: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_1, num_sites).to_vec() };
        let dims2: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_2, num_sites).to_vec() };
        let site_dims: Vec<(usize, usize)> = dims1.into_iter().zip(dims2).collect();
        let mpo = MPO::<f64>::constant(&site_dims, value);
        Box::into_raw(Box::new(t4a_mpo_f64::new(mpo)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create an identity MPO (f64)
///
/// Only valid when site_dim_1 == site_dim_2 at each site
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_new_identity(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_mpo_f64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        match MPO::<f64>::identity(&dims) {
            Ok(mpo) => Box::into_raw(Box::new(t4a_mpo_f64::new(mpo))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Accessors - f64
// ============================================================================

/// Get the number of sites in an MPO (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_len(
    ptr: *const t4a_mpo_f64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        unsafe { *out_len = mpo.inner().len() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the site dimensions of an MPO (f64)
///
/// # Arguments
/// - `ptr`: MPO handle
/// - `out_dims_1`: Buffer for first site dimensions
/// - `out_dims_2`: Buffer for second site dimensions
/// - `buf_len`: Length of the buffers
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_site_dims(
    ptr: *const t4a_mpo_f64,
    out_dims_1: *mut libc::size_t,
    out_dims_2: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims_1.is_null() || out_dims_2.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let dims = mpo.inner().site_dims();

        if buf_len < dims.len() {
            return T4A_BUFFER_TOO_SMALL;
        }

        unsafe {
            for (i, &(d1, d2)) in dims.iter().enumerate() {
                *out_dims_1.add(i) = d1;
                *out_dims_2.add(i) = d2;
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the link dimensions (bond dimensions) of an MPO (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_link_dims(
    ptr: *const t4a_mpo_f64,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let dims = mpo.inner().link_dims();

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

/// Get the maximum bond dimension (rank) of an MPO (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_rank(
    ptr: *const t4a_mpo_f64,
    out_rank: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_rank.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        unsafe { *out_rank = mpo.inner().rank() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Evaluate an MPO at a given index set (f64)
///
/// # Arguments
/// - `ptr`: MPO handle
/// - `indices`: Array of indices (length = 2 * num_sites, alternating i, j)
/// - `num_indices`: Number of indices (must equal 2 * num_sites)
/// - `out_value`: Output: evaluated value
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_evaluate(
    ptr: *const t4a_mpo_f64,
    indices: *const libc::size_t,
    num_indices: libc::size_t,
    out_value: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_value.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let idx: Vec<usize> = unsafe { std::slice::from_raw_parts(indices, num_indices).to_vec() };

        match mpo.inner().evaluate(&idx) {
            Ok(value) => {
                unsafe { *out_value = value };
                T4A_SUCCESS
            }
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the sum of all elements in an MPO (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_sum(
    ptr: *const t4a_mpo_f64,
    out_sum: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_sum.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        unsafe { *out_sum = mpo.inner().sum() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Modifiers - f64
// ============================================================================

/// Scale an MPO by a factor in place (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_scale_inplace(
    ptr: *mut t4a_mpo_f64,
    factor: libc::c_double,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &mut *ptr };
        mpo.inner_mut().scale(factor);
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Scale an MPO by a factor, returning a new object (f64)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_scaled(
    ptr: *const t4a_mpo_f64,
    factor: libc::c_double,
) -> *mut t4a_mpo_f64 {
    if ptr.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let scaled = mpo.inner().scaled(factor);
        Box::into_raw(Box::new(t4a_mpo_f64::new(scaled)))
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Contraction - f64
// ============================================================================

/// Contract two MPOs using naive (exact) algorithm (f64)
///
/// This computes the exact contraction but may be memory-intensive for large MPOs.
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_contract_naive(
    a: *const t4a_mpo_f64,
    b: *const t4a_mpo_f64,
) -> *mut t4a_mpo_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        match contract_naive(mpo_a.inner(), mpo_b.inner(), None) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_f64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using zip-up algorithm with compression (f64)
///
/// This performs on-the-fly compression during contraction, reducing memory usage.
///
/// # Arguments
/// - `a`: First MPO
/// - `b`: Second MPO
/// - `tolerance`: Relative tolerance for truncation
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_contract_zipup(
    a: *const t4a_mpo_f64,
    b: *const t4a_mpo_f64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_mpo_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let options = ContractionOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            factorize_method: FactorizeMethod::SVD,
        };

        match contract_zipup(mpo_a.inner(), mpo_b.inner(), &options) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_f64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using variational fitting algorithm (f64)
///
/// This is the most memory-efficient method for large MPOs.
///
/// # Arguments
/// - `a`: First MPO
/// - `b`: Second MPO
/// - `tolerance`: Convergence tolerance
/// - `max_bond_dim`: Maximum bond dimension of result
/// - `max_sweeps`: Maximum number of optimization sweeps (0 for default)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_contract_fit(
    a: *const t4a_mpo_f64,
    b: *const t4a_mpo_f64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
    max_sweeps: libc::size_t,
) -> *mut t4a_mpo_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let fit_options = FitOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            max_sweeps: if max_sweeps == 0 { 10 } else { max_sweeps },
            convergence_tol: tolerance,
            factorize_method: FactorizeMethod::SVD,
        };

        match contract_fit(mpo_a.inner(), mpo_b.inner(), &fit_options, None) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_f64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using the specified algorithm (f64)
///
/// Unified dispatch function that selects the appropriate contraction algorithm.
///
/// # Arguments
/// - `a`: First MPO
/// - `b`: Second MPO
/// - `algorithm`: Contraction algorithm (0=Naive, 1=ZipUp, 2=Fit)
/// - `tolerance`: Relative tolerance for truncation
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
#[no_mangle]
pub extern "C" fn t4a_mpo_f64_contract(
    a: *const t4a_mpo_f64,
    b: *const t4a_mpo_f64,
    algorithm: crate::t4a_contraction_algorithm,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_mpo_f64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let rust_algorithm: ContractionAlgorithm = algorithm.into();
        let options = ContractionOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            factorize_method: FactorizeMethod::SVD,
        };

        match contract(mpo_a.inner(), mpo_b.inner(), rust_algorithm, &options) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_f64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Constructors - Complex64
// ============================================================================

/// Create a new MPO representing the zero operator (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_new_zeros(
    site_dims_1: *const libc::size_t,
    site_dims_2: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_mpo_c64 {
    if site_dims_1.is_null() || site_dims_2.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims1: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_1, num_sites).to_vec() };
        let dims2: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_2, num_sites).to_vec() };
        let site_dims: Vec<(usize, usize)> = dims1.into_iter().zip(dims2).collect();
        let mpo = MPO::<Complex64>::zeros(&site_dims);
        Box::into_raw(Box::new(t4a_mpo_c64::new(mpo)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create a new MPO representing a constant operator (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_new_constant(
    site_dims_1: *const libc::size_t,
    site_dims_2: *const libc::size_t,
    num_sites: libc::size_t,
    value_re: libc::c_double,
    value_im: libc::c_double,
) -> *mut t4a_mpo_c64 {
    if site_dims_1.is_null() || site_dims_2.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims1: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_1, num_sites).to_vec() };
        let dims2: Vec<usize> =
            unsafe { std::slice::from_raw_parts(site_dims_2, num_sites).to_vec() };
        let site_dims: Vec<(usize, usize)> = dims1.into_iter().zip(dims2).collect();
        let value = Complex64::new(value_re, value_im);
        let mpo = MPO::<Complex64>::constant(&site_dims, value);
        Box::into_raw(Box::new(t4a_mpo_c64::new(mpo)))
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Create an identity MPO (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_new_identity(
    site_dims: *const libc::size_t,
    num_sites: libc::size_t,
) -> *mut t4a_mpo_c64 {
    if site_dims.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dims: Vec<usize> = unsafe { std::slice::from_raw_parts(site_dims, num_sites).to_vec() };
        match MPO::<Complex64>::identity(&dims) {
            Ok(mpo) => Box::into_raw(Box::new(t4a_mpo_c64::new(mpo))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Accessors - Complex64
// ============================================================================

/// Get the number of sites in an MPO (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_len(
    ptr: *const t4a_mpo_c64,
    out_len: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_len.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        unsafe { *out_len = mpo.inner().len() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Evaluate an MPO at a given index set (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_evaluate(
    ptr: *const t4a_mpo_c64,
    indices: *const libc::size_t,
    num_indices: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || indices.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let idx: Vec<usize> = unsafe { std::slice::from_raw_parts(indices, num_indices).to_vec() };

        match mpo.inner().evaluate(&idx) {
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

/// Compute the sum of all elements in an MPO (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_sum(
    ptr: *const t4a_mpo_c64,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo = unsafe { &*ptr };
        let sum = mpo.inner().sum();
        unsafe {
            *out_re = sum.re;
            *out_im = sum.im;
        };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Contraction - Complex64
// ============================================================================

/// Contract two MPOs using naive (exact) algorithm (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_contract_naive(
    a: *const t4a_mpo_c64,
    b: *const t4a_mpo_c64,
) -> *mut t4a_mpo_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        match contract_naive(mpo_a.inner(), mpo_b.inner(), None) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_c64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using zip-up algorithm with compression (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_contract_zipup(
    a: *const t4a_mpo_c64,
    b: *const t4a_mpo_c64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_mpo_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let options = ContractionOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            factorize_method: FactorizeMethod::SVD,
        };

        match contract_zipup(mpo_a.inner(), mpo_b.inner(), &options) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_c64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using variational fitting algorithm (Complex64)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_contract_fit(
    a: *const t4a_mpo_c64,
    b: *const t4a_mpo_c64,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
    max_sweeps: libc::size_t,
) -> *mut t4a_mpo_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let fit_options = FitOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            max_sweeps: if max_sweeps == 0 { 10 } else { max_sweeps },
            convergence_tol: tolerance,
            factorize_method: FactorizeMethod::SVD,
        };

        match contract_fit(mpo_a.inner(), mpo_b.inner(), &fit_options, None) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_c64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

/// Contract two MPOs using the specified algorithm (Complex64)
///
/// Unified dispatch function that selects the appropriate contraction algorithm.
///
/// # Arguments
/// - `a`: First MPO
/// - `b`: Second MPO
/// - `algorithm`: Contraction algorithm (0=Naive, 1=ZipUp, 2=Fit)
/// - `tolerance`: Relative tolerance for truncation
/// - `max_bond_dim`: Maximum bond dimension (0 for unlimited)
#[no_mangle]
pub extern "C" fn t4a_mpo_c64_contract(
    a: *const t4a_mpo_c64,
    b: *const t4a_mpo_c64,
    algorithm: crate::t4a_contraction_algorithm,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> *mut t4a_mpo_c64 {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mpo_a = unsafe { &*a };
        let mpo_b = unsafe { &*b };

        let rust_algorithm: ContractionAlgorithm = algorithm.into();
        let options = ContractionOptions {
            tolerance,
            max_bond_dim: if max_bond_dim == 0 {
                usize::MAX
            } else {
                max_bond_dim
            },
            factorize_method: FactorizeMethod::SVD,
        };

        match contract(mpo_a.inner(), mpo_b.inner(), rust_algorithm, &options) {
            Ok(contracted) => Box::into_raw(Box::new(t4a_mpo_c64::new(contracted))),
            Err(_) => ptr::null_mut(),
        }
    }));

    result.unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpo_f64_lifecycle() {
        let dims1 = [2_usize, 3];
        let dims2 = [2_usize, 3];
        let mpo = t4a_mpo_f64_new_zeros(dims1.as_ptr(), dims2.as_ptr(), 2);
        assert!(!mpo.is_null());

        assert_eq!(t4a_mpo_f64_is_assigned(mpo as *const _), 1);

        let cloned = t4a_mpo_f64_clone(mpo as *const _);
        assert!(!cloned.is_null());

        t4a_mpo_f64_release(cloned);
        t4a_mpo_f64_release(mpo);
    }

    #[test]
    fn test_mpo_f64_constant() {
        let dims1 = [2_usize, 2];
        let dims2 = [2_usize, 2];
        let mpo = t4a_mpo_f64_new_constant(dims1.as_ptr(), dims2.as_ptr(), 2, 5.0);
        assert!(!mpo.is_null());

        let mut len: usize = 0;
        assert_eq!(t4a_mpo_f64_len(mpo as *const _, &mut len), T4A_SUCCESS);
        assert_eq!(len, 2);

        // Sum should be 5.0 * (2*2) * (2*2) = 80.0
        let mut sum: f64 = 0.0;
        assert_eq!(t4a_mpo_f64_sum(mpo as *const _, &mut sum), T4A_SUCCESS);
        assert!((sum - 80.0).abs() < 1e-10);

        t4a_mpo_f64_release(mpo);
    }

    #[test]
    fn test_mpo_f64_identity() {
        let dims = [2_usize, 3];
        let mpo = t4a_mpo_f64_new_identity(dims.as_ptr(), 2);
        assert!(!mpo.is_null());

        // Identity[0, 0, 0, 0] = 1.0
        let indices = [0_usize, 0, 0, 0];
        let mut value: f64 = 0.0;
        assert_eq!(
            t4a_mpo_f64_evaluate(mpo as *const _, indices.as_ptr(), 4, &mut value),
            T4A_SUCCESS
        );
        assert!((value - 1.0).abs() < 1e-10);

        // Identity[0, 1, 0, 0] = 0.0 (off-diagonal)
        let indices2 = [0_usize, 1, 0, 0];
        assert_eq!(
            t4a_mpo_f64_evaluate(mpo as *const _, indices2.as_ptr(), 4, &mut value),
            T4A_SUCCESS
        );
        assert!(value.abs() < 1e-10);

        t4a_mpo_f64_release(mpo);
    }

    #[test]
    fn test_mpo_f64_contract_naive() {
        // Contract two identity MPOs: I * I = I
        let dims = [2_usize, 2];
        let mpo_a = t4a_mpo_f64_new_identity(dims.as_ptr(), 2);
        let mpo_b = t4a_mpo_f64_new_identity(dims.as_ptr(), 2);
        assert!(!mpo_a.is_null());
        assert!(!mpo_b.is_null());

        let result = t4a_mpo_f64_contract_naive(mpo_a as *const _, mpo_b as *const _);
        assert!(!result.is_null());

        // Result[0, 0, 0, 0] = 1.0
        let indices = [0_usize, 0, 0, 0];
        let mut value: f64 = 0.0;
        assert_eq!(
            t4a_mpo_f64_evaluate(result as *const _, indices.as_ptr(), 4, &mut value),
            T4A_SUCCESS
        );
        assert!((value - 1.0).abs() < 1e-10);

        t4a_mpo_f64_release(result);
        t4a_mpo_f64_release(mpo_b);
        t4a_mpo_f64_release(mpo_a);
    }

    #[test]
    fn test_mpo_f64_contract_zipup() {
        let dims = [2_usize, 2];
        let mpo_a = t4a_mpo_f64_new_identity(dims.as_ptr(), 2);
        let mpo_b = t4a_mpo_f64_new_identity(dims.as_ptr(), 2);

        let result = t4a_mpo_f64_contract_zipup(mpo_a as *const _, mpo_b as *const _, 1e-12, 0);
        assert!(!result.is_null());

        let indices = [0_usize, 0, 0, 0];
        let mut value: f64 = 0.0;
        t4a_mpo_f64_evaluate(result as *const _, indices.as_ptr(), 4, &mut value);
        assert!((value - 1.0).abs() < 1e-10);

        t4a_mpo_f64_release(result);
        t4a_mpo_f64_release(mpo_b);
        t4a_mpo_f64_release(mpo_a);
    }
}
