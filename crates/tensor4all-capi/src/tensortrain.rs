//! C API for TensorTrain operations
//!
//! Provides functions to create, manipulate, and query tensor trains.

use crate::types::{
    t4a_canonical_form, t4a_index, t4a_tensor, t4a_tensortrain, InternalTensorTrain,
};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::panic::{catch_unwind, AssertUnwindSafe};

use tensor4all_itensorlike::{CanonicalForm, TruncateOptions};

// ============================================================================
// Lifecycle functions
// ============================================================================

impl_opaque_type_common!(tensortrain);

// ============================================================================
// Constructors
// ============================================================================

/// Create a tensor train from an array of tensors.
///
/// # Arguments
/// * `tensors` - Array of tensor pointers
/// * `num_tensors` - Number of tensors in the array
///
/// # Returns
/// A new tensor train, or NULL on error.
///
/// # Safety
/// - `tensors` must be a valid pointer to an array of `num_tensors` t4a_tensor pointers
/// - All tensor pointers must be valid
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_new(
    tensors: *const *const t4a_tensor,
    num_tensors: libc::size_t,
) -> *mut t4a_tensortrain {
    if tensors.is_null() && num_tensors > 0 {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Collect tensors
        let tensor_vec: Vec<_> = (0..num_tensors)
            .map(|i| unsafe {
                let tensor_ptr = *tensors.add(i);
                (*tensor_ptr).inner().clone()
            })
            .collect();

        // Create tensor train
        match InternalTensorTrain::new(tensor_vec) {
            Ok(tt) => Box::into_raw(Box::new(t4a_tensortrain::new(tt))),
            Err(_) => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

/// Create an empty tensor train.
///
/// # Returns
/// A new empty tensor train, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_new_empty() -> *mut t4a_tensortrain {
    let result = catch_unwind(AssertUnwindSafe(|| {
        match InternalTensorTrain::new(vec![]) {
            Ok(tt) => Box::into_raw(Box::new(t4a_tensortrain::new(tt))),
            Err(_) => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the number of sites in the tensor train.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_len` - Output pointer for the length
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_len(
    ptr: *const t4a_tensortrain,
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Check if the tensor train is empty.
///
/// # Arguments
/// * `ptr` - Tensor train handle
///
/// # Returns
/// 1 if empty, 0 if not empty, negative on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_is_empty(ptr: *const t4a_tensortrain) -> libc::c_int {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        if tt.inner().is_empty() {
            1
        } else {
            0
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the tensor at a specific site.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `site` - Site index (0-based)
///
/// # Returns
/// A new tensor handle (caller owns it), or NULL on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_tensor(
    ptr: *const t4a_tensortrain,
    site: libc::size_t,
) -> *mut t4a_tensor {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        if site >= tt.inner().len() {
            return std::ptr::null_mut();
        }
        let tensor = tt.inner().tensor(site).clone();
        Box::into_raw(Box::new(t4a_tensor::new(tensor)))
    }));

    result.unwrap_or(std::ptr::null_mut())
}

/// Set the tensor at a specific site.
///
/// This replaces the tensor at the given site and invalidates orthogonality.
///
/// # Arguments
/// * `ptr` - Tensor train handle (modified in place)
/// * `site` - Site index (0-based)
/// * `tensor` - New tensor to set at the site (the tensor is cloned)
///
/// # Returns
/// Status code
///
/// # Safety
/// - `ptr` must be a valid tensor train pointer
/// - `tensor` must be a valid tensor pointer
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_set_tensor(
    ptr: *mut t4a_tensortrain,
    site: libc::size_t,
    tensor: *const t4a_tensor,
) -> StatusCode {
    if ptr.is_null() || tensor.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        let tensor_inner = unsafe { &*tensor };

        if site >= tt.inner().len() {
            return T4A_INVALID_ARGUMENT;
        }

        tt.inner_mut()
            .set_tensor(site, tensor_inner.inner().clone());
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the bond dimensions of the tensor train.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_dims` - Output buffer for bond dimensions (length = len - 1)
/// * `buf_len` - Length of the output buffer
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_bond_dims(
    ptr: *const t4a_tensortrain,
    out_dims: *mut libc::size_t,
    buf_len: libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_dims.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        let dims = tt.inner().bond_dims();

        if buf_len < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }

        for (i, &d) in dims.iter().enumerate() {
            unsafe { *out_dims.add(i) = d };
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the maximum bond dimension of the tensor train.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_max` - Output pointer for the maximum bond dimension
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_maxbonddim(
    ptr: *const t4a_tensortrain,
    out_max: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_max.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_max = tt.inner().maxbonddim() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the link index between two adjacent sites.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `site` - Site index (returns link between site and site+1)
///
/// # Returns
/// A new index handle (caller owns it), or NULL if no link exists or on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_linkind(
    ptr: *const t4a_tensortrain,
    site: libc::size_t,
) -> *mut t4a_index {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        match tt.inner().linkind(site) {
            Some(idx) => Box::into_raw(Box::new(t4a_index::new(idx))),
            None => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

// ============================================================================
// Orthogonality tracking
// ============================================================================

/// Check if the tensor train has a single orthogonality center.
///
/// # Arguments
/// * `ptr` - Tensor train handle
///
/// # Returns
/// 1 if orthogonalized, 0 if not, negative on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_isortho(ptr: *const t4a_tensortrain) -> libc::c_int {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        if tt.inner().isortho() {
            1
        } else {
            0
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the orthogonality center (0-indexed).
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_center` - Output pointer for the center site
///
/// # Returns
/// Status code (T4A_INVALID_ARGUMENT if no single center exists)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_orthocenter(
    ptr: *const t4a_tensortrain,
    out_center: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_center.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        match tt.inner().orthocenter() {
            Some(center) => {
                unsafe { *out_center = center };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the left orthogonality limit.
///
/// Sites 0..llim are guaranteed to be left-orthogonal.
/// Returns -1 if no sites are left-orthogonal.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_llim` - Output pointer for llim
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_llim(
    ptr: *const t4a_tensortrain,
    out_llim: *mut libc::c_int,
) -> StatusCode {
    if ptr.is_null() || out_llim.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_llim = tt.inner().llim() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the right orthogonality limit.
///
/// Sites rlim..len are guaranteed to be right-orthogonal.
/// Returns len+1 if no sites are right-orthogonal.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_rlim` - Output pointer for rlim
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_rlim(
    ptr: *const t4a_tensortrain,
    out_rlim: *mut libc::c_int,
) -> StatusCode {
    if ptr.is_null() || out_rlim.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_rlim = tt.inner().rlim() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Get the canonical form used for the tensor train.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_form` - Output pointer for the canonical form
///
/// # Returns
/// Status code (T4A_INVALID_ARGUMENT if no canonical form is set)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_canonical_form(
    ptr: *const t4a_tensortrain,
    out_form: *mut t4a_canonical_form,
) -> StatusCode {
    if ptr.is_null() || out_form.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        match tt.inner().canonical_form() {
            Some(form) => {
                unsafe { *out_form = form.into() };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Operations
// ============================================================================

/// Orthogonalize the tensor train to have orthogonality center at the given site.
///
/// Uses QR decomposition (Unitary canonical form) by default.
///
/// # Arguments
/// * `ptr` - Tensor train handle (modified in place)
/// * `site` - Target site for orthogonality center (0-indexed)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_orthogonalize(
    ptr: *mut t4a_tensortrain,
    site: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        match tt.inner_mut().orthogonalize(site) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Orthogonalize the tensor train with a specific canonical form.
///
/// # Arguments
/// * `ptr` - Tensor train handle (modified in place)
/// * `site` - Target site for orthogonality center (0-indexed)
/// * `form` - Canonical form to use
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_orthogonalize_with(
    ptr: *mut t4a_tensortrain,
    site: libc::size_t,
    form: t4a_canonical_form,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };
        let rust_form: CanonicalForm = form.into();
        match tt.inner_mut().orthogonalize_with(site, rust_form) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Truncate the tensor train bond dimensions.
///
/// # Arguments
/// * `ptr` - Tensor train handle (modified in place)
/// * `rtol` - Relative tolerance for truncation (use 0.0 for default)
/// * `max_rank` - Maximum bond dimension (use 0 for no limit)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_truncate(
    ptr: *mut t4a_tensortrain,
    rtol: libc::c_double,
    max_rank: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &mut *ptr };

        let mut options = TruncateOptions::svd();
        if rtol > 0.0 {
            options = options.with_rtol(rtol);
        }
        if max_rank > 0 {
            options = options.with_max_rank(max_rank);
        }

        match tt.inner_mut().truncate(&options) {
            Ok(()) => T4A_SUCCESS,
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the norm of the tensor train.
///
/// # Arguments
/// * `ptr` - Tensor train handle
/// * `out_norm` - Output pointer for the norm
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_norm(
    ptr: *const t4a_tensortrain,
    out_norm: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out_norm.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tt = unsafe { &*ptr };
        unsafe { *out_norm = tt.inner().norm() };
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Compute the inner product of two tensor trains.
///
/// Computes <self | other> = sum over all indices of conj(self) * other.
///
/// # Arguments
/// * `ptr1` - First tensor train handle
/// * `ptr2` - Second tensor train handle
/// * `out_re` - Output pointer for real part
/// * `out_im` - Output pointer for imaginary part
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_inner(
    ptr1: *const t4a_tensortrain,
    ptr2: *const t4a_tensortrain,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr1.is_null() || ptr2.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_core::AnyScalar;

        let tt1 = unsafe { &*ptr1 };
        let tt2 = unsafe { &*ptr2 };

        let inner = tt1.inner().inner(tt2.inner());
        unsafe {
            match inner {
                AnyScalar::F64(x) => {
                    *out_re = x;
                    *out_im = 0.0;
                }
                AnyScalar::C64(z) => {
                    *out_re = z.re;
                    *out_im = z.im;
                }
            }
        }
        T4A_SUCCESS
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

/// Contract two tensor trains.
///
/// Both tensor trains must have the same site indices.
///
/// # Arguments
/// * `ptr1` - First tensor train handle
/// * `ptr2` - Second tensor train handle
/// * `method` - Contract method (Zipup=0, Fit=1)
/// * `max_rank` - Maximum bond dimension (0 for no limit)
/// * `rtol` - Relative tolerance (0.0 for default)
/// * `nhalfsweeps` - Number of half-sweeps for Fit method (must be a multiple of 2)
///
/// # Returns
/// A new tensor train handle, or NULL on error
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tt_contract(
    ptr1: *const t4a_tensortrain,
    ptr2: *const t4a_tensortrain,
    method: crate::types::t4a_contract_method,
    max_rank: libc::size_t,
    rtol: libc::c_double,
    nhalfsweeps: libc::size_t,
) -> *mut t4a_tensortrain {
    if ptr1.is_null() || ptr2.is_null() {
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_itensorlike::{ContractMethod, ContractOptions};

        let tt1 = unsafe { &*ptr1 };
        let tt2 = unsafe { &*ptr2 };

        // Build options
        let rust_method: ContractMethod = method.into();
        let mut options = match rust_method {
            ContractMethod::Zipup => ContractOptions::zipup(),
            ContractMethod::Fit => ContractOptions::fit(),
            ContractMethod::Naive => ContractOptions::naive(),
        };

        if max_rank > 0 {
            options = options.with_max_rank(max_rank);
        }
        if rtol > 0.0 {
            options = options.with_rtol(rtol);
        }
        if nhalfsweeps > 0 {
            // nhalfsweeps must be a multiple of 2
            // Round up to nearest even number if odd
            let nhalfsweeps_even = if nhalfsweeps.is_multiple_of(2) {
                nhalfsweeps
            } else {
                nhalfsweeps + 1
            };
            options = options.with_nhalfsweeps(nhalfsweeps_even);
        }

        match tt1.inner().contract(tt2.inner(), &options) {
            Ok(result) => Box::into_raw(Box::new(t4a_tensortrain::new(result))),
            Err(_) => std::ptr::null_mut(),
        }
    }));

    result.unwrap_or(std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_lifecycle() {
        // Create empty tensor train
        let tt = t4a_tt_new_empty();
        assert!(!tt.is_null());

        // Check it's empty
        let is_empty = t4a_tt_is_empty(tt);
        assert_eq!(is_empty, 1);

        // Get length
        let mut len: libc::size_t = 0;
        let status = t4a_tt_len(tt, &mut len);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(len, 0);

        // Clone
        let tt2 = t4a_tensortrain_clone(tt);
        assert!(!tt2.is_null());

        // Release both
        t4a_tensortrain_release(tt);
        t4a_tensortrain_release(tt2);
    }

    #[test]
    fn test_tt_from_tensors() {
        use crate::{t4a_index_new, t4a_tensor_new_dense_f64};

        // Create indices
        let s0 = t4a_index_new(2);
        let l01 = t4a_index_new(3);
        let s1 = t4a_index_new(2);

        // Create tensor data
        let data0: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let data1: Vec<f64> = (0..6).map(|i| i as f64).collect();

        // Create tensors - need const pointers
        let inds0: [*const t4a_index; 2] = [s0, l01];
        let dims0: [libc::size_t; 2] = [2, 3];
        let t0 = t4a_tensor_new_dense_f64(2, inds0.as_ptr(), dims0.as_ptr(), data0.as_ptr(), 6);

        let l01_clone = crate::t4a_index_clone(l01);
        let inds1: [*const t4a_index; 2] = [l01_clone, s1];
        let dims1: [libc::size_t; 2] = [3, 2];
        let t1 = t4a_tensor_new_dense_f64(2, inds1.as_ptr(), dims1.as_ptr(), data1.as_ptr(), 6);

        // Create tensor train
        let tensors: [*const t4a_tensor; 2] = [t0, t1];
        let tt = t4a_tt_new(tensors.as_ptr(), 2);
        assert!(!tt.is_null());

        // Check length
        let mut len: libc::size_t = 0;
        t4a_tt_len(tt, &mut len);
        assert_eq!(len, 2);

        // Check bond dimension
        let mut max_bond: libc::size_t = 0;
        t4a_tt_maxbonddim(tt, &mut max_bond);
        assert_eq!(max_bond, 3);

        // Cleanup
        t4a_tensortrain_release(tt);
        crate::t4a_tensor_release(t0);
        crate::t4a_tensor_release(t1);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(l01);
        crate::t4a_index_release(l01_clone);
        crate::t4a_index_release(s1);
    }

    #[test]
    fn test_tt_contract_rounds_up_odd_nhalfsweeps() {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        // Build two 1-site tensor trains sharing the same site index id
        let s0 = t4a_index_new(2);
        let s0_clone = t4a_index_clone(s0);

        let data_a: Vec<f64> = vec![1.0, 2.0];
        let data_b: Vec<f64> = vec![3.0, 4.0];

        let inds_a: [*const t4a_index; 1] = [s0];
        let dims_a: [libc::size_t; 1] = [2];
        let t0 = t4a_tensor_new_dense_f64(1, inds_a.as_ptr(), dims_a.as_ptr(), data_a.as_ptr(), 2);

        let inds_b: [*const t4a_index; 1] = [s0_clone];
        let dims_b: [libc::size_t; 1] = [2];
        let t1 = t4a_tensor_new_dense_f64(1, inds_b.as_ptr(), dims_b.as_ptr(), data_b.as_ptr(), 2);

        let tensors0: [*const t4a_tensor; 1] = [t0];
        let tensors1: [*const t4a_tensor; 1] = [t1];
        let tt0 = t4a_tt_new(tensors0.as_ptr(), 1);
        let tt1 = t4a_tt_new(tensors1.as_ptr(), 1);
        assert!(!tt0.is_null());
        assert!(!tt1.is_null());

        // Pass odd nhalfsweeps (= 1). The C API should round it up to 2 and not panic.
        let result = t4a_tt_contract(
            tt0,
            tt1,
            crate::types::t4a_contract_method::Zipup,
            0,
            0.0,
            1,
        );
        assert!(!result.is_null());
        let mut len: libc::size_t = 0;
        let status = t4a_tt_len(result, &mut len);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(len, 1);

        // Cleanup
        t4a_tensortrain_release(result);
        t4a_tensortrain_release(tt0);
        t4a_tensortrain_release(tt1);
        crate::t4a_tensor_release(t0);
        crate::t4a_tensor_release(t1);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(s0_clone);
    }

    #[test]
    fn test_tt_contract_with_even_nhalfsweeps() {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        // Build two 1-site tensor trains sharing the same site index id
        let s0 = t4a_index_new(2);
        let s0_clone = t4a_index_clone(s0);

        let data_a: Vec<f64> = vec![1.0, 2.0];
        let data_b: Vec<f64> = vec![3.0, 4.0];

        let inds_a: [*const t4a_index; 1] = [s0];
        let dims_a: [libc::size_t; 1] = [2];
        let t0 = t4a_tensor_new_dense_f64(1, inds_a.as_ptr(), dims_a.as_ptr(), data_a.as_ptr(), 2);

        let inds_b: [*const t4a_index; 1] = [s0_clone];
        let dims_b: [libc::size_t; 1] = [2];
        let t1 = t4a_tensor_new_dense_f64(1, inds_b.as_ptr(), dims_b.as_ptr(), data_b.as_ptr(), 2);

        let tensors0: [*const t4a_tensor; 1] = [t0];
        let tensors1: [*const t4a_tensor; 1] = [t1];
        let tt0 = t4a_tt_new(tensors0.as_ptr(), 1);
        let tt1 = t4a_tt_new(tensors1.as_ptr(), 1);
        assert!(!tt0.is_null());
        assert!(!tt1.is_null());

        // Pass even nhalfsweeps (= 4). The C API should use it as-is without rounding.
        let result = t4a_tt_contract(
            tt0,
            tt1,
            crate::types::t4a_contract_method::Fit,
            10,
            0.0,
            4, // Even number, should not be rounded
        );
        assert!(!result.is_null());
        let mut len: libc::size_t = 0;
        let status = t4a_tt_len(result, &mut len);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(len, 1);

        // Cleanup
        t4a_tensortrain_release(result);
        t4a_tensortrain_release(tt0);
        t4a_tensortrain_release(tt1);
        crate::t4a_tensor_release(t0);
        crate::t4a_tensor_release(t1);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(s0_clone);
    }
}
