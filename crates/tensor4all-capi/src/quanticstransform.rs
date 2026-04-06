//! C API for QuanticsTransform operators
//!
//! This provides C-compatible interface for constructing and applying quantics
//! transformation operators (shift, flip, phase rotation, cumulative sum, Fourier).
//! These wrap `tensor4all-quanticstransform` which provides `LinearOperator` constructors.

use crate::types::{t4a_boundary_condition, t4a_linop, t4a_treetn};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use num_rational::Rational64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_quanticstransform::{
    affine_operator, binaryop_operator, cumsum_operator, flip_operator, flip_operator_multivar,
    phase_rotation_operator, phase_rotation_operator_multivar, quantics_fourier_operator,
    shift_operator, shift_operator_multivar, AffineParams, BinaryCoeffs, BoundaryCondition,
    FourierOptions,
};
use tensor4all_treetn::treetn::contraction::ContractionMethod;
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

// ============================================================================
// Lifecycle functions
// ============================================================================

impl_opaque_type_common!(linop);

fn boundary_conditions_from_raw(
    ptr: *const t4a_boundary_condition,
    len: usize,
) -> Result<Vec<BoundaryCondition>, StatusCode> {
    if ptr.is_null() {
        return Err(T4A_NULL_POINTER);
    }

    let bc = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bc.iter().copied().map(Into::into).collect())
}

fn rationals_from_raw(
    num_ptr: *const i64,
    den_ptr: *const i64,
    len: usize,
) -> Result<Vec<Rational64>, StatusCode> {
    if num_ptr.is_null() || den_ptr.is_null() {
        return Err(T4A_NULL_POINTER);
    }

    let nums = unsafe { std::slice::from_raw_parts(num_ptr, len) };
    let dens = unsafe { std::slice::from_raw_parts(den_ptr, len) };

    let mut out = Vec::with_capacity(len);
    for (i, (&num, &den)) in nums.iter().zip(dens.iter()).enumerate() {
        if den == 0 {
            crate::set_last_error(&format!(
                "Rational denominator at index {i} must be nonzero"
            ));
            return Err(T4A_INVALID_ARGUMENT);
        }
        out.push(Rational64::new(num, den));
    }
    Ok(out)
}

// ============================================================================
// Operator construction
// ============================================================================

/// Create a shift operator: f(x) = g(x + offset) mod 2^r
///
/// # Arguments
/// * `r` - Number of quantics bits
/// * `offset` - Shift offset (can be negative)
/// * `bc` - Boundary condition (Periodic or Open)
/// * `out` - Output: new linop handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_shift(
    r: libc::size_t,
    offset: i64,
    bc: t4a_boundary_condition,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let bc_rust: BoundaryCondition = bc.into();
        match shift_operator(r, offset, bc_rust) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a shift operator for one variable in a multi-variable system.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_shift_multivar(
    r: libc::size_t,
    offset: i64,
    bc: t4a_boundary_condition,
    nvariables: libc::size_t,
    target_var: libc::size_t,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 || nvariables == 0 {
        return T4A_INVALID_ARGUMENT;
    }
    if target_var >= nvariables {
        return crate::err_status(
            "target_var must be smaller than nvariables",
            T4A_INVALID_ARGUMENT,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let bc_rust: BoundaryCondition = bc.into();
        match shift_operator_multivar(r, offset, bc_rust, nvariables, target_var) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a flip operator: f(x) = g(2^r - x)
///
/// # Arguments
/// * `r` - Number of quantics bits
/// * `bc` - Boundary condition (Periodic or Open)
/// * `out` - Output: new linop handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_flip(
    r: libc::size_t,
    bc: t4a_boundary_condition,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let bc_rust: BoundaryCondition = bc.into();
        match flip_operator(r, bc_rust) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a flip operator for one variable in a multi-variable system.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_flip_multivar(
    r: libc::size_t,
    bc: t4a_boundary_condition,
    nvariables: libc::size_t,
    target_var: libc::size_t,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 || nvariables == 0 {
        return T4A_INVALID_ARGUMENT;
    }
    if target_var >= nvariables {
        return crate::err_status(
            "target_var must be smaller than nvariables",
            T4A_INVALID_ARGUMENT,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let bc_rust: BoundaryCondition = bc.into();
        match flip_operator_multivar(r, bc_rust, nvariables, target_var) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a phase rotation operator: f(x) = exp(i*theta*x) * g(x)
///
/// # Arguments
/// * `r` - Number of quantics bits
/// * `theta` - Phase angle
/// * `out` - Output: new linop handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_phase_rotation(
    r: libc::size_t,
    theta: f64,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        match phase_rotation_operator(r, theta) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a phase rotation operator for one variable in a multi-variable system.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_phase_rotation_multivar(
    r: libc::size_t,
    theta: f64,
    nvariables: libc::size_t,
    target_var: libc::size_t,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 || nvariables == 0 {
        return T4A_INVALID_ARGUMENT;
    }
    if target_var >= nvariables {
        return crate::err_status(
            "target_var must be smaller than nvariables",
            T4A_INVALID_ARGUMENT,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| match phase_rotation_operator_multivar(
        r, theta, nvariables, target_var,
    ) {
        Ok(op) => {
            unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
            T4A_SUCCESS
        }
        Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
    }));

    crate::unwrap_catch(result)
}

/// Create a cumulative sum operator: y_i = sum_{j<i} x_j
///
/// # Arguments
/// * `r` - Number of quantics bits
/// * `out` - Output: new linop handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_cumsum(r: libc::size_t, out: *mut *mut t4a_linop) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| match cumsum_operator(r) {
        Ok(op) => {
            unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
            T4A_SUCCESS
        }
        Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
    }));

    crate::unwrap_catch(result)
}

/// Create a general affine transformation operator.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_affine(
    r: libc::size_t,
    a_num: *const i64,
    a_den: *const i64,
    b_num: *const i64,
    b_den: *const i64,
    m: libc::size_t,
    n: libc::size_t,
    bc: *const t4a_boundary_condition,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 || m == 0 || n == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let a = match rationals_from_raw(a_num, a_den, m * n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let b = match rationals_from_raw(b_num, b_den, m) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let bc = match boundary_conditions_from_raw(bc, m) {
        Ok(v) => v,
        Err(code) => return code,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let params = match AffineParams::new(a, b, m, n) {
            Ok(params) => params,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };
        match affine_operator(r, &params, &bc) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a two-output binary operation operator.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_binaryop(
    r: libc::size_t,
    a1: i8,
    b1: i8,
    a2: i8,
    b2: i8,
    bc1: t4a_boundary_condition,
    bc2: t4a_boundary_condition,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let coeffs1 = match BinaryCoeffs::new(a1, b1) {
            Ok(coeffs) => coeffs,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };
        let coeffs2 = match BinaryCoeffs::new(a2, b2) {
            Ok(coeffs) => coeffs,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };
        let bc = [bc1.into(), bc2.into()];

        match binaryop_operator(r, coeffs1, coeffs2, bc) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Create a Fourier transform operator.
///
/// # Arguments
/// * `r` - Number of quantics bits
/// * `forward` - 1 for forward transform, 0 for inverse
/// * `maxbonddim` - Maximum bond dimension (0 = default of 12)
/// * `tolerance` - Compression tolerance (0.0 = default of 1e-14)
/// * `out` - Output: new linop handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_fourier(
    r: libc::size_t,
    forward: i32,
    maxbonddim: libc::size_t,
    tolerance: f64,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if r == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut options = if forward != 0 {
            FourierOptions::forward()
        } else {
            FourierOptions::inverse()
        };

        if maxbonddim > 0 {
            options.maxbonddim = maxbonddim;
        }
        if tolerance > 0.0 {
            options.tolerance = tolerance;
        }

        match quantics_fourier_operator(r, options) {
            Ok(op) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Operator application
// ============================================================================

/// Apply a linear operator to a TreeTN (MPS) state.
///
/// Computes `result = operator * state`.
///
/// # Arguments
/// * `op` - Linear operator handle
/// * `state` - Input TreeTN state
/// * `method` - Contraction method: 0=Naive, 1=Zipup, 2=Fit
/// * `rtol` - Relative tolerance for truncation (0.0 = no truncation)
/// * `maxdim` - Maximum bond dimension (0 = unlimited)
/// * `out` - Output: new TreeTN handle (caller owns)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_linop_apply(
    op: *const t4a_linop,
    state: *const t4a_treetn,
    method: i32,
    rtol: f64,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if op.is_null() || state.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let contraction_method = match method {
        0 => ContractionMethod::Naive,
        1 => ContractionMethod::Zipup,
        2 => ContractionMethod::Fit,
        _ => return T4A_INVALID_ARGUMENT,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let op_ref = unsafe { &*op };
        let state_ref = unsafe { &*state };

        let mut options = ApplyOptions {
            method: contraction_method,
            ..ApplyOptions::default()
        };

        if rtol > 0.0 {
            options.rtol = Some(rtol);
        }
        if maxdim > 0 {
            options.max_rank = Some(maxdim);
        }

        match apply_linear_operator(op_ref.inner(), state_ref.inner(), options) {
            Ok(result_treetn) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(result_treetn))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Reset the operator's true input site indices to match a TreeTN state.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_linop_set_input_space(
    op: *mut t4a_linop,
    state: *const t4a_treetn,
) -> StatusCode {
    if op.is_null() || state.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let op_ref = unsafe { &mut *op };
        let state_ref = unsafe { &*state };
        match op_ref
            .inner_mut()
            .set_input_space_from_state(state_ref.inner())
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

/// Reset the operator's true output site indices to match a TreeTN state.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_linop_set_output_space(
    op: *mut t4a_linop,
    state: *const t4a_treetn,
) -> StatusCode {
    if op.is_null() || state.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let op_ref = unsafe { &mut *op };
        let state_ref = unsafe { &*state };
        match op_ref
            .inner_mut()
            .set_output_space_from_state(state_ref.inner())
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
