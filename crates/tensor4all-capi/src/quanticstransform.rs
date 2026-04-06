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
    affine_pullback_operator, cumsum_operator, flip_operator, phase_rotation_operator,
    quantics_fourier_operator, shift_operator, AffineParams, BoundaryCondition, FourierOptions,
};
use tensor4all_treetn::treetn::contraction::ContractionMethod;
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

// ============================================================================
// Lifecycle functions
// ============================================================================

impl_opaque_type_common!(linop);

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

/// Create an affine pullback operator: f(y) = g(A*y + b).
///
/// `a_num` and `a_den` encode an MxN matrix in column-major order.
/// `b_num` and `b_den` encode an M-vector.
/// `bc` must have length M and applies to the transformed source coordinates.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_affine_pullback(
    r: libc::size_t,
    m: libc::size_t,
    n: libc::size_t,
    a_num: *const i64,
    a_den: *const i64,
    b_num: *const i64,
    b_den: *const i64,
    bc: *const t4a_boundary_condition,
    out: *mut *mut t4a_linop,
) -> StatusCode {
    if out.is_null()
        || a_num.is_null()
        || a_den.is_null()
        || b_num.is_null()
        || b_den.is_null()
        || bc.is_null()
    {
        return T4A_NULL_POINTER;
    }
    if r == 0 || m == 0 || n == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let a_len = m * n;
        let a_num_slice = unsafe { std::slice::from_raw_parts(a_num, a_len) };
        let a_den_slice = unsafe { std::slice::from_raw_parts(a_den, a_len) };
        let b_num_slice = unsafe { std::slice::from_raw_parts(b_num, m) };
        let b_den_slice = unsafe { std::slice::from_raw_parts(b_den, m) };
        let bc_slice = unsafe { std::slice::from_raw_parts(bc, m) };

        let a = a_num_slice
            .iter()
            .zip(a_den_slice.iter())
            .map(|(&num, &den)| {
                if den == 0 {
                    Err(anyhow::anyhow!("Affine matrix denominator must be non-zero"))
                } else {
                    Ok(Rational64::new(num, den))
                }
            })
            .collect::<Result<Vec<_>, _>>();
        let b = b_num_slice
            .iter()
            .zip(b_den_slice.iter())
            .map(|(&num, &den)| {
                if den == 0 {
                    Err(anyhow::anyhow!("Affine shift denominator must be non-zero"))
                } else {
                    Ok(Rational64::new(num, den))
                }
            })
            .collect::<Result<Vec<_>, _>>();

        let bc_rust: Vec<BoundaryCondition> = bc_slice.iter().copied().map(Into::into).collect();

        match (a, b) {
            (Ok(a_vals), Ok(b_vals)) => match AffineParams::new(a_vals, b_vals, m, n)
                .and_then(|params| affine_pullback_operator(r, &params, &bc_rust))
            {
                Ok(op) => {
                    unsafe { *out = Box::into_raw(Box::new(t4a_linop::new(op))) };
                    T4A_SUCCESS
                }
                Err(e) => crate::err_status(e, T4A_INTERNAL_ERROR),
            },
            (Err(e), _) | (_, Err(e)) => crate::err_status(e, T4A_INVALID_ARGUMENT),
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
        let mut aligned_op = op_ref.inner().clone();

        for (node, mapping) in aligned_op.input_mapping.iter_mut() {
            let state_site = match state_ref.inner().site_space(node) {
                Some(indices) => indices,
                None => {
                    return crate::err_status(
                        anyhow::anyhow!("State node {:?} has no site index", node),
                        T4A_INTERNAL_ERROR,
                    )
                }
            };
            if state_site.len() != 1 {
                return crate::err_status(
                    anyhow::anyhow!(
                        "Expected exactly one site index at node {:?}, got {}",
                        node,
                        state_site.len()
                    ),
                    T4A_INTERNAL_ERROR,
                );
            }
            mapping.true_index = state_site.iter().next().unwrap().clone();
        }

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

        match apply_linear_operator(&aligned_op, state_ref.inner(), options) {
            Ok(result_treetn) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(result_treetn))) };
                T4A_SUCCESS
            }
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
