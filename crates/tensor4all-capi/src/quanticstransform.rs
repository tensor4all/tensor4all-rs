//! C API for QuanticsTransform operators
//!
//! This provides C-compatible interface for constructing and applying quantics
//! transformation operators (shift, flip, phase rotation, cumulative sum, Fourier).
//! These wrap `tensor4all-quanticstransform` which provides `LinearOperator` constructors.

use crate::types::{t4a_boundary_condition, t4a_linop, t4a_treetn};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_quanticstransform::{
    cumsum_operator, flip_operator, phase_rotation_operator, quantics_fourier_operator,
    shift_operator, BoundaryCondition, FourierOptions,
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_operator_construction() {
        let mut op: *mut t4a_linop = std::ptr::null_mut();

        // Periodic shift by +1 on 4-bit quantics
        let status = t4a_qtransform_shift(4, 1, t4a_boundary_condition::Periodic, &mut op);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op.is_null());
        t4a_linop_release(op);

        // Open shift
        let mut op2: *mut t4a_linop = std::ptr::null_mut();
        let status = t4a_qtransform_shift(4, -3, t4a_boundary_condition::Open, &mut op2);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op2.is_null());
        t4a_linop_release(op2);
    }

    #[test]
    fn test_flip_operator_construction() {
        let mut op: *mut t4a_linop = std::ptr::null_mut();

        let status = t4a_qtransform_flip(4, t4a_boundary_condition::Periodic, &mut op);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op.is_null());
        t4a_linop_release(op);
    }

    #[test]
    fn test_phase_rotation_construction() {
        let mut op: *mut t4a_linop = std::ptr::null_mut();

        let status = t4a_qtransform_phase_rotation(4, std::f64::consts::PI / 4.0, &mut op);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op.is_null());
        t4a_linop_release(op);
    }

    #[test]
    fn test_cumsum_construction() {
        let mut op: *mut t4a_linop = std::ptr::null_mut();

        let status = t4a_qtransform_cumsum(4, &mut op);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op.is_null());
        t4a_linop_release(op);
    }

    #[test]
    fn test_fourier_operator_construction() {
        // Forward Fourier
        let mut op: *mut t4a_linop = std::ptr::null_mut();
        let status = t4a_qtransform_fourier(4, 1, 0, 0.0, &mut op);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op.is_null());
        t4a_linop_release(op);

        // Inverse Fourier with custom params
        let mut op2: *mut t4a_linop = std::ptr::null_mut();
        let status = t4a_qtransform_fourier(4, 0, 16, 1e-12, &mut op2);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!op2.is_null());
        t4a_linop_release(op2);
    }

    #[test]
    fn test_linop_apply_shift() {
        use num_complex::Complex64;
        use tensor4all_core::index::{DynId, Index, TagSet};
        use tensor4all_core::TensorDynLen;
        use tensor4all_simplett::{
            types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain,
        };
        use tensor4all_treetn::{apply_linear_operator, ApplyOptions, TreeTN};

        type DynIndex = Index<DynId, TagSet>;

        let r = 3;

        // Build the shift operator via Rust API
        let op = shift_operator(r, 1, BoundaryCondition::Periodic)
            .expect("Failed to create shift operator");

        // Create a product state |0⟩ = |0⟩⊗|0⟩⊗|0⟩ (all bits zero)
        let mut tensors_mps: Vec<_> = Vec::with_capacity(r);
        for _ in 0..r {
            let mut t = tensor3_zeros::<Complex64>(1, 2, 1);
            *t.get3_mut(0, 0, 0) = Complex64::new(1.0, 0.0); // bit = 0
            tensors_mps.push(t);
        }
        let mps = TensorTrain::new(tensors_mps).expect("Failed to create MPS");

        // Convert MPS to TreeTN with indices matching operator's input (true_index)
        let n = mps.len();
        let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let dim = if i == 0 {
                1
            } else {
                mps.site_tensor(i - 1).right_dim()
            };
            bond_indices.push(Index::new_dyn(dim));
        }

        let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
        let node_names: Vec<usize> = (0..n).collect();

        for i in 0..n {
            let t = mps.site_tensor(i);
            let site_dim = t.site_dim();
            let right_dim = t.right_dim();
            let left_dim = t.left_dim();

            // Use operator's true_index as the state's site index
            let op_input = op
                .get_input_mapping(&i)
                .expect("input mapping")
                .true_index
                .clone();

            let mut indices: Vec<DynIndex> = Vec::new();
            let mut dims_vec: Vec<usize> = Vec::new();

            if i > 0 {
                indices.push(bond_indices[i].clone());
                dims_vec.push(left_dim);
            }
            indices.push(op_input);
            dims_vec.push(site_dim);
            if i < n - 1 {
                indices.push(bond_indices[i + 1].clone());
                dims_vec.push(right_dim);
            }

            let total_size: usize = dims_vec.iter().product();
            let mut data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total_size];

            #[allow(clippy::needless_range_loop)]
            if i == 0 && n == 1 {
                for s in 0..site_dim {
                    data[s] = *t.get3(0, s, 0);
                }
            } else if i == 0 {
                for s in 0..site_dim {
                    for rv in 0..right_dim {
                        data[s * right_dim + rv] = *t.get3(0, s, rv);
                    }
                }
            } else if i == n - 1 {
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        data[l * site_dim + s] = *t.get3(l, s, 0);
                    }
                }
            } else {
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        for rv in 0..right_dim {
                            data[(l * site_dim + s) * right_dim + rv] = *t.get3(l, s, rv);
                        }
                    }
                }
            }

            tensors.push(TensorDynLen::from_dense_c64(indices, data));
        }

        let treetn = TreeTN::from_tensors(tensors, node_names).expect("Failed to create TreeTN");

        // Test via Rust API first to verify our TreeTN construction is correct
        let result =
            apply_linear_operator(&op, &treetn, ApplyOptions::naive()).expect("Rust apply failed");
        assert!(!result.node_names().is_empty());

        // Now test via C API: wrap in opaque types
        let c_op = Box::into_raw(Box::new(t4a_linop::new(op)));
        let c_state = Box::into_raw(Box::new(t4a_treetn::new(treetn)));

        let mut c_result: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_linop_apply(
            c_op,
            c_state,
            0, // Naive
            0.0,
            0,
            &mut c_result,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!c_result.is_null());

        // Clean up
        unsafe {
            let _ = Box::from_raw(c_result);
            let _ = Box::from_raw(c_state);
            let _ = Box::from_raw(c_op);
        }
    }

    #[test]
    fn test_null_pointer_guards() {
        let mut op: *mut t4a_linop = std::ptr::null_mut();

        // Null output pointer
        assert_eq!(
            t4a_qtransform_shift(4, 1, t4a_boundary_condition::Periodic, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_qtransform_flip(4, t4a_boundary_condition::Periodic, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_qtransform_phase_rotation(4, 1.0, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_qtransform_cumsum(4, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_qtransform_fourier(4, 1, 0, 0.0, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );

        // Invalid r=0
        assert_eq!(
            t4a_qtransform_shift(0, 1, t4a_boundary_condition::Periodic, &mut op),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_qtransform_flip(0, t4a_boundary_condition::Periodic, &mut op),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(
            t4a_qtransform_phase_rotation(0, 1.0, &mut op),
            T4A_INVALID_ARGUMENT
        );
        assert_eq!(t4a_qtransform_cumsum(0, &mut op), T4A_INVALID_ARGUMENT);
        assert_eq!(
            t4a_qtransform_fourier(0, 1, 0, 0.0, &mut op),
            T4A_INVALID_ARGUMENT
        );

        // Apply null guards
        let mut out: *mut t4a_treetn = std::ptr::null_mut();
        assert_eq!(
            t4a_linop_apply(std::ptr::null(), std::ptr::null(), 1, 0.0, 0, &mut out),
            T4A_NULL_POINTER
        );

        // Invalid method
        // We need a valid op and state for this, so skip method check with nulls
        // (it returns NULL_POINTER first)

        // Release null should not crash
        t4a_linop_release(std::ptr::null_mut());
    }
}
