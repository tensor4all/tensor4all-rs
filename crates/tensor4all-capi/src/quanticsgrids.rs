//! C API for quanticsgrids crate
//!
//! Provides functions for creating and using DiscretizedGrid and InherentDiscreteGrid
//! for quantics tensor train representations.

use crate::types::{t4a_qgrid_disc, t4a_qgrid_int, t4a_unfolding_scheme};
use crate::{StatusCode, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};

// ============================================================================
// DiscretizedGrid lifecycle functions
// ============================================================================

impl_opaque_type_common!(qgrid_disc);

/// Create a new DiscretizedGrid.
///
/// # Arguments
/// * `ndims` - Number of dimensions
/// * `rs_arr` - Resolution (bits) per dimension, array of length `ndims`
/// * `lower_arr` - Lower bounds per dimension, array of length `ndims`
/// * `upper_arr` - Upper bounds per dimension, array of length `ndims`
/// * `unfolding` - Unfolding scheme (0=Fused, 1=Interleaved)
/// * `out` - Output pointer for the created grid
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_new(
    ndims: libc::size_t,
    rs_arr: *const libc::size_t,
    lower_arr: *const libc::c_double,
    upper_arr: *const libc::c_double,
    unfolding: t4a_unfolding_scheme,
    out: *mut *mut t4a_qgrid_disc,
) -> StatusCode {
    if rs_arr.is_null() || lower_arr.is_null() || upper_arr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }
    if ndims == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let rs: Vec<usize> = unsafe { std::slice::from_raw_parts(rs_arr, ndims) }.to_vec();
        let lower: Vec<f64> = unsafe { std::slice::from_raw_parts(lower_arr, ndims) }.to_vec();
        let upper: Vec<f64> = unsafe { std::slice::from_raw_parts(upper_arr, ndims) }.to_vec();
        let scheme: quanticsgrids::UnfoldingScheme = unfolding.into();

        let grid = quanticsgrids::DiscretizedGrid::builder(&rs)
            .with_lower_bound(&lower)
            .with_upper_bound(&upper)
            .with_unfolding_scheme(scheme)
            .build();

        match grid {
            Ok(g) => {
                let boxed = Box::new(t4a_qgrid_disc::new(g));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// DiscretizedGrid property accessors
// ============================================================================

/// Get the number of dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_ndims(
    grid: *const t4a_qgrid_disc,
    out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        *out = g.inner().ndims();
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the resolution (bits) per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_rs(
    grid: *const t4a_qgrid_disc,
    out_arr: *mut libc::size_t,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let rs = g.inner().rs();
        if buf_size < rs.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, rs.len());
        out_slice.copy_from_slice(rs);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the local dimensions of all tensor sites.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_local_dims(
    grid: *const t4a_qgrid_disc,
    out_arr: *mut libc::size_t,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let dims = g.inner().local_dimensions();
        *n_out = dims.len();
        if buf_size < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, dims.len());
        out_slice.copy_from_slice(&dims);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the lower bounds per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_lower_bound(
    grid: *const t4a_qgrid_disc,
    out_arr: *mut libc::c_double,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let lb = g.inner().lower_bound();
        if buf_size < lb.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, lb.len());
        out_slice.copy_from_slice(lb);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the upper bounds per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_upper_bound(
    grid: *const t4a_qgrid_disc,
    out_arr: *mut libc::c_double,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let ub = g.inner().upper_bound();
        if buf_size < ub.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, ub.len());
        out_slice.copy_from_slice(ub);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the grid step per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_grid_step(
    grid: *const t4a_qgrid_disc,
    out_arr: *mut libc::c_double,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let step = g.inner().grid_step();
        if buf_size < step.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, step.len());
        out_slice.copy_from_slice(&step);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

// ============================================================================
// DiscretizedGrid coordinate conversions
// ============================================================================

/// Convert original (continuous) coordinates to quantics indices.
///
/// # Arguments
/// * `grid` - Grid handle
/// * `coord_arr` - Input coordinates, flat array of length `ndims`
/// * `ndims` - Number of dimensions (length of coord_arr)
/// * `out_arr` - Output quantics indices buffer
/// * `buf_size` - Size of output buffer
/// * `n_out` - Actual number of quantics indices written
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_origcoord_to_quantics(
    grid: *const t4a_qgrid_disc,
    coord_arr: *const libc::c_double,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || coord_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let coord = std::slice::from_raw_parts(coord_arr, ndims);
        match g.inner().origcoord_to_quantics(coord) {
            Ok(quantics) => {
                *n_out = quantics.len();
                if buf_size < quantics.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, quantics.len());
                out_slice.copy_from_slice(&quantics);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert quantics indices to original (continuous) coordinates.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_quantics_to_origcoord(
    grid: *const t4a_qgrid_disc,
    quantics_arr: *const i64,
    n_quantics: libc::size_t,
    out_arr: *mut libc::c_double,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || quantics_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let quantics = std::slice::from_raw_parts(quantics_arr, n_quantics);
        match g.inner().quantics_to_origcoord(quantics) {
            Ok(coord) => {
                if buf_size < coord.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, coord.len());
                out_slice.copy_from_slice(&coord);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert original coordinates to grid indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_origcoord_to_grididx(
    grid: *const t4a_qgrid_disc,
    coord_arr: *const libc::c_double,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || coord_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let coord = std::slice::from_raw_parts(coord_arr, ndims);
        match g.inner().origcoord_to_grididx(coord) {
            Ok(grididx) => {
                if buf_size < grididx.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, grididx.len());
                out_slice.copy_from_slice(&grididx);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert grid indices to original coordinates.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_grididx_to_origcoord(
    grid: *const t4a_qgrid_disc,
    grididx_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut libc::c_double,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || grididx_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let grididx = std::slice::from_raw_parts(grididx_arr, ndims);
        match g.inner().grididx_to_origcoord(grididx) {
            Ok(coord) => {
                if buf_size < coord.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, coord.len());
                out_slice.copy_from_slice(&coord);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert grid indices to quantics indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_grididx_to_quantics(
    grid: *const t4a_qgrid_disc,
    grididx_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || grididx_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let grididx = std::slice::from_raw_parts(grididx_arr, ndims);
        match g.inner().grididx_to_quantics(grididx) {
            Ok(quantics) => {
                *n_out = quantics.len();
                if buf_size < quantics.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, quantics.len());
                out_slice.copy_from_slice(&quantics);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert quantics indices to grid indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_disc_quantics_to_grididx(
    grid: *const t4a_qgrid_disc,
    quantics_arr: *const i64,
    n_quantics: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || quantics_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let quantics = std::slice::from_raw_parts(quantics_arr, n_quantics);
        match g.inner().quantics_to_grididx(quantics) {
            Ok(grididx) => {
                *n_out = grididx.len();
                if buf_size < grididx.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, grididx.len());
                out_slice.copy_from_slice(&grididx);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

// ============================================================================
// InherentDiscreteGrid lifecycle functions
// ============================================================================

impl_opaque_type_common!(qgrid_int);

/// Create a new InherentDiscreteGrid.
///
/// # Arguments
/// * `ndims` - Number of dimensions
/// * `rs_arr` - Resolution (bits) per dimension, array of length `ndims`
/// * `origin_arr` - Origin per dimension (or null for default [1,1,...])
/// * `unfolding` - Unfolding scheme (0=Fused, 1=Interleaved)
/// * `out` - Output pointer for the created grid
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_new(
    ndims: libc::size_t,
    rs_arr: *const libc::size_t,
    origin_arr: *const i64,
    unfolding: t4a_unfolding_scheme,
    out: *mut *mut t4a_qgrid_int,
) -> StatusCode {
    if rs_arr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }
    if ndims == 0 {
        return T4A_INVALID_ARGUMENT;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let rs: Vec<usize> = unsafe { std::slice::from_raw_parts(rs_arr, ndims) }.to_vec();
        let scheme: quanticsgrids::UnfoldingScheme = unfolding.into();

        let mut builder =
            quanticsgrids::InherentDiscreteGrid::builder(&rs).with_unfolding_scheme(scheme);

        if !origin_arr.is_null() {
            let origin: Vec<i64> =
                unsafe { std::slice::from_raw_parts(origin_arr, ndims) }.to_vec();
            builder = builder.with_origin(&origin);
        }

        match builder.build() {
            Ok(g) => {
                let boxed = Box::new(t4a_qgrid_int::new(g));
                unsafe { *out = Box::into_raw(boxed) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// InherentDiscreteGrid property accessors
// ============================================================================

/// Get the number of dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_ndims(
    grid: *const t4a_qgrid_int,
    out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        *out = g.inner().ndims();
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the resolution (bits) per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_rs(
    grid: *const t4a_qgrid_int,
    out_arr: *mut libc::size_t,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let rs = g.inner().rs();
        if buf_size < rs.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, rs.len());
        out_slice.copy_from_slice(rs);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the local dimensions of all tensor sites.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_local_dims(
    grid: *const t4a_qgrid_int,
    out_arr: *mut libc::size_t,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let dims = g.inner().local_dimensions();
        *n_out = dims.len();
        if buf_size < dims.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, dims.len());
        out_slice.copy_from_slice(&dims);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

/// Get the origin per dimension.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_origin(
    grid: *const t4a_qgrid_int,
    out_arr: *mut i64,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let origin = g.inner().origin();
        if buf_size < origin.len() {
            return T4A_INVALID_ARGUMENT;
        }
        let out_slice = std::slice::from_raw_parts_mut(out_arr, origin.len());
        out_slice.copy_from_slice(origin);
        T4A_SUCCESS
    }));
    crate::unwrap_catch(result)
}

// ============================================================================
// InherentDiscreteGrid coordinate conversions
// ============================================================================

/// Convert original (integer) coordinates to quantics indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_origcoord_to_quantics(
    grid: *const t4a_qgrid_int,
    coord_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || coord_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let coord = std::slice::from_raw_parts(coord_arr, ndims);
        match g.inner().origcoord_to_quantics(coord) {
            Ok(quantics) => {
                *n_out = quantics.len();
                if buf_size < quantics.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, quantics.len());
                out_slice.copy_from_slice(&quantics);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert quantics indices to original (integer) coordinates.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_quantics_to_origcoord(
    grid: *const t4a_qgrid_int,
    quantics_arr: *const i64,
    n_quantics: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || quantics_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let quantics = std::slice::from_raw_parts(quantics_arr, n_quantics);
        match g.inner().quantics_to_origcoord(quantics) {
            Ok(coord) => {
                if buf_size < coord.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, coord.len());
                out_slice.copy_from_slice(&coord);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert original coordinates to grid indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_origcoord_to_grididx(
    grid: *const t4a_qgrid_int,
    coord_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || coord_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let coord = std::slice::from_raw_parts(coord_arr, ndims);
        match g.inner().origcoord_to_grididx(coord) {
            Ok(grididx) => {
                if buf_size < grididx.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, grididx.len());
                out_slice.copy_from_slice(&grididx);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert grid indices to original coordinates.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_grididx_to_origcoord(
    grid: *const t4a_qgrid_int,
    grididx_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
) -> StatusCode {
    if grid.is_null() || grididx_arr.is_null() || out_arr.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let grididx = std::slice::from_raw_parts(grididx_arr, ndims);
        match g.inner().grididx_to_origcoord(grididx) {
            Ok(coord) => {
                if buf_size < coord.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, coord.len());
                out_slice.copy_from_slice(&coord);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert grid indices to quantics indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_grididx_to_quantics(
    grid: *const t4a_qgrid_int,
    grididx_arr: *const i64,
    ndims: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || grididx_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let grididx = std::slice::from_raw_parts(grididx_arr, ndims);
        match g.inner().grididx_to_quantics(grididx) {
            Ok(quantics) => {
                *n_out = quantics.len();
                if buf_size < quantics.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, quantics.len());
                out_slice.copy_from_slice(&quantics);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

/// Convert quantics indices to grid indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qgrid_int_quantics_to_grididx(
    grid: *const t4a_qgrid_int,
    quantics_arr: *const i64,
    n_quantics: libc::size_t,
    out_arr: *mut i64,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if grid.is_null() || quantics_arr.is_null() || out_arr.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let g = &*grid;
        let quantics = std::slice::from_raw_parts(quantics_arr, n_quantics);
        match g.inner().quantics_to_grididx(quantics) {
            Ok(grididx) => {
                *n_out = grididx.len();
                if buf_size < grididx.len() {
                    return T4A_INVALID_ARGUMENT;
                }
                let out_slice = std::slice::from_raw_parts_mut(out_arr, grididx.len());
                out_slice.copy_from_slice(&grididx);
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));
    crate::unwrap_catch(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
