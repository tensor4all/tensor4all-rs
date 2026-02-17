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
mod tests {
    use super::*;

    // Helper to create a 1D DiscretizedGrid via C API
    fn create_disc_grid_1d(r: usize) -> *mut t4a_qgrid_disc {
        let rs = [r];
        let lower = [0.0f64];
        let upper = [1.0f64];
        let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();
        let status = t4a_qgrid_disc_new(
            1,
            rs.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            t4a_unfolding_scheme::Fused,
            &mut out,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!out.is_null());
        out
    }

    #[test]
    fn test_disc_grid_1d_properties() {
        let grid = create_disc_grid_1d(3);

        // ndims
        let mut ndims: libc::size_t = 0;
        assert_eq!(t4a_qgrid_disc_ndims(grid, &mut ndims), T4A_SUCCESS);
        assert_eq!(ndims, 1);

        // rs
        let mut rs = [0usize; 1];
        assert_eq!(t4a_qgrid_disc_rs(grid, rs.as_mut_ptr(), 1), T4A_SUCCESS);
        assert_eq!(rs[0], 3);

        // lower_bound
        let mut lb = [0.0f64; 1];
        assert_eq!(
            t4a_qgrid_disc_lower_bound(grid, lb.as_mut_ptr(), 1),
            T4A_SUCCESS
        );
        assert!((lb[0] - 0.0).abs() < 1e-10);

        // upper_bound
        let mut ub = [0.0f64; 1];
        assert_eq!(
            t4a_qgrid_disc_upper_bound(grid, ub.as_mut_ptr(), 1),
            T4A_SUCCESS
        );
        assert!((ub[0] - 1.0).abs() < 1e-10);

        // grid_step
        let mut step = [0.0f64; 1];
        assert_eq!(
            t4a_qgrid_disc_grid_step(grid, step.as_mut_ptr(), 1),
            T4A_SUCCESS
        );
        assert!((step[0] - 0.125).abs() < 1e-10); // 1/8

        // local_dims
        let mut dims = [0usize; 8];
        let mut n_out: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_local_dims(grid, dims.as_mut_ptr(), 8, &mut n_out),
            T4A_SUCCESS
        );
        assert_eq!(n_out, 3);
        assert_eq!(&dims[..3], &[2, 2, 2]);

        t4a_qgrid_disc_release(grid);
    }

    #[test]
    fn test_disc_grid_1d_roundtrip() {
        let grid = create_disc_grid_1d(3);
        let ndims = 1;

        // Test origcoord -> quantics -> origcoord roundtrip
        let coord = [0.5f64];
        let mut quantics = [0i64; 8];
        let mut n_out: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_origcoord_to_quantics(
                grid,
                coord.as_ptr(),
                ndims,
                quantics.as_mut_ptr(),
                8,
                &mut n_out
            ),
            T4A_SUCCESS
        );

        let mut coord_back = [0.0f64; 1];
        assert_eq!(
            t4a_qgrid_disc_quantics_to_origcoord(
                grid,
                quantics.as_ptr(),
                n_out,
                coord_back.as_mut_ptr(),
                1
            ),
            T4A_SUCCESS
        );
        assert!((coord_back[0] - 0.5).abs() < 1e-10);

        // Test origcoord -> grididx -> origcoord roundtrip
        let mut grididx = [0i64; 1];
        assert_eq!(
            t4a_qgrid_disc_origcoord_to_grididx(
                grid,
                coord.as_ptr(),
                ndims,
                grididx.as_mut_ptr(),
                1
            ),
            T4A_SUCCESS
        );
        assert_eq!(grididx[0], 5); // 1-indexed, 0.5 maps to grid index 5

        let mut coord_back2 = [0.0f64; 1];
        assert_eq!(
            t4a_qgrid_disc_grididx_to_origcoord(
                grid,
                grididx.as_ptr(),
                ndims,
                coord_back2.as_mut_ptr(),
                1
            ),
            T4A_SUCCESS
        );
        assert!((coord_back2[0] - 0.5).abs() < 1e-10);

        t4a_qgrid_disc_release(grid);
    }

    #[test]
    fn test_disc_grid_2d_interleaved() {
        let rs = [3usize, 2];
        let lower = [0.0f64, 0.0];
        let upper = [1.0f64, 1.0];
        let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();

        let status = t4a_qgrid_disc_new(
            2,
            rs.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            t4a_unfolding_scheme::Interleaved,
            &mut out,
        );
        assert_eq!(status, T4A_SUCCESS);

        let mut ndims: libc::size_t = 0;
        assert_eq!(t4a_qgrid_disc_ndims(out, &mut ndims), T4A_SUCCESS);
        assert_eq!(ndims, 2);

        // Test roundtrip for a 2D coordinate
        let coord = [0.25f64, 0.5];
        let mut quantics = [0i64; 16];
        let mut n_out: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_origcoord_to_quantics(
                out,
                coord.as_ptr(),
                2,
                quantics.as_mut_ptr(),
                16,
                &mut n_out
            ),
            T4A_SUCCESS
        );

        let mut coord_back = [0.0f64; 2];
        assert_eq!(
            t4a_qgrid_disc_quantics_to_origcoord(
                out,
                quantics.as_ptr(),
                n_out,
                coord_back.as_mut_ptr(),
                2
            ),
            T4A_SUCCESS
        );
        assert!((coord_back[0] - 0.25).abs() < 1e-10);
        assert!((coord_back[1] - 0.5).abs() < 1e-10);

        t4a_qgrid_disc_release(out);
    }

    #[test]
    fn test_disc_grid_grididx_quantics_roundtrip() {
        let grid = create_disc_grid_1d(3);

        // Test all grid indices roundtrip through quantics
        for idx in 1..=8i64 {
            let grididx = [idx];
            let mut quantics = [0i64; 8];
            let mut n_q: libc::size_t = 0;
            assert_eq!(
                t4a_qgrid_disc_grididx_to_quantics(
                    grid,
                    grididx.as_ptr(),
                    1,
                    quantics.as_mut_ptr(),
                    8,
                    &mut n_q
                ),
                T4A_SUCCESS
            );

            let mut grididx_back = [0i64; 1];
            let mut n_g: libc::size_t = 0;
            assert_eq!(
                t4a_qgrid_disc_quantics_to_grididx(
                    grid,
                    quantics.as_ptr(),
                    n_q,
                    grididx_back.as_mut_ptr(),
                    1,
                    &mut n_g
                ),
                T4A_SUCCESS
            );
            assert_eq!(grididx_back[0], idx);
        }

        t4a_qgrid_disc_release(grid);
    }

    #[test]
    fn test_int_grid_1d_properties() {
        let rs = [3usize];
        let mut out: *mut t4a_qgrid_int = std::ptr::null_mut();
        let status = t4a_qgrid_int_new(
            1,
            rs.as_ptr(),
            std::ptr::null(),
            t4a_unfolding_scheme::Fused,
            &mut out,
        );
        assert_eq!(status, T4A_SUCCESS);

        let mut ndims: libc::size_t = 0;
        assert_eq!(t4a_qgrid_int_ndims(out, &mut ndims), T4A_SUCCESS);
        assert_eq!(ndims, 1);

        let mut rs_out = [0usize; 1];
        assert_eq!(t4a_qgrid_int_rs(out, rs_out.as_mut_ptr(), 1), T4A_SUCCESS);
        assert_eq!(rs_out[0], 3);

        let mut origin = [0i64; 1];
        assert_eq!(
            t4a_qgrid_int_origin(out, origin.as_mut_ptr(), 1),
            T4A_SUCCESS
        );
        assert_eq!(origin[0], 1); // default origin

        t4a_qgrid_int_release(out);
    }

    #[test]
    fn test_int_grid_1d_roundtrip() {
        let rs = [3usize];
        let origin = [0i64];
        let mut out: *mut t4a_qgrid_int = std::ptr::null_mut();
        let status = t4a_qgrid_int_new(
            1,
            rs.as_ptr(),
            origin.as_ptr(),
            t4a_unfolding_scheme::Fused,
            &mut out,
        );
        assert_eq!(status, T4A_SUCCESS);

        // origcoord -> quantics -> origcoord
        let coord = [3i64];
        let mut quantics = [0i64; 8];
        let mut n_out: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_int_origcoord_to_quantics(
                out,
                coord.as_ptr(),
                1,
                quantics.as_mut_ptr(),
                8,
                &mut n_out
            ),
            T4A_SUCCESS
        );

        let mut coord_back = [0i64; 1];
        assert_eq!(
            t4a_qgrid_int_quantics_to_origcoord(
                out,
                quantics.as_ptr(),
                n_out,
                coord_back.as_mut_ptr(),
                1
            ),
            T4A_SUCCESS
        );
        assert_eq!(coord_back[0], 3);

        // grididx -> quantics -> grididx
        for idx in 1..=8i64 {
            let grididx = [idx];
            let mut q = [0i64; 8];
            let mut nq: libc::size_t = 0;
            assert_eq!(
                t4a_qgrid_int_grididx_to_quantics(
                    out,
                    grididx.as_ptr(),
                    1,
                    q.as_mut_ptr(),
                    8,
                    &mut nq
                ),
                T4A_SUCCESS
            );

            let mut g_back = [0i64; 1];
            let mut ng: libc::size_t = 0;
            assert_eq!(
                t4a_qgrid_int_quantics_to_grididx(
                    out,
                    q.as_ptr(),
                    nq,
                    g_back.as_mut_ptr(),
                    1,
                    &mut ng
                ),
                T4A_SUCCESS
            );
            assert_eq!(g_back[0], idx);
        }

        t4a_qgrid_int_release(out);
    }

    #[test]
    fn test_null_pointer_guards() {
        let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();
        assert_eq!(
            t4a_qgrid_disc_new(
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                t4a_unfolding_scheme::Fused,
                &mut out
            ),
            T4A_NULL_POINTER
        );

        let mut ndims: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_ndims(std::ptr::null(), &mut ndims),
            T4A_NULL_POINTER
        );

        let mut out_int: *mut t4a_qgrid_int = std::ptr::null_mut();
        assert_eq!(
            t4a_qgrid_int_new(
                1,
                std::ptr::null(),
                std::ptr::null(),
                t4a_unfolding_scheme::Fused,
                &mut out_int
            ),
            T4A_NULL_POINTER
        );
    }

    #[test]
    fn test_disc_grid_clone() {
        let grid = create_disc_grid_1d(3);
        let cloned = t4a_qgrid_disc_clone(grid);
        assert!(!cloned.is_null());

        let mut ndims: libc::size_t = 0;
        assert_eq!(t4a_qgrid_disc_ndims(cloned, &mut ndims), T4A_SUCCESS);
        assert_eq!(ndims, 1);

        t4a_qgrid_disc_release(grid);
        t4a_qgrid_disc_release(cloned);
    }
}
