//! C API for the reduced TreeTN surface.

use crate::types::{
    t4a_canonical_form, t4a_contract_method, t4a_index, t4a_tensor, t4a_treetn, InternalIndex,
    InternalTreeTN,
};
use crate::{
    capi_error, clone_opaque, is_assigned_opaque, panic_message, release_opaque, run_catching,
    set_last_error, CapiResult, StatusCode, T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR,
    T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};
use num_complex::Complex64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetn::treetn::contraction::{self, ContractionMethod, ContractionOptions};
use tensor4all_treetn::{CanonicalizationOptions, TruncationOptions};

/// Release a TreeTN handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_release(obj: *mut t4a_treetn) {
    release_opaque(obj);
}

/// Clone a TreeTN handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_clone(
    src: *const t4a_treetn,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    clone_opaque(src, out)
}

/// Check whether a TreeTN handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_is_assigned(obj: *const t4a_treetn) -> i32 {
    is_assigned_opaque(obj)
}

#[derive(Clone, Copy, Debug)]
enum Tol {
    None,
    Rtol(f64),
    Cutoff(f64),
}

#[inline]
fn select_tol(rtol: f64, cutoff: f64) -> Tol {
    if cutoff > 0.0 {
        Tol::Cutoff(cutoff)
    } else if rtol > 0.0 {
        Tol::Rtol(rtol)
    } else {
        Tol::None
    }
}

#[inline]
fn cutoff_to_rtol(cutoff: f64) -> f64 {
    cutoff.sqrt()
}

#[inline]
fn resolve_rtol(rtol: f64, cutoff: f64) -> Option<f64> {
    match select_tol(rtol, cutoff) {
        Tol::Cutoff(c) => Some(cutoff_to_rtol(c)),
        Tol::Rtol(r) => Some(r),
        Tol::None => None,
    }
}

fn run_status<F>(f: F) -> StatusCode
where
    F: FnOnce() -> CapiResult<()>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => T4A_SUCCESS,
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

fn require_tree<'a>(ptr: *const t4a_treetn) -> CapiResult<&'a t4a_treetn> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "treetn is null"));
    }
    Ok(unsafe { &*ptr })
}

fn require_tree_mut<'a>(ptr: *mut t4a_treetn) -> CapiResult<&'a mut t4a_treetn> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "treetn is null"));
    }
    Ok(unsafe { &mut *ptr })
}

fn require_node(tn: &InternalTreeTN, vertex: usize) -> CapiResult<()> {
    if tn.node_index(&vertex).is_none() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("vertex {vertex} does not exist"),
        ));
    }
    Ok(())
}

fn query_then_fill_copy<T: Copy>(
    values: &[T],
    buf: *mut T,
    buf_len: usize,
    out_len: *mut usize,
    what: &str,
) -> CapiResult<()> {
    if out_len.is_null() {
        return Err(capi_error(
            T4A_NULL_POINTER,
            format!("{what}: out_len is null"),
        ));
    }

    unsafe { *out_len = values.len() };
    if buf.is_null() {
        return Ok(());
    }
    if buf_len < values.len() {
        return Err(capi_error(
            T4A_BUFFER_TOO_SMALL,
            format!(
                "{what}: buffer too small, need {}, got {}",
                values.len(),
                buf_len
            ),
        ));
    }

    for (i, value) in values.iter().enumerate() {
        unsafe { *buf.add(i) = *value };
    }
    Ok(())
}

fn collect_indices(
    index_ptrs: *const *const t4a_index,
    n_indices: usize,
) -> CapiResult<Vec<InternalIndex>> {
    if index_ptrs.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "indices is null"));
    }

    let mut indices = Vec::with_capacity(n_indices);
    for i in 0..n_indices {
        let ptr = unsafe { *index_ptrs.add(i) };
        if ptr.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                format!("indices[{i}] is null"),
            ));
        }
        indices.push(unsafe { &*ptr }.inner().clone());
    }
    Ok(indices)
}

/// Create a tree tensor network from an array of tensors.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_new(
    tensors: *const *const t4a_tensor,
    n_tensors: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    run_catching(out, || {
        if n_tensors == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "n_tensors must be greater than zero",
            ));
        }
        if tensors.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "tensors is null"));
        }

        let mut tensor_vec = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let tensor_ptr = unsafe { *tensors.add(i) };
            if tensor_ptr.is_null() {
                return Err(capi_error(
                    T4A_NULL_POINTER,
                    format!("tensors[{i}] is null"),
                ));
            }
            tensor_vec.push(unsafe { &*tensor_ptr }.inner().clone());
        }

        let node_names: Vec<usize> = (0..n_tensors).collect();
        let treetn = InternalTreeTN::from_tensors(tensor_vec, node_names)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(treetn))
    })
}

/// Get the number of vertices in the tree tensor network.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_num_vertices(
    treetn: *const t4a_treetn,
    out_n: *mut libc::size_t,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree(treetn)?;
        if out_n.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_n is null"));
        }
        unsafe { *out_n = tn.inner().node_count() };
        Ok(())
    })
}

/// Get the tensor at a specific vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_tensor(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let tensor = tn.inner().tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} has no tensor"),
            )
        })?;
        Ok(t4a_tensor::new(tensor.clone()))
    })
}

/// Replace the tensor at a specific vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_set_tensor(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    tensor: *const t4a_tensor,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        if tensor.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "tensor is null"));
        }
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        tn.inner_mut()
            .replace_tensor(node_idx, unsafe { &*tensor }.inner().clone())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(())
    })
}

/// Get the neighbors of a vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_neighbors(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    buf: *mut libc::size_t,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree(treetn)?;
        require_node(tn.inner(), vertex)?;
        let neighbors: Vec<_> = tn.inner().site_index_network().neighbors(&vertex).collect();
        query_then_fill_copy(&neighbors, buf, buf_len, out_len, "t4a_treetn_neighbors")
    })
}

/// Get the site indices attached to a vertex.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_siteinds(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    buf: *mut *mut t4a_index,
    buf_len: libc::size_t,
    out_len: *mut libc::size_t,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree(treetn)?;
        let node_idx = tn.inner().node_index(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let tensor = tn.inner().tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} has no tensor"),
            )
        })?;
        let site_space = tn.inner().site_space(&vertex).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertex {vertex} does not exist"),
            )
        })?;
        let ordered_indices: Vec<_> = tensor
            .indices
            .iter()
            .filter(|index| site_space.contains(*index))
            .cloned()
            .collect();

        if out_len.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_len is null"));
        }
        unsafe { *out_len = ordered_indices.len() };

        if buf.is_null() {
            return Ok(());
        }
        if buf_len < ordered_indices.len() {
            return Err(capi_error(
                T4A_BUFFER_TOO_SMALL,
                format!(
                    "t4a_treetn_siteinds: buffer too small, need {}, got {}",
                    ordered_indices.len(),
                    buf_len
                ),
            ));
        }

        for (i, index) in ordered_indices.iter().enumerate() {
            unsafe { *buf.add(i) = Box::into_raw(Box::new(t4a_index::new(index.clone()))) };
        }
        Ok(())
    })
}

/// Get the bond index on the edge between two vertices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linkind(
    treetn: *const t4a_treetn,
    v1: libc::size_t,
    v2: libc::size_t,
    out: *mut *mut t4a_index,
) -> StatusCode {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let edge = tn.inner().edge_between(&v1, &v2).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("vertices {v1} and {v2} are not adjacent"),
            )
        })?;
        let index = tn.inner().bond_index(edge).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("missing bond index between {v1} and {v2}"),
            )
        })?;
        Ok(t4a_index::new(index.clone()))
    })
}

/// Orthogonalize the tree tensor network to a vertex using the requested form.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_orthogonalize(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    form: t4a_canonical_form,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        require_node(tn.inner(), vertex)?;
        let options = CanonicalizationOptions::forced().with_form(form.into());
        tn.inner_mut()
            .canonicalize_mut(std::iter::once(vertex), options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(())
    })
}

/// Truncate the tree tensor network bond dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_truncate(
    treetn: *mut t4a_treetn,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;

        let mut options = TruncationOptions::new();
        if let Some(rtol) = resolve_rtol(rtol, cutoff) {
            options = options.with_rtol(rtol);
        }
        if maxdim > 0 {
            options = options.with_max_rank(maxdim);
        }

        let center =
            tn.inner().node_names().into_iter().min().ok_or_else(|| {
                capi_error(T4A_INVALID_ARGUMENT, "cannot truncate an empty TreeTN")
            })?;
        tn.inner_mut()
            .truncate_mut(std::iter::once(center), options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(())
    })
}

/// Evaluate a TreeTN at one or more points using explicit index handles.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluate(
    treetn: *const t4a_treetn,
    indices: *const *const t4a_index,
    n_indices: libc::size_t,
    values_col_major: *const libc::size_t,
    n_points: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree(treetn)?;
        if n_indices == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate requires n_indices > 0",
            ));
        }
        if n_points == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate requires n_points > 0",
            ));
        }
        if values_col_major.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "values_col_major is null"));
        }
        if out_re.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_re is null"));
        }

        let indices = collect_indices(indices, n_indices)?;
        let n_values = n_indices.checked_mul(n_points).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                "t4a_treetn_evaluate value array size overflowed size_t",
            )
        })?;
        let values_slice = unsafe { std::slice::from_raw_parts(values_col_major, n_values) };
        let shape = [n_indices, n_points];
        let values = ColMajorArrayRef::new(values_slice, &shape);
        let results = tn
            .inner()
            .evaluate_at(&indices, values)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        for (i, scalar) in results.iter().enumerate() {
            let z: Complex64 = scalar.clone().into();
            unsafe { *out_re.add(i) = z.re };
            if z.im != 0.0 {
                if out_im.is_null() {
                    return Err(capi_error(
                        T4A_NULL_POINTER,
                        "out_im is required for complex-valued evaluation results",
                    ));
                }
                unsafe { *out_im.add(i) = z.im };
            } else if !out_im.is_null() {
                unsafe { *out_im.add(i) = 0.0 };
            }
        }
        Ok(())
    })
}

/// Compute the inner product of two tree tensor networks.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_inner(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    run_status(|| {
        let tn_a = require_tree(a)?;
        let tn_b = require_tree(b)?;
        if out_re.is_null() || out_im.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_re or out_im is null"));
        }
        let z: Complex64 = tn_a
            .inner()
            .inner(tn_b.inner())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?
            .into();
        unsafe {
            *out_re = z.re;
            *out_im = z.im;
        }
        Ok(())
    })
}

/// Compute the norm of the tree tensor network.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_norm(
    treetn: *mut t4a_treetn,
    out_norm: *mut libc::c_double,
) -> StatusCode {
    run_status(|| {
        let tn = require_tree_mut(treetn)?;
        if out_norm.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_norm is null"));
        }
        unsafe {
            *out_norm = tn
                .inner_mut()
                .norm()
                .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        }
        Ok(())
    })
}

/// Contract two tree tensor networks with the requested method.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_contract(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    method: t4a_contract_method,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    let tn_a = match require_tree(a) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let tn_b = match require_tree(b) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let center =
            tn_a.inner().node_names().into_iter().min().ok_or_else(|| {
                capi_error(T4A_INVALID_ARGUMENT, "cannot contract an empty TreeTN")
            })?;

        let rust_method: ContractionMethod = method.into();
        let mut options = ContractionOptions::new(rust_method);
        if let Some(rtol) = resolve_rtol(rtol, cutoff) {
            options = options.with_rtol(rtol);
        }
        if maxdim > 0 {
            options = options.with_max_rank(maxdim);
        }

        let result = contraction::contract(tn_a.inner(), tn_b.inner(), &center, options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(result))
    })
}

/// Contract all bonds and materialize the TreeTN as a dense tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_to_dense(
    treetn: *const t4a_treetn,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    let tn = match require_tree(treetn) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let tensor = tn
            .inner()
            .to_dense()
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

#[cfg(test)]
mod tests;
