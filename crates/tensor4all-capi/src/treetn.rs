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
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_core::{ColMajorArrayRef, IndexLike};
use tensor4all_treetn::treetn::contraction::{self, ContractionMethod, ContractionOptions};
use tensor4all_treetn::{
    apply_linear_operator, ApplyOptions, CanonicalizationOptions, IndexMapping, LinearOperator,
    TruncationOptions,
};

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
    what: &str,
) -> CapiResult<Vec<InternalIndex>> {
    if index_ptrs.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }

    let mut indices = Vec::with_capacity(n_indices);
    for i in 0..n_indices {
        let ptr = unsafe { *index_ptrs.add(i) };
        if ptr.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, format!("{what}[{i}] is null")));
        }
        indices.push(unsafe { &*ptr }.inner().clone());
    }
    Ok(indices)
}

fn collect_positions(
    positions: *const libc::size_t,
    len: usize,
    what: &str,
) -> CapiResult<Vec<usize>> {
    if positions.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(unsafe { *positions.add(i) });
    }
    Ok(out)
}

fn require_chain_layout(tn: &InternalTreeTN, what: &str) -> CapiResult<usize> {
    let n = tn.node_count();
    if n == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{what} must not be empty"),
        ));
    }
    if tn.edge_count() != n.saturating_sub(1) {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "{what} must be a chain with {} edges, got {}",
                n.saturating_sub(1),
                tn.edge_count()
            ),
        ));
    }

    for position in 0..n {
        if tn.node_index(&position).is_none() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} must use dense node names 0..{}, missing {position}",
                    n - 1
                ),
            ));
        }

        let mut neighbors: Vec<usize> = tn.site_index_network().neighbors(&position).collect();
        neighbors.sort_unstable();
        let mut expected = Vec::with_capacity(2);
        if position > 0 {
            expected.push(position - 1);
        }
        if position + 1 < n {
            expected.push(position + 1);
        }
        if neighbors != expected {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} must follow chain adjacency at node {position}, got neighbors {:?}, expected {:?}",
                    neighbors, expected
                ),
            ));
        }
    }

    Ok(n)
}

fn collect_single_site_indices(tn: &InternalTreeTN, what: &str) -> CapiResult<Vec<InternalIndex>> {
    let n = require_chain_layout(tn, what)?;
    let mut site_indices = Vec::with_capacity(n);
    for position in 0..n {
        let site_space = tn.site_space(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{what} node {position} has no site index"),
            )
        })?;
        if site_space.len() != 1 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "{what} node {position} must have exactly one site index, got {}",
                    site_space.len()
                ),
            ));
        }
        site_indices.push(site_space.iter().next().unwrap().clone());
    }
    Ok(site_indices)
}

fn validate_mapped_positions(
    mapped_positions: &[usize],
    state_nsites: usize,
    operator_nsites: usize,
) -> CapiResult<()> {
    if mapped_positions.is_empty() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "mapped node positions must not be empty",
        ));
    }
    if mapped_positions.len() != operator_nsites {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "mapped node positions length {} does not match operator node count {}",
                mapped_positions.len(),
                operator_nsites
            ),
        ));
    }
    for (i, &position) in mapped_positions.iter().enumerate() {
        if position >= state_nsites {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "mapped node position {position} is outside the state chain with {state_nsites} nodes"
                ),
            ));
        }
        if i > 0 && mapped_positions[i - 1] >= position {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "mapped node positions must be strictly increasing",
            ));
        }
    }
    Ok(())
}

fn remap_operator_nodes_to_positions(
    operator: &InternalTreeTN,
    mapped_positions: &[usize],
) -> CapiResult<InternalTreeTN> {
    let operator_nsites = require_chain_layout(operator, "operator")?;
    if operator_nsites != mapped_positions.len() {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!(
                "operator node count {} does not match mapped node positions length {}",
                operator_nsites,
                mapped_positions.len()
            ),
        ));
    }

    let mut tensors = Vec::with_capacity(operator_nsites);
    for position in 0..operator_nsites {
        let node_idx = operator.node_index(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator must use dense node names 0..{}, missing {position}",
                    operator_nsites - 1
                ),
            )
        })?;
        let tensor = operator.tensor(node_idx).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("operator node {position} has no tensor"),
            )
        })?;
        tensors.push(tensor.clone());
    }

    InternalTreeTN::from_tensors(tensors, mapped_positions.to_vec())
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))
}

fn build_chain_linear_operator(
    operator: &InternalTreeTN,
    state_true_inputs: &[InternalIndex],
    mapped_positions: &[usize],
    internal_inputs: &[InternalIndex],
    internal_outputs: &[InternalIndex],
    true_outputs: &[InternalIndex],
) -> CapiResult<LinearOperator<crate::types::InternalTensor, usize>> {
    let remapped_operator = remap_operator_nodes_to_positions(operator, mapped_positions)?;
    let mut input_mapping = HashMap::with_capacity(mapped_positions.len());
    let mut output_mapping = HashMap::with_capacity(mapped_positions.len());

    for (slot, &position) in mapped_positions.iter().enumerate() {
        let site_space = remapped_operator.site_space(&position).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("operator node {position} has no site indices"),
            )
        })?;
        if site_space.len() != 2 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} must have exactly two site indices, got {}",
                    site_space.len()
                ),
            ));
        }

        let internal_input = &internal_inputs[slot];
        let internal_output = &internal_outputs[slot];
        if !site_space.contains(internal_input) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} does not contain the provided internal input index"
                ),
            ));
        }
        if !site_space.contains(internal_output) {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "operator node {position} does not contain the provided internal output index"
                ),
            ));
        }

        let true_input = &state_true_inputs[position];
        let true_output = &true_outputs[slot];
        if true_input.dim() != internal_input.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "input index dimension mismatch at node {position}: state has {}, operator internal input has {}",
                    true_input.dim(),
                    internal_input.dim()
                ),
            ));
        }
        if true_output.dim() != internal_output.dim() {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "output index dimension mismatch at node {position}: requested true output has {}, operator internal output has {}",
                    true_output.dim(),
                    internal_output.dim()
                ),
            ));
        }

        input_mapping.insert(
            position,
            IndexMapping {
                true_index: true_input.clone(),
                internal_index: internal_input.clone(),
            },
        );
        output_mapping.insert(
            position,
            IndexMapping {
                true_index: true_output.clone(),
                internal_index: internal_output.clone(),
            },
        );
    }

    Ok(LinearOperator::new(
        remapped_operator,
        input_mapping,
        output_mapping,
    ))
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

        let indices = collect_indices(indices, n_indices, "indices")?;
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

/// Apply a chain-compatible operator TreeTN to a chain state TreeTN.
///
/// The operator must use dense node names `0..m-1`. `mapped_positions[i]`
/// specifies which state-chain node the operator node `i` acts on. Each mapped
/// operator node must have exactly two site indices, and the corresponding
/// `internal_input_indices[i]` and `internal_output_indices[i]` must point to
/// those indices. Unmapped state nodes are filled with identities by
/// `apply_linear_operator`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_apply_operator_chain(
    operator: *const t4a_treetn,
    state: *const t4a_treetn,
    mapped_positions: *const libc::size_t,
    n_mapped_positions: libc::size_t,
    internal_input_indices: *const *const t4a_index,
    internal_output_indices: *const *const t4a_index,
    true_output_indices: *const *const t4a_index,
    method: t4a_contract_method,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    let op = match require_tree(operator) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };
    let state = match require_tree(state) {
        Ok(tn) => tn,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        let mapped_positions =
            collect_positions(mapped_positions, n_mapped_positions, "mapped_positions")?;
        let internal_inputs = collect_indices(
            internal_input_indices,
            n_mapped_positions,
            "internal_input_indices",
        )?;
        let internal_outputs = collect_indices(
            internal_output_indices,
            n_mapped_positions,
            "internal_output_indices",
        )?;
        let true_outputs = collect_indices(
            true_output_indices,
            n_mapped_positions,
            "true_output_indices",
        )?;

        let state_true_inputs = collect_single_site_indices(state.inner(), "state")?;
        validate_mapped_positions(
            &mapped_positions,
            state_true_inputs.len(),
            op.inner().node_count(),
        )?;
        let linear_operator = build_chain_linear_operator(
            op.inner(),
            &state_true_inputs,
            &mapped_positions,
            &internal_inputs,
            &internal_outputs,
            &true_outputs,
        )?;

        let result = apply_linear_operator(
            &linear_operator,
            state.inner(),
            ApplyOptions {
                method: method.into(),
                max_rank: (maxdim > 0).then_some(maxdim),
                rtol: resolve_rtol(rtol, cutoff),
                ..Default::default()
            },
        )
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
