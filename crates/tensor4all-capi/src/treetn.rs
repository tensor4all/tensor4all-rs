//! C API for TreeTN (Tree Tensor Network) operations
//!
//! Provides functions to create, manipulate, and query tree tensor networks.
//! This replaces the old `tensortrain.rs` module with a direct wrapper around
//! `tensor4all_treetn::DefaultTreeTN<usize>`.

use crate::types::{t4a_canonical_form, t4a_index, t4a_tensor, t4a_treetn, InternalTreeTN};
use crate::{StatusCode, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_core::ColMajorArrayRef;

use tensor4all_treetn::treetn::contraction::{ContractionMethod, ContractionOptions};
use tensor4all_treetn::{CanonicalizationOptions, TruncationOptions};

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

/// Convert cutoff to rtol: rtol = sqrt(cutoff).
#[inline]
fn cutoff_to_rtol(cutoff: f64) -> f64 {
    cutoff.sqrt()
}

/// Resolve rtol from either rtol or cutoff convention.
/// Returns None if neither is set.
#[inline]
fn resolve_rtol(rtol: f64, cutoff: f64) -> Option<f64> {
    match select_tol(rtol, cutoff) {
        Tol::Cutoff(c) => Some(cutoff_to_rtol(c)),
        Tol::Rtol(r) => Some(r),
        Tol::None => None,
    }
}

// ============================================================================
// Lifecycle functions
// ============================================================================

impl_opaque_type_common!(treetn);

// ============================================================================
// Constructors
// ============================================================================

/// Create a tree tensor network from an array of tensors.
///
/// Node names are assigned as 0, 1, 2, ..., n_tensors-1.
/// Tensors are connected by matching index IDs (einsum rule).
///
/// # Arguments
/// * `tensors` - Array of tensor pointers
/// * `n_tensors` - Number of tensors in the array
/// * `out` - Output pointer for the new TreeTN
///
/// # Returns
/// Status code
///
/// # Safety
/// - `tensors` must be a valid pointer to an array of `n_tensors` t4a_tensor pointers
/// - All tensor pointers must be valid
/// - `out` must be a valid pointer
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_new(
    tensors: *const *const t4a_tensor,
    n_tensors: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if out.is_null() {
        return T4A_NULL_POINTER;
    }
    if tensors.is_null() && n_tensors > 0 {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Collect tensors
        let mut tensor_vec = Vec::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let tensor_ptr = unsafe { *tensors.add(i) };
            if tensor_ptr.is_null() {
                return T4A_NULL_POINTER;
            }
            let tensor_ref = unsafe { &*tensor_ptr };
            tensor_vec.push(tensor_ref.inner().clone());
        }

        // Create node names: 0, 1, 2, ...
        let node_names: Vec<usize> = (0..n_tensors).collect();

        // Create TreeTN
        match InternalTreeTN::from_tensors(tensor_vec, node_names) {
            Ok(treetn) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(treetn))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Accessors
// ============================================================================

/// Get the number of vertices (nodes) in the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output pointer for the vertex count
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_num_vertices(
    treetn: *const t4a_treetn,
    out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        unsafe { *out = tn.inner().node_count() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the number of edges (bonds) in the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output pointer for the edge count
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_num_edges(
    treetn: *const t4a_treetn,
    out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        unsafe { *out = tn.inner().edge_count() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the tensor at a specific vertex.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `vertex` - Vertex name (node name as usize)
/// * `out` - Output pointer for the tensor
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_tensor(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let node_idx = match tn.inner().node_index(&vertex) {
            Some(idx) => idx,
            None => return T4A_INVALID_ARGUMENT,
        };
        match tn.inner().tensor(node_idx) {
            Some(tensor) => {
                let cloned = tensor.clone();
                unsafe { *out = Box::into_raw(Box::new(t4a_tensor::new(cloned))) };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    crate::unwrap_catch(result)
}

/// Set the tensor at a specific vertex.
///
/// # Arguments
/// * `treetn` - TreeTN handle (modified in place)
/// * `vertex` - Vertex name (node name as usize)
/// * `tensor` - New tensor to set (cloned)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_set_tensor(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    tensor: *const t4a_tensor,
) -> StatusCode {
    if treetn.is_null() || tensor.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };
        let tensor_inner = unsafe { &*tensor };
        let node_idx = match tn.inner().node_index(&vertex) {
            Some(idx) => idx,
            None => return T4A_INVALID_ARGUMENT,
        };
        match tn
            .inner_mut()
            .replace_tensor(node_idx, tensor_inner.inner().clone())
        {
            Ok(_) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Get the neighbors of a vertex.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `vertex` - Vertex name (node name as usize)
/// * `out_buf` - Output buffer for neighbor vertex names
/// * `buf_size` - Size of the output buffer
/// * `n_out` - Output: number of neighbors written
///
/// # Returns
/// Status code (T4A_BUFFER_TOO_SMALL if buffer is too small)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_neighbors(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    out_buf: *mut libc::size_t,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out_buf.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let neighbors: Vec<usize> = tn.inner().site_index_network().neighbors(&vertex).collect();

        if buf_size < neighbors.len() {
            unsafe { *n_out = neighbors.len() };
            return crate::T4A_BUFFER_TOO_SMALL;
        }

        for (i, &n) in neighbors.iter().enumerate() {
            unsafe { *out_buf.add(i) = n };
        }
        unsafe { *n_out = neighbors.len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the link (bond) index on the edge between two vertices.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `v1` - First vertex name
/// * `v2` - Second vertex name
/// * `out` - Output pointer for the index
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linkind(
    treetn: *const t4a_treetn,
    v1: libc::size_t,
    v2: libc::size_t,
    out: *mut *mut t4a_index,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let edge = match tn.inner().edge_between(&v1, &v2) {
            Some(e) => e,
            None => return T4A_INVALID_ARGUMENT,
        };
        match tn.inner().bond_index(edge) {
            Some(idx) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_index::new(idx.clone()))) };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    crate::unwrap_catch(result)
}

/// Get the site (physical) indices at a vertex.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `vertex` - Vertex name
/// * `out_buf` - Output buffer for index pointers
/// * `buf_size` - Size of the output buffer
/// * `n_out` - Output: number of site indices written
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_siteinds(
    treetn: *const t4a_treetn,
    vertex: libc::size_t,
    out_buf: *mut *mut t4a_index,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out_buf.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let site_space = match tn.inner().site_space(&vertex) {
            Some(s) => s,
            None => return T4A_INVALID_ARGUMENT,
        };

        let indices: Vec<_> = site_space.iter().cloned().collect();
        if buf_size < indices.len() {
            unsafe { *n_out = indices.len() };
            return crate::T4A_BUFFER_TOO_SMALL;
        }

        for (i, idx) in indices.into_iter().enumerate() {
            unsafe { *out_buf.add(i) = Box::into_raw(Box::new(t4a_index::new(idx))) };
        }
        unsafe { *n_out = site_space.len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the bond dimension on the edge between two vertices.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `v1` - First vertex name
/// * `v2` - Second vertex name
/// * `out` - Output pointer for the bond dimension
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_bond_dim(
    treetn: *const t4a_treetn,
    v1: libc::size_t,
    v2: libc::size_t,
    out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let edge = match tn.inner().edge_between(&v1, &v2) {
            Some(e) => e,
            None => return T4A_INVALID_ARGUMENT,
        };
        match tn.inner().bond_index(edge) {
            Some(idx) => {
                use tensor4all_core::IndexLike;
                unsafe { *out = idx.dim() };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// MPS convenience functions (linear topology: vertices 0, 1, ..., n-1)
// ============================================================================

/// Get the link index between vertex i and i+1 (MPS convention).
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `i` - Site index (returns link between site i and site i+1)
/// * `out` - Output pointer for the index
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linkind_at(
    treetn: *const t4a_treetn,
    i: libc::size_t,
    out: *mut *mut t4a_index,
) -> StatusCode {
    t4a_treetn_linkind(treetn, i, i + 1, out)
}

/// Get the bond dimension between vertex i and i+1 (MPS convention).
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `i` - Site index (returns bond dim between site i and site i+1)
/// * `out` - Output pointer for the bond dimension
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_bond_dim_at(
    treetn: *const t4a_treetn,
    i: libc::size_t,
    out: *mut libc::size_t,
) -> StatusCode {
    t4a_treetn_bond_dim(treetn, i, i + 1, out)
}

/// Get all bond dimensions for an MPS-like TreeTN (vertices 0, 1, ..., n-1).
///
/// Writes n-1 bond dimensions: bond_dim(0,1), bond_dim(1,2), ..., bond_dim(n-2,n-1).
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output buffer for bond dimensions (must have room for at least n-1 elements)
/// * `n` - Size of the output buffer
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_bond_dims(
    treetn: *const t4a_treetn,
    out: *mut libc::size_t,
    n: libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let num_vertices = tn.inner().node_count();
        if num_vertices == 0 {
            return T4A_SUCCESS;
        }
        let num_bonds = num_vertices - 1;
        if n < num_bonds {
            return T4A_INVALID_ARGUMENT;
        }

        for i in 0..num_bonds {
            let edge = match tn.inner().edge_between(&i, &(i + 1)) {
                Some(e) => e,
                None => return T4A_INVALID_ARGUMENT,
            };
            match tn.inner().bond_index(edge) {
                Some(idx) => {
                    use tensor4all_core::IndexLike;
                    unsafe { *out.add(i) = idx.dim() };
                }
                None => return T4A_INVALID_ARGUMENT,
            }
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum bond dimension of an MPS-like TreeTN.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output pointer for the maximum bond dimension
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_maxbonddim(
    treetn: *const t4a_treetn,
    out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let num_vertices = tn.inner().node_count();
        if num_vertices <= 1 {
            unsafe { *out = 0 };
            return T4A_SUCCESS;
        }

        let mut max_dim: usize = 0;
        let num_bonds = num_vertices - 1;
        for i in 0..num_bonds {
            let edge = match tn.inner().edge_between(&i, &(i + 1)) {
                Some(e) => e,
                None => return T4A_INVALID_ARGUMENT,
            };
            match tn.inner().bond_index(edge) {
                Some(idx) => {
                    use tensor4all_core::IndexLike;
                    let dim = idx.dim();
                    if dim > max_dim {
                        max_dim = dim;
                    }
                }
                None => return T4A_INVALID_ARGUMENT,
            }
        }
        unsafe { *out = max_dim };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Orthogonalization
// ============================================================================

/// Orthogonalize the tree tensor network to a single vertex.
///
/// Uses QR decomposition (Unitary canonical form) by default.
///
/// # Arguments
/// * `treetn` - TreeTN handle (modified in place)
/// * `vertex` - Target vertex for orthogonality center
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_orthogonalize(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
) -> StatusCode {
    if treetn.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };
        let options = CanonicalizationOptions::forced();
        match tn
            .inner_mut()
            .canonicalize_mut(std::iter::once(vertex), options)
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Orthogonalize the tree tensor network to a vertex with a specific canonical form.
///
/// # Arguments
/// * `treetn` - TreeTN handle (modified in place)
/// * `vertex` - Target vertex for orthogonality center
/// * `form` - Canonical form to use
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_orthogonalize_with(
    treetn: *mut t4a_treetn,
    vertex: libc::size_t,
    form: t4a_canonical_form,
) -> StatusCode {
    if treetn.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };
        let rust_form: tensor4all_treetn::CanonicalForm = form.into();
        let options = CanonicalizationOptions::forced().with_form(rust_form);
        match tn
            .inner_mut()
            .canonicalize_mut(std::iter::once(vertex), options)
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Get the canonical (orthogonality) region of the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out_vertices` - Output buffer for vertex names in the canonical region
/// * `buf_size` - Size of the output buffer
/// * `n_out` - Output: number of vertices in the canonical region
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_ortho_center(
    treetn: *const t4a_treetn,
    out_vertices: *mut libc::size_t,
    buf_size: libc::size_t,
    n_out: *mut libc::size_t,
) -> StatusCode {
    if treetn.is_null() || out_vertices.is_null() || n_out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        let region = tn.inner().canonical_region();

        if buf_size < region.len() {
            unsafe { *n_out = region.len() };
            return crate::T4A_BUFFER_TOO_SMALL;
        }

        let mut sorted_region: Vec<usize> = region.iter().copied().collect();
        sorted_region.sort();

        for (i, &v) in sorted_region.iter().enumerate() {
            unsafe { *out_vertices.add(i) = v };
        }
        unsafe { *n_out = sorted_region.len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the canonical form used for the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output pointer for the canonical form
///
/// # Returns
/// Status code (T4A_INVALID_ARGUMENT if no canonical form is set)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_canonical_form(
    treetn: *const t4a_treetn,
    out: *mut t4a_canonical_form,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        match tn.inner().canonical_form() {
            Some(form) => {
                unsafe { *out = form.into() };
                T4A_SUCCESS
            }
            None => T4A_INVALID_ARGUMENT,
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Operations
// ============================================================================

/// Truncate the tree tensor network bond dimensions.
///
/// # Arguments
/// * `treetn` - TreeTN handle (modified in place)
/// * `rtol` - Relative tolerance (use -1.0 for default, 0.0 for not set)
/// * `cutoff` - ITensorMPS.jl cutoff (0.0 for not set). Converted to rtol = sqrt(cutoff).
///   If both rtol and cutoff are positive, cutoff takes precedence.
/// * `maxdim` - Maximum bond dimension (use 0 for no limit)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_truncate(
    treetn: *mut t4a_treetn,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
) -> StatusCode {
    if treetn.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };

        let mut options = TruncationOptions::new();
        if let Some(r) = resolve_rtol(rtol, cutoff) {
            options = options.with_rtol(r);
        }
        if maxdim > 0 {
            options = options.with_max_rank(maxdim);
        }

        // For truncation, we need a center vertex. Use vertex 0.
        let num_vertices = tn.inner().node_count();
        if num_vertices == 0 {
            return T4A_SUCCESS;
        }

        // Find the first node name
        let node_names = tn.inner().node_names();
        let center = node_names.into_iter().min().unwrap_or(0);

        match tn
            .inner_mut()
            .truncate_mut(std::iter::once(center), options)
        {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Get all site index IDs and their owning vertex names.
///
/// Uses query-then-fill: pass NULL buffers to get `out_n_indices` only.
///
/// # Arguments
/// * `ptr` - TreeTN handle
/// * `out_index_ids` - Output buffer for index IDs (as u64), or NULL to query size
/// * `out_vertex_names` - Output buffer for vertex names (as usize), or NULL to query size
/// * `buf_len` - Size of the output buffers
/// * `out_n_indices` - Output: number of site indices
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_all_site_index_ids(
    ptr: *const t4a_treetn,
    out_index_ids: *mut u64,
    out_vertex_names: *mut libc::size_t,
    buf_len: libc::size_t,
    out_n_indices: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_n_indices.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*ptr };
        let inner = tn.inner();

        let (index_ids, vertex_names) = match inner.all_site_index_ids() {
            Ok(v) => v,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        let n_indices = index_ids.len();
        unsafe { *out_n_indices = n_indices };

        // If buffers are NULL, just return the count
        if out_index_ids.is_null() && out_vertex_names.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < n_indices {
            return crate::err_status(
                format!("Buffer too small: need {}, got {}", n_indices, buf_len),
                crate::T4A_BUFFER_TOO_SMALL,
            );
        }

        if !out_index_ids.is_null() {
            for (i, id) in index_ids.iter().enumerate() {
                unsafe { *out_index_ids.add(i) = id.0 };
            }
        }

        if !out_vertex_names.is_null() {
            for (i, name) in vertex_names.iter().enumerate() {
                unsafe { *out_vertex_names.add(i) = *name };
            }
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Evaluate a TreeTN at one or more multi-indices using IndexId-based API.
///
/// `index_ids`: `n_indices` index IDs (from `t4a_treetn_all_site_index_ids`).
/// `values`: column-major `[n_indices, n_points]` array.
///   `values[i + n_indices * p]` = value of `index_ids[i]` at point `p`.
///
/// # Arguments
/// * `ptr` - TreeTN handle
/// * `index_ids` - Array of index IDs (n_indices entries)
/// * `n_indices` - Number of index IDs
/// * `values` - Column-major `(n_indices, n_points)` index value array
/// * `n_points` - Number of evaluation points
/// * `out_re` - Output buffer for real parts (`n_points` entries)
/// * `out_im` - Output buffer for imaginary parts (`n_points` entries), or NULL for real-only
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluate(
    ptr: *const t4a_treetn,
    index_ids: *const u64,
    n_indices: libc::size_t,
    values: *const libc::size_t,
    n_points: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || index_ids.is_null() || values.is_null() || out_re.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_core::DynId;

        let tn = unsafe { &*ptr };
        let inner = tn.inner();

        if n_indices == 0 {
            return crate::err_status(
                "t4a_treetn_evaluate requires n_indices > 0",
                T4A_INVALID_ARGUMENT,
            );
        }
        if n_points == 0 {
            return crate::err_status(
                "t4a_treetn_evaluate requires n_points > 0",
                T4A_INVALID_ARGUMENT,
            );
        }

        let Some(n_entries) = n_indices.checked_mul(n_points) else {
            return crate::err_status(
                "t4a_treetn_evaluate index array size overflowed size_t",
                T4A_INVALID_ARGUMENT,
            );
        };

        // Build DynId vector from raw u64 values
        let ids: Vec<DynId> = (0..n_indices)
            .map(|i| DynId(unsafe { *index_ids.add(i) }))
            .collect();

        // Build ColMajorArrayRef from the flat values
        let values_slice = unsafe { std::slice::from_raw_parts(values, n_entries) };
        let shape = [n_indices, n_points];
        let values_arr = ColMajorArrayRef::new(values_slice, &shape);

        let scalars = match inner.evaluate(&ids, values_arr) {
            Ok(v) => v,
            Err(e) => return crate::err_status(e, T4A_INVALID_ARGUMENT),
        };

        for (point, scalar) in scalars.iter().enumerate() {
            unsafe {
                *out_re.add(point) = scalar.real();
            }
            if scalar.is_complex() && out_im.is_null() {
                return crate::err_status(
                    "t4a_treetn_evaluate requires out_im for complex-valued TreeTN results",
                    T4A_NULL_POINTER,
                );
            }
            if !out_im.is_null() {
                unsafe {
                    *out_im.add(point) = scalar.imag();
                }
            }
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Compute the inner product of two tree tensor networks.
///
/// Computes <a | b> = sum over all indices of conj(a) * b.
///
/// # Arguments
/// * `a` - First TreeTN handle
/// * `b` - Second TreeTN handle
/// * `out_re` - Output pointer for real part
/// * `out_im` - Output pointer for imaginary part
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_inner(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode {
    if a.is_null() || b.is_null() || out_re.is_null() || out_im.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use num_complex::Complex64;

        let tn_a = unsafe { &*a };
        let tn_b = unsafe { &*b };

        match tn_a.inner().inner(tn_b.inner()) {
            Ok(scalar) => {
                let z: Complex64 = scalar.into();
                unsafe {
                    *out_re = z.re;
                    *out_im = z.im;
                }
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the norm of the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle (may be modified internally for canonicalization)
/// * `out` - Output pointer for the norm
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_norm(treetn: *mut t4a_treetn, out: *mut libc::c_double) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };
        match tn.inner_mut().norm() {
            Ok(norm) => {
                unsafe { *out = norm };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Compute the log-norm of the tree tensor network.
///
/// # Arguments
/// * `treetn` - TreeTN handle (may be modified internally for canonicalization)
/// * `out` - Output pointer for the log-norm
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_lognorm(
    treetn: *mut t4a_treetn,
    out: *mut libc::c_double,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &mut *treetn };
        match tn.inner_mut().log_norm() {
            Ok(lognorm) => {
                unsafe { *out = lognorm };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Add two tree tensor networks using direct-sum construction.
///
/// # Arguments
/// * `a` - First TreeTN handle
/// * `b` - Second TreeTN handle
/// * `out` - Output pointer for the result
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_add(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn_a = unsafe { &*a };
        let tn_b = unsafe { &*b };

        match tn_a.inner().add(tn_b.inner()) {
            Ok(result) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(result))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Contract two tree tensor networks.
///
/// # Arguments
/// * `a` - First TreeTN handle
/// * `b` - Second TreeTN handle
/// * `method` - Contract method (Zipup=0, Fit=1, Naive=2)
/// * `rtol` - Relative tolerance (0.0 for not set)
/// * `cutoff` - ITensorMPS.jl cutoff (0.0 for not set). Converted to rtol = sqrt(cutoff).
/// * `maxdim` - Maximum bond dimension (0 for no limit)
/// * `out` - Output pointer for the result
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_contract(
    a: *const t4a_treetn,
    b: *const t4a_treetn,
    method: crate::types::t4a_contract_method,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if a.is_null() || b.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn_a = unsafe { &*a };
        let tn_b = unsafe { &*b };

        let rust_method: ContractionMethod = method.into();
        let mut options = ContractionOptions::new(rust_method);

        if let Some(r) = resolve_rtol(rtol, cutoff) {
            options = options.with_rtol(r);
        }
        if maxdim > 0 {
            options = options.with_max_rank(maxdim);
        }

        // Use the first (min) node name as center
        let node_names = tn_a.inner().node_names();
        let center = match node_names.iter().min() {
            Some(&c) => c,
            None => return T4A_INVALID_ARGUMENT,
        };

        match tensor4all_treetn::treetn::contraction::contract(
            tn_a.inner(),
            tn_b.inner(),
            &center,
            options,
        ) {
            Ok(result) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(result))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Convert tree tensor network to a dense tensor by contracting all link indices.
///
/// # Arguments
/// * `treetn` - TreeTN handle
/// * `out` - Output pointer for the dense tensor
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_to_dense(
    treetn: *const t4a_treetn,
    out: *mut *mut t4a_tensor,
) -> StatusCode {
    if treetn.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let tn = unsafe { &*treetn };
        match tn.inner().to_dense() {
            Ok(tensor) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_tensor::new(tensor))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

/// Solve `(a0 + a1 * A) * x = b` for `x` using DMRG-like sweeps.
///
/// # Arguments
/// * `operator` - The operator A (TTN operator)
/// * `rhs` - The right-hand side b (TTN state)
/// * `init` - Initial guess for x (TTN state, cloned internally)
/// * `a0` - Coefficient a0
/// * `a1` - Coefficient a1
/// * `nsweeps` - Number of full sweeps (0 = default)
/// * `rtol` - Relative tolerance (0.0 for not set)
/// * `cutoff` - ITensorMPS.jl cutoff (0.0 for not set)
/// * `maxdim` - Maximum bond dimension (0 = no limit)
/// * `out` - Output pointer for the result
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_linsolve(
    operator: *const t4a_treetn,
    rhs: *const t4a_treetn,
    init: *const t4a_treetn,
    a0: libc::c_double,
    a1: libc::c_double,
    nsweeps: libc::size_t,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: libc::size_t,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    if operator.is_null() || rhs.is_null() || init.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use tensor4all_treetn::LinsolveOptions;

        let op = unsafe { &*operator };
        let b = unsafe { &*rhs };
        let x0 = unsafe { &*init };

        let x0_clone = x0.inner().clone();

        let mut options = LinsolveOptions::default();
        options = options.with_coefficients(a0, a1);

        if nsweeps > 0 {
            options = options.with_nfullsweeps(nsweeps);
        }
        if let Some(r) = resolve_rtol(rtol, cutoff) {
            options = options.with_rtol(r);
        }
        if maxdim > 0 {
            options = options.with_max_rank(maxdim);
        }

        // Use the first (min) node name of init as center
        let node_names = x0.inner().node_names();
        let center = match node_names.iter().min() {
            Some(&c) => c,
            None => return T4A_INVALID_ARGUMENT,
        };

        match tensor4all_treetn::square_linsolve(op.inner(), b.inner(), x0_clone, &center, options)
        {
            Ok(result) => {
                unsafe { *out = Box::into_raw(Box::new(t4a_treetn::new(result.solution))) };
                T4A_SUCCESS
            }
            Err(e) => crate::err_status(e, T4A_INVALID_ARGUMENT),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Swap site indices
// ============================================================================

/// Swap site indices on a TreeTN to a target assignment.
///
/// Rearranges site indices so that each specified index ends up at its
/// target vertex. Values are preserved via SVD factorization.
///
/// # Arguments
/// * `ttn` - Tree tensor network (modified in place)
/// * `index_ids` - Array of index IDs (u64) to move
/// * `target_vertices` - Array of target vertex numbers (same length as index_ids)
/// * `n_pairs` - Number of (index_id, target_vertex) pairs
/// * `max_rank` - Maximum bond dimension after SVD (0 = no limit)
/// * `rtol` - Relative SVD truncation tolerance (0.0 = no truncation)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_swap_site_indices(
    ttn: *mut t4a_treetn,
    index_ids: *const u64,
    target_vertices: *const libc::size_t,
    n_pairs: libc::size_t,
    max_rank: libc::size_t,
    rtol: libc::c_double,
) -> StatusCode {
    if ttn.is_null() {
        return T4A_NULL_POINTER;
    }
    if n_pairs > 0 && (index_ids.is_null() || target_vertices.is_null()) {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        use std::collections::HashMap;
        use tensor4all_core::defaults::index::DynId;
        use tensor4all_treetn::treetn::SwapOptions;

        let inner = unsafe { &mut *ttn }.inner_mut();

        let mut target_map: HashMap<DynId, usize> = HashMap::new();
        for i in 0..n_pairs {
            let id = DynId(unsafe { *index_ids.add(i) });
            let vertex = unsafe { *target_vertices.add(i) };
            target_map.insert(id, vertex);
        }

        let opts = SwapOptions {
            max_rank: if max_rank == 0 { None } else { Some(max_rank) },
            rtol: if rtol == 0.0 { None } else { Some(rtol) },
        };

        match inner.swap_site_indices(&target_map, &opts) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => crate::err_status(e, crate::T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
