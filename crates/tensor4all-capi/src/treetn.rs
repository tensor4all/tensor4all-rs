//! C API for TreeTN (Tree Tensor Network) operations
//!
//! Provides functions to create, manipulate, and query tree tensor networks.
//! This replaces the old `tensortrain.rs` module with a direct wrapper around
//! `tensor4all_treetn::DefaultTreeTN<usize>`.

use crate::types::{t4a_canonical_form, t4a_index, t4a_tensor, t4a_treetn, InternalTreeTN};
use crate::{StatusCode, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS};
use std::panic::{catch_unwind, AssertUnwindSafe};

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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
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
            Err(_) => T4A_INVALID_ARGUMENT,
        }
    }));

    result.unwrap_or(T4A_INTERNAL_ERROR)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Helper: create a 2-site MPS-like TreeTN with site dims (2, 2) and bond dim 3.
    fn make_two_site_treetn() -> (*mut t4a_treetn, Vec<*mut t4a_tensor>, Vec<*mut t4a_index>) {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        let s0 = t4a_index_new(2);
        let l01 = t4a_index_new(3);
        let s1 = t4a_index_new(2);

        let data0: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
        let data1: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();

        let inds0: [*const t4a_index; 2] = [s0, l01];
        let dims0: [libc::size_t; 2] = [2, 3];
        let t0 = t4a_tensor_new_dense_f64(2, inds0.as_ptr(), dims0.as_ptr(), data0.as_ptr(), 6);

        let l01_clone = t4a_index_clone(l01);
        let inds1: [*const t4a_index; 2] = [l01_clone, s1];
        let dims1: [libc::size_t; 2] = [3, 2];
        let t1 = t4a_tensor_new_dense_f64(2, inds1.as_ptr(), dims1.as_ptr(), data1.as_ptr(), 6);

        let tensors: [*const t4a_tensor; 2] = [t0, t1];
        let mut out: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetn_new(tensors.as_ptr(), 2, &mut out);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!out.is_null());

        (out, vec![t0, t1], vec![s0, l01, l01_clone, s1])
    }

    fn cleanup_treetn(
        tt: *mut t4a_treetn,
        tensors: Vec<*mut t4a_tensor>,
        indices: Vec<*mut t4a_index>,
    ) {
        t4a_treetn_release(tt);
        for t in tensors {
            crate::t4a_tensor_release(t);
        }
        for i in indices {
            crate::t4a_index_release(i);
        }
    }

    // ========================================================================
    // Lifecycle tests
    // ========================================================================

    #[test]
    fn test_treetn_new_and_release() {
        let (tt, tensors, indices) = make_two_site_treetn();

        // Check vertex count
        let mut n_vertices: libc::size_t = 0;
        let status = t4a_treetn_num_vertices(tt, &mut n_vertices);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_vertices, 2);

        // Check edge count
        let mut n_edges: libc::size_t = 0;
        let status = t4a_treetn_num_edges(tt, &mut n_edges);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_edges, 1);

        // Clone
        let tt2 = t4a_treetn_clone(tt);
        assert!(!tt2.is_null());
        t4a_treetn_release(tt2);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_tensor_access() {
        let (tt, tensors, indices) = make_two_site_treetn();

        // Get tensor at vertex 0
        let mut tensor_out: *mut t4a_tensor = std::ptr::null_mut();
        let status = t4a_treetn_tensor(tt, 0, &mut tensor_out);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!tensor_out.is_null());

        // Check tensor rank
        let mut rank: libc::size_t = 0;
        crate::t4a_tensor_get_rank(tensor_out, &mut rank);
        assert_eq!(rank, 2); // site + link
        crate::t4a_tensor_release(tensor_out);

        // Invalid vertex
        let mut tensor_out2: *mut t4a_tensor = std::ptr::null_mut();
        let status = t4a_treetn_tensor(tt, 99, &mut tensor_out2);
        assert_eq!(status, T4A_INVALID_ARGUMENT);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_neighbors() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let mut buf = [0usize; 4];
        let mut n_out: libc::size_t = 0;
        let status = t4a_treetn_neighbors(tt, 0, buf.as_mut_ptr(), 4, &mut n_out);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_out, 1);
        assert_eq!(buf[0], 1);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_linkind_and_bond_dim() {
        let (tt, tensors, indices) = make_two_site_treetn();

        // Link index between vertex 0 and 1
        let mut idx_out: *mut t4a_index = std::ptr::null_mut();
        let status = t4a_treetn_linkind(tt, 0, 1, &mut idx_out);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!idx_out.is_null());
        crate::t4a_index_release(idx_out);

        // Bond dimension
        let mut dim: libc::size_t = 0;
        let status = t4a_treetn_bond_dim(tt, 0, 1, &mut dim);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(dim, 3);

        // MPS convenience
        let mut dim2: libc::size_t = 0;
        let status = t4a_treetn_bond_dim_at(tt, 0, &mut dim2);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(dim2, 3);

        // maxbonddim
        let mut max_dim: libc::size_t = 0;
        let status = t4a_treetn_maxbonddim(tt, &mut max_dim);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(max_dim, 3);

        // bond_dims
        let mut dims = [0usize; 1];
        let status = t4a_treetn_bond_dims(tt, dims.as_mut_ptr(), 1);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(dims[0], 3);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_siteinds() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let mut idx_buf: [*mut t4a_index; 4] = [std::ptr::null_mut(); 4];
        let mut n_out: libc::size_t = 0;
        let status = t4a_treetn_siteinds(tt, 0, idx_buf.as_mut_ptr(), 4, &mut n_out);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_out, 1); // one site index at vertex 0

        // Clean up returned indices
        for i in 0..n_out {
            if !idx_buf[i].is_null() {
                crate::t4a_index_release(idx_buf[i]);
            }
        }

        cleanup_treetn(tt, tensors, indices);
    }

    // ========================================================================
    // Orthogonalization tests
    // ========================================================================

    #[test]
    fn test_treetn_orthogonalize() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let status = t4a_treetn_orthogonalize(tt, 0);
        assert_eq!(status, T4A_SUCCESS);

        // Check canonical form
        let mut form: t4a_canonical_form = t4a_canonical_form::LU;
        let status = t4a_treetn_canonical_form(tt, &mut form);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(form, t4a_canonical_form::Unitary);

        // Check ortho center
        let mut center_buf = [0usize; 4];
        let mut n_center: libc::size_t = 0;
        let status = t4a_treetn_ortho_center(tt, center_buf.as_mut_ptr(), 4, &mut n_center);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_center, 1);
        assert_eq!(center_buf[0], 0);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_orthogonalize_with() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let status = t4a_treetn_orthogonalize_with(tt, 1, t4a_canonical_form::LU);
        assert_eq!(status, T4A_SUCCESS);

        let mut form: t4a_canonical_form = t4a_canonical_form::Unitary;
        let status = t4a_treetn_canonical_form(tt, &mut form);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(form, t4a_canonical_form::LU);

        cleanup_treetn(tt, tensors, indices);
    }

    // ========================================================================
    // Operation tests
    // ========================================================================

    #[test]
    fn test_treetn_truncate() {
        let (tt, tensors, indices) = make_two_site_treetn();

        // Truncate with maxdim=1
        let status = t4a_treetn_truncate(tt, 0.0, 0.0, 1);
        assert_eq!(status, T4A_SUCCESS);

        let mut max_dim: libc::size_t = 0;
        t4a_treetn_maxbonddim(tt, &mut max_dim);
        assert_eq!(max_dim, 1);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_truncate_with_cutoff() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let status = t4a_treetn_truncate(tt, 0.0, 1e-10, 0);
        assert_eq!(status, T4A_SUCCESS);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_truncate_with_rtol() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let status = t4a_treetn_truncate(tt, 1e-5, 0.0, 0);
        assert_eq!(status, T4A_SUCCESS);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_inner() {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        // Build two 1-site TreeTNs
        let s0 = t4a_index_new(2);
        let s0_clone = t4a_index_clone(s0);

        let data_a: Vec<f64> = vec![1.0, 0.0];
        let data_b: Vec<f64> = vec![0.0, 1.0];

        let inds_a: [*const t4a_index; 1] = [s0];
        let dims_a: [libc::size_t; 1] = [2];
        let t0 = t4a_tensor_new_dense_f64(1, inds_a.as_ptr(), dims_a.as_ptr(), data_a.as_ptr(), 2);

        let inds_b: [*const t4a_index; 1] = [s0_clone];
        let dims_b: [libc::size_t; 1] = [2];
        let t1 = t4a_tensor_new_dense_f64(1, inds_b.as_ptr(), dims_b.as_ptr(), data_b.as_ptr(), 2);

        let tensors0: [*const t4a_tensor; 1] = [t0];
        let tensors1: [*const t4a_tensor; 1] = [t1];

        let mut tt0: *mut t4a_treetn = std::ptr::null_mut();
        let mut tt1: *mut t4a_treetn = std::ptr::null_mut();
        t4a_treetn_new(tensors0.as_ptr(), 1, &mut tt0);
        t4a_treetn_new(tensors1.as_ptr(), 1, &mut tt1);

        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        let status = t4a_treetn_inner(tt0, tt1, &mut re, &mut im);
        assert_eq!(status, T4A_SUCCESS);
        assert!((re - 0.0).abs() < 1e-10);
        assert!((im - 0.0).abs() < 1e-10);

        t4a_treetn_release(tt0);
        t4a_treetn_release(tt1);
        crate::t4a_tensor_release(t0);
        crate::t4a_tensor_release(t1);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(s0_clone);
    }

    #[test]
    fn test_treetn_norm() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let mut norm: f64 = 0.0;
        let status = t4a_treetn_norm(tt, &mut norm);
        assert_eq!(status, T4A_SUCCESS);
        assert!(norm > 0.0);

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_lognorm() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let mut lognorm: f64 = 0.0;
        let status = t4a_treetn_lognorm(tt, &mut lognorm);
        assert_eq!(status, T4A_SUCCESS);
        assert!(lognorm.is_finite());

        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_add() {
        let (tt1, tensors1, indices1) = make_two_site_treetn();
        let (tt2, tensors2, indices2) = make_two_site_treetn();

        let mut result: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetn_add(tt1, tt2, &mut result);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!result.is_null());

        // Result should have 2 vertices
        let mut n_vertices: libc::size_t = 0;
        t4a_treetn_num_vertices(result, &mut n_vertices);
        assert_eq!(n_vertices, 2);

        // Bond dimension should be sum (3 + 3 = 6)
        let mut max_bond: libc::size_t = 0;
        t4a_treetn_maxbonddim(result, &mut max_bond);
        assert_eq!(max_bond, 6);

        t4a_treetn_release(result);
        cleanup_treetn(tt1, tensors1, indices1);
        cleanup_treetn(tt2, tensors2, indices2);
    }

    #[test]
    fn test_treetn_to_dense() {
        let (tt, tensors, indices) = make_two_site_treetn();

        let mut dense: *mut t4a_tensor = std::ptr::null_mut();
        let status = t4a_treetn_to_dense(tt, &mut dense);
        assert_eq!(status, T4A_SUCCESS);
        assert!(!dense.is_null());

        let mut rank: libc::size_t = 0;
        crate::t4a_tensor_get_rank(dense, &mut rank);
        assert_eq!(rank, 2); // two site indices

        crate::t4a_tensor_release(dense);
        cleanup_treetn(tt, tensors, indices);
    }

    #[test]
    fn test_treetn_contract() {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        let s0 = t4a_index_new(2);
        let s0_clone = t4a_index_clone(s0);

        let data: Vec<f64> = vec![1.0, 2.0];

        let inds0: [*const t4a_index; 1] = [s0];
        let dims: [libc::size_t; 1] = [2];
        let t0 = t4a_tensor_new_dense_f64(1, inds0.as_ptr(), dims.as_ptr(), data.as_ptr(), 2);

        let inds1: [*const t4a_index; 1] = [s0_clone];
        let t1 = t4a_tensor_new_dense_f64(1, inds1.as_ptr(), dims.as_ptr(), data.as_ptr(), 2);

        let tensors0: [*const t4a_tensor; 1] = [t0];
        let tensors1: [*const t4a_tensor; 1] = [t1];

        let mut tt0: *mut t4a_treetn = std::ptr::null_mut();
        let mut tt1: *mut t4a_treetn = std::ptr::null_mut();
        t4a_treetn_new(tensors0.as_ptr(), 1, &mut tt0);
        t4a_treetn_new(tensors1.as_ptr(), 1, &mut tt1);

        let mut result: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetn_contract(
            tt0,
            tt1,
            crate::types::t4a_contract_method::Zipup,
            0.0,
            0.0,
            0,
            &mut result,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!result.is_null());

        let mut n_vertices: libc::size_t = 0;
        t4a_treetn_num_vertices(result, &mut n_vertices);
        assert_eq!(n_vertices, 1);

        t4a_treetn_release(result);
        t4a_treetn_release(tt0);
        t4a_treetn_release(tt1);
        crate::t4a_tensor_release(t0);
        crate::t4a_tensor_release(t1);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(s0_clone);
    }

    #[test]
    fn test_treetn_linsolve() {
        use crate::{t4a_index_clone, t4a_index_new, t4a_tensor_new_dense_f64};

        // Build 1-site identity MPO and 1-site MPS for I * x = b
        let s0 = t4a_index_new(2);
        let s0p = t4a_index_new(2);

        let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let mpo_inds: [*const t4a_index; 2] = [s0, s0p];
        let mpo_dims: [libc::size_t; 2] = [2, 2];
        let mpo_t = t4a_tensor_new_dense_f64(
            2,
            mpo_inds.as_ptr(),
            mpo_dims.as_ptr(),
            identity.as_ptr(),
            4,
        );
        let mpo_tensors: [*const t4a_tensor; 1] = [mpo_t];
        let mut mpo: *mut t4a_treetn = std::ptr::null_mut();
        t4a_treetn_new(mpo_tensors.as_ptr(), 1, &mut mpo);
        assert!(!mpo.is_null());

        // RHS with index s0p
        let s0p_clone = t4a_index_clone(s0p);
        let rhs_data: Vec<f64> = vec![3.0, 4.0];
        let rhs_inds: [*const t4a_index; 1] = [s0p_clone];
        let rhs_dims: [libc::size_t; 1] = [2];
        let rhs_t = t4a_tensor_new_dense_f64(
            1,
            rhs_inds.as_ptr(),
            rhs_dims.as_ptr(),
            rhs_data.as_ptr(),
            2,
        );
        let rhs_tensors: [*const t4a_tensor; 1] = [rhs_t];
        let mut rhs: *mut t4a_treetn = std::ptr::null_mut();
        t4a_treetn_new(rhs_tensors.as_ptr(), 1, &mut rhs);
        assert!(!rhs.is_null());

        // Initial guess with index s0
        let s0_clone = t4a_index_clone(s0);
        let init_data: Vec<f64> = vec![1.0, 1.0];
        let init_inds: [*const t4a_index; 1] = [s0_clone];
        let init_dims: [libc::size_t; 1] = [2];
        let init_t = t4a_tensor_new_dense_f64(
            1,
            init_inds.as_ptr(),
            init_dims.as_ptr(),
            init_data.as_ptr(),
            2,
        );
        let init_tensors: [*const t4a_tensor; 1] = [init_t];
        let mut init: *mut t4a_treetn = std::ptr::null_mut();
        t4a_treetn_new(init_tensors.as_ptr(), 1, &mut init);
        assert!(!init.is_null());

        let mut result: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetn_linsolve(
            mpo,
            rhs,
            init,
            0.0, // a0
            1.0, // a1
            4,   // nsweeps
            1e-10,
            0.0,
            10,
            &mut result,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!result.is_null());

        let mut n_vertices: libc::size_t = 0;
        t4a_treetn_num_vertices(result, &mut n_vertices);
        assert_eq!(n_vertices, 1);

        t4a_treetn_release(result);
        t4a_treetn_release(mpo);
        t4a_treetn_release(rhs);
        t4a_treetn_release(init);
        crate::t4a_tensor_release(mpo_t);
        crate::t4a_tensor_release(rhs_t);
        crate::t4a_tensor_release(init_t);
        crate::t4a_index_release(s0);
        crate::t4a_index_release(s0p);
        crate::t4a_index_release(s0p_clone);
        crate::t4a_index_release(s0_clone);
    }

    // ========================================================================
    // Null pointer guard tests
    // ========================================================================

    #[test]
    fn test_treetn_null_guards() {
        let mut dummy_out: *mut t4a_treetn = std::ptr::null_mut();
        let mut dummy_tensor: *mut t4a_tensor = std::ptr::null_mut();
        let mut dummy_index: *mut t4a_index = std::ptr::null_mut();
        let mut dummy_size: libc::size_t = 0;
        let mut dummy_double: f64 = 0.0;
        let mut dummy_form = t4a_canonical_form::Unitary;

        // t4a_treetn_new with null out
        assert_eq!(
            t4a_treetn_new(std::ptr::null(), 0, std::ptr::null_mut()),
            T4A_NULL_POINTER
        );

        // t4a_treetn_new with null tensors but non-zero count
        assert_eq!(
            t4a_treetn_new(std::ptr::null(), 3, &mut dummy_out),
            T4A_NULL_POINTER
        );

        // Accessors
        assert_eq!(
            t4a_treetn_num_vertices(std::ptr::null(), &mut dummy_size),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_num_edges(std::ptr::null(), &mut dummy_size),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_tensor(std::ptr::null(), 0, &mut dummy_tensor),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_set_tensor(std::ptr::null_mut(), 0, std::ptr::null()),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_neighbors(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                &mut dummy_size
            ),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_linkind(std::ptr::null(), 0, 1, &mut dummy_index),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_siteinds(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                &mut dummy_size
            ),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_bond_dim(std::ptr::null(), 0, 1, &mut dummy_size),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_maxbonddim(std::ptr::null(), &mut dummy_size),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_bond_dims(std::ptr::null(), std::ptr::null_mut(), 0),
            T4A_NULL_POINTER
        );

        // Orthogonalization
        assert_eq!(
            t4a_treetn_orthogonalize(std::ptr::null_mut(), 0),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_orthogonalize_with(std::ptr::null_mut(), 0, t4a_canonical_form::Unitary),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_ortho_center(std::ptr::null(), std::ptr::null_mut(), 0, &mut dummy_size),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_canonical_form(std::ptr::null(), &mut dummy_form),
            T4A_NULL_POINTER
        );

        // Operations
        assert_eq!(
            t4a_treetn_truncate(std::ptr::null_mut(), 0.0, 0.0, 0),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_inner(
                std::ptr::null(),
                std::ptr::null(),
                &mut dummy_double,
                &mut dummy_double
            ),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_norm(std::ptr::null_mut(), &mut dummy_double),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_lognorm(std::ptr::null_mut(), &mut dummy_double),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_add(std::ptr::null(), std::ptr::null(), &mut dummy_out),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_contract(
                std::ptr::null(),
                std::ptr::null(),
                crate::types::t4a_contract_method::Zipup,
                0.0,
                0.0,
                0,
                &mut dummy_out
            ),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_to_dense(std::ptr::null(), &mut dummy_tensor),
            T4A_NULL_POINTER
        );
        assert_eq!(
            t4a_treetn_linsolve(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.0,
                1.0,
                0,
                0.0,
                0.0,
                0,
                &mut dummy_out
            ),
            T4A_NULL_POINTER
        );
    }
}
