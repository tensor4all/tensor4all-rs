//! C API for TreeTCI (tree-structured tensor cross interpolation)

use crate::t4a_treetn;
use crate::types::{t4a_treetci_f64, t4a_treetci_graph, t4a_treetci_proposer_kind};
use crate::{
    err_status, set_last_error, StatusCode, T4A_INTERNAL_ERROR, T4A_NULL_POINTER, T4A_SUCCESS,
};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tensor4all_treetci::{
    DefaultProposer, GlobalIndexBatch, SimpleProposer, SimpleTreeTci, TreeTciEdge, TreeTciGraph,
    TreeTciOptions, TruncatedDefaultProposer,
};

// ============================================================================
// Callback type
// ============================================================================

/// Batch evaluation callback for TreeTCI.
///
/// Evaluates the target function at multiple points simultaneously.
/// When `n_points == 1`, this acts as a single-point evaluation.
///
/// # Arguments
/// * `batch_data` - Column-major (n_sites, n_points) index array.
///   Element at (site, point) is at `batch_data[site + n_sites * point]`.
/// * `n_sites` - Number of sites
/// * `n_points` - Number of evaluation points
/// * `results` - Output buffer for `n_points` f64 values
/// * `user_data` - User data pointer passed through from the calling function
///
/// # Returns
/// 0 on success, non-zero on error
pub type TreeTciBatchEvalCallback = extern "C" fn(
    batch_data: *const libc::size_t,
    n_sites: libc::size_t,
    n_points: libc::size_t,
    results: *mut libc::c_double,
    user_data: *mut c_void,
) -> i32;

// ============================================================================
// Internal helpers
// ============================================================================

/// Create a batch eval closure from the C callback.
///
/// Returns a closure compatible with `Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>>`.
fn make_batch_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    move |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        let mut results = vec![0.0f64; batch.n_points()];
        let status = eval_fn(
            batch.data().as_ptr(),
            batch.n_sites(),
            batch.n_points(),
            results.as_mut_ptr(),
            user_data,
        );
        if status != 0 {
            anyhow::bail!(
                "TreeTCI batch eval callback returned error status {}",
                status
            );
        }
        Ok(results)
    }
}

/// Create a point eval closure from the C batch callback (n_points=1).
fn make_point_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(&[usize]) -> f64 {
    move |indices: &[usize]| -> f64 {
        let mut result: f64 = 0.0;
        let status = eval_fn(indices.as_ptr(), indices.len(), 1, &mut result, user_data);
        if status != 0 {
            f64::NAN
        } else {
            result
        }
    }
}

/// Convert C API parameters to TreeTciOptions.
fn make_options(
    tolerance: f64,
    max_bond_dim: libc::size_t,
    max_iter: libc::size_t,
    normalize_error: bool,
) -> TreeTciOptions {
    TreeTciOptions {
        tolerance,
        max_bond_dim: if max_bond_dim == 0 {
            usize::MAX
        } else {
            max_bond_dim
        },
        max_iter,
        normalize_error,
    }
}

// ============================================================================
// Graph lifecycle
// ============================================================================

impl_opaque_type_common!(treetci_graph);

/// Create a new tree graph.
///
/// # Arguments
/// - `n_sites`: Number of sites (>= 1)
/// - `edges_flat`: Edge pairs [u0, v0, u1, v1, ...] (length = n_edges * 2)
/// - `n_edges`: Number of edges (must equal n_sites - 1 for a tree)
///
/// # Returns
/// New graph handle, or NULL on error (invalid tree structure).
///
/// # Safety
/// `edges_flat` must point to a valid buffer of `n_edges * 2` elements.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_new(
    n_sites: libc::size_t,
    edges_flat: *const libc::size_t,
    n_edges: libc::size_t,
) -> *mut t4a_treetci_graph {
    if edges_flat.is_null() && n_edges > 0 {
        set_last_error("edges_flat is null");
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let edges: Vec<TreeTciEdge> = (0..n_edges)
            .map(|i| {
                let u = unsafe { *edges_flat.add(2 * i) };
                let v = unsafe { *edges_flat.add(2 * i + 1) };
                TreeTciEdge::new(u, v)
            })
            .collect();

        match TreeTciGraph::new(n_sites, &edges) {
            Ok(graph) => Box::into_raw(Box::new(t4a_treetci_graph::new(graph))),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    }));

    crate::unwrap_catch_ptr(result)
}

/// Get the number of sites in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_sites(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode {
    if graph.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &*graph };
        unsafe { *out = g.inner().n_sites() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the number of edges in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_edges(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode {
    if graph.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let g = unsafe { &*graph };
        unsafe { *out = g.inner().edges().len() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// State lifecycle
// ============================================================================

/// Create a new TreeTCI state.
///
/// # Arguments
/// - `local_dims`: Local dimension at each site (length = n_sites)
/// - `n_sites`: Number of sites (must match graph)
/// - `graph`: Tree graph handle (not consumed; cloned internally)
///
/// # Returns
/// New state handle, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_new(
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
) -> *mut t4a_treetci_f64 {
    if local_dims.is_null() || graph.is_null() {
        set_last_error("local_dims or graph is null");
        return std::ptr::null_mut();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();
        let g = unsafe { &*graph };
        let graph_clone = g.inner().clone();

        match SimpleTreeTci::new(dims, graph_clone) {
            Ok(state) => Box::into_raw(Box::new(t4a_treetci_f64::new(state))),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    }));

    crate::unwrap_catch_ptr(result)
}

/// Release a TreeTCI state.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_release(ptr: *mut t4a_treetci_f64) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ============================================================================
// Pivot management
// ============================================================================

/// Add global pivots to the TreeTCI state.
///
/// Each pivot is a multi-index over all sites. The pivots are projected
/// to per-edge pivot sets internally.
///
/// # Arguments
/// - `ptr`: State handle
/// - `pivots_flat`: Column-major (n_sites, n_pivots) index array
/// - `n_sites`: Number of sites (must match state)
/// - `n_pivots`: Number of pivots
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_add_global_pivots(
    ptr: *mut t4a_treetci_f64,
    pivots_flat: *const libc::size_t,
    n_sites: libc::size_t,
    n_pivots: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }
    if pivots_flat.is_null() && n_pivots > 0 {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &mut *ptr };
        let state_inner = state.inner_mut();

        // Unpack column-major (n_sites, n_pivots) to Vec<Vec<usize>>
        let pivots: Vec<Vec<usize>> = (0..n_pivots)
            .map(|p| {
                (0..n_sites)
                    .map(|s| unsafe { *pivots_flat.add(s + n_sites * p) })
                    .collect()
            })
            .collect();

        match state_inner.add_global_pivots(&pivots) {
            Ok(()) => T4A_SUCCESS,
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Sweep execution
// ============================================================================

/// Run one optimization iteration (visit all edges once).
///
/// Internally calls `optimize_with_proposer` with `max_iter=1`.
///
/// # Arguments
/// - `ptr`: State handle (mutable)
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `proposer_kind`: Proposer selection
/// - `tolerance`: Relative tolerance for this iteration
/// - `max_bond_dim`: Maximum bond dimension (0 = unlimited)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_sweep(
    ptr: *mut t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> StatusCode {
    if ptr.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &mut *ptr };
        let state_inner = state.inner_mut();
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);
        let options = make_options(tolerance, max_bond_dim, 1, true);

        let res = match proposer_kind {
            t4a_treetci_proposer_kind::Default => {
                let proposer = DefaultProposer;
                tensor4all_treetci::optimize_with_proposer(
                    state_inner,
                    batch_eval,
                    &options,
                    &proposer,
                )
            }
            t4a_treetci_proposer_kind::Simple => {
                let proposer = SimpleProposer::default();
                tensor4all_treetci::optimize_with_proposer(
                    state_inner,
                    batch_eval,
                    &options,
                    &proposer,
                )
            }
            t4a_treetci_proposer_kind::TruncatedDefault => {
                let proposer = TruncatedDefaultProposer::default();
                tensor4all_treetci::optimize_with_proposer(
                    state_inner,
                    batch_eval,
                    &options,
                    &proposer,
                )
            }
        };

        match res {
            Ok(_) => T4A_SUCCESS,
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// State inspection
// ============================================================================

/// Get the maximum bond error across all edges.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_bond_error(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_bond_error() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum rank (bond dimension) across all edges.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_rank(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_rank() };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the maximum observed sample value (used for normalization).
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_sample_value(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode {
    if ptr.is_null() || out.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        unsafe { *out = state.inner().max_sample_value };
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the bond dimensions (ranks) at each edge.
///
/// Uses query-then-fill: pass `out_ranks = NULL` to query `out_n_edges` only.
///
/// # Arguments
/// - `out_ranks`: Output buffer (length >= n_edges), or NULL to query size
/// - `buf_len`: Buffer capacity
/// - `out_n_edges`: Outputs the number of edges
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_bond_dims(
    ptr: *const t4a_treetci_f64,
    out_ranks: *mut libc::size_t,
    buf_len: libc::size_t,
    out_n_edges: *mut libc::size_t,
) -> StatusCode {
    if ptr.is_null() || out_n_edges.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        let inner = state.inner();
        let edges = inner.graph.edges();
        let n_edges = edges.len();

        unsafe { *out_n_edges = n_edges };

        if out_ranks.is_null() {
            return T4A_SUCCESS;
        }

        if buf_len < n_edges {
            return err_status(
                format!("Buffer too small: need {}, got {}", n_edges, buf_len),
                crate::T4A_BUFFER_TOO_SMALL,
            );
        }

        for (i, edge) in edges.iter().enumerate() {
            // Bond dim = number of pivot rows for either side of this edge
            let (key_u, _key_v) = inner.graph.subregion_vertices(*edge).unwrap();
            let rank = inner.ijset.get(&key_u).map_or(0, |v| v.len());
            unsafe { *out_ranks.add(i) = rank };
        }

        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// Materialization
// ============================================================================

/// Materialize the converged TreeTCI state into a TreeTN.
///
/// Internally re-evaluates tensor values using the batch callback and
/// performs LU factorization to construct per-vertex tensors.
///
/// # Arguments
/// - `ptr`: State handle (const -- state is not modified)
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `center_site`: BFS root site for materialization
/// - `out_treetn`: Output TreeTN handle pointer
///
/// # Returns
/// The result is a `t4a_treetn` handle. Release with `t4a_treetn_release`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_to_treetn(
    ptr: *const t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
) -> StatusCode {
    if ptr.is_null() || out_treetn.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { &*ptr };
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);

        match tensor4all_treetci::to_treetn(state.inner(), batch_eval, Some(center_site)) {
            Ok(treetn) => {
                unsafe { *out_treetn = Box::into_raw(Box::new(t4a_treetn::new(treetn))) };
                T4A_SUCCESS
            }
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

// ============================================================================
// High-level convenience function
// ============================================================================

/// Run TreeTCI to convergence and return a TreeTN.
///
/// Equivalent to: new -> add_pivots -> sweep loop -> materialize.
///
/// # Arguments
/// - `eval_cb`: Batch evaluation callback
/// - `user_data`: User data passed to callback
/// - `local_dims`: Local dimension at each site (length = n_sites)
/// - `n_sites`: Number of sites
/// - `graph`: Tree graph handle
/// - `initial_pivots_flat`: Column-major (n_sites, n_pivots), or NULL for empty
/// - `n_pivots`: Number of initial pivots
/// - `proposer_kind`: Proposer selection
/// - `tolerance`: Relative tolerance
/// - `max_bond_dim`: Maximum bond dimension (0 = unlimited)
/// - `max_iter`: Maximum number of iterations
/// - `normalize_error`: Whether to normalize errors (0=false, 1=true)
/// - `center_site`: Materialization center site
/// - `out_treetn`: Output TreeTN handle
/// - `out_ranks`: Buffer for max rank per iteration (length >= max_iter), or NULL
/// - `out_errors`: Buffer for normalized error per iteration (length >= max_iter), or NULL
/// - `out_n_iters`: Output: actual number of iterations performed
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn t4a_crossinterpolate_tree_f64(
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
    initial_pivots_flat: *const libc::size_t,
    n_pivots: libc::size_t,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
    max_iter: libc::size_t,
    normalize_error: libc::c_int,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode {
    if local_dims.is_null() || graph.is_null() || out_treetn.is_null() || out_n_iters.is_null() {
        return T4A_NULL_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Parse local_dims
        let dims: Vec<usize> = (0..n_sites)
            .map(|i| unsafe { *local_dims.add(i) })
            .collect();

        // Clone graph
        let g = unsafe { &*graph };
        let graph_clone = g.inner().clone();

        // Create state
        let mut state = match SimpleTreeTci::new(dims, graph_clone) {
            Ok(s) => s,
            Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
        };

        // Add initial pivots
        if !initial_pivots_flat.is_null() && n_pivots > 0 {
            let pivots: Vec<Vec<usize>> = (0..n_pivots)
                .map(|p| {
                    (0..n_sites)
                        .map(|s| unsafe { *initial_pivots_flat.add(s + n_sites * p) })
                        .collect()
                })
                .collect();
            if let Err(e) = state.add_global_pivots(&pivots) {
                return err_status(e, T4A_INTERNAL_ERROR);
            }
        }

        // Evaluate max_sample_value at initial pivots using the point eval closure
        let point_eval = make_point_eval_closure(eval_cb, user_data);
        if !initial_pivots_flat.is_null() && n_pivots > 0 {
            for p in 0..n_pivots {
                let pivot: Vec<usize> = (0..n_sites)
                    .map(|s| unsafe { *initial_pivots_flat.add(s + n_sites * p) })
                    .collect();
                let val = point_eval(&pivot);
                state.max_sample_value = state.max_sample_value.max(val.abs());
            }
        }

        // Run optimization
        let batch_eval = make_batch_eval_closure(eval_cb, user_data);
        let options = make_options(tolerance, max_bond_dim, max_iter, normalize_error != 0);

        let (ranks, errors) = match proposer_kind {
            t4a_treetci_proposer_kind::Default => {
                let proposer = DefaultProposer;
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state,
                    &batch_eval,
                    &options,
                    &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
            t4a_treetci_proposer_kind::Simple => {
                let proposer = SimpleProposer::default();
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state,
                    &batch_eval,
                    &options,
                    &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
            t4a_treetci_proposer_kind::TruncatedDefault => {
                let proposer = TruncatedDefaultProposer::default();
                match tensor4all_treetci::optimize_with_proposer(
                    &mut state,
                    &batch_eval,
                    &options,
                    &proposer,
                ) {
                    Ok(r) => r,
                    Err(e) => return err_status(e, T4A_INTERNAL_ERROR),
                }
            }
        };

        let n_iters = ranks.len();
        unsafe { *out_n_iters = n_iters };

        // Copy ranks and errors to output buffers
        if !out_ranks.is_null() {
            for (i, &r) in ranks.iter().enumerate() {
                unsafe { *out_ranks.add(i) = r };
            }
        }
        if !out_errors.is_null() {
            for (i, &e) in errors.iter().enumerate() {
                unsafe { *out_errors.add(i) = e };
            }
        }

        // Materialize
        match tensor4all_treetci::to_treetn(&state, &batch_eval, Some(center_site)) {
            Ok(treetn) => {
                unsafe { *out_treetn = Box::into_raw(Box::new(t4a_treetn::new(treetn))) };
                T4A_SUCCESS
            }
            Err(e) => err_status(e, T4A_INTERNAL_ERROR),
        }
    }));

    crate::unwrap_catch(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 7-site branching tree
    ///     0
    ///     |
    ///     1---2
    ///     |
    ///     3
    ///     |
    ///     4
    ///    / \
    ///   5   6
    fn sample_edges() -> Vec<libc::size_t> {
        // flat: [u0,v0, u1,v1, ...]
        vec![0, 1, 1, 2, 1, 3, 3, 4, 4, 5, 4, 6]
    }

    #[test]
    fn test_graph_new_and_query() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        assert!(!graph.is_null());

        let mut n_sites: libc::size_t = 0;
        let status = t4a_treetci_graph_n_sites(graph, &mut n_sites);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_sites, 7);

        let mut n_edges: libc::size_t = 0;
        let status = t4a_treetci_graph_n_edges(graph, &mut n_edges);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_edges, 6);

        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_graph_invalid_disconnected() {
        // 4 sites but only 2 edges, disconnected
        let edges: Vec<libc::size_t> = vec![0, 1, 2, 3];
        let graph = t4a_treetci_graph_new(4, edges.as_ptr(), 2);
        assert!(graph.is_null()); // should fail validation
    }

    /// Product function: f(idx) = prod(idx[s] + 1.0)
    /// This has an exact TT representation with bond dim 1.
    extern "C" fn product_batch_eval(
        batch_data: *const libc::size_t,
        n_sites: libc::size_t,
        n_points: libc::size_t,
        results: *mut libc::c_double,
        _user_data: *mut c_void,
    ) -> i32 {
        for p in 0..n_points {
            let mut val = 1.0f64;
            for s in 0..n_sites {
                let idx = unsafe { *batch_data.add(s + n_sites * p) };
                val *= (idx as f64) + 1.0;
            }
            unsafe { *results.add(p) = val };
        }
        0
    }

    #[test]
    fn test_state_new_and_add_pivots() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        assert!(!graph.is_null());

        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);
        assert!(!state.is_null());

        // Add one pivot: all zeros (column-major, n_sites=7, n_pivots=1)
        let pivot: Vec<libc::size_t> = vec![0; 7];
        let status = t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);
        assert_eq!(status, T4A_SUCCESS);

        // Add two pivots at once (column-major)
        // pivot0 = [0,0,0,0,0,0,0], pivot1 = [1,0,1,0,1,0,1]
        let pivots: Vec<libc::size_t> = vec![
            0, 0, 0, 0, 0, 0, 0, // column 0 (sites 0-6 for point 0)
            1, 0, 1, 0, 1, 0, 1, // column 1 (sites 0-6 for point 1)
        ];
        let status = t4a_treetci_f64_add_global_pivots(state, pivots.as_ptr(), 7, 2);
        assert_eq!(status, T4A_SUCCESS);

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_sweep() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        // Add initial pivot
        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        // Run one sweep
        let status = t4a_treetci_f64_sweep(
            state,
            product_batch_eval,
            std::ptr::null_mut(), // no user_data needed
            t4a_treetci_proposer_kind::Default,
            1e-12,
            0, // unlimited bond dim
        );
        assert_eq!(status, T4A_SUCCESS);

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_state_inspection() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        // Run a few sweeps
        for _ in 0..4 {
            t4a_treetci_f64_sweep(
                state,
                product_batch_eval,
                std::ptr::null_mut(),
                t4a_treetci_proposer_kind::Default,
                1e-12,
                0,
            );
        }

        // max_bond_error
        let mut error: libc::c_double = 0.0;
        let status = t4a_treetci_f64_max_bond_error(state, &mut error);
        assert_eq!(status, T4A_SUCCESS);
        assert!(error < 1e-10, "error = {}", error);

        // max_rank
        let mut rank: libc::size_t = 0;
        let status = t4a_treetci_f64_max_rank(state, &mut rank);
        assert_eq!(status, T4A_SUCCESS);
        assert!(rank >= 1);

        // max_sample_value
        let mut max_val: libc::c_double = 0.0;
        let status = t4a_treetci_f64_max_sample_value(state, &mut max_val);
        assert_eq!(status, T4A_SUCCESS);
        assert!(max_val > 0.0);

        // bond_dims: query size first
        let mut n_edges: libc::size_t = 0;
        let status = t4a_treetci_f64_bond_dims(state, std::ptr::null_mut(), 0, &mut n_edges);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_edges, 6);

        // bond_dims: fill buffer
        let mut dims = vec![0usize; n_edges];
        let status = t4a_treetci_f64_bond_dims(state, dims.as_mut_ptr(), n_edges, &mut n_edges);
        assert_eq!(status, T4A_SUCCESS);
        for &d in &dims {
            assert!(d >= 1);
        }

        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_to_treetn() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let state = t4a_treetci_f64_new(local_dims.as_ptr(), 7, graph);

        let pivot: Vec<libc::size_t> = vec![0; 7];
        t4a_treetci_f64_add_global_pivots(state, pivot.as_ptr(), 7, 1);

        for _ in 0..4 {
            t4a_treetci_f64_sweep(
                state,
                product_batch_eval,
                std::ptr::null_mut(),
                t4a_treetci_proposer_kind::Default,
                1e-12,
                0,
            );
        }

        // Materialize to TreeTN
        let mut treetn_ptr: *mut t4a_treetn = std::ptr::null_mut();
        let status = t4a_treetci_f64_to_treetn(
            state,
            product_batch_eval,
            std::ptr::null_mut(),
            0, // center_site
            &mut treetn_ptr,
        );
        assert_eq!(status, T4A_SUCCESS);
        assert!(!treetn_ptr.is_null());

        // Verify TreeTN is valid by checking vertex count
        let mut n_vertices: libc::size_t = 0;
        let status = crate::t4a_treetn_num_vertices(treetn_ptr, &mut n_vertices);
        assert_eq!(status, T4A_SUCCESS);
        assert_eq!(n_vertices, 7);

        crate::t4a_treetn_release(treetn_ptr);
        t4a_treetci_f64_release(state);
        t4a_treetci_graph_release(graph);
    }

    #[test]
    fn test_crossinterpolate_tree_f64() {
        let edges = sample_edges();
        let graph = t4a_treetci_graph_new(7, edges.as_ptr(), 6);
        let local_dims: Vec<libc::size_t> = vec![2; 7];
        let initial_pivot: Vec<libc::size_t> = vec![0; 7];

        let max_iter: libc::size_t = 10;
        let mut out_treetn: *mut t4a_treetn = std::ptr::null_mut();
        let mut out_ranks = vec![0usize; max_iter];
        let mut out_errors = vec![0.0f64; max_iter];
        let mut out_n_iters: libc::size_t = 0;

        let status = t4a_crossinterpolate_tree_f64(
            product_batch_eval,
            std::ptr::null_mut(),
            local_dims.as_ptr(),
            7,
            graph,
            initial_pivot.as_ptr(),
            1, // n_pivots
            t4a_treetci_proposer_kind::Default,
            1e-12, // tolerance
            0,     // max_bond_dim (unlimited)
            max_iter,
            1, // normalize_error = true
            0, // center_site
            &mut out_treetn,
            out_ranks.as_mut_ptr(),
            out_errors.as_mut_ptr(),
            &mut out_n_iters,
        );

        assert_eq!(status, T4A_SUCCESS);
        assert!(!out_treetn.is_null());
        assert!(out_n_iters > 0);
        assert!(out_n_iters <= max_iter);

        // Verify convergence
        let actual_iters = out_n_iters;
        let last_error = out_errors[actual_iters - 1];
        assert!(last_error < 1e-10, "last_error = {}", last_error);

        // Verify TreeTN
        let mut n_vertices: libc::size_t = 0;
        crate::t4a_treetn_num_vertices(out_treetn, &mut n_vertices);
        assert_eq!(n_vertices, 7);

        crate::t4a_treetn_release(out_treetn);
        t4a_treetci_graph_release(graph);
    }
}
