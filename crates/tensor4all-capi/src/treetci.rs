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
}
