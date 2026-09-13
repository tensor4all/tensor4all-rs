//! Python bindings for tree tensor cross interpolation.
//!
//! The only evaluator boundary is a **batched** Python callable. TreeTCI asks
//! for many points at once (candidate matrices, site-tensor fills, and global
//! pivot searches are all batched), so a per-point Python API would multiply
//! interpreter overhead by the batch size. There is therefore no point-wise
//! adapter here: `evaluate` receives a whole batch.
//!
//! # Layout rule
//!
//! Rust hands the evaluator a column-major `(n_sites, n_points)` buffer holding
//! one contiguous `n_sites`-long block per point (`data[site + n_sites * point]`).
//! Exposing the same bytes as a C-contiguous `(n_points, n_sites)` array means
//! `points[point, site]` addresses exactly that element: the batch axis is axis
//! 0, the site axis is axis 1, and nothing is transposed or copied twice.
//!
//! The batch axis is always present, even for a single point, so `points.shape`
//! is always `(n_points, n_sites)` and the returned array must be `(n_points,)`.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions,
};

use crate::treetn::PyTreeTensorNetwork;

/// Initial evaluator result, which fixes the value type of the whole run.
enum ProbeValues {
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

/// One already-evaluated batch, served once so the probe does not cost an extra
/// Python call.
struct Probe<T> {
    points: Vec<usize>,
    values: Vec<T>,
}

/// Build the `(n_points, n_sites)` int64 array handed to the Python evaluator.
fn points_to_numpy<'py>(
    py: Python<'py>,
    batch: &GlobalIndexBatch<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    let n_sites = batch.n_sites();
    let n_points = batch.n_points();
    let flat: Vec<i64> = batch.data().iter().map(|&value| value as i64).collect();
    let array = PyArray1::<i64>::from_vec(py, flat);
    // The source is C-contiguous, so this is a view of the same buffer.
    Ok(array.reshape((n_points, n_sites))?.into_any())
}

/// Copy one typed evaluator result out of a Python object.
///
/// `Ok(None)` means "not a 1-D array of this dtype"; the caller decides how to
/// report that.
fn extract_values<T: numpy::Element + Copy>(
    value: &Bound<'_, PyAny>,
    n_points: usize,
) -> PyResult<Option<Vec<T>>> {
    let Ok(array) = value.extract::<PyReadonlyArray1<'_, T>>() else {
        return Ok(None);
    };
    let values: Vec<T> = array.as_array().iter().copied().collect();
    if values.len() != n_points {
        return Err(PyValueError::new_err(format!(
            "evaluate returned {} values for a batch of {n_points} points",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn unexpected_result(value: &Bound<'_, PyAny>, n_points: usize, required: &str) -> PyErr {
    let describe = |name: &str| {
        value
            .getattr(name)
            .map(|item| item.to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    };
    PyTypeError::new_err(format!(
        "evaluate must return a numpy array of shape ({n_points},) with dtype {required}; \
         got shape={} dtype={} (for example np.asarray(values, dtype=np.float64))",
        describe("shape"),
        describe("dtype"),
    ))
}

/// Keep the original Python exception so it can be re-raised after
/// `crossinterpolate2` reports an error of its own.
fn stash(slot: &Rc<RefCell<Option<PyErr>>>, error: PyErr) -> anyhow::Error {
    *slot.borrow_mut() = Some(error);
    anyhow::anyhow!("the evaluate callback failed")
}

fn take_error(slot: &Rc<RefCell<Option<PyErr>>>, error: anyhow::Error) -> PyErr {
    slot.borrow_mut()
        .take()
        .unwrap_or_else(|| PyValueError::new_err(error.to_string()))
}

fn panic_error(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown payload".to_string());
    PyRuntimeError::new_err(format!("TreeTCI panicked inside Rust: {message}"))
}

/// Produce the values for one batch: serve the cached probe batch when it is
/// the batch being asked for, otherwise call `evaluate`.
fn next_values<T, Extract>(
    py: Python<'_>,
    evaluate: &Bound<'_, PyAny>,
    batch: &GlobalIndexBatch<'_>,
    cache: &RefCell<Option<Probe<T>>>,
    slot: &Rc<RefCell<Option<PyErr>>>,
    extract: Extract,
    required: &str,
) -> anyhow::Result<Vec<T>>
where
    Extract: Fn(&Bound<'_, PyAny>, usize) -> PyResult<Option<Vec<T>>>,
{
    let n_points = batch.n_points();
    if let Some(probe) = cache.borrow_mut().take() {
        if probe.points.as_slice() == batch.data() {
            return Ok(probe.values);
        }
    }
    let points = points_to_numpy(py, batch).map_err(|error| stash(slot, error))?;
    let value = evaluate
        .call1((points,))
        .map_err(|error| stash(slot, error))?;
    match extract(&value, n_points) {
        Ok(Some(values)) => Ok(values),
        Ok(None) => Err(stash(slot, unexpected_result(&value, n_points, required))),
        Err(error) => Err(stash(slot, error)),
    }
}

/// Interpolate a discrete function as a tree tensor network.
///
/// Args:
///     evaluate: Batched target function. It is called with a C-contiguous
///         ``int64`` array of shape ``(n_points, n_sites)``, where row ``p``
///         holds the site values of point ``p``, and must return a numpy array
///         of shape ``(n_points,)`` with dtype ``float64`` or ``complex128``.
///         The dtype is fixed by the first call and must not change afterwards.
///         The array is a fresh copy that the callee may modify. The function
///         must be pure: TreeTCI may request the same point repeatedly and
///         gives no ordering guarantee. The batch axis is always present, so a
///         single-point batch still has shape ``(1, n_sites)``.
///     local_dims: Local dimension of each site.
///     edges: Bonds as ``(u, v)`` site pairs. Defaults to a linear chain.
///     initial_pivots: Optional starting points, each listing one site value
///         per site. Defaults to the all-zero point, which must not evaluate to
///         zero.
///     tolerance: Relative stopping tolerance on the normalized bond error.
///     max_iter: Maximum number of edge-order sweeps.
///     max_bond_dim: Optional cap on the number of pivots per edge.
///     seed: Optional seed for the global pivot search. ``None`` (the default)
///         seeds from OS entropy, so runs are not reproducible unless a seed is
///         given.
///
/// Returns:
///     ``(network, ranks, errors)``: the interpolated
///     :class:`TreeTensorNetwork` (node ``k`` is site ``k`` and its site leg is
///     the first index of ``network.tensor(k)``), the maximum bond dimension
///     after each sweep, and the normalized bond error after each sweep.
///     Not converging is not an error; inspect ``errors``.
#[pyfunction]
#[pyo3(signature = (evaluate, local_dims, edges=None, initial_pivots=None,
                    tolerance=1e-8, max_iter=20, max_bond_dim=None, seed=None))]
#[allow(clippy::too_many_arguments)]
fn crossinterpolate(
    py: Python<'_>,
    evaluate: Bound<'_, PyAny>,
    local_dims: Vec<usize>,
    edges: Option<Vec<(usize, usize)>>,
    initial_pivots: Option<Vec<Vec<usize>>>,
    tolerance: f64,
    max_iter: usize,
    max_bond_dim: Option<usize>,
    seed: Option<u64>,
) -> PyResult<(PyTreeTensorNetwork, Vec<usize>, Vec<f64>)> {
    let n_sites = local_dims.len();
    let graph = match edges {
        Some(edges) => {
            let edges: Vec<TreeTciEdge> =
                edges.iter().map(|&(u, v)| TreeTciEdge::new(u, v)).collect();
            TreeTciGraph::new(n_sites, &edges)
        }
        None => TreeTciGraph::linear_chain(n_sites),
    }
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let make_options = || TreeTciOptions {
        tolerance,
        max_iter,
        max_bond_dim,
        seed,
        ..Default::default()
    };

    // Probe the initial-pivot batch first: it is the batch TreeTCI evaluates
    // first, and its dtype fixes the value type for the whole run. The probed
    // values are cached and served to TreeTCI, so the callback is called once.
    let pivots = match initial_pivots {
        Some(pivots) if !pivots.is_empty() => pivots,
        _ => vec![vec![0; n_sites]],
    };
    if pivots.iter().any(|pivot| pivot.len() != n_sites) {
        return Err(PyValueError::new_err(
            "each initial pivot must list exactly one value per site",
        ));
    }
    let probe_points: Vec<usize> = pivots.iter().flatten().copied().collect();
    let probe_batch = GlobalIndexBatch::new(&probe_points, n_sites, pivots.len())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let probe_array = points_to_numpy(py, &probe_batch)?;
    let probe_value = evaluate.call1((probe_array,))?;
    let probe = match extract_values::<f64>(&probe_value, pivots.len()) {
        Ok(Some(values)) => ProbeValues::Real(values),
        Ok(None) => match extract_values::<Complex64>(&probe_value, pivots.len()) {
            Ok(Some(values)) => ProbeValues::Complex(values),
            Ok(None) => {
                return Err(unexpected_result(
                    &probe_value,
                    pivots.len(),
                    "float64 or complex128",
                ))
            }
            Err(error) => return Err(error),
        },
        Err(error) => return Err(error),
    };

    let py_error: Rc<RefCell<Option<PyErr>>> = Rc::new(RefCell::new(None));
    let run = match probe {
        ProbeValues::Real(values) => {
            let slot = py_error.clone();
            let evaluate = evaluate.clone();
            let cache = RefCell::new(Some(Probe {
                points: probe_points,
                values,
            }));
            let evaluate_fn = move |batch: GlobalIndexBatch<'_>| {
                next_values(
                    py,
                    &evaluate,
                    &batch,
                    &cache,
                    &slot,
                    extract_values::<f64>,
                    "float64 (the first call returned float64)",
                )
            };
            catch_unwind(AssertUnwindSafe(|| {
                crossinterpolate2::<f64, _, _>(
                    evaluate_fn,
                    local_dims,
                    graph,
                    pivots,
                    make_options(),
                    None,
                    &DefaultProposer,
                )
            }))
        }
        ProbeValues::Complex(values) => {
            let slot = py_error.clone();
            let evaluate = evaluate.clone();
            let cache = RefCell::new(Some(Probe {
                points: probe_points,
                values,
            }));
            let evaluate_fn = move |batch: GlobalIndexBatch<'_>| {
                next_values(
                    py,
                    &evaluate,
                    &batch,
                    &cache,
                    &slot,
                    extract_values::<Complex64>,
                    "complex128 (the first call returned complex128)",
                )
            };
            catch_unwind(AssertUnwindSafe(|| {
                crossinterpolate2::<Complex64, _, _>(
                    evaluate_fn,
                    local_dims,
                    graph,
                    pivots,
                    make_options(),
                    None,
                    &DefaultProposer,
                )
            }))
        }
    };

    let (treetn, ranks, errors) = match run {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => return Err(take_error(&py_error, error.into())),
        Err(payload) => return Err(panic_error(payload)),
    };
    Ok((PyTreeTensorNetwork::from_inner(treetn), ranks, errors))
}

/// Register the TreeTCI bindings on the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(crossinterpolate, module)?)
}
