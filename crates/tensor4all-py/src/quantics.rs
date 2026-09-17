//! Python bindings for quantics tensor cross interpolation.
//!
//! The evaluator boundary is batched, matching `tensor4all-treetci` and the
//! crate's Rust entry points: one call per batch of grid points, never one call
//! per point.
//!
//! * Continuous grid: `evaluate` receives a C-contiguous `float64` array of
//!   shape `(n_points, n_dims)` holding original coordinates.
//! * Discrete grid: `evaluate` receives the same shape with `int64` 0-indexed
//!   grid indices.
//!
//! Both are the bytes of the Rust column-major `(n_dims, n_points)` batch, so
//! the batch axis is axis 0 and nothing is transposed.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use num_complex::Complex64;
use numpy::ndarray::IxDyn;
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat};
use tensor4all_quanticstci::{
    quanticscrossinterpolate_batch, quanticscrossinterpolate_discrete_batch, DiscretizedGrid,
    QtciOptions, QuanticsBatch, QuanticsTensorCI2, UnfoldingScheme,
};

use crate::batch::{
    batch_to_numpy, extract_values, index_batch_to_numpy, panic_error, probe_dtype, stash,
    take_error, unexpected_result, ValueDtype,
};
use crate::tensor::write_flat;

/// Cap on the number of dense elements `to_numpy()` will materialize.
const MAX_DENSE_ELEMENTS: usize = 1 << 22;

/// The interpolation result, in whichever value type the evaluator fixed.
enum QuanticsValues {
    Real(QuanticsTensorCI2<f64>),
    Complex(QuanticsTensorCI2<Complex64>),
}

/// A quantics tensor train interpolating a function on a grid.
#[pyclass(name = "QuanticsTCI", module = "tensor4all")]
pub struct PyQuanticsTCI {
    inner: QuanticsValues,
    shape: Vec<usize>,
}

impl PyQuanticsTCI {
    fn new(inner: QuanticsValues, shape: Vec<usize>) -> Self {
        Self { inner, shape }
    }
}

fn rank_of(values: &QuanticsValues) -> usize {
    match values {
        QuanticsValues::Real(qtci) => qtci.rank(),
        QuanticsValues::Complex(qtci) => qtci.rank(),
    }
}

fn qtci_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

/// Advance a multi-index in C order (last axis fastest).
fn advance_index(index: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

#[pymethods]
impl PyQuanticsTCI {
    /// Value at one grid point, given one 0-indexed index per dimension.
    fn evaluate(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            QuanticsValues::Real(qtci) => {
                let value = qtci
                    .evaluate(&indices)
                    .map_err(|error| qtci_error("evaluate failed", error))?;
                Ok(PyFloat::new(py, value).into_any().unbind())
            }
            QuanticsValues::Complex(qtci) => {
                let value = qtci
                    .evaluate(&indices)
                    .map_err(|error| qtci_error("evaluate failed", error))?;
                Ok(PyComplex::from_doubles(py, value.re, value.im)
                    .into_any()
                    .unbind())
            }
        }
    }

    /// Factorized sum over all grid points.
    ///
    /// The sum is computed from the tensor train structure, so it never visits
    /// every grid point.
    fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            QuanticsValues::Real(qtci) => {
                let value = qtci
                    .sum()
                    .map_err(|error| qtci_error("sum failed", error))?;
                Ok(PyFloat::new(py, value).into_any().unbind())
            }
            QuanticsValues::Complex(qtci) => {
                let value = qtci
                    .sum()
                    .map_err(|error| qtci_error("sum failed", error))?;
                Ok(PyComplex::from_doubles(py, value.re, value.im)
                    .into_any()
                    .unbind())
            }
        }
    }

    /// Integral over the domain; only available for continuous grids.
    fn integral(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            QuanticsValues::Real(qtci) => {
                let value = qtci
                    .integral()
                    .map_err(|error| qtci_error("integral failed", error))?;
                Ok(PyFloat::new(py, value).into_any().unbind())
            }
            QuanticsValues::Complex(qtci) => {
                let value = qtci
                    .integral()
                    .map_err(|error| qtci_error("integral failed", error))?;
                Ok(PyComplex::from_doubles(py, value.re, value.im)
                    .into_any()
                    .unbind())
            }
        }
    }

    /// Maximum bond dimension of the interpolated tensor train.
    #[getter]
    fn rank(&self) -> usize {
        rank_of(&self.inner)
    }

    /// Grid point counts, one per dimension.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// All grid values as a C-ordered array of shape `(2**bits, ...)`.
    ///
    /// Materialization is exponential in the number of bits; above
    /// `MAX_DENSE_ELEMENTS` this raises instead of allocating, and callers
    /// should use `sum()` or `evaluate()`.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let total = self
            .shape
            .iter()
            .try_fold(1usize, |total, &size| total.checked_mul(size))
            .ok_or_else(|| PyValueError::new_err("grid shape overflowed usize"))?;
        if total > MAX_DENSE_ELEMENTS {
            return Err(PyValueError::new_err(format!(
                "to_numpy() would materialize {total} elements, above the {MAX_DENSE_ELEMENTS} \
                 element limit; use sum() or evaluate() instead"
            )));
        }
        let mut index = vec![0usize; self.shape.len()];
        match &self.inner {
            QuanticsValues::Real(qtci) => {
                let mut flat = Vec::with_capacity(total);
                for _ in 0..total {
                    flat.push(
                        qtci.evaluate(&index)
                            .map_err(|error| qtci_error("evaluate failed", error))?,
                    );
                    advance_index(&mut index, &self.shape);
                }
                let array = PyArrayDyn::<f64>::zeros(py, IxDyn(&self.shape), false);
                write_flat(&array, &flat)?;
                Ok(array.into_any().unbind())
            }
            QuanticsValues::Complex(qtci) => {
                let mut flat = Vec::with_capacity(total);
                for _ in 0..total {
                    flat.push(
                        qtci.evaluate(&index)
                            .map_err(|error| qtci_error("evaluate failed", error))?,
                    );
                    advance_index(&mut index, &self.shape);
                }
                let array = PyArrayDyn::<Complex64>::zeros(py, IxDyn(&self.shape), false);
                write_flat(&array, &flat)?;
                Ok(array.into_any().unbind())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QuanticsTCI(shape={:?}, rank={})",
            self.shape,
            rank_of(&self.inner)
        )
    }
}

/// Build the batch evaluator closure around one Python callable.
fn python_evaluator<'py, T, V, ToArray>(
    py: Python<'py>,
    evaluate: Bound<'py, PyAny>,
    slot: Rc<RefCell<Option<PyErr>>>,
    required: &'static str,
    to_array: ToArray,
) -> impl Fn(QuanticsBatch<'_, T>) -> anyhow::Result<Vec<V>> + 'py
where
    T: Copy,
    V: numpy::Element + Copy,
    ToArray: Fn(Python<'py>, &QuanticsBatch<'_, T>) -> PyResult<Bound<'py, PyAny>> + 'py,
{
    move |batch: QuanticsBatch<'_, T>| -> anyhow::Result<Vec<V>> {
        let n_points = batch.n_points();
        let points = to_array(py, &batch).map_err(|error| stash(&slot, error))?;
        let value = evaluate
            .call1((points,))
            .map_err(|error| stash(&slot, error))?;
        match extract_values::<V>(&value, n_points) {
            Ok(Some(values)) => Ok(values),
            Ok(None) => Err(stash(&slot, unexpected_result(&value, n_points, required))),
            Err(error) => Err(stash(&slot, error)),
        }
    }
}

fn usize_vec(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<usize>> {
    if let Ok(scalar) = value.extract::<usize>() {
        return Ok(vec![scalar]);
    }
    value
        .extract::<Vec<usize>>()
        .map_err(|_| PyValueError::new_err(format!("{what} must be an int or a sequence of ints")))
}

fn f64_vec(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(vec![scalar]);
    }
    value.extract::<Vec<f64>>().map_err(|_| {
        PyValueError::new_err(format!("{what} must be a float or a sequence of floats"))
    })
}

fn parse_unfolding(name: &str) -> PyResult<UnfoldingScheme> {
    match name {
        "interleaved" => Ok(UnfoldingScheme::Interleaved),
        "fused" => Ok(UnfoldingScheme::Fused),
        "grouped" => Ok(UnfoldingScheme::Grouped),
        other => Err(PyValueError::new_err(format!(
            "unknown unfolding scheme {other:?}: expected \"interleaved\", \"fused\", or \"grouped\""
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_options(
    tolerance: f64,
    max_iter: usize,
    max_bond_dim: Option<usize>,
    random_init_pivots: usize,
    scheme: UnfoldingScheme,
) -> QtciOptions {
    let options = QtciOptions::default()
        .with_tolerance(tolerance)
        .with_maxiter(max_iter)
        .with_nrandominitpivot(random_init_pivots)
        .with_unfoldingscheme(scheme);
    match max_bond_dim {
        Some(max_bond_dim) => options.with_max_bond_dim(max_bond_dim),
        None => options,
    }
}

fn prepare_pivots(
    initial_pivots: Option<Vec<Vec<usize>>>,
    n_dims: usize,
) -> PyResult<Vec<Vec<usize>>> {
    match initial_pivots {
        Some(pivots) if !pivots.is_empty() => {
            if pivots.iter().any(|pivot| pivot.len() != n_dims) {
                return Err(PyValueError::new_err(
                    "each initial pivot must list exactly one grid index per dimension",
                ));
            }
            Ok(pivots)
        }
        _ => Ok(vec![vec![0; n_dims]]),
    }
}

fn grid_shape(bits: &[usize]) -> PyResult<Vec<usize>> {
    bits.iter()
        .map(|&bit| {
            1usize.checked_shl(bit as u32).ok_or_else(|| {
                PyValueError::new_err(format!("bits value {bit} is too large for a grid"))
            })
        })
        .collect()
}

/// Interpolate a function on a uniform quantics grid.
///
/// `evaluate` is called **in batches**: it receives a C-contiguous `float64`
/// array of shape `(n_points, n_dims)` whose row `p` holds the coordinates of
/// one grid point, and must return a `(n_points,)` array with dtype `float64`
/// or `complex128`. The dtype is fixed by the first call and must not change.
/// The function must be pure; the batch axis is always present, so a
/// one-point batch has shape `(1, n_dims)`.
///
/// Args:
///     evaluate: Batched target function of the coordinates.
///     bits: Number of quantics bits per dimension (int or one int per
///         dimension); each dimension gets `2**bits` grid points.
///     lower: Lower domain bound (float or one float per dimension).
///     upper: Upper domain bound (float or one float per dimension).
///     initial_pivots: Optional starting points, one 0-indexed grid index per
///         dimension. Defaults to the all-zero point, which must not evaluate
///         to zero.
///     tolerance: Relative convergence tolerance.
///     max_iter: Maximum number of half sweeps.
///     max_bond_dim: Optional cap on the bond dimension.
///     random_init_pivots: Number of random starting pivots. Defaults to 0 so
///         runs are reproducible (the Rust default is 5).
///     unfolding: Tensor-train layout, one of ``"interleaved"``, ``"fused"``,
///         or ``"grouped"``.
///
/// Returns:
///     ``(result, ranks, errors)``: a :class:`QuanticsTCI`, the ranks per
///     sweep, and the errors per sweep. Not converging is not an error;
///     inspect ``errors``.
#[pyfunction]
#[pyo3(signature = (evaluate, bits, lower, upper, initial_pivots=None, tolerance=1e-8,
                    max_iter=200, max_bond_dim=None, random_init_pivots=0, unfolding="interleaved"))]
#[allow(clippy::too_many_arguments)]
fn quanticscrossinterpolate<'py>(
    py: Python<'py>,
    evaluate: Bound<'py, PyAny>,
    bits: Bound<'py, PyAny>,
    lower: Bound<'py, PyAny>,
    upper: Bound<'py, PyAny>,
    initial_pivots: Option<Vec<Vec<usize>>>,
    tolerance: f64,
    max_iter: usize,
    max_bond_dim: Option<usize>,
    random_init_pivots: usize,
    unfolding: &str,
) -> PyResult<(PyQuanticsTCI, Vec<usize>, Vec<f64>)> {
    let bits = usize_vec(&bits, "bits")?;
    if bits.is_empty() {
        return Err(PyValueError::new_err(
            "bits must contain at least one entry",
        ));
    }
    let mut lower = f64_vec(&lower, "lower")?;
    let mut upper = f64_vec(&upper, "upper")?;
    if lower.len() == 1 && bits.len() > 1 {
        lower = vec![lower[0]; bits.len()];
    }
    if upper.len() == 1 && bits.len() > 1 {
        upper = vec![upper[0]; bits.len()];
    }
    if lower.len() != bits.len() || upper.len() != bits.len() {
        return Err(PyValueError::new_err(format!(
            "lower ({}) and upper ({}) must have one entry per dimension, but bits has {}",
            lower.len(),
            upper.len(),
            bits.len()
        )));
    }

    let n_dims = bits.len();
    let scheme = parse_unfolding(unfolding)?;
    let grid = DiscretizedGrid::builder(&bits)
        .with_lower_bound(&lower)
        .with_upper_bound(&upper)
        .with_unfolding_scheme(scheme)
        .build()
        .map_err(|error| PyValueError::new_err(format!("invalid quantics grid: {error}")))?;
    let shape = grid_shape(&bits)?;
    let options = make_options(
        tolerance,
        max_iter,
        max_bond_dim,
        random_init_pivots,
        scheme,
    );
    let pivots = prepare_pivots(initial_pivots, n_dims)?;

    // Probe the initial-pivot coordinates: the first batch TreeTCI evaluates
    // fixes the value type of the whole run.
    let mut probe_flat = Vec::with_capacity(n_dims * pivots.len());
    for pivot in &pivots {
        let quantics = grid.grididx_to_quantics(pivot).map_err(|error| {
            PyValueError::new_err(format!("invalid initial pivot {pivot:?}: {error}"))
        })?;
        let coordinates = grid.quantics_to_origcoord(&quantics).map_err(|error| {
            PyValueError::new_err(format!("invalid initial pivot {pivot:?}: {error}"))
        })?;
        probe_flat.extend_from_slice(&coordinates);
    }
    let probe_array = batch_to_numpy(py, &probe_flat, n_dims, pivots.len())?;
    let probe_value = evaluate.call1((probe_array,))?;
    let dtype = probe_dtype(&probe_value, pivots.len())?;

    let py_error: Rc<RefCell<Option<PyErr>>> = Rc::new(RefCell::new(None));
    let run = match dtype {
        ValueDtype::Real => {
            let evaluator = python_evaluator::<f64, f64, _>(
                py,
                evaluate.clone(),
                py_error.clone(),
                "float64 (the first call returned float64)",
                |py, batch| batch_to_numpy(py, batch.data(), batch.n_dims(), batch.n_points()),
            );
            let pivots = pivots.clone();
            let options = options.clone();
            catch_unwind(AssertUnwindSafe(|| {
                quanticscrossinterpolate_batch::<f64, _>(&grid, evaluator, Some(pivots), options)
            }))
            .map(|run| run.map(|(qtci, ranks, errors)| (QuanticsValues::Real(qtci), ranks, errors)))
        }
        ValueDtype::Complex => {
            let evaluator = python_evaluator::<f64, Complex64, _>(
                py,
                evaluate.clone(),
                py_error.clone(),
                "complex128 (the first call returned complex128)",
                |py, batch| batch_to_numpy(py, batch.data(), batch.n_dims(), batch.n_points()),
            );
            let pivots = pivots.clone();
            let options = options.clone();
            catch_unwind(AssertUnwindSafe(|| {
                quanticscrossinterpolate_batch::<Complex64, _>(
                    &grid,
                    evaluator,
                    Some(pivots),
                    options,
                )
            }))
            .map(|run| {
                run.map(|(qtci, ranks, errors)| (QuanticsValues::Complex(qtci), ranks, errors))
            })
        }
    };

    let (inner, ranks, errors) = match run {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return Err(take_error(&py_error, error.into())),
        Err(payload) => return Err(panic_error("Quantics TCI", payload)),
    };
    Ok((PyQuanticsTCI::new(inner, shape), ranks, errors))
}

/// Interpolate a function on a discrete integer grid.
///
/// `evaluate` is called **in batches**: it receives a C-contiguous `int64`
/// array of shape `(n_points, n_dims)` whose row `p` holds the 0-indexed grid
/// indices of one point, and must return a `(n_points,)` array with dtype
/// `float64` or `complex128`. All dimensions must have the same number of
/// points, and that number must be a power of two.
///
/// Args:
///     evaluate: Batched target function of the grid indices.
///     sizes: Number of grid points per dimension (int or sequence of ints).
///     initial_pivots: Optional starting points, one 0-indexed grid index per
///         dimension.
///     tolerance, max_iter, max_bond_dim, random_init_pivots, unfolding: See
///         :func:`quanticscrossinterpolate`.
///
/// Returns:
///     ``(result, ranks, errors)`` as in :func:`quanticscrossinterpolate`.
#[pyfunction]
#[pyo3(signature = (evaluate, sizes, initial_pivots=None, tolerance=1e-8,
                    max_iter=200, max_bond_dim=None, random_init_pivots=0, unfolding="interleaved"))]
#[allow(clippy::too_many_arguments)]
fn quanticscrossinterpolate_discrete<'py>(
    py: Python<'py>,
    evaluate: Bound<'py, PyAny>,
    sizes: Bound<'py, PyAny>,
    initial_pivots: Option<Vec<Vec<usize>>>,
    tolerance: f64,
    max_iter: usize,
    max_bond_dim: Option<usize>,
    random_init_pivots: usize,
    unfolding: &str,
) -> PyResult<(PyQuanticsTCI, Vec<usize>, Vec<f64>)> {
    let sizes = usize_vec(&sizes, "sizes")?;
    if sizes.is_empty() {
        return Err(PyValueError::new_err(
            "sizes must contain at least one entry",
        ));
    }
    let n_dims = sizes.len();
    let scheme = parse_unfolding(unfolding)?;
    let options = make_options(
        tolerance,
        max_iter,
        max_bond_dim,
        random_init_pivots,
        scheme,
    );
    let pivots = prepare_pivots(initial_pivots, n_dims)?;

    // Grid indices are what the evaluator receives, so the initial pivots can
    // be probed directly.
    let probe_flat: Vec<usize> = pivots.iter().flatten().copied().collect();
    let probe_array = index_batch_to_numpy(py, &probe_flat, n_dims, pivots.len())?;
    let probe_value = evaluate.call1((probe_array,))?;
    let dtype = probe_dtype(&probe_value, pivots.len())?;

    let py_error: Rc<RefCell<Option<PyErr>>> = Rc::new(RefCell::new(None));
    let run = match dtype {
        ValueDtype::Real => {
            let evaluator = python_evaluator::<usize, f64, _>(
                py,
                evaluate.clone(),
                py_error.clone(),
                "float64 (the first call returned float64)",
                |py, batch| {
                    index_batch_to_numpy(py, batch.data(), batch.n_dims(), batch.n_points())
                },
            );
            let pivots = pivots.clone();
            let options = options.clone();
            catch_unwind(AssertUnwindSafe(|| {
                quanticscrossinterpolate_discrete_batch::<f64, _>(
                    &sizes,
                    evaluator,
                    Some(pivots),
                    options,
                )
            }))
            .map(|run| run.map(|(qtci, ranks, errors)| (QuanticsValues::Real(qtci), ranks, errors)))
        }
        ValueDtype::Complex => {
            let evaluator = python_evaluator::<usize, Complex64, _>(
                py,
                evaluate.clone(),
                py_error.clone(),
                "complex128 (the first call returned complex128)",
                |py, batch| {
                    index_batch_to_numpy(py, batch.data(), batch.n_dims(), batch.n_points())
                },
            );
            let pivots = pivots.clone();
            let options = options.clone();
            catch_unwind(AssertUnwindSafe(|| {
                quanticscrossinterpolate_discrete_batch::<Complex64, _>(
                    &sizes,
                    evaluator,
                    Some(pivots),
                    options,
                )
            }))
            .map(|run| {
                run.map(|(qtci, ranks, errors)| (QuanticsValues::Complex(qtci), ranks, errors))
            })
        }
    };

    let (inner, ranks, errors) = match run {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return Err(take_error(&py_error, error.into())),
        Err(payload) => return Err(panic_error("Quantics TCI", payload)),
    };
    Ok((PyQuanticsTCI::new(inner, sizes), ranks, errors))
}

/// Register the quantics TCI bindings on the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyQuanticsTCI>()?;
    module.add_function(wrap_pyfunction!(quanticscrossinterpolate, module)?)?;
    module.add_function(wrap_pyfunction!(quanticscrossinterpolate_discrete, module)?)
}
