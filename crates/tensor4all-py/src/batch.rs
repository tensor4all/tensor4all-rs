//! Shared helpers for batched evaluator callbacks.
//!
//! Both TCI entry points hand Python a C-contiguous `(n_points, n_dims)` array
//! whose bytes are the Rust column-major `(n_dims, n_points)` batch, and expect
//! a 1-D `(n_points,)` array of `float64` or `complex128` back (`int64` for
//! index batches).

use std::cell::RefCell;
use std::rc::Rc;

use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Wrap column-major flat data as a C-contiguous `(n_points, n_dims)` array.
pub(crate) fn batch_to_numpy<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    data: &[T],
    n_dims: usize,
    n_points: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let array = PyArray1::<T>::from_vec(py, data.to_vec());
    // The source is C-contiguous, so this is a view of the same buffer.
    Ok(array.reshape((n_points, n_dims))?.into_any())
}

/// Same, for index batches, which Python receives as `int64`.
pub(crate) fn index_batch_to_numpy<'py>(
    py: Python<'py>,
    data: &[usize],
    n_dims: usize,
    n_points: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let values: Vec<i64> = data.iter().map(|&value| value as i64).collect();
    batch_to_numpy(py, &values, n_dims, n_points)
}

/// Copy one typed evaluator result out of a Python object.
///
/// `Ok(None)` means "not a 1-D array of this dtype"; the caller decides how to
/// report that.
pub(crate) fn extract_values<T: numpy::Element + Copy>(
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

pub(crate) fn unexpected_result(
    value: &Bound<'_, PyAny>,
    n_points: usize,
    required: &str,
) -> PyErr {
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

/// Keep the original Python exception so it can be re-raised after the Rust
/// entry point reports an error of its own.
pub(crate) fn stash(slot: &Rc<RefCell<Option<PyErr>>>, error: PyErr) -> anyhow::Error {
    *slot.borrow_mut() = Some(error);
    anyhow::anyhow!("the evaluate callback failed")
}

pub(crate) fn take_error(slot: &Rc<RefCell<Option<PyErr>>>, error: anyhow::Error) -> PyErr {
    slot.borrow_mut()
        .take()
        .unwrap_or_else(|| PyValueError::new_err(error.to_string()))
}

/// Convert a caught Rust panic into a Python exception.
///
/// Without this the panic would surface as `pyo3`'s `PanicException`, which
/// derives from `BaseException` and therefore escapes `except Exception`.
pub(crate) fn panic_error(context: &str, payload: Box<dyn std::any::Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown payload".to_string());
    PyRuntimeError::new_err(format!("{context} panicked inside Rust: {message}"))
}

/// The value type fixed by the first evaluator call.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueDtype {
    Real,
    Complex,
}

/// Decide the value type from one already-computed evaluator result.
pub(crate) fn probe_dtype(value: &Bound<'_, PyAny>, n_points: usize) -> PyResult<ValueDtype> {
    if extract_values::<f64>(value, n_points)?.is_some() {
        return Ok(ValueDtype::Real);
    }
    if extract_values::<Complex64>(value, n_points)?.is_some() {
        return Ok(ValueDtype::Complex);
    }
    Err(unexpected_result(value, n_points, "float64 or complex128"))
}
