//! Python wrapper around [`tensor4all_core::IdxTensor`], including the explicit
//! NumPy interchange boundary.
//!
//! Data contract:
//!
//! * NumPy input is copied into Rust-owned storage; later mutation of the input
//!   array never affects the tensor.
//! * `to_numpy()` allocates a fresh NumPy array, so the result never aliases
//!   internal tensor state.
//! * The Rust dense convention is column-major (first index varies fastest).
//!   NumPy arguments are accepted in logical index order for any layout
//!   (C-contiguous, F-contiguous, sliced, negative strides), and results are
//!   returned in logical index order as a Fortran-ordered array. The logical
//!   tensor is never transposed.
//! * Only `float64` and `complex128` are accepted. Other dtypes raise
//!   `TypeError`; no imaginary component is ever dropped silently.

use num_complex::Complex64;
use numpy::ndarray::IxDyn;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use tensor4all_core::{contract, DynIndex, IdxTensor};

use crate::index::PyIndex;

/// Dense tensor storage kind.
enum PyTensorData {
    Real(IdxTensor),
    Complex(IdxTensor),
}

/// A dense tensor with labelled indices.
#[pyclass(name = "Tensor", module = "tensor4all")]
pub struct PyTensor {
    data: PyTensorData,
}

impl PyTensor {
    pub(crate) fn from_idx_tensor(tensor: IdxTensor) -> Self {
        let data = if tensor.is_complex() {
            PyTensorData::Complex(tensor)
        } else {
            PyTensorData::Real(tensor)
        };
        Self { data }
    }

    /// Borrow the underlying Rust tensor.
    pub(crate) fn inner(&self) -> &IdxTensor {
        match &self.data {
            PyTensorData::Real(tensor) | PyTensorData::Complex(tensor) => tensor,
        }
    }
}

/// Copy any strided NumPy view into column-major (first index fastest) order.
fn col_major_flat<T: Copy>(view: numpy::ndarray::ArrayViewD<'_, T>) -> Vec<T> {
    view.reversed_axes().iter().copied().collect()
}

fn shape_mismatch(dims: &[usize], shape: &[usize]) -> PyErr {
    PyValueError::new_err(format!(
        "data shape {shape:?} does not match index dimensions {dims:?}"
    ))
}

fn tensor_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {err}"))
}

#[pymethods]
impl PyTensor {
    /// Create a tensor from indices and a NumPy array.
    ///
    /// Args:
    ///     indices: One `Index` per array axis, in logical order.
    ///     data: `float64` or `complex128` array with `shape == [i.dim for i in
    ///         indices]`. Any memory layout is accepted and copied.
    #[new]
    fn new(indices: Vec<PyRef<'_, PyIndex>>, data: Bound<'_, PyAny>) -> PyResult<Self> {
        let dims: Vec<usize> = indices.iter().map(|index| index.inner.size()).collect();
        let tensor = if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            if array.shape() != dims.as_slice() {
                return Err(shape_mismatch(&dims, array.shape()));
            }
            let flat = col_major_flat(array.as_array());
            drop(array);
            IdxTensor::from_dense(index_buffers(&indices), flat)
                .map_err(|err| tensor_error("Tensor construction failed", err))?
        } else if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, Complex64>>() {
            if array.shape() != dims.as_slice() {
                return Err(shape_mismatch(&dims, array.shape()));
            }
            let flat = col_major_flat(array.as_array());
            drop(array);
            IdxTensor::from_dense(index_buffers(&indices), flat)
                .map_err(|err| tensor_error("Tensor construction failed", err))?
        } else {
            let message = match data.getattr("dtype") {
                Ok(dtype) => format!(
                    "data must be a numpy array with dtype float64 or complex128, got {dtype}"
                ),
                Err(_) => "data must be a numpy array with dtype float64 or complex128".to_string(),
            };
            return Err(PyTypeError::new_err(message));
        };
        Ok(Self::from_idx_tensor(tensor))
    }

    /// Indices of the tensor, in axis order.
    #[getter]
    fn indices(&self) -> Vec<PyIndex> {
        self.inner()
            .indices()
            .iter()
            .cloned()
            .map(PyIndex::from_inner)
            .collect()
    }

    /// Axis dimensions, in axis order.
    #[getter]
    fn dims(&self) -> Vec<usize> {
        self.inner().dims()
    }

    /// Copy the tensor into a new NumPy array.
    ///
    /// The returned array owns its storage and is independent of the tensor.
    /// It is returned in logical index order as a Fortran-ordered array.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dims = self.inner().dims();
        match &self.data {
            PyTensorData::Real(tensor) => {
                let flat = tensor
                    .to_vec::<f64>()
                    .map_err(|err| tensor_error("to_numpy failed", err))?;
                let array = PyArrayDyn::<f64>::zeros(py, IxDyn(&dims), true);
                write_flat(&array, &flat)?;
                Ok(array.into_any().unbind())
            }
            PyTensorData::Complex(tensor) => {
                let flat = tensor
                    .to_vec::<Complex64>()
                    .map_err(|err| tensor_error("to_numpy failed", err))?;
                let array = PyArrayDyn::<Complex64>::zeros(py, IxDyn(&dims), true);
                write_flat(&array, &flat)?;
                Ok(array.into_any().unbind())
            }
        }
    }

    /// Contract this tensor with `other` over their shared indices.
    ///
    /// The result keeps the non-contracted indices: this tensor's first, then
    /// `other`'s. Tensors with no shared index are rejected rather than
    /// silently forming an outer product.
    fn contract(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let tensor = contract(&[self.inner(), other.inner()])
            .map_err(|err| tensor_error("Tensor contraction failed", err))?;
        Ok(Self::from_idx_tensor(tensor))
    }

    fn __repr__(&self) -> String {
        let kind = match self.data {
            PyTensorData::Real(_) => "float64",
            PyTensorData::Complex(_) => "complex128",
        };
        format!("Tensor(dims={:?}, dtype={kind})", self.inner().dims())
    }
}

/// Clone the Rust indices out of their Python wrappers.
fn index_buffers(indices: &[PyRef<'_, PyIndex>]) -> Vec<DynIndex> {
    indices.iter().map(|index| index.inner.clone()).collect()
}

/// Copy column-major flat data into a freshly allocated Fortran-ordered array.
pub(crate) fn write_flat<T: numpy::Element + Copy>(
    array: &Bound<'_, PyArrayDyn<T>>,
    flat: &[T],
) -> PyResult<()> {
    let mut guard = array.readwrite();
    let target = guard
        .as_slice_mut()
        .map_err(|_| PyRuntimeError::new_err("to_numpy produced a non-contiguous array"))?;
    if target.len() != flat.len() {
        return Err(PyRuntimeError::new_err(
            "to_numpy produced an array of unexpected length",
        ));
    }
    target.copy_from_slice(flat);
    Ok(())
}
