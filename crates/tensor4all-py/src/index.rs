//! Python wrapper around [`tensor4all_core::DynIndex`].
//!
//! Identity is preserved exactly: `Index` equality is (id, prime level, tags),
//! matching the Rust contract. Cloning a Python `Index` (for example by passing
//! the same object to two tensors) clones the Rust index and therefore keeps
//! the id, so the two tensor legs stay contractable.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use tensor4all_core::{DynIndex, IndexLike, TagSet, TagSetLike};

/// A tensor index: dimension, identity, prime level, and tags.
#[pyclass(name = "Index", module = "tensor4all", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyIndex {
    pub(crate) inner: DynIndex,
}

impl PyIndex {
    pub(crate) fn from_inner(inner: DynIndex) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIndex {
    /// Create a new index with a fresh identity.
    ///
    /// Args:
    ///     dim: Dimension (must be positive).
    ///     tags: Optional tag names.
    ///     plev: Prime level (default 0).
    #[new]
    #[pyo3(signature = (dim, tags = None, plev = 0))]
    fn new(dim: usize, tags: Option<Vec<String>>, plev: i64) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("Index dimension must be positive"));
        }
        let tag_set = match tags {
            Some(tags) => {
                let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                TagSet::from_tags(&refs).map_err(|err| PyValueError::new_err(err.to_string()))?
            }
            None => TagSet::new(),
        };
        let mut inner = DynIndex::new_dyn_with_tags(dim, tag_set);
        inner.plev = plev;
        Ok(Self { inner })
    }

    /// Dimension of the index.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.size()
    }

    /// Prime level of the index.
    #[getter]
    fn plev(&self) -> i64 {
        self.inner.plev
    }

    /// Tag names of the index.
    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.iter().collect()
    }

    /// Same index with the prime level incremented by one.
    fn prime(&self) -> Self {
        Self::from_inner(self.inner.prime())
    }

    /// Same index with the prime level reset to zero.
    fn noprime(&self) -> Self {
        Self::from_inner(self.inner.noprime())
    }

    /// Whether two indices were created from the same identity.
    ///
    /// Differs from `==`: indices that share an id but differ in prime level,
    /// dimension, or tags are not equal, yet `same_id` holds.
    fn same_id(&self, other: &PyIndex) -> bool {
        self.inner.same_id(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Index(dim={}, plev={}, tags={:?})",
            self.inner.size(),
            self.inner.plev,
            self.inner.tags.iter().collect::<Vec<String>>()
        )
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __richcmp__(&self, other: &PyIndex, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyTypeError::new_err(
                "Index does not support ordering comparisons",
            )),
        }
    }
}
