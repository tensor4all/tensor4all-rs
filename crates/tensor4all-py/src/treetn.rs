//! Python wrapper around [`tensor4all_treetn::DefaultTreeTN`].
//!
//! `TreeTensorNetwork` is the only network type exposed: chains (MPS/MPO-like
//! networks) are ordinary tree networks with a path-shaped topology, so no
//! separate chain type or legacy `TensorTrain` wrapper exists here.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use tensor4all_core::IdxTensor;
use tensor4all_treetn::contraction::{self, ContractionMethod, ContractionOptions};
use tensor4all_treetn::DefaultTreeTN;

use crate::tensor::PyTensor;

/// Default dense-element cap for reference (`naive`) contraction.
const DEFAULT_DENSE_REFERENCE_LIMIT: usize = 1 << 20;

/// A tree tensor network with `usize` node names.
#[pyclass(name = "TreeTensorNetwork", module = "tensor4all")]
pub struct PyTreeTensorNetwork {
    inner: DefaultTreeTN<usize>,
}

impl PyTreeTensorNetwork {
    pub(crate) fn from_inner(inner: DefaultTreeTN<usize>) -> Self {
        Self { inner }
    }
}

fn tree_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {err}"))
}

fn parse_method(method: &str) -> PyResult<ContractionMethod> {
    match method {
        "naive" => Ok(ContractionMethod::Naive),
        "zipup" => Ok(ContractionMethod::Zipup),
        other => Err(PyValueError::new_err(format!(
            "unsupported contraction method {other:?}: expected \"naive\" or \"zipup\""
        ))),
    }
}

#[pymethods]
impl PyTreeTensorNetwork {
    /// Build a network from tensors and optional node names.
    ///
    /// Nodes are connected automatically: an index occurring in exactly two
    /// tensors becomes the bond between them, and an index occurring once stays
    /// a site (physical) leg. Any other multiplicity is an error.
    ///
    /// Args:
    ///     tensors: Node tensors.
    ///     names: Optional unique node names; defaults to `0..len(tensors)`.
    #[new]
    #[pyo3(signature = (tensors, names = None))]
    fn new(tensors: Vec<PyRef<'_, PyTensor>>, names: Option<Vec<usize>>) -> PyResult<Self> {
        let nodes: Vec<IdxTensor> = tensors
            .iter()
            .map(|tensor| tensor.inner().clone())
            .collect();
        if nodes.is_empty() {
            return Err(PyValueError::new_err(
                "TreeTensorNetwork requires at least one tensor",
            ));
        }
        let names = match names {
            Some(names) if names.len() != nodes.len() => {
                return Err(PyValueError::new_err(format!(
                    "names has {} entries but {} tensors were given",
                    names.len(),
                    nodes.len()
                )));
            }
            Some(names) => names,
            None => (0..nodes.len()).collect(),
        };
        let inner = DefaultTreeTN::from_tensors(nodes, names)
            .map_err(|err| tree_error("TreeTensorNetwork construction failed", err))?;
        Ok(Self { inner })
    }

    /// Number of nodes.
    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of bonds.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.edge_count()
    }

    /// Node names, in the order the network stores them.
    fn node_names(&self) -> Vec<usize> {
        self.inner.node_names()
    }

    /// Tensor stored at `name`.
    fn tensor(&self, name: usize) -> PyResult<PyTensor> {
        let node = self
            .inner
            .node_index(&name)
            .ok_or_else(|| PyKeyError::new_err(format!("no node named {name}")))?;
        let tensor = self
            .inner
            .tensor(node)
            .ok_or_else(|| PyKeyError::new_err(format!("node {name} has no tensor")))?;
        Ok(PyTensor::from_idx_tensor(tensor.clone()))
    }

    /// Contract the whole network into one dense tensor over its site legs.
    fn contract_to_tensor(&self) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .contract_to_tensor()
            .map_err(|err| tree_error("contract_to_tensor failed", err))?;
        Ok(PyTensor::from_idx_tensor(tensor))
    }

    /// Contract this network with `other`.
    ///
    /// Both networks must describe the same nodes: ``"zipup"`` validates the
    /// graph topology, while ``"naive"`` reuses this network's topology for
    /// the result. Indices shared by the two networks are summed; the remaining
    /// site legs form the result.
    ///
    /// Args:
    ///     other: Network to contract with.
    ///     method: ``"naive"`` materializes both networks as dense tensors and
    ///         is limited by `dense_reference_limit`; ``"zipup"`` uses the
    ///         structural algorithm.
    ///     maxdim: Optional maximum bond dimension for ``"zipup"``.
    ///     dense_reference_limit: Maximum dense elements per network and result
    ///         for ``"naive"``.
    #[pyo3(signature = (other, method = "naive", maxdim = None, dense_reference_limit = DEFAULT_DENSE_REFERENCE_LIMIT))]
    fn contract(
        &self,
        other: &PyTreeTensorNetwork,
        method: &str,
        maxdim: Option<usize>,
        dense_reference_limit: usize,
    ) -> PyResult<Self> {
        let rust_method = parse_method(method)?;
        let center =
            self.inner.node_names().into_iter().min().ok_or_else(|| {
                PyValueError::new_err("cannot contract an empty TreeTensorNetwork")
            })?;
        let mut options = ContractionOptions::new(rust_method);
        if let Some(maxdim) = maxdim {
            options = options.with_max_bond_dim(maxdim);
        }
        if rust_method == ContractionMethod::Naive {
            options = options.with_dense_reference_limit(dense_reference_limit);
        }
        let result = contraction::contract(&self.inner, &other.inner, &center, options)
            .map_err(|err| tree_error("TreeTensorNetwork contraction failed", err))?;
        Ok(Self::from_inner(result))
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeTensorNetwork(num_vertices={}, nodes={:?})",
            self.inner.node_count(),
            self.inner.node_names()
        )
    }
}
