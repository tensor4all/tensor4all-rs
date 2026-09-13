//! PyO3 bindings for tensor4all-rs.
//!
//! The Python layer is intentionally thin. Every public operation forwards to
//! the Rust API of `tensor4all-core` / `tensor4all-treetn`; no tensor or
//! network algorithm is implemented here. NumPy interchange is an explicit
//! copy boundary (see `tensor`).

use pyo3::prelude::*;

mod index;
mod tensor;
mod treetn;

#[pymodule]
fn tensor4all(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<index::PyIndex>()?;
    module.add_class::<tensor::PyTensor>()?;
    module.add_class::<treetn::PyTreeTensorNetwork>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
