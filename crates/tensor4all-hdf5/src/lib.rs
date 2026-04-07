//! HDF5 serialization for tensor4all-rs (ITensors.jl compatible format).
//!
//! This crate provides read/write functionality for tensor4all-rs data structures
//! using the HDF5 format compatible with ITensors.jl / ITensorMPS.jl.
//!
//! # Supported types
//!
//! - [`TensorDynLen`] ↔ ITensors.jl `ITensor`
//! - [`TensorTrain`] ↔ ITensorMPS.jl `MPS`
//!
//! # Data layout
//!
//! tensor4all-rs and ITensors.jl both use column-major dense linearization.
//! This crate therefore preserves dense flat buffers as-is when serializing and
//! deserializing ITensors.jl-compatible payloads.
//!
//! # Backend selection
//!
//! - `link` feature (default): uses `hdf5-metno` with compile-time linking
//! - `runtime-loading` feature: uses `hdf5-rt` with dlopen (for Julia/Python FFI)

pub(crate) mod backend;
mod compat;
mod index;
mod itensor;
mod mps;
mod schema;

use anyhow::Result;
use backend::File;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::TensorTrain;

// Re-export the HDF5 initialization functions (runtime-loading mode only)
#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::sys::{
    init as hdf5_init, is_initialized as hdf5_is_initialized, library_path as hdf5_library_path,
};

/// Save a [`TensorDynLen`] as an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_hdf5::{save_itensor, load_itensor};
/// use tensor4all_core::index::Index;
/// use tensor4all_core::TensorDynLen;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0
/// ]).unwrap();
///
/// let path = "/tmp/test_itensor.h5";
/// save_itensor(path, "my_tensor", &tensor).unwrap();
/// let loaded = load_itensor(path, "my_tensor").unwrap();
/// assert_eq!(loaded.ndim(), 2);
/// ```
pub fn save_itensor(filepath: &str, name: &str, tensor: &TensorDynLen) -> Result<()> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    itensor::write_itensor(&group, tensor)
}

/// Load a [`TensorDynLen`] from an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_hdf5::{save_itensor, load_itensor};
/// use tensor4all_core::index::Index;
/// use tensor4all_core::TensorDynLen;
///
/// // (Requires a previously saved file — see `save_itensor` for a full round-trip example.)
/// let path = "/tmp/test_itensor.h5";
/// let loaded = load_itensor(path, "my_tensor").unwrap();
/// assert_eq!(loaded.ndim(), 2);
/// ```
pub fn load_itensor(filepath: &str, name: &str) -> Result<TensorDynLen> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    itensor::read_itensor(&group)
}

/// Save a [`TensorTrain`] as an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_hdf5::{save_mps, load_mps};
/// use tensor4all_core::index::Index;
/// use tensor4all_core::TensorDynLen;
/// use tensor4all_itensorlike::TensorTrain;
///
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0]).unwrap();
/// let t1 = TensorDynLen::from_dense(vec![bond, s1], vec![1.0, 0.0]).unwrap();
/// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
///
/// let path = "/tmp/test_mps.h5";
/// save_mps(path, "my_mps", &tt).unwrap();
/// let loaded = load_mps(path, "my_mps").unwrap();
/// assert_eq!(loaded.len(), 2);
/// ```
pub fn save_mps(filepath: &str, name: &str, tt: &TensorTrain) -> Result<()> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    mps::write_mps(&group, tt)
}

/// Load a [`TensorTrain`] from an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_hdf5::load_mps;
///
/// // (Requires a previously saved file — see `save_mps` for a full round-trip example.)
/// let path = "/tmp/test_mps.h5";
/// let tt = load_mps(path, "my_mps").unwrap();
/// assert_eq!(tt.len(), 2);
/// ```
pub fn load_mps(filepath: &str, name: &str) -> Result<TensorTrain> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    mps::read_mps(&group)
}
