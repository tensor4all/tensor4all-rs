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
//! tensor4all-rs uses row-major (C order) while ITensors.jl uses column-major
//! (Fortran order). This crate handles the conversion transparently.

mod compat;
mod index;
mod itensor;
mod layout;
mod mps;
mod schema;

use anyhow::Result;
use hdf5_rt::File;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::TensorTrain;

// Re-export the HDF5 initialization functions for users
pub use hdf5_rt::sys::{
    init as hdf5_init, is_initialized as hdf5_is_initialized, library_path as hdf5_library_path,
};

/// Save a [`TensorDynLen`] as an ITensors.jl-compatible `ITensor` in an HDF5 file.
pub fn save_itensor(filepath: &str, name: &str, tensor: &TensorDynLen) -> Result<()> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    itensor::write_itensor(&group, tensor)
}

/// Load a [`TensorDynLen`] from an ITensors.jl-compatible `ITensor` in an HDF5 file.
pub fn load_itensor(filepath: &str, name: &str) -> Result<TensorDynLen> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    itensor::read_itensor(&group)
}

/// Save a [`TensorTrain`] as an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
pub fn save_mps(filepath: &str, name: &str, tt: &TensorTrain) -> Result<()> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    mps::write_mps(&group, tt)
}

/// Load a [`TensorTrain`] from an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
pub fn load_mps(filepath: &str, name: &str) -> Result<TensorTrain> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    mps::read_mps(&group)
}
