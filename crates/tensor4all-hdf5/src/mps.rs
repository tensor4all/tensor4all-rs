//! Layer 2: MPS (TensorTrain) HDF5 read/write (ITensorMPS.jl compatible).
//!
//! MPS is simply metadata (length, llim, rlim) + a sequence of ITensors,
//! so this module is a thin wrapper around [`crate::itensor`].

use crate::backend::Group;
use anyhow::{Context, Result};
use tensor4all_itensorlike::TensorTrain;

use crate::itensor;
use crate::schema;

/// Write a [`TensorTrain`] as an ITensorMPS.jl `MPS` to an HDF5 group.
///
/// Schema:
/// ```text
/// <group>/
///   @type = "MPS"
///   @version = 1
///   length: Int64
///   llim: Int64
///   rlim: Int64
///   MPS[1]/ ...   (ITensor, 1-indexed)
///   MPS[2]/ ...
/// ```
pub(crate) fn write_mps(group: &Group, tt: &TensorTrain) -> Result<()> {
    schema::write_type_version(group, "MPS", 1)?;

    // Metadata datasets
    let length = tt.len() as i64;
    let length_ds = group.new_dataset::<i64>().shape(()).create("length")?;
    length_ds.as_writer().write_scalar(&length)?;

    let llim_ds = group.new_dataset::<i64>().shape(()).create("llim")?;
    llim_ds.as_writer().write_scalar(&(tt.llim() as i64))?;

    let rlim_ds = group.new_dataset::<i64>().shape(()).create("rlim")?;
    rlim_ds.as_writer().write_scalar(&(tt.rlim() as i64))?;

    // Write each tensor (1-indexed, Julia convention)
    let tensors = tt.tensors();
    for (i, tensor) in tensors.iter().enumerate() {
        let name = format!("MPS[{}]", i + 1);
        let tensor_group = group.create_group(&name)?;
        itensor::write_itensor(&tensor_group, tensor)?;
    }

    Ok(())
}

/// Read a [`TensorTrain`] from an ITensorMPS.jl `MPS` in an HDF5 group.
pub(crate) fn read_mps(group: &Group) -> Result<TensorTrain> {
    schema::require_type_version(group, "MPS", 1)?;

    let length: i64 = group
        .dataset("length")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS length")?;

    let llim: i64 = group
        .dataset("llim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS llim")?;

    let rlim: i64 = group
        .dataset("rlim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS rlim")?;

    let mut tensors = Vec::with_capacity(length as usize);
    for i in 1..=length {
        let name = format!("MPS[{}]", i);
        let tensor_group = group.group(&name)?;
        tensors.push(itensor::read_itensor(&tensor_group)?);
    }

    Ok(TensorTrain::with_ortho(
        tensors,
        llim as i32,
        rlim as i32,
        None,
    )?)
}
