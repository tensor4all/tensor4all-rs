//! Compatibility helpers for reading HDF5 files from different writers.
//!
//! ITensors.jl writes fixed-length UTF-8 strings for attributes and datasets,
//! while our Rust code uses variable-length Unicode strings. This module provides
//! functions that can read both formats.

use anyhow::{bail, Result};
use tensor4all_hdf5_ffi::types::{FixedUnicode, VarLenAscii, VarLenUnicode};
use tensor4all_hdf5_ffi::{Attribute, Dataset, Group};

/// Try multiple string-reading strategies, returning the first success.
///
/// HDF5 strings may be stored as VarLenUnicode (our format), FixedUnicode
/// (ITensors.jl format), or VarLenAscii. This helper abstracts that logic.
fn try_read_string<F1, F2, F3>(try_varlen: F1, try_fixed: F2, try_ascii: F3) -> Result<String>
where
    F1: FnOnce() -> tensor4all_hdf5_ffi::Result<VarLenUnicode>,
    F2: FnOnce() -> tensor4all_hdf5_ffi::Result<FixedUnicode<256>>,
    F3: FnOnce() -> tensor4all_hdf5_ffi::Result<VarLenAscii>,
{
    if let Ok(val) = try_varlen() {
        return Ok(val.as_str().to_string());
    }
    if let Ok(val) = try_fixed() {
        return Ok(val.as_str().trim_end_matches('\0').to_string());
    }
    if let Ok(val) = try_ascii() {
        return Ok(val.as_str().to_string());
    }
    bail!("Failed to read HDF5 string: unsupported string type (tried VarLenUnicode, FixedUnicode, VarLenAscii)")
}

/// Read a string attribute that may be stored as either fixed-length or
/// variable-length Unicode.
pub(crate) fn read_string_attr(attr: &Attribute) -> Result<String> {
    try_read_string(
        || attr.as_reader().read_scalar::<VarLenUnicode>(),
        || attr.as_reader().read_scalar::<FixedUnicode<256>>(),
        || attr.as_reader().read_scalar::<VarLenAscii>(),
    )
}

/// Read a string scalar dataset that may be stored as either fixed-length or
/// variable-length Unicode.
pub(crate) fn read_string_dataset(ds: &Dataset) -> Result<String> {
    try_read_string(
        || ds.as_reader().read_scalar::<VarLenUnicode>(),
        || ds.as_reader().read_scalar::<FixedUnicode<256>>(),
        || ds.as_reader().read_scalar::<VarLenAscii>(),
    )
}

/// Read a string attribute by name from a group.
pub(crate) fn read_string_attr_by_name(group: &Group, name: &str) -> Result<String> {
    let attr = group
        .attr(name)
        .map_err(|_| anyhow::anyhow!("Attribute '{}' not found", name))?;
    read_string_attr(&attr)
}
