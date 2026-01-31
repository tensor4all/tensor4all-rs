//! Compatibility helpers for reading HDF5 files from different writers.
//!
//! ITensors.jl writes fixed-length UTF-8 strings for attributes and datasets,
//! while our Rust code uses variable-length Unicode strings. This module provides
//! functions that can read both formats.

use anyhow::{Context, Result};
use hdf5::types::{FixedUnicode, VarLenUnicode};
use hdf5::{Attribute, Dataset};

/// Read a string attribute that may be stored as either fixed-length or
/// variable-length Unicode.
pub(crate) fn read_string_attr(attr: &Attribute) -> Result<String> {
    // Try variable-length Unicode first (our format)
    if let Ok(val) = attr.as_reader().read_scalar::<VarLenUnicode>() {
        return Ok(val.as_str().to_string());
    }

    // Try fixed-length Unicode with a large buffer (ITensors.jl format).
    // HDF5 will pad shorter strings with nulls, which we trim.
    if let Ok(val) = attr.as_reader().read_scalar::<FixedUnicode<256>>() {
        return Ok(val.as_str().trim_end_matches('\0').to_string());
    }

    // Fallback: try reading as VarLenAscii
    if let Ok(val) = attr.as_reader().read_scalar::<hdf5::types::VarLenAscii>() {
        return Ok(val.as_str().to_string());
    }

    anyhow::bail!("Failed to read string attribute: unsupported string type")
}

/// Read a string scalar dataset that may be stored as either fixed-length or
/// variable-length Unicode.
pub(crate) fn read_string_dataset(ds: &Dataset) -> Result<String> {
    // Try variable-length Unicode first (our format)
    if let Ok(val) = ds.as_reader().read_scalar::<VarLenUnicode>() {
        return Ok(val.as_str().to_string());
    }

    // Try fixed-length Unicode with a large buffer (ITensors.jl format)
    if let Ok(val) = ds.as_reader().read_scalar::<FixedUnicode<256>>() {
        return Ok(val.as_str().trim_end_matches('\0').to_string());
    }

    // Fallback: try reading as VarLenAscii
    if let Ok(val) = ds.as_reader().read_scalar::<hdf5::types::VarLenAscii>() {
        return Ok(val.as_str().to_string());
    }

    anyhow::bail!("Failed to read string dataset: unsupported string type")
}

/// Read a string attribute by name from a group.
pub(crate) fn read_string_attr_by_name(group: &hdf5::Group, name: &str) -> Result<String> {
    let attr = group
        .attr(name)
        .context(format!("Attribute '{}' not found", name))?;
    read_string_attr(&attr)
}
