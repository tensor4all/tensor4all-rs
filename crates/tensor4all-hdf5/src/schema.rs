//! HDF5 type/version attribute helpers (ITensors.jl compatible).
//!
//! Every serialized object in the ITensors.jl HDF5 schema carries
//! `@type` (string) and `@version` (i64) attributes. This module
//! provides helpers to write and read them, replacing copy-pasted
//! boilerplate across index, itensor, and mps modules.

use crate::backend::types::VarLenUnicode;
use crate::backend::Group;
use anyhow::{bail, Result};
use std::str::FromStr;

/// Write `@type` and `@version` attributes to an HDF5 group.
pub(crate) fn write_type_version(group: &Group, type_name: &str, version: i64) -> Result<()> {
    let type_attr = group.new_attr::<VarLenUnicode>().shape(()).create("type")?;
    type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str(type_name)?)?;

    let version_attr = group.new_attr::<i64>().shape(()).create("version")?;
    version_attr.as_writer().write_scalar(&version)?;

    Ok(())
}

/// Read `@type` and `@version` attributes from an HDF5 group.
pub(crate) fn read_type_version(group: &Group) -> Result<(String, i64)> {
    let type_str = crate::compat::read_string_attr_by_name(group, "type")?;

    let version: i64 = group
        .attr("version")?
        .as_reader()
        .read_scalar()
        .map_err(|e| anyhow::anyhow!("Failed to read version attribute: {}", e))?;

    Ok((type_str, version))
}

/// Read and validate `@type` and `@version` attributes.
///
/// Returns the version number on success, or an error if:
/// - The type doesn't match `expected_type`
/// - The version exceeds `max_version`
pub(crate) fn require_type_version(
    group: &Group,
    expected_type: &str,
    max_version: i64,
) -> Result<i64> {
    let (type_str, version) = read_type_version(group)?;

    if type_str != expected_type {
        bail!(
            "Expected HDF5 type '{}', found '{}'",
            expected_type,
            type_str
        );
    }
    if version > max_version {
        bail!(
            "Unsupported {} version {} (max supported: {})",
            expected_type,
            version,
            max_version
        );
    }

    Ok(version)
}

#[cfg(test)]
mod tests;
