//! High-level HDF5 API.
//!
//! This module provides a simplified high-level Rust API for HDF5.
//! For now, it provides basic functionality; more features will be added later.

mod attribute;
mod dataset;
mod dataspace;
mod datatype;
mod file;
mod group;

pub use attribute::{Attribute, AttributeReader, AttributeWriter};
pub use dataset::{Dataset, DatasetReader, DatasetWriter, H5Type};
pub use dataspace::Dataspace;
pub use datatype::Datatype;
pub use file::{File, FileBuilder, OpenMode};
pub use group::{AttributeBuilder, DatasetBuilder, Group, H5TypeBuilder};

// Re-export types module
pub use crate::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
