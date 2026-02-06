//! HDF5 backend abstraction layer.
//!
//! Selects between link-time (hdf5-metno) and runtime-loading (hdf5-rt) backends
//! based on feature flags. All other modules import HDF5 types through this module.

#[cfg(feature = "link")]
pub use hdf5_metno::{types, Attribute, Dataset, File, Group, Result};

#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::{types, Attribute, Dataset, File, Group, Result};
