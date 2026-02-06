//! HDF5 backend abstraction layer.
//!
//! Selects between link-time (hdf5-metno) and runtime-loading (hdf5-rt) backends
//! based on feature flags. All other modules import HDF5 types through this module.

// When both features are active (due to Cargo feature unification),
// runtime-loading takes priority.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub use hdf5_metno::{types, Attribute, Dataset, File, Group, Result};

#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::{types, Attribute, Dataset, File, Group, Result};
