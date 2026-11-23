//! HDF5 backend abstraction layer.
//!
//! Selects between link-time (`hdf5-metno`) and runtime-loading (`hdf5-rt`) backends
//! based on Cargo feature flags. All other modules import HDF5 types through this
//! module, so the backend choice is transparent to the rest of the crate.
//!
//! When both features are active (due to Cargo feature unification),
//! `runtime-loading` takes priority.

use anyhow::Context;
use std::ffi::CString;

// When both features are active (due to Cargo feature unification),
// runtime-loading takes priority.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub use hdf5_metno::{types, Attribute, Dataset, File, Group, LinkType, Result};

#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::{types, Attribute, Dataset, File, Group, LinkType, Result};

// H5Aexists checks one known name without enumerating or allocating all attributes.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub(crate) fn attribute_exists(group: &Group, name: &str) -> anyhow::Result<bool> {
    hdf5_sync(|| {
        let name = CString::new(name).context("HDF5 attribute name contains NUL")?;
        let result = unsafe { hdf5_metno_sys::h5a::H5Aexists(group.id(), name.as_ptr()) };
        hdf5_metno::h5check(result)
            .map(|result| result > 0)
            .context("H5Aexists failed")
    })
}

#[cfg(feature = "runtime-loading")]
pub(crate) fn attribute_exists(group: &Group, name: &str) -> anyhow::Result<bool> {
    hdf5_sync(|| {
        let name = CString::new(name).context("HDF5 attribute name contains NUL")?;
        let result = unsafe { hdf5_rt::sys::h5a::H5Aexists(group.id(), name.as_ptr()) };
        hdf5_rt::h5check(result)
            .map(|result| result > 0)
            .context("H5Aexists failed")
    })
}

/// Run one whole public HDF5 operation under the backend's process-wide
/// reentrant lock.
///
/// This is the crate's single thread-safety choke point: every public
/// `save_*`/`load_*`/`append_*` entry point wraps its body in [`hdf5_sync`]
/// and no internal helper locks on its own. The backend lock is reentrant, so
/// nested HDF5 calls inside the operation are safe, and it also serializes
/// against direct hdf5-metno/hdf5-rt usage elsewhere in the process that goes
/// through the same lock.
///
/// This function never reads or writes the process environment. In
/// particular it does not touch `HDF5_USE_FILE_LOCKING`: whether HDF5 uses OS
/// file locking is the caller's decision, made before the first HDF5 call
/// (see the crate-level "Thread safety" docs).
pub(crate) fn hdf5_sync<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(all(feature = "link", not(feature = "runtime-loading")))]
    {
        hdf5_metno::sync::sync(f)
    }
    #[cfg(feature = "runtime-loading")]
    {
        hdf5_rt::sync::sync(f)
    }
}
