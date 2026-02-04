//! HDF5 File handle.

use std::path::Path;

use crate::class::ObjectClass;
use crate::error::Result;
use crate::globals::H5P_DEFAULT;
use crate::h5call;
use crate::handle::Handle;
use crate::init::ensure_hdf5_init;
use crate::sync::sync;
use crate::sys::h5f::{H5F_ACC_EXCL, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_TRUNC};
use crate::sys::h5i::H5I_FILE;
use crate::sys::{hid_t, H5Fclose, H5Fcreate, H5Fopen};
use crate::util::to_cstring;

use super::Group;

/// File open mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenMode {
    /// Read-only access.
    Read,
    /// Read-write access.
    ReadWrite,
    /// Create new file, fail if exists.
    Create,
    /// Create new file, truncate if exists.
    Truncate,
}

/// An HDF5 file handle.
#[repr(transparent)]
pub struct File(Handle);

impl ObjectClass for File {
    const NAME: &'static str = "file";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_FILE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl File {
    /// Create a new HDF5 file.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_options().mode(OpenMode::Truncate).open(path)
    }

    /// Open an existing HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_options().mode(OpenMode::Read).open(path)
    }

    /// Open an existing HDF5 file for read-write access.
    pub fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_options().mode(OpenMode::ReadWrite).open(path)
    }

    /// Create a FileBuilder for more options.
    pub fn with_options() -> FileBuilder {
        FileBuilder::new()
    }

    /// Create a new group in this file.
    pub fn create_group(&self, name: &str) -> Result<Group> {
        Group::create(self, name)
    }

    /// Open an existing group in this file.
    pub fn group(&self, name: &str) -> Result<Group> {
        Group::open(self, name)
    }

    /// Get the file's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Fclose(self.0.id()) });
        }
    }
}

/// Builder for opening/creating HDF5 files.
pub struct FileBuilder {
    mode: OpenMode,
}

impl FileBuilder {
    /// Create a new FileBuilder with default options.
    pub fn new() -> Self {
        Self {
            mode: OpenMode::Read,
        }
    }

    /// Set the open mode.
    pub fn mode(mut self, mode: OpenMode) -> Self {
        self.mode = mode;
        self
    }

    /// Open or create the file.
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<File> {
        ensure_hdf5_init();

        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| crate::Error::Internal("Invalid path encoding".into()))?;
        let c_path = to_cstring(path_str)?;

        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe {
            match self.mode {
                OpenMode::Read => H5Fopen(c_path.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT),
                OpenMode::ReadWrite => H5Fopen(c_path.as_ptr(), H5F_ACC_RDWR, H5P_DEFAULT),
                OpenMode::Create => {
                    H5Fcreate(c_path.as_ptr(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT)
                }
                OpenMode::Truncate => {
                    H5Fcreate(c_path.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
                }
            }
        })?;
        File::from_id(id)
    }
}

impl Default for FileBuilder {
    fn default() -> Self {
        Self::new()
    }
}
