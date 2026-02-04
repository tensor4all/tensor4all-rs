//! HDF5 Dataspace handle.

use crate::class::ObjectClass;
use crate::dim::Dimension;
use crate::error::Result;
use crate::handle::Handle;
use crate::sync::sync;
use crate::sys::h5i::H5I_DATASPACE;
use crate::sys::h5s::H5S_SCALAR;
use crate::sys::{
    hid_t, hsize_t, H5Sclose, H5Screate, H5Screate_simple, H5Sget_simple_extent_dims,
    H5Sget_simple_extent_ndims,
};

/// An HDF5 dataspace handle.
#[repr(transparent)]
pub struct Dataspace(Handle);

impl ObjectClass for Dataspace {
    const NAME: &'static str = "dataspace";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_DATASPACE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for Dataspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl Dataspace {
    /// Create a scalar dataspace.
    pub fn new_scalar() -> Result<Self> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Screate(H5S_SCALAR as i32) })?;
        Dataspace::from_id(id)
    }

    /// Create a simple dataspace with the given dimensions.
    pub fn new<D: Dimension>(dims: D) -> Result<Self> {
        let dims_vec = dims.dims();
        if dims_vec.is_empty() {
            return Self::new_scalar();
        }

        let h5_dims: Vec<hsize_t> = dims_vec.iter().map(|&d| d as hsize_t).collect();
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe {
            H5Screate_simple(h5_dims.len() as i32, h5_dims.as_ptr(), std::ptr::null())
        })?;
        Dataspace::from_id(id)
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        let n = sync(|| unsafe { H5Sget_simple_extent_ndims(self.id()) });
        if n < 0 {
            0
        } else {
            n as usize
        }
    }

    /// Get the dimensions.
    pub fn dims(&self) -> Vec<usize> {
        let ndim = self.ndim();
        if ndim == 0 {
            return vec![];
        }

        let mut dims = vec![0 as hsize_t; ndim];
        sync(|| unsafe {
            H5Sget_simple_extent_dims(self.id(), dims.as_mut_ptr(), std::ptr::null_mut());
        });
        dims.iter().map(|&d| d as usize).collect()
    }

    /// Get the total number of elements.
    pub fn size(&self) -> usize {
        let dims = self.dims();
        if dims.is_empty() {
            1 // scalar
        } else {
            dims.iter().product()
        }
    }

    /// Get the dataspace's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }
}

impl Drop for Dataspace {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Sclose(self.0.id()) });
        }
    }
}
