//! HDF5 Attribute handle.

use crate::class::ObjectClass;
use crate::error::Result;
use crate::globals::H5P_DEFAULT;
use crate::handle::Handle;
use crate::sync::sync;
use crate::sys::h5i::H5I_ATTR;
use crate::sys::{
    hid_t, H5Aclose, H5Acreate2, H5Aget_space, H5Aget_type, H5Aopen, H5Aread, H5Awrite,
};
use crate::util::to_cstring;

use super::dataset::H5Type;
use super::{Dataspace, Datatype};

/// An HDF5 attribute handle.
#[repr(transparent)]
pub struct Attribute(Handle);

impl ObjectClass for Attribute {
    const NAME: &'static str = "attribute";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_ATTR];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for Attribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl Attribute {
    /// Open an existing attribute.
    pub fn open(loc_id: hid_t, name: &str) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Call H5Aopen with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Aopen(loc_id, c_name.as_ptr(), H5P_DEFAULT) })?;
        Attribute::from_id(id)
    }

    /// Create a new attribute.
    pub fn create(loc_id: hid_t, name: &str, dtype: &Datatype, space: &Dataspace) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Call H5Acreate2 with lock, then release lock before from_id
        // (matching hdf5-metno's pattern: h5try! releases lock before from_id)
        let id = h5call!(unsafe {
            H5Acreate2(
                loc_id,
                c_name.as_ptr(),
                dtype.id(),
                space.id(),
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        })?;
        Attribute::from_id(id)
    }

    /// Get the attribute's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }

    /// Get the dataspace.
    pub fn space(&self) -> Result<Dataspace> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Aget_space(self.id()) })?;
        Dataspace::from_id(id)
    }

    /// Get the datatype.
    pub fn dtype(&self) -> Result<Datatype> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Aget_type(self.id()) })?;
        Datatype::from_id(id)
    }

    /// Read raw data into a buffer.
    ///
    /// # Safety
    /// The buffer must be large enough to hold all data.
    pub unsafe fn read_raw(&self, mem_type: hid_t, buf: *mut std::ffi::c_void) -> Result<()> {
        let ret = sync(|| H5Aread(self.id(), mem_type, buf));
        if ret < 0 {
            return Err(crate::Error::Hdf5("Failed to read attribute".into()));
        }
        Ok(())
    }

    /// Write raw data from a buffer.
    ///
    /// # Safety
    /// The buffer must contain valid data matching the attribute's type and size.
    pub unsafe fn write_raw(&self, mem_type: hid_t, buf: *const std::ffi::c_void) -> Result<()> {
        let ret = sync(|| H5Awrite(self.id(), mem_type, buf));
        if ret < 0 {
            return Err(crate::Error::Hdf5("Failed to write attribute".into()));
        }
        Ok(())
    }

    /// Create a reader for this attribute.
    pub fn as_reader(&self) -> AttributeReader<'_> {
        AttributeReader { attr: self }
    }

    /// Create a writer for this attribute.
    pub fn as_writer(&self) -> AttributeWriter<'_> {
        AttributeWriter { attr: self }
    }
}

impl Drop for Attribute {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Aclose(self.0.id()) });
        }
    }
}

/// Reader for an attribute.
pub struct AttributeReader<'a> {
    attr: &'a Attribute,
}

impl<'a> AttributeReader<'a> {
    /// Read a scalar value.
    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        let mut value = T::default_value();
        unsafe {
            self.attr
                .read_raw(T::type_id()?, &mut value as *mut T as *mut std::ffi::c_void)?;
        }
        Ok(value)
    }
}

/// Writer for an attribute.
pub struct AttributeWriter<'a> {
    attr: &'a Attribute,
}

impl<'a> AttributeWriter<'a> {
    /// Write a scalar value.
    pub fn write_scalar<T: H5Type>(&self, value: &T) -> Result<()> {
        unsafe {
            self.attr
                .write_raw(T::type_id()?, value as *const T as *const std::ffi::c_void)?;
        }
        Ok(())
    }
}
