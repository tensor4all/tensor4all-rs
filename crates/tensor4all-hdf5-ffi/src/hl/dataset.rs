//! HDF5 Dataset handle.

use crate::class::ObjectClass;
use crate::error::Result;
use crate::globals::H5P_DEFAULT;
use crate::handle::Handle;
use crate::sync::sync;
use crate::sys::h5i::H5I_DATASET;
use crate::sys::{
    hid_t, H5Dclose, H5Dcreate2, H5Dget_space, H5Dget_type, H5Dopen2, H5Dread, H5Dwrite,
};
use crate::util::to_cstring;

use super::{Dataspace, Datatype};

/// An HDF5 dataset handle.
#[repr(transparent)]
pub struct Dataset(Handle);

impl ObjectClass for Dataset {
    const NAME: &'static str = "dataset";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_DATASET];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl Dataset {
    /// Open an existing dataset.
    pub fn open(loc_id: hid_t, name: &str) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Dopen2(loc_id, c_name.as_ptr(), H5P_DEFAULT) })?;
        Dataset::from_id(id)
    }

    /// Create a new dataset.
    pub fn create(loc_id: hid_t, name: &str, dtype: &Datatype, space: &Dataspace) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe {
            H5Dcreate2(
                loc_id,
                c_name.as_ptr(),
                dtype.id(),
                space.id(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        })?;
        Dataset::from_id(id)
    }

    /// Get the dataset's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }

    /// Get the dataspace.
    pub fn space(&self) -> Result<Dataspace> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Dget_space(self.id()) })?;
        Dataspace::from_id(id)
    }

    /// Get the datatype.
    pub fn dtype(&self) -> Result<Datatype> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Dget_type(self.id()) })?;
        Datatype::from_id(id)
    }

    /// Get the shape dimensions.
    pub fn shape(&self) -> Vec<usize> {
        self.space().map(|s| s.dims()).unwrap_or_default()
    }

    /// Get the total number of elements.
    pub fn size(&self) -> usize {
        self.space().map(|s| s.size()).unwrap_or(0)
    }

    /// Read raw data into a buffer.
    ///
    /// # Safety
    /// The buffer must be large enough to hold all data.
    pub unsafe fn read_raw(&self, mem_type: hid_t, buf: *mut std::ffi::c_void) -> Result<()> {
        let ret = sync(|| {
            H5Dread(
                self.id(),
                mem_type,
                crate::sys::h5s::H5S_ALL,
                crate::sys::h5s::H5S_ALL,
                H5P_DEFAULT,
                buf,
            )
        });
        if ret < 0 {
            return Err(crate::Error::Hdf5("Failed to read dataset".into()));
        }
        Ok(())
    }

    /// Write raw data from a buffer.
    ///
    /// # Safety
    /// The buffer must contain valid data matching the dataset's type and size.
    pub unsafe fn write_raw(&self, mem_type: hid_t, buf: *const std::ffi::c_void) -> Result<()> {
        let ret = sync(|| {
            H5Dwrite(
                self.id(),
                mem_type,
                crate::sys::h5s::H5S_ALL,
                crate::sys::h5s::H5S_ALL,
                H5P_DEFAULT,
                buf,
            )
        });
        if ret < 0 {
            return Err(crate::Error::Hdf5("Failed to write dataset".into()));
        }
        Ok(())
    }

    /// Create a reader for this dataset.
    pub fn as_reader(&self) -> DatasetReader<'_> {
        DatasetReader { dataset: self }
    }

    /// Create a writer for this dataset.
    pub fn as_writer(&self) -> DatasetWriter<'_> {
        DatasetWriter { dataset: self }
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Dclose(self.0.id()) });
        }
    }
}

/// Reader for a dataset.
pub struct DatasetReader<'a> {
    dataset: &'a Dataset,
}

impl<'a> DatasetReader<'a> {
    /// Read a scalar value.
    pub fn read_scalar<T: H5Type>(&self) -> Result<T> {
        let mut value = T::default_value();
        unsafe {
            self.dataset
                .read_raw(T::type_id()?, &mut value as *mut T as *mut std::ffi::c_void)?;
        }
        Ok(value)
    }

    /// Read data into a vector.
    pub fn read_1d<T: H5Type>(&self) -> Result<Vec<T>> {
        let size = self.dataset.size();
        let mut data = vec![T::default_value(); size];
        unsafe {
            self.dataset
                .read_raw(T::type_id()?, data.as_mut_ptr() as *mut std::ffi::c_void)?;
        }
        Ok(data)
    }
}

/// Writer for a dataset.
pub struct DatasetWriter<'a> {
    dataset: &'a Dataset,
}

impl<'a> DatasetWriter<'a> {
    /// Write a scalar value.
    pub fn write_scalar<T: H5Type>(&self, value: &T) -> Result<()> {
        unsafe {
            self.dataset
                .write_raw(T::type_id()?, value as *const T as *const std::ffi::c_void)?;
        }
        Ok(())
    }

    /// Write a slice of data (alias for write_slice).
    pub fn write<T: H5Type>(&self, data: &[T]) -> Result<()> {
        self.write_slice(data)
    }

    /// Write a slice of data.
    pub fn write_slice<T: H5Type>(&self, data: &[T]) -> Result<()> {
        unsafe {
            self.dataset
                .write_raw(T::type_id()?, data.as_ptr() as *const std::ffi::c_void)?;
        }
        Ok(())
    }
}

/// Trait for types that can be read/written to HDF5.
pub trait H5Type: Clone + Default {
    fn type_id() -> Result<hid_t>;
    fn default_value() -> Self {
        Self::default()
    }
}

// Implement H5Type for primitive types
impl H5Type for i8 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_INT8())
    }
}

impl H5Type for i16 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_INT16())
    }
}

impl H5Type for i32 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_INT32())
    }
}

impl H5Type for i64 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_INT64())
    }
}

impl H5Type for u8 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_UINT8())
    }
}

impl H5Type for u16 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_UINT16())
    }
}

impl H5Type for u32 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_UINT32())
    }
}

impl H5Type for u64 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_UINT64())
    }
}

impl H5Type for f32 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_FLOAT())
    }
}

impl H5Type for f64 {
    fn type_id() -> Result<hid_t> {
        Ok(crate::globals::H5T_NATIVE_DOUBLE())
    }
}

impl H5Type for crate::types::VarLenUnicode {
    fn type_id() -> Result<hid_t> {
        // Create a variable-length UTF-8 string type
        let dtype = super::Datatype::varlen_string()?;
        let id = dtype.id();
        // Leak the type so the ID remains valid
        // This is acceptable because:
        // 1. HDF5 maintains internal ref counts for types
        // 2. Programs typically create a small number of string types
        std::mem::forget(dtype);
        Ok(id)
    }
}

impl H5Type for crate::types::VarLenAscii {
    fn type_id() -> Result<hid_t> {
        // Use same type as VarLenUnicode - both are variable-length strings
        let dtype = super::Datatype::varlen_string()?;
        let id = dtype.id();
        std::mem::forget(dtype);
        Ok(id)
    }
}

impl<const N: usize> H5Type for crate::types::FixedUnicode<N> {
    fn type_id() -> Result<hid_t> {
        let dtype = super::Datatype::fixed_string(N)?;
        let id = dtype.id();
        std::mem::forget(dtype);
        Ok(id)
    }
}

impl<const N: usize> H5Type for crate::types::FixedAscii<N> {
    fn type_id() -> Result<hid_t> {
        let dtype = super::Datatype::fixed_string(N)?;
        let id = dtype.id();
        std::mem::forget(dtype);
        Ok(id)
    }
}

#[cfg(feature = "complex")]
impl H5Type for num_complex::Complex<f64> {
    fn type_id() -> Result<hid_t> {
        // Create compound type for Complex64 compatible with ITensors.jl
        let dtype = super::Datatype::complex64()?;
        let id = dtype.id();
        std::mem::forget(dtype);
        Ok(id)
    }
}

#[cfg(feature = "complex")]
impl H5Type for num_complex::Complex<f32> {
    fn type_id() -> Result<hid_t> {
        let dtype = super::Datatype::complex32()?;
        let id = dtype.id();
        std::mem::forget(dtype);
        Ok(id)
    }
}
