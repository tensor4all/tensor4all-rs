//! HDF5 Datatype handle.

use crate::class::ObjectClass;
use crate::error::Result;
use crate::globals;
use crate::h5call;
use crate::handle::Handle;
use crate::sync::sync;
use crate::sys::h5i::H5I_DATATYPE;
use crate::sys::h5t::{H5T_class_t, H5T_COMPOUND, H5T_CSET_UTF8, H5T_STR_NULLTERM, H5T_VARIABLE};
use crate::sys::{
    hid_t, H5Tclose, H5Tcopy, H5Tcreate, H5Tget_class, H5Tget_size, H5Tinsert, H5Tset_cset,
    H5Tset_size, H5Tset_strpad,
};
use crate::util::to_cstring;

/// An HDF5 datatype handle.
#[repr(transparent)]
pub struct Datatype(Handle);

impl ObjectClass for Datatype {
    const NAME: &'static str = "datatype";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_DATATYPE];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for Datatype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl Datatype {
    /// Create a datatype from native HDF5 type ID.
    pub fn from_type_id(type_id: hid_t) -> Result<Self> {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = h5call!(unsafe { H5Tcopy(type_id) })?;
        Datatype::from_id(id)
    }

    /// Get the datatype's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }

    /// Get the size of the datatype in bytes.
    pub fn size(&self) -> usize {
        sync(|| unsafe { H5Tget_size(self.id()) })
    }

    /// Get the class of the datatype.
    pub fn class(&self) -> H5T_class_t {
        // SAFETY: We convert the i32 return value to the enum
        let val = sync(|| unsafe { H5Tget_class(self.id()) });
        unsafe { std::mem::transmute(val) }
    }

    // Native type constructors

    /// Create i8 datatype.
    pub fn native_i8() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_INT8())
    }

    /// Create i16 datatype.
    pub fn native_i16() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_INT16())
    }

    /// Create i32 datatype.
    pub fn native_i32() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_INT32())
    }

    /// Create i64 datatype.
    pub fn native_i64() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_INT64())
    }

    /// Create u8 datatype.
    pub fn native_u8() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_UINT8())
    }

    /// Create u16 datatype.
    pub fn native_u16() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_UINT16())
    }

    /// Create u32 datatype.
    pub fn native_u32() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_UINT32())
    }

    /// Create u64 datatype.
    pub fn native_u64() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_UINT64())
    }

    /// Create f32 datatype.
    pub fn native_f32() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_FLOAT())
    }

    /// Create f64 datatype.
    pub fn native_f64() -> Result<Self> {
        Self::from_type_id(globals::H5T_NATIVE_DOUBLE())
    }

    /// Create a variable-length UTF-8 string datatype.
    pub fn varlen_string() -> Result<Self> {
        // Multiple HDF5 calls need the lock, but from_id should be outside
        // (matching hdf5-metno's pattern)
        let id = sync(|| {
            let id = unsafe { H5Tcopy(globals::H5T_C_S1()) };
            if id < 0 {
                return Err(crate::Error::Hdf5(
                    "Failed to create string datatype".into(),
                ));
            }

            // Set variable length
            if unsafe { H5Tset_size(id, H5T_VARIABLE) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to set variable length".into()));
            }

            // Set UTF-8 charset
            if unsafe { H5Tset_cset(id, H5T_CSET_UTF8 as i32) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to set UTF-8 charset".into()));
            }

            // Set null-terminated
            if unsafe { H5Tset_strpad(id, H5T_STR_NULLTERM as i32) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to set null-terminated".into()));
            }

            Ok(id)
        })?;
        Datatype::from_id(id)
    }

    /// Create a fixed-length UTF-8 string datatype.
    pub fn fixed_string(len: usize) -> Result<Self> {
        // Multiple HDF5 calls need the lock, but from_id should be outside
        // (matching hdf5-metno's pattern)
        let id = sync(|| {
            let id = unsafe { H5Tcopy(globals::H5T_C_S1()) };
            if id < 0 {
                return Err(crate::Error::Hdf5(
                    "Failed to create string datatype".into(),
                ));
            }

            // Set fixed length
            if unsafe { H5Tset_size(id, len) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to set string length".into()));
            }

            // Set UTF-8 charset
            if unsafe { H5Tset_cset(id, H5T_CSET_UTF8 as i32) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to set UTF-8 charset".into()));
            }

            Ok(id)
        })?;
        Datatype::from_id(id)
    }

    /// Create a Complex64 compound datatype (compatible with ITensors.jl).
    ///
    /// This creates a compound type with two f64 fields: "r" (real) and "i" (imaginary).
    pub fn complex64() -> Result<Self> {
        let r_name = to_cstring("r")?;
        let i_name = to_cstring("i")?;

        // Multiple HDF5 calls need the lock, but from_id should be outside
        // (matching hdf5-metno's pattern)
        let id = sync(|| {
            // Create compound type with size of two f64
            let id = unsafe { H5Tcreate(H5T_COMPOUND as i32, 16) }; // 2 * sizeof(f64)
            if id < 0 {
                return Err(crate::Error::Hdf5(
                    "Failed to create compound datatype".into(),
                ));
            }

            // Insert "r" (real) field at offset 0
            if unsafe { H5Tinsert(id, r_name.as_ptr(), 0, globals::H5T_NATIVE_DOUBLE()) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to insert 'r' field".into()));
            }

            // Insert "i" (imaginary) field at offset 8
            if unsafe { H5Tinsert(id, i_name.as_ptr(), 8, globals::H5T_NATIVE_DOUBLE()) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to insert 'i' field".into()));
            }

            Ok(id)
        })?;
        Datatype::from_id(id)
    }

    /// Create a Complex32 compound datatype.
    ///
    /// This creates a compound type with two f32 fields: "r" (real) and "i" (imaginary).
    pub fn complex32() -> Result<Self> {
        let r_name = to_cstring("r")?;
        let i_name = to_cstring("i")?;

        // Multiple HDF5 calls need the lock, but from_id should be outside
        // (matching hdf5-metno's pattern)
        let id = sync(|| {
            // Create compound type with size of two f32
            let id = unsafe { H5Tcreate(H5T_COMPOUND as i32, 8) }; // 2 * sizeof(f32)
            if id < 0 {
                return Err(crate::Error::Hdf5(
                    "Failed to create compound datatype".into(),
                ));
            }

            // Insert "r" (real) field at offset 0
            if unsafe { H5Tinsert(id, r_name.as_ptr(), 0, globals::H5T_NATIVE_FLOAT()) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to insert 'r' field".into()));
            }

            // Insert "i" (imaginary) field at offset 4
            if unsafe { H5Tinsert(id, i_name.as_ptr(), 4, globals::H5T_NATIVE_FLOAT()) } < 0 {
                unsafe { H5Tclose(id) };
                return Err(crate::Error::Hdf5("Failed to insert 'i' field".into()));
            }

            Ok(id)
        })?;
        Datatype::from_id(id)
    }
}

impl Clone for Datatype {
    fn clone(&self) -> Self {
        // Call HDF5 API with lock, then release lock before from_id
        // (matching hdf5-metno's pattern)
        let id = sync(|| unsafe { H5Tcopy(self.id()) });
        if id < 0 {
            Self::from_handle(Handle::invalid())
        } else {
            Self::from_id(id).unwrap_or_else(|_| Self::from_handle(Handle::invalid()))
        }
    }
}

impl Drop for Datatype {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Tclose(self.0.id()) });
        }
    }
}
