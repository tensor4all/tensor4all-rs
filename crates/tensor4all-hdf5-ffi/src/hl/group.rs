//! HDF5 Group handle.

use crate::class::ObjectClass;
use crate::dim::Dimension;
use crate::error::Result;
use crate::globals::H5P_DEFAULT;
use crate::handle::Handle;
use crate::sync::sync;
use crate::sys::h5i::H5I_GROUP;
use crate::sys::{hid_t, H5Gclose, H5Gcreate2, H5Gopen2};
use crate::util::to_cstring;

use super::dataset::H5Type;
use super::{Attribute, Dataset, Dataspace, Datatype, File};

/// An HDF5 group handle.
#[repr(transparent)]
pub struct Group(Handle);

impl ObjectClass for Group {
    const NAME: &'static str = "group";
    const VALID_TYPES: &'static [crate::sys::h5i::H5I_type_t] = &[H5I_GROUP];

    fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    fn handle(&self) -> &Handle {
        &self.0
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)
    }
}

impl Group {
    /// Create a new group within a file.
    pub fn create(file: &File, name: &str) -> Result<Self> {
        Self::create_in(file.id(), name)
    }

    /// Open an existing group within a file.
    pub fn open(file: &File, name: &str) -> Result<Self> {
        Self::open_in(file.id(), name)
    }

    /// Create a group within any location (file or group).
    fn create_in(loc_id: hid_t, name: &str) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Keep the entire creation and validation in a single sync block
        sync(|| {
            let id = unsafe {
                H5Gcreate2(
                    loc_id,
                    c_name.as_ptr(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                )
            };

            if id < 0 {
                return Err(crate::Error::Hdf5(format!(
                    "Failed to create group: {}",
                    name
                )));
            }

            Group::from_id(id)
        })
    }

    /// Open a group within any location (file or group).
    fn open_in(loc_id: hid_t, name: &str) -> Result<Self> {
        let c_name = to_cstring(name)?;
        // Keep the entire open and validation in a single sync block
        sync(|| {
            let id = unsafe { H5Gopen2(loc_id, c_name.as_ptr(), H5P_DEFAULT) };

            if id < 0 {
                return Err(crate::Error::Hdf5(format!(
                    "Failed to open group: {}",
                    name
                )));
            }

            Group::from_id(id)
        })
    }

    /// Get the group's ID.
    pub fn id(&self) -> hid_t {
        self.0.id()
    }

    /// Create a sub-group.
    pub fn create_group(&self, name: &str) -> Result<Group> {
        Self::create_in(self.id(), name)
    }

    /// Open a sub-group.
    pub fn group(&self, name: &str) -> Result<Group> {
        Self::open_in(self.id(), name)
    }

    /// Open an existing attribute.
    pub fn attr(&self, name: &str) -> Result<Attribute> {
        Attribute::open(self.id(), name)
    }

    /// Open an existing dataset.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        Dataset::open(self.id(), name)
    }

    /// Create an attribute builder.
    pub fn new_attr<T: H5TypeBuilder>(&self) -> AttributeBuilder<T> {
        AttributeBuilder::new(self.id())
    }

    /// Create a dataset builder.
    pub fn new_dataset<T: H5TypeBuilder>(&self) -> DatasetBuilder<T> {
        DatasetBuilder::new(self.id())
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.0.is_valid_user_id() {
            sync(|| unsafe { H5Gclose(self.0.id()) });
        }
    }
}

/// Trait for types that can be used in attribute/dataset builders.
pub trait H5TypeBuilder: H5Type {
    fn build_type() -> Result<Datatype>;
}

// Implement for primitive types
impl H5TypeBuilder for i8 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_i8()
    }
}

impl H5TypeBuilder for i16 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_i16()
    }
}

impl H5TypeBuilder for i32 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_i32()
    }
}

impl H5TypeBuilder for i64 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_i64()
    }
}

impl H5TypeBuilder for u8 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_u8()
    }
}

impl H5TypeBuilder for u16 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_u16()
    }
}

impl H5TypeBuilder for u32 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_u32()
    }
}

impl H5TypeBuilder for u64 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_u64()
    }
}

impl H5TypeBuilder for f32 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_f32()
    }
}

impl H5TypeBuilder for f64 {
    fn build_type() -> Result<Datatype> {
        Datatype::native_f64()
    }
}

impl H5TypeBuilder for crate::types::VarLenUnicode {
    fn build_type() -> Result<Datatype> {
        Datatype::varlen_string()
    }
}

impl H5TypeBuilder for crate::types::VarLenAscii {
    fn build_type() -> Result<Datatype> {
        // Use same type as VarLenUnicode
        Datatype::varlen_string()
    }
}

impl<const N: usize> H5TypeBuilder for crate::types::FixedUnicode<N> {
    fn build_type() -> Result<Datatype> {
        Datatype::fixed_string(N)
    }
}

impl<const N: usize> H5TypeBuilder for crate::types::FixedAscii<N> {
    fn build_type() -> Result<Datatype> {
        Datatype::fixed_string(N)
    }
}

#[cfg(feature = "complex")]
impl H5TypeBuilder for num_complex::Complex<f64> {
    fn build_type() -> Result<Datatype> {
        Datatype::complex64()
    }
}

#[cfg(feature = "complex")]
impl H5TypeBuilder for num_complex::Complex<f32> {
    fn build_type() -> Result<Datatype> {
        Datatype::complex32()
    }
}

/// Builder for creating attributes.
pub struct AttributeBuilder<T> {
    loc_id: hid_t,
    shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: H5TypeBuilder> AttributeBuilder<T> {
    fn new(loc_id: hid_t) -> Self {
        Self {
            loc_id,
            shape: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the shape (use () for scalar).
    pub fn shape<D: Dimension>(mut self, dims: D) -> Self {
        self.shape = dims.dims();
        self
    }

    /// Create the attribute.
    pub fn create(self, name: &str) -> Result<Attribute> {
        let dtype = T::build_type()?;
        let space = if self.shape.is_empty() {
            Dataspace::new_scalar()?
        } else {
            Dataspace::new(&self.shape)?
        };
        Attribute::create(self.loc_id, name, &dtype, &space)
    }
}

/// Builder for creating datasets.
pub struct DatasetBuilder<T> {
    loc_id: hid_t,
    shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: H5TypeBuilder> DatasetBuilder<T> {
    fn new(loc_id: hid_t) -> Self {
        Self {
            loc_id,
            shape: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the shape (use () for scalar).
    pub fn shape<D: Dimension>(mut self, dims: D) -> Self {
        self.shape = dims.dims();
        self
    }

    /// Create the dataset.
    pub fn create(self, name: &str) -> Result<Dataset> {
        let dtype = T::build_type()?;
        let space = if self.shape.is_empty() {
            Dataspace::new_scalar()?
        } else {
            Dataspace::new(&self.shape)?
        };
        Dataset::create(self.loc_id, name, &dtype, &space)
    }
}
