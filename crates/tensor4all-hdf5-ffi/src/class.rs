//! Object class trait for HDF5 objects.

use std::fmt;
use std::mem;
use std::ptr::{self, addr_of};

use crate::error::Result;
use crate::handle::Handle;
use crate::sys::h5i::H5I_type_t;
use crate::sys::hid_t;

pub trait ObjectClass: Sized {
    const NAME: &'static str;
    const VALID_TYPES: &'static [H5I_type_t];

    fn from_handle(handle: Handle) -> Self;

    fn handle(&self) -> &Handle;

    fn short_repr(&self) -> Option<String> {
        None
    }

    fn validate(&self) -> Result<()> {
        Ok(())
    }

    fn from_id(id: hid_t) -> Result<Self> {
        h5lock!({
            let handle = Handle::try_new(id)?;
            if Self::is_valid_id_type(handle.id_type()) {
                let obj = Self::from_handle(handle);
                obj.validate().map(|()| obj)
            } else {
                Err(From::from(format!("Invalid {} id: {}", Self::NAME, id)))
            }
        })
    }

    fn invalid() -> Self {
        Self::from_handle(Handle::invalid())
    }

    fn is_valid_id_type(tp: H5I_type_t) -> bool {
        Self::VALID_TYPES.is_empty() || Self::VALID_TYPES.contains(&tp)
    }

    unsafe fn transmute<T: ObjectClass>(&self) -> &T {
        &*(self as *const Self).cast::<T>()
    }

    unsafe fn transmute_mut<T: ObjectClass>(&mut self) -> &mut T {
        &mut *(self as *mut Self).cast::<T>()
    }

    unsafe fn cast_unchecked<T: ObjectClass>(self) -> T {
        let obj = ptr::read(addr_of!(self).cast());
        mem::forget(self);
        obj
    }

    fn cast<T: ObjectClass>(self) -> Result<T> {
        let id_type = self.handle().id_type();
        if Self::is_valid_id_type(id_type) {
            Ok(unsafe { self.cast_unchecked() })
        } else {
            Err(format!(
                "unable to cast {} ({:?}) into {}",
                Self::NAME,
                id_type,
                T::NAME
            )
            .into())
        }
    }

    fn debug_fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        h5lock!({
            if !self.handle().is_valid_user_id() {
                write!(f, "<HDF5 {}: invalid id>", Self::NAME)
            } else if let Some(d) = self.short_repr() {
                write!(f, "<HDF5 {}: {}>", Self::NAME, d)
            } else {
                write!(f, "<HDF5 {}>", Self::NAME)
            }
        })
    }
}

/// Takes ownership of an object via its identifier.
///
/// # Safety
///
/// This should only be called with an identifier obtained from an object constructor in the HDF5 C
/// library.
pub unsafe fn from_id<T: ObjectClass>(id: hid_t) -> Result<T> {
    T::from_id(id)
}
