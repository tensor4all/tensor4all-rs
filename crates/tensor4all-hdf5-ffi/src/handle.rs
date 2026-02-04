//! Handle to an HDF5 object.

use std::mem;

use crate::error::Result;
use crate::sys::h5i::{H5I_type_t, H5I_BADID, H5I_NTYPES};
use crate::sys::{
    hid_t, H5Idec_ref, H5Iget_ref, H5Iget_type, H5Iinc_ref, H5Iis_valid, H5I_INVALID_HID,
};

/// A handle to an HDF5 object
#[derive(Debug)]
pub struct Handle {
    id: hid_t,
}

impl Handle {
    /// Create a handle from object ID, taking ownership of it.
    ///
    /// Uses `H5Iis_valid` to check if the ID is valid. If `H5Iis_valid` is
    /// unreliable (as seen on some HDF5 builds), use `try_new_trusted` instead.
    pub fn try_new(id: hid_t) -> Result<Self> {
        let handle = Self { id };
        if handle.is_valid_user_id() {
            Ok(handle)
        } else {
            // Drop on an invalid handle could cause closing an unrelated object
            // in the destructor, hence it's important to prevent the drop here.
            mem::forget(handle);
            Err(From::from(format!("Invalid handle id: {id}")))
        }
    }

    /// Create a handle from an ID returned directly from an HDF5 API call.
    ///
    /// This method trusts that the caller has verified the ID is valid
    /// (e.g., the HDF5 function returned a non-negative value). It does NOT
    /// call any HDF5 validation functions (H5Iis_valid, H5Iget_type), which
    /// may be unreliable on some HDF5 builds.
    ///
    /// # Safety
    /// Caller must ensure the ID was just returned from an HDF5 API call
    /// that returned success (non-negative value).
    pub fn try_new_trusted(id: hid_t) -> Result<Self> {
        // Trust the ID if it's positive - no HDF5 validation calls
        if id > 0 {
            Ok(Self { id })
        } else {
            Err(From::from(format!("Invalid handle id: {id}")))
        }
    }

    /// Create a handle from object ID by cloning it
    pub fn try_borrow(id: hid_t) -> Result<Self> {
        let handle = Self::try_new(id)?;
        handle.incref();
        Ok(handle)
    }

    pub const fn invalid() -> Self {
        Self {
            id: H5I_INVALID_HID,
        }
    }

    pub const fn id(&self) -> hid_t {
        self.id
    }

    /// Increment the reference count of the handle
    pub fn incref(&self) {
        if self.is_valid_user_id() {
            h5lock!(unsafe { H5Iinc_ref(self.id) });
        }
    }

    /// Decrease the reference count of the handle
    pub fn decref(&self) {
        h5lock!({
            if self.is_valid_id() {
                unsafe { H5Idec_ref(self.id) };
            }
        });
    }

    /// Returns `true` if the object has a valid unlocked identifier
    pub fn is_valid_user_id(&self) -> bool {
        h5lock!(unsafe { H5Iis_valid(self.id) }) == 1
    }

    pub fn is_valid_id(&self) -> bool {
        matches!(self.id_type(), tp if tp > H5I_BADID && tp < H5I_NTYPES)
    }

    /// Return the reference count of the object
    pub fn refcount(&self) -> u32 {
        h5call!(unsafe { H5Iget_ref(self.id) })
            .map(|x| x as _)
            .unwrap_or(0) as _
    }

    /// Get HDF5 object type as a native enum.
    pub fn id_type(&self) -> H5I_type_t {
        if self.id <= 0 {
            H5I_BADID
        } else {
            let tp = h5lock!(unsafe { H5Iget_type(self.id) });
            // Convert i32 to enum by checking if in valid range
            if tp > (H5I_BADID as i32) && tp < (H5I_NTYPES as i32) {
                // SAFETY: We've verified tp is in the valid range of H5I_type_t
                unsafe { std::mem::transmute(tp) }
            } else {
                H5I_BADID
            }
        }
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self::try_borrow(self.id).unwrap_or_else(|_| Self::invalid())
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        h5lock!(self.decref());
    }
}
