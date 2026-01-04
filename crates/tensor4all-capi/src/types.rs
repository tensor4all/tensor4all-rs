//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::ffi::c_void;
use tensor4all_core_common::index::{DefaultIndex, DynId, NoSymmSpace};
use tensor4all_core_tensor::{TensorDynLen, Storage};

/// The internal index type we're wrapping
pub(crate) type InternalIndex = DefaultIndex<DynId, NoSymmSpace>;

/// The internal tensor type we're wrapping
pub(crate) type InternalTensor = TensorDynLen<DynId, NoSymmSpace>;

/// Opaque index type for C API
///
/// Wraps `DefaultIndex<DynId, NoSymmSpace>` which corresponds to ITensors.jl's `Index{Int}`.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_index {
    pub(crate) _private: *const c_void,
}

impl t4a_index {
    /// Create a new t4a_index from an InternalIndex
    pub(crate) fn new(index: InternalIndex) -> Self {
        Self {
            _private: Box::into_raw(Box::new(index)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalIndex
    pub(crate) fn inner(&self) -> &InternalIndex {
        unsafe { &*(self._private as *const InternalIndex) }
    }

    /// Get a mutable reference to the inner InternalIndex
    pub(crate) fn inner_mut(&mut self) -> &mut InternalIndex {
        unsafe { &mut *(self._private as *mut InternalIndex) }
    }
}

impl Clone for t4a_index {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_index {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalIndex);
            }
        }
    }
}

// Safety: t4a_index is Send + Sync because InternalIndex is Send + Sync
unsafe impl Send for t4a_index {}
unsafe impl Sync for t4a_index {}

/// Storage kind enum for C API
///
/// Represents the type of storage backend used by a tensor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_storage_kind {
    /// Dense storage with f64 elements
    DenseF64 = 0,
    /// Dense storage with Complex64 elements
    DenseC64 = 1,
    /// Diagonal storage with f64 elements
    DiagF64 = 2,
    /// Diagonal storage with Complex64 elements
    DiagC64 = 3,
}

impl t4a_storage_kind {
    /// Convert from Rust Storage to t4a_storage_kind
    pub(crate) fn from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(_) => Self::DenseF64,
            Storage::DenseC64(_) => Self::DenseC64,
            Storage::DiagF64(_) => Self::DiagF64,
            Storage::DiagC64(_) => Self::DiagC64,
        }
    }
}

/// Opaque tensor type for C API
///
/// Wraps `TensorDynLen<DynId, NoSymmSpace>` which corresponds to ITensors.jl's `ITensor`.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_tensor {
    pub(crate) _private: *const c_void,
}

impl t4a_tensor {
    /// Create a new t4a_tensor from an InternalTensor
    pub(crate) fn new(tensor: InternalTensor) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tensor)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalTensor
    pub(crate) fn inner(&self) -> &InternalTensor {
        unsafe { &*(self._private as *const InternalTensor) }
    }

    /// Get a mutable reference to the inner InternalTensor
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTensor {
        unsafe { &mut *(self._private as *mut InternalTensor) }
    }
}

impl Clone for t4a_tensor {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_tensor {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalTensor);
            }
        }
    }
}

// Safety: t4a_tensor is Send + Sync because InternalTensor is Send + Sync
unsafe impl Send for t4a_tensor {}
unsafe impl Sync for t4a_tensor {}
