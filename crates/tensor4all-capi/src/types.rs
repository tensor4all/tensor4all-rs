//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::ffi::c_void;
use tensor4all_core_common::index::{DefaultIndex, DynId, NoSymmSpace};
use tensor4all_core_tensor::{TensorDynLen, Storage};
use tensor4all_itensorlike::TensorTrain;

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

// ============================================================================
// TensorTrain type
// ============================================================================

/// The internal tensor train type we're wrapping
pub(crate) type InternalTensorTrain = TensorTrain<DynId, NoSymmSpace>;

/// Opaque tensor train type for C API
///
/// Wraps `TensorTrain<DynId, NoSymmSpace>` which corresponds to ITensorMPS.jl's `MPS`.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_tensortrain {
    pub(crate) _private: *const c_void,
}

impl t4a_tensortrain {
    /// Create a new t4a_tensortrain from an InternalTensorTrain
    pub(crate) fn new(tt: InternalTensorTrain) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tt)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalTensorTrain
    pub(crate) fn inner(&self) -> &InternalTensorTrain {
        unsafe { &*(self._private as *const InternalTensorTrain) }
    }

    /// Get a mutable reference to the inner InternalTensorTrain
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTensorTrain {
        unsafe { &mut *(self._private as *mut InternalTensorTrain) }
    }
}

impl Clone for t4a_tensortrain {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_tensortrain {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalTensorTrain);
            }
        }
    }
}

// Safety: t4a_tensortrain is Send + Sync because InternalTensorTrain is Send + Sync
unsafe impl Send for t4a_tensortrain {}
unsafe impl Sync for t4a_tensortrain {}

/// Canonical form enum for C API
///
/// Represents the method used for canonicalization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_canonical_form {
    /// Unitary (QR-based) canonicalization - tensors are isometric
    Unitary = 0,
    /// LU-based canonicalization - one factor has unit diagonal
    LU = 1,
    /// Cross Interpolation based canonicalization
    CI = 2,
}

impl From<tensor4all_itensorlike::CanonicalForm> for t4a_canonical_form {
    fn from(form: tensor4all_itensorlike::CanonicalForm) -> Self {
        match form {
            tensor4all_itensorlike::CanonicalForm::Unitary => Self::Unitary,
            tensor4all_itensorlike::CanonicalForm::LU => Self::LU,
            tensor4all_itensorlike::CanonicalForm::CI => Self::CI,
        }
    }
}

impl From<t4a_canonical_form> for tensor4all_itensorlike::CanonicalForm {
    fn from(form: t4a_canonical_form) -> Self {
        match form {
            t4a_canonical_form::Unitary => Self::Unitary,
            t4a_canonical_form::LU => Self::LU,
            t4a_canonical_form::CI => Self::CI,
        }
    }
}

// ============================================================================
// Algorithm types
// ============================================================================

/// Factorization algorithm for C API
///
/// Used for matrix decomposition in compression and truncation operations.
///
/// Corresponds to `FactorizeAlgorithm` in Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_factorize_algorithm {
    /// Singular Value Decomposition (default)
    SVD = 0,
    /// LU decomposition with partial pivoting
    LU = 1,
    /// Cross Interpolation / Skeleton decomposition
    CI = 2,
}

impl Default for t4a_factorize_algorithm {
    fn default() -> Self {
        Self::SVD
    }
}

impl From<tensor4all_core_common::FactorizeAlgorithm> for t4a_factorize_algorithm {
    fn from(alg: tensor4all_core_common::FactorizeAlgorithm) -> Self {
        match alg {
            tensor4all_core_common::FactorizeAlgorithm::SVD => Self::SVD,
            tensor4all_core_common::FactorizeAlgorithm::LU => Self::LU,
            tensor4all_core_common::FactorizeAlgorithm::CI => Self::CI,
        }
    }
}

impl From<t4a_factorize_algorithm> for tensor4all_core_common::FactorizeAlgorithm {
    fn from(alg: t4a_factorize_algorithm) -> Self {
        match alg {
            t4a_factorize_algorithm::SVD => Self::SVD,
            t4a_factorize_algorithm::LU => Self::LU,
            t4a_factorize_algorithm::CI => Self::CI,
        }
    }
}

/// Contraction algorithm for C API
///
/// Used for tensor train contraction (TT-TT or MPO-MPO).
///
/// Corresponds to `ContractionAlgorithm` in Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_contraction_algorithm {
    /// Naive contraction followed by compression (default)
    Naive = 0,
    /// Zip-up contraction with on-the-fly compression
    ZipUp = 1,
    /// Variational fitting algorithm
    Fit = 2,
}

impl Default for t4a_contraction_algorithm {
    fn default() -> Self {
        Self::Naive
    }
}

impl From<tensor4all_core_common::ContractionAlgorithm> for t4a_contraction_algorithm {
    fn from(alg: tensor4all_core_common::ContractionAlgorithm) -> Self {
        match alg {
            tensor4all_core_common::ContractionAlgorithm::Naive => Self::Naive,
            tensor4all_core_common::ContractionAlgorithm::ZipUp => Self::ZipUp,
            tensor4all_core_common::ContractionAlgorithm::Fit => Self::Fit,
        }
    }
}

impl From<t4a_contraction_algorithm> for tensor4all_core_common::ContractionAlgorithm {
    fn from(alg: t4a_contraction_algorithm) -> Self {
        match alg {
            t4a_contraction_algorithm::Naive => Self::Naive,
            t4a_contraction_algorithm::ZipUp => Self::ZipUp,
            t4a_contraction_algorithm::Fit => Self::Fit,
        }
    }
}

/// Compression algorithm for C API
///
/// Used for tensor train compression.
///
/// Corresponds to `CompressionAlgorithm` in Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_compression_algorithm {
    /// SVD-based compression (default)
    SVD = 0,
    /// LU-based compression
    LU = 1,
    /// Cross Interpolation based compression
    CI = 2,
    /// Variational compression
    Variational = 3,
}

impl Default for t4a_compression_algorithm {
    fn default() -> Self {
        Self::SVD
    }
}

impl From<tensor4all_core_common::CompressionAlgorithm> for t4a_compression_algorithm {
    fn from(alg: tensor4all_core_common::CompressionAlgorithm) -> Self {
        match alg {
            tensor4all_core_common::CompressionAlgorithm::SVD => Self::SVD,
            tensor4all_core_common::CompressionAlgorithm::LU => Self::LU,
            tensor4all_core_common::CompressionAlgorithm::CI => Self::CI,
            tensor4all_core_common::CompressionAlgorithm::Variational => Self::Variational,
        }
    }
}

impl From<t4a_compression_algorithm> for tensor4all_core_common::CompressionAlgorithm {
    fn from(alg: t4a_compression_algorithm) -> Self {
        match alg {
            t4a_compression_algorithm::SVD => Self::SVD,
            t4a_compression_algorithm::LU => Self::LU,
            t4a_compression_algorithm::CI => Self::CI,
            t4a_compression_algorithm::Variational => Self::Variational,
        }
    }
}

/// Contract method for tensor train contraction in C API
///
/// Used for tensor train contraction operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_contract_method {
    /// Zip-up contraction (faster, one-pass) - default
    Zipup = 0,
    /// Fit/variational contraction (iterative optimization)
    Fit = 1,
    /// Naive contraction: contract to full tensor, then decompose back.
    /// Useful for debugging and testing, but O(exp(n)) in memory.
    Naive = 2,
}

impl Default for t4a_contract_method {
    fn default() -> Self {
        Self::Zipup
    }
}

impl From<tensor4all_itensorlike::ContractMethod> for t4a_contract_method {
    fn from(method: tensor4all_itensorlike::ContractMethod) -> Self {
        match method {
            tensor4all_itensorlike::ContractMethod::Zipup => Self::Zipup,
            tensor4all_itensorlike::ContractMethod::Fit => Self::Fit,
            tensor4all_itensorlike::ContractMethod::Naive => Self::Naive,
        }
    }
}

impl From<t4a_contract_method> for tensor4all_itensorlike::ContractMethod {
    fn from(method: t4a_contract_method) -> Self {
        match method {
            t4a_contract_method::Zipup => Self::Zipup,
            t4a_contract_method::Fit => Self::Fit,
            t4a_contract_method::Naive => Self::Naive,
        }
    }
}
