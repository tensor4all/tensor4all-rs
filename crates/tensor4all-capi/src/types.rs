//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::ffi::c_void;
use tensor4all_core::{DynIndex, Storage, TensorDynLen};
use tensor4all_quanticstci::QuanticsTensorCI2;
use tensor4all_treetn::{DefaultTreeTN, LinearOperator};

/// The internal index type we're wrapping (DynIndex = Index<DynId, TagSet>)
pub(crate) type InternalIndex = DynIndex;

/// The internal tensor type we're wrapping (TensorDynLen is a concrete type, not generic)
pub(crate) type InternalTensor = TensorDynLen;

/// Opaque index type for C API
///
/// Wraps `DynIndex` (= `Index<DynId, TagSet>`) which corresponds to ITensors.jl's `Index{Int}`.
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
        if storage.is_diag() {
            if storage.is_f64() {
                Self::DiagF64
            } else {
                Self::DiagC64
            }
        } else if storage.is_f64() {
            Self::DenseF64
        } else {
            Self::DenseC64
        }
    }
}

/// Opaque tensor type for C API
///
/// Wraps `TensorDynLen` which corresponds to ITensors.jl's `ITensor`.
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
// TreeTN type
// ============================================================================

/// The internal tree tensor network type we're wrapping.
pub(crate) type InternalTreeTN = DefaultTreeTN<usize>;

/// Opaque tree tensor network type for C API
///
/// Wraps `DefaultTreeTN<usize>` which is a tree-shaped tensor network
/// generalizing MPS/TT to tree topologies.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_treetn {
    pub(crate) _private: *const c_void,
}

impl t4a_treetn {
    /// Create a new t4a_treetn from an InternalTreeTN
    pub(crate) fn new(treetn: InternalTreeTN) -> Self {
        Self {
            _private: Box::into_raw(Box::new(treetn)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalTreeTN
    pub(crate) fn inner(&self) -> &InternalTreeTN {
        unsafe { &*(self._private as *const InternalTreeTN) }
    }

    /// Get a mutable reference to the inner InternalTreeTN
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTreeTN {
        unsafe { &mut *(self._private as *mut InternalTreeTN) }
    }
}

impl Clone for t4a_treetn {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_treetn {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalTreeTN);
            }
        }
    }
}

// Safety: t4a_treetn is Send + Sync because InternalTreeTN is Send + Sync
unsafe impl Send for t4a_treetn {}
unsafe impl Sync for t4a_treetn {}

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

// Note: tensor4all_itensorlike::CanonicalForm IS tensor4all_treetn::CanonicalForm
// (re-exported), so the From impls above also cover the treetn crate.

// ============================================================================
// Algorithm types
// ============================================================================

/// Factorization algorithm for C API
///
/// Used for matrix decomposition in compression and truncation operations.
///
/// Corresponds to `FactorizeAlg` in Rust.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_factorize_algorithm {
    /// Singular Value Decomposition (default)
    #[default]
    SVD = 0,
    /// LU decomposition with partial pivoting
    LU = 1,
    /// Cross Interpolation / Skeleton decomposition
    CI = 2,
    /// QR decomposition
    QR = 3,
}

impl From<tensor4all_core::FactorizeAlg> for t4a_factorize_algorithm {
    fn from(alg: tensor4all_core::FactorizeAlg) -> Self {
        match alg {
            tensor4all_core::FactorizeAlg::SVD => Self::SVD,
            tensor4all_core::FactorizeAlg::LU => Self::LU,
            tensor4all_core::FactorizeAlg::CI => Self::CI,
            tensor4all_core::FactorizeAlg::QR => Self::QR,
        }
    }
}

impl From<t4a_factorize_algorithm> for tensor4all_core::FactorizeAlg {
    fn from(alg: t4a_factorize_algorithm) -> Self {
        match alg {
            t4a_factorize_algorithm::SVD => Self::SVD,
            t4a_factorize_algorithm::LU => Self::LU,
            t4a_factorize_algorithm::CI => Self::CI,
            t4a_factorize_algorithm::QR => Self::QR,
        }
    }
}

/// Contraction algorithm for C API
///
/// Used for tensor train contraction (TT-TT or MPO-MPO).
///
/// This is a standalone C API type (not mapped to tensor4all-core type).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_contraction_algorithm {
    /// Naive contraction followed by compression (default)
    #[default]
    Naive = 0,
    /// Zip-up contraction with on-the-fly compression
    ZipUp = 1,
    /// Variational fitting algorithm
    Fit = 2,
}

/// Compression algorithm for C API
///
/// Used for tensor train compression.
///
/// This is a standalone C API type (not mapped to tensor4all-core type).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_compression_algorithm {
    /// SVD-based compression (default)
    #[default]
    SVD = 0,
    /// LU-based compression
    LU = 1,
    /// Cross Interpolation based compression
    CI = 2,
    /// Variational compression
    Variational = 3,
}

/// Contract method for tensor train contraction in C API
///
/// Used for tensor train contraction operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_contract_method {
    /// Zip-up contraction (faster, one-pass) - default
    #[default]
    Zipup = 0,
    /// Fit/variational contraction (iterative optimization)
    Fit = 1,
    /// Naive contraction: contract to full tensor, then decompose back.
    /// Useful for debugging and testing, but O(exp(n)) in memory.
    Naive = 2,
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

impl From<tensor4all_treetn::treetn::contraction::ContractionMethod> for t4a_contract_method {
    fn from(method: tensor4all_treetn::treetn::contraction::ContractionMethod) -> Self {
        match method {
            tensor4all_treetn::treetn::contraction::ContractionMethod::Zipup => Self::Zipup,
            tensor4all_treetn::treetn::contraction::ContractionMethod::Fit => Self::Fit,
            tensor4all_treetn::treetn::contraction::ContractionMethod::Naive => Self::Naive,
        }
    }
}

impl From<t4a_contract_method> for tensor4all_treetn::treetn::contraction::ContractionMethod {
    fn from(method: t4a_contract_method) -> Self {
        match method {
            t4a_contract_method::Zipup => Self::Zipup,
            t4a_contract_method::Fit => Self::Fit,
            t4a_contract_method::Naive => Self::Naive,
        }
    }
}

// ============================================================================
// QuanticsGrids types
// ============================================================================

/// The internal DiscretizedGrid type we're wrapping.
pub(crate) type InternalDiscretizedGrid = quanticsgrids::DiscretizedGrid;

/// Opaque discretized grid type for C API
///
/// Wraps `quanticsgrids::DiscretizedGrid` which provides coordinate conversions
/// between continuous coordinates, grid indices, and quantics indices.
#[repr(C)]
pub struct t4a_qgrid_disc {
    pub(crate) _private: *const c_void,
}

impl t4a_qgrid_disc {
    /// Create a new t4a_qgrid_disc from an InternalDiscretizedGrid
    pub(crate) fn new(grid: InternalDiscretizedGrid) -> Self {
        Self {
            _private: Box::into_raw(Box::new(grid)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalDiscretizedGrid
    pub(crate) fn inner(&self) -> &InternalDiscretizedGrid {
        unsafe { &*(self._private as *const InternalDiscretizedGrid) }
    }
}

impl Clone for t4a_qgrid_disc {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_qgrid_disc {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalDiscretizedGrid);
            }
        }
    }
}

// Safety: t4a_qgrid_disc is Send + Sync because InternalDiscretizedGrid is Send + Sync
unsafe impl Send for t4a_qgrid_disc {}
unsafe impl Sync for t4a_qgrid_disc {}

/// The internal InherentDiscreteGrid type we're wrapping.
pub(crate) type InternalInherentDiscreteGrid = quanticsgrids::InherentDiscreteGrid;

/// Opaque inherent discrete grid type for C API
///
/// Wraps `quanticsgrids::InherentDiscreteGrid` which provides coordinate conversions
/// between integer coordinates, grid indices, and quantics indices.
#[repr(C)]
pub struct t4a_qgrid_int {
    pub(crate) _private: *const c_void,
}

impl t4a_qgrid_int {
    /// Create a new t4a_qgrid_int from an InternalInherentDiscreteGrid
    pub(crate) fn new(grid: InternalInherentDiscreteGrid) -> Self {
        Self {
            _private: Box::into_raw(Box::new(grid)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalInherentDiscreteGrid
    pub(crate) fn inner(&self) -> &InternalInherentDiscreteGrid {
        unsafe { &*(self._private as *const InternalInherentDiscreteGrid) }
    }
}

impl Clone for t4a_qgrid_int {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_qgrid_int {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalInherentDiscreteGrid);
            }
        }
    }
}

// Safety: t4a_qgrid_int is Send + Sync because InternalInherentDiscreteGrid is Send + Sync
unsafe impl Send for t4a_qgrid_int {}
unsafe impl Sync for t4a_qgrid_int {}

/// Unfolding scheme enum for C API
///
/// Represents the tensor train layout for quantics grids.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_unfolding_scheme {
    /// Fused scheme: indices at same bit level grouped together
    Fused = 0,
    /// Interleaved scheme: indices alternate between dimensions
    Interleaved = 1,
}

impl From<t4a_unfolding_scheme> for quanticsgrids::UnfoldingScheme {
    fn from(scheme: t4a_unfolding_scheme) -> Self {
        match scheme {
            t4a_unfolding_scheme::Fused => Self::Fused,
            t4a_unfolding_scheme::Interleaved => Self::Interleaved,
        }
    }
}

impl From<quanticsgrids::UnfoldingScheme> for t4a_unfolding_scheme {
    fn from(scheme: quanticsgrids::UnfoldingScheme) -> Self {
        match scheme {
            quanticsgrids::UnfoldingScheme::Fused => Self::Fused,
            quanticsgrids::UnfoldingScheme::Interleaved => Self::Interleaved,
        }
    }
}

// ============================================================================
// QuanticsTCI type
// ============================================================================

/// The internal QuanticsTensorCI2 type we're wrapping.
pub(crate) type InternalQuanticsTCI = QuanticsTensorCI2<f64>;

/// Opaque quantics TCI type for C API
///
/// Wraps `QuanticsTensorCI2<f64>` which combines TCI with quantics grid information
/// for seamless interpolation of functions on quantics grids.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_qtci_f64 {
    pub(crate) _private: *const c_void,
}

impl t4a_qtci_f64 {
    /// Create a new t4a_qtci_f64 from an InternalQuanticsTCI
    pub(crate) fn new(qtci: InternalQuanticsTCI) -> Self {
        Self {
            _private: Box::into_raw(Box::new(qtci)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalQuanticsTCI
    pub(crate) fn inner(&self) -> &InternalQuanticsTCI {
        unsafe { &*(self._private as *const InternalQuanticsTCI) }
    }

    /// Get a mutable reference to the inner InternalQuanticsTCI
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut InternalQuanticsTCI {
        unsafe { &mut *(self._private as *mut InternalQuanticsTCI) }
    }
}

impl Drop for t4a_qtci_f64 {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalQuanticsTCI);
            }
        }
    }
}

// Safety: t4a_qtci_f64 is Send + Sync because InternalQuanticsTCI is Send + Sync
unsafe impl Send for t4a_qtci_f64 {}
unsafe impl Sync for t4a_qtci_f64 {}

// ============================================================================
// LinearOperator (QuanticsTransform) type
// ============================================================================

/// The internal LinearOperator type we're wrapping.
/// This is the same as `QuanticsOperator` in tensor4all-quanticstransform.
pub(crate) type InternalLinearOperator = LinearOperator<TensorDynLen, usize>;

/// Opaque linear operator type for C API
///
/// Wraps `LinearOperator<TensorDynLen, usize>` (= `QuanticsOperator`) which represents
/// a quantics transformation operator (shift, flip, Fourier, etc.).
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_linop {
    pub(crate) _private: *const c_void,
}

impl t4a_linop {
    /// Create a new t4a_linop from an InternalLinearOperator
    pub(crate) fn new(op: InternalLinearOperator) -> Self {
        Self {
            _private: Box::into_raw(Box::new(op)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalLinearOperator
    pub(crate) fn inner(&self) -> &InternalLinearOperator {
        unsafe { &*(self._private as *const InternalLinearOperator) }
    }

    /// Get a mutable reference to the inner InternalLinearOperator
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut InternalLinearOperator {
        unsafe { &mut *(self._private as *mut InternalLinearOperator) }
    }
}

impl Clone for t4a_linop {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_linop {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalLinearOperator);
            }
        }
    }
}

// Safety: t4a_linop is Send + Sync because InternalLinearOperator is Send + Sync
unsafe impl Send for t4a_linop {}
unsafe impl Sync for t4a_linop {}

/// Boundary condition enum for C API
///
/// Used for quantics transformation operators (shift, flip).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_boundary_condition {
    /// Periodic boundary condition
    Periodic = 0,
    /// Open boundary condition
    Open = 1,
}

impl From<t4a_boundary_condition> for tensor4all_quanticstransform::BoundaryCondition {
    fn from(bc: t4a_boundary_condition) -> Self {
        match bc {
            t4a_boundary_condition::Periodic => {
                tensor4all_quanticstransform::BoundaryCondition::Periodic
            }
            t4a_boundary_condition::Open => tensor4all_quanticstransform::BoundaryCondition::Open,
        }
    }
}

impl From<tensor4all_quanticstransform::BoundaryCondition> for t4a_boundary_condition {
    fn from(bc: tensor4all_quanticstransform::BoundaryCondition) -> Self {
        match bc {
            tensor4all_quanticstransform::BoundaryCondition::Periodic => Self::Periodic,
            tensor4all_quanticstransform::BoundaryCondition::Open => Self::Open,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tensor4all_core::storage::{DiagStorageC64, DiagStorageF64};

    #[test]
    fn test_storage_kind_from_storage_diag() {
        // Test DiagF64
        let diag_f64 = Storage::DiagF64(DiagStorageF64::from_vec(vec![1.0, 2.0]));
        assert_eq!(
            t4a_storage_kind::from_storage(&diag_f64),
            t4a_storage_kind::DiagF64
        );

        // Test DiagC64
        let diag_c64 = Storage::DiagC64(DiagStorageC64::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ]));
        assert_eq!(
            t4a_storage_kind::from_storage(&diag_c64),
            t4a_storage_kind::DiagC64
        );
    }
}
