//! Opaque types and enums for the reduced C API surface.

use std::ffi::c_void;

use tensor4all_core::{DynIndex, FactorizeAlg, TensorDynLen};
use tensor4all_quanticstransform::BoundaryCondition as QuanticsBoundaryCondition;
use tensor4all_treetn::treetn::contraction::ContractionMethod;
use tensor4all_treetn::{CanonicalForm as TreeCanonicalForm, DefaultTreeTN};

/// Internal dynamic index type wrapped by `t4a_index`.
pub(crate) type InternalIndex = DynIndex;

/// Internal tensor type wrapped by `t4a_tensor`.
pub(crate) type InternalTensor = TensorDynLen;

/// Internal tree tensor network type wrapped by `t4a_treetn`.
pub(crate) type InternalTreeTN = DefaultTreeTN<usize>;

/// Opaque index type for the C API.
#[repr(C)]
pub struct t4a_index {
    pub(crate) _private: *const c_void,
}

impl t4a_index {
    /// Create a wrapper from an internal index value.
    pub(crate) fn new(index: InternalIndex) -> Self {
        Self {
            _private: Box::into_raw(Box::new(index)) as *const c_void,
        }
    }

    /// Borrow the wrapped index.
    pub(crate) fn inner(&self) -> &InternalIndex {
        unsafe { &*(self._private as *const InternalIndex) }
    }
}

impl Clone for t4a_index {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_index {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalIndex);
            }
        }
    }
}

unsafe impl Send for t4a_index {}
unsafe impl Sync for t4a_index {}

/// Scalar kind used by the reduced tensor surface.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_scalar_kind {
    /// Real-valued `f64` data.
    F64 = 0,
    /// Complex-valued `Complex64` data represented as interleaved doubles.
    C64 = 1,
}

impl t4a_scalar_kind {
    /// Classify a tensor by the scalar family exposed through the C API.
    pub(crate) fn from_tensor(tensor: &TensorDynLen) -> Self {
        if tensor.is_complex() {
            Self::C64
        } else {
            Self::F64
        }
    }
}

/// Opaque tensor type for the C API.
#[repr(C)]
pub struct t4a_tensor {
    pub(crate) _private: *const c_void,
}

impl t4a_tensor {
    /// Create a wrapper from an internal tensor value.
    pub(crate) fn new(tensor: InternalTensor) -> Self {
        Self {
            _private: Box::into_raw(Box::new(tensor)) as *const c_void,
        }
    }

    /// Borrow the wrapped tensor.
    pub(crate) fn inner(&self) -> &InternalTensor {
        unsafe { &*(self._private as *const InternalTensor) }
    }
}

impl Clone for t4a_tensor {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_tensor {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalTensor);
            }
        }
    }
}

unsafe impl Send for t4a_tensor {}
unsafe impl Sync for t4a_tensor {}

/// Opaque tree tensor network type for the C API.
#[repr(C)]
pub struct t4a_treetn {
    pub(crate) _private: *const c_void,
}

impl t4a_treetn {
    /// Create a wrapper from an internal TreeTN value.
    pub(crate) fn new(treetn: InternalTreeTN) -> Self {
        Self {
            _private: Box::into_raw(Box::new(treetn)) as *const c_void,
        }
    }

    /// Borrow the wrapped TreeTN.
    pub(crate) fn inner(&self) -> &InternalTreeTN {
        unsafe { &*(self._private as *const InternalTreeTN) }
    }

    /// Mutably borrow the wrapped TreeTN.
    pub(crate) fn inner_mut(&mut self) -> &mut InternalTreeTN {
        unsafe { &mut *(self._private as *mut InternalTreeTN) }
    }
}

impl Clone for t4a_treetn {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_treetn {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalTreeTN);
            }
        }
    }
}

unsafe impl Send for t4a_treetn {}
unsafe impl Sync for t4a_treetn {}

/// Canonical form used for orthogonalization/truncation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_canonical_form {
    /// Unitary / QR-based canonicalization.
    Unitary = 0,
    /// LU-based canonicalization.
    LU = 1,
    /// Cross-interpolation-based canonicalization.
    CI = 2,
}

impl From<TreeCanonicalForm> for t4a_canonical_form {
    fn from(form: TreeCanonicalForm) -> Self {
        match form {
            TreeCanonicalForm::Unitary => Self::Unitary,
            TreeCanonicalForm::LU => Self::LU,
            TreeCanonicalForm::CI => Self::CI,
        }
    }
}

impl From<t4a_canonical_form> for TreeCanonicalForm {
    fn from(form: t4a_canonical_form) -> Self {
        match form {
            t4a_canonical_form::Unitary => Self::Unitary,
            t4a_canonical_form::LU => Self::LU,
            t4a_canonical_form::CI => Self::CI,
        }
    }
}

/// Factorization algorithms exposed through the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_factorize_alg {
    /// Singular value decomposition.
    #[default]
    SVD = 0,
    /// QR decomposition.
    QR = 1,
    /// Rank-revealing LU decomposition.
    LU = 2,
    /// Cross interpolation.
    CI = 3,
}

impl From<FactorizeAlg> for t4a_factorize_alg {
    fn from(alg: FactorizeAlg) -> Self {
        match alg {
            FactorizeAlg::SVD => Self::SVD,
            FactorizeAlg::QR => Self::QR,
            FactorizeAlg::LU => Self::LU,
            FactorizeAlg::CI => Self::CI,
        }
    }
}

impl From<t4a_factorize_alg> for FactorizeAlg {
    fn from(alg: t4a_factorize_alg) -> Self {
        match alg {
            t4a_factorize_alg::SVD => Self::SVD,
            t4a_factorize_alg::QR => Self::QR,
            t4a_factorize_alg::LU => Self::LU,
            t4a_factorize_alg::CI => Self::CI,
        }
    }
}

/// TreeTN contraction method exposed through the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum t4a_contract_method {
    /// Zip-up contraction with on-the-fly truncation.
    #[default]
    Zipup = 0,
    /// Variational fit-based contraction.
    Fit = 1,
    /// Naive dense contraction.
    Naive = 2,
}

impl From<ContractionMethod> for t4a_contract_method {
    fn from(method: ContractionMethod) -> Self {
        match method {
            ContractionMethod::Zipup => Self::Zipup,
            ContractionMethod::Fit => Self::Fit,
            ContractionMethod::Naive => Self::Naive,
        }
    }
}

impl From<t4a_contract_method> for ContractionMethod {
    fn from(method: t4a_contract_method) -> Self {
        match method {
            t4a_contract_method::Zipup => Self::Zipup,
            t4a_contract_method::Fit => Self::Fit,
            t4a_contract_method::Naive => Self::Naive,
        }
    }
}

/// Boundary-condition choices for quantics transform materialization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_boundary_condition {
    /// Periodic boundary conditions.
    Periodic = 0,
    /// Open boundary conditions.
    Open = 1,
}

impl From<t4a_boundary_condition> for QuanticsBoundaryCondition {
    fn from(value: t4a_boundary_condition) -> Self {
        match value {
            t4a_boundary_condition::Periodic => Self::Periodic,
            t4a_boundary_condition::Open => Self::Open,
        }
    }
}

impl From<QuanticsBoundaryCondition> for t4a_boundary_condition {
    fn from(value: QuanticsBoundaryCondition) -> Self {
        match value {
            QuanticsBoundaryCondition::Periodic => Self::Periodic,
            QuanticsBoundaryCondition::Open => Self::Open,
        }
    }
}

/// Canonical binary QTT layout kinds supported by the reduced C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_qtt_layout_kind {
    /// Variable-major grouped layout.
    Grouped = 0,
    /// Level-major interleaved layout.
    Interleaved = 1,
    /// Fused-per-level layout.
    Fused = 2,
}

/// Internal immutable descriptor for canonical binary QTT layouts.
#[derive(Debug, Clone)]
pub(crate) struct InternalQttLayout {
    kind: t4a_qtt_layout_kind,
    variable_resolutions: Vec<usize>,
    nsites: usize,
}

impl InternalQttLayout {
    /// Validate and build a canonical layout descriptor.
    pub(crate) fn new(
        kind: t4a_qtt_layout_kind,
        variable_resolutions: Vec<usize>,
    ) -> Result<Self, String> {
        if variable_resolutions.is_empty() {
            return Err("nvariables must be greater than zero".to_string());
        }
        if variable_resolutions.contains(&0) {
            return Err("variable_resolutions must all be greater than zero".to_string());
        }

        let first = variable_resolutions[0];
        let all_equal = variable_resolutions.iter().all(|&r| r == first);
        let nsites = match kind {
            t4a_qtt_layout_kind::Grouped => variable_resolutions.iter().sum(),
            t4a_qtt_layout_kind::Interleaved => {
                if !all_equal {
                    return Err(
                        "interleaved layouts require all variable_resolutions to match".to_string(),
                    );
                }
                first * variable_resolutions.len()
            }
            t4a_qtt_layout_kind::Fused => {
                if !all_equal {
                    return Err(
                        "fused layouts require all variable_resolutions to match".to_string()
                    );
                }
                first
            }
        };

        Ok(Self {
            kind,
            variable_resolutions,
            nsites,
        })
    }

    /// Number of logical variables.
    pub(crate) fn nvariables(&self) -> usize {
        self.variable_resolutions.len()
    }

    /// Number of physical sites in the canonical layout.
    pub(crate) fn nsites(&self) -> usize {
        self.nsites
    }

    /// Per-variable bit resolutions in canonical variable order.
    pub(crate) fn variable_resolutions(&self) -> &[usize] {
        &self.variable_resolutions
    }

    /// Bit resolution of a single logical variable.
    pub(crate) fn resolution(&self, variable: usize) -> usize {
        self.variable_resolutions[variable]
    }

    /// Layout kind.
    pub(crate) fn kind(&self) -> t4a_qtt_layout_kind {
        self.kind
    }
}

/// Opaque canonical QTT layout descriptor.
#[repr(C)]
pub struct t4a_qtt_layout {
    pub(crate) _private: *const c_void,
}

impl t4a_qtt_layout {
    /// Wrap an internal layout descriptor.
    pub(crate) fn new(layout: InternalQttLayout) -> Self {
        Self {
            _private: Box::into_raw(Box::new(layout)) as *const c_void,
        }
    }

    /// Borrow the wrapped layout descriptor.
    pub(crate) fn inner(&self) -> &InternalQttLayout {
        unsafe { &*(self._private as *const InternalQttLayout) }
    }
}

impl Clone for t4a_qtt_layout {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_qtt_layout {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut InternalQttLayout);
            }
        }
    }
}

unsafe impl Send for t4a_qtt_layout {}
unsafe impl Sync for t4a_qtt_layout {}

#[cfg(test)]
mod tests;
