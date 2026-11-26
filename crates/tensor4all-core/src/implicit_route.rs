//! The implicit (context-free) route's only access to process-global execution state.
//!
//! A context-free entry point has no caller-supplied execution context. When it
//! needs an engine or an eager runtime that its operands do not already carry, it
//! takes it from here, and nowhere else in the workspace:
//!
//! - [`eager_runtime`] adopts an owner-less host value into an eager runtime
//!   (design doc family F5);
//! - [`default_cpu_execution_context`] hands the global context to code that
//!   builds an `ExecutionContext` itself (family F3);
//! - the numerical helpers ([`mat_mul`], [`solve_matrix`], [`svd_backend`], the
//!   native einsum bridge, ...) run on an engine nobody named.
//!
//! With the `global-defaults` feature these are the process-global
//! `tensor4all-tensorbackend` items themselves, re-exported, so the global build
//! runs exactly the code it ran before. Without it there is no process-global
//! context, and each item refuses with
//! [`ExecutionContextEntryError::NoContext`](tensor4all_tensorbackend::ExecutionContextEntryError::NoContext)
//! instead of choosing an engine. Context-taking (`*_in`) entry points never
//! reach this module.
//!
//! Downstream crates reach the same items as `tensor4all_core::legacy`, a
//! `#[doc(hidden)]` migration bridge (tensor4all-rs#1376 §7.0). Every use is a
//! call site still on the global path:
//!
//! ```text
//! grep -rn "implicit_route::" crates/tensor4all-core/src
//! grep -rn "tensor4all_core::legacy" crates --include=*.rs
//! ```

use std::sync::Arc;

use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContextEntryError};

/// The process-global CPU context, for context-free library code that has to
/// hand a context on (design doc family F3).
///
/// With `global-defaults` this is the process-global context; without it the
/// call refuses. It follows `tensor4all-core`'s feature, so a downstream crate
/// and core never disagree about whether the global route exists.
///
/// # Errors
///
/// Returns [`ExecutionContextEntryError::NoContext`] in a build without
/// `global-defaults`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::legacy::default_cpu_execution_context;
///
/// match default_cpu_execution_context() {
///     // `global-defaults` build: the same context on every call.
///     Ok(first) => {
///         let second = default_cpu_execution_context().unwrap();
///         assert!(std::sync::Arc::ptr_eq(&first, &second));
///     }
///     // Explicit-only build: there is no process-global context.
///     Err(error) => assert!(error.to_string().contains("no execution context")),
/// }
/// ```
pub fn default_cpu_execution_context(
) -> Result<Arc<CpuExecutionContext>, ExecutionContextEntryError> {
    #[cfg(feature = "global-defaults")]
    {
        Ok(tensor4all_tensorbackend::default_cpu_execution_context())
    }
    #[cfg(not(feature = "global-defaults"))]
    {
        Err(ExecutionContextEntryError::NoContext)
    }
}

/// Run `f` on the engine of the context owning `runtime`, or, with
/// `global-defaults`, on the process-global CPU context when no live context
/// owns it (tensor4all-rs#1378).
///
/// The strict [`tensor4all_tensorbackend::with_runtime_engine`] refuses an
/// owner-less runtime; this bridge is the one explicit place where an implicit
/// entry still accepts the process-global engine. Without `global-defaults` it
/// refuses with `ExecutionContextEntryError::NoContext`.
pub use tensor4all_tensorbackend::with_runtime_engine_or_global;

/// Whether this build of `tensor4all-core` has the process-global route.
///
/// A crate cannot read another crate's Cargo features with `cfg`, so a test or
/// a downstream configuration check that must know which build of core it links
/// reads this constant. Library code should not switch implementations on it:
/// run on the engine that owns the operands (for example
/// [`IdxTensor::with_owning_cpu_engine`](crate::IdxTensor::with_owning_cpu_engine))
/// and take the implicit route only when no operand has one.
///
/// # Examples
///
/// ```
/// use tensor4all_core::legacy::{default_cpu_execution_context, GLOBAL_DEFAULTS};
///
/// assert_eq!(GLOBAL_DEFAULTS, default_cpu_execution_context().is_ok());
/// ```
pub const GLOBAL_DEFAULTS: bool = cfg!(feature = "global-defaults");

/// The refusal an implicit entry returns in a build without `global-defaults`.
///
/// Only the explicit-only build constructs it outside tests; the
/// `global-defaults` build never refuses.
#[cfg_attr(feature = "global-defaults", allow(dead_code))]
pub(crate) fn refusal() -> anyhow::Error {
    anyhow::Error::new(ExecutionContextEntryError::NoContext)
}

#[cfg(feature = "global-defaults")]
mod global {
    use std::sync::Arc;

    use tenferro::{Tensor as NativeTensor, TensorRead};
    use tenferro_ad::EagerRuntime;
    use tensor4all_tensorbackend::{BackendLinalgError, FullPivLuScalar, NativeEinsumPlan};

    pub use tensor4all_tensorbackend::{
        batched_mat_mul_same_shape_owned, einsum_native_subscripts, einsum_native_tensor_reads,
        einsum_native_tensors, einsum_native_tensors_owned, hermitian_eigendecomposition,
        hermitian_exponential_first_column, lowest_hermitian_eigenpair, mat_mul, mat_mul_owned,
        native_tensor_primal_to_diag, qr_backend, solve_matrix, solve_matrix_owned,
        src_error_estimate, src_error_estimate_general, svd_backend, triangular_solve_matrix_owned,
    };

    /// Error returned by [`eager_runtime`].
    pub type EagerRuntimeError = tensor4all_tensorbackend::EagerContextError;

    /// The process-global eager runtime that owner-less values join.
    ///
    /// # Errors
    ///
    /// Returns the eager-runtime initialisation failure.
    pub fn eager_runtime() -> Result<Arc<EagerRuntime>, EagerRuntimeError> {
        tensor4all_tensorbackend::default_eager_ctx()
    }

    /// [`FullPivLuScalar::solve_right_full_piv_lu`] on the process-global context.
    ///
    /// # Errors
    ///
    /// Returns the solve's shape or backend failure.
    pub fn solve_right_full_piv_lu<T: FullPivLuScalar>(
        lhs_values: &[T],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[T],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> Result<Vec<T>, BackendLinalgError> {
        T::solve_right_full_piv_lu(
            lhs_values,
            lhs_rows,
            lhs_cols,
            pivot_values,
            pivot_rows,
            pivot_cols,
        )
    }

    /// [`NativeEinsumPlan::execute_reads`] on the process-global context.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`NativeEinsumPlan::execute_reads`].
    pub fn native_einsum_plan_execute_reads(
        plan: &NativeEinsumPlan,
        inputs: &[TensorRead<'_>],
    ) -> anyhow::Result<NativeTensor> {
        plan.execute_reads(inputs)
    }
}

#[cfg(not(feature = "global-defaults"))]
mod refuse {
    use std::sync::Arc;

    use num_complex::{Complex64, ComplexFloat};
    use tenferro::{Tensor as NativeTensor, TensorRead, TensorScalar, TypedTensor};
    use tenferro_ad::EagerRuntime;
    use tensor4all_tensorbackend::{
        BackendLinalgError, BackendLinalgScalar, BlasMul, BridgeError, ExecutionContextEntryError,
        FullPivLuScalar, HermitianEigenError, HermitianEigenScalar, HermitianEigendecomposition,
        HermitianEigenpair, Matrix, MatrixMulError, MatrixSolveScalar, MatrixTriangularSolveScalar,
        NativeEinsumPlan, NativeTensorReadInput, SrcErrorEstimate, SvdResult, TensorElement,
    };

    use super::refusal;

    fn eigen_refusal() -> HermitianEigenError {
        HermitianEigenError::Backend {
            source: Box::new(ExecutionContextEntryError::NoContext),
        }
    }

    /// Error returned by [`eager_runtime`].
    pub type EagerRuntimeError = ExecutionContextEntryError;

    /// Refuse: there is no process-global eager runtime to adopt into.
    ///
    /// # Errors
    ///
    /// Always [`ExecutionContextEntryError::NoContext`].
    pub fn eager_runtime() -> Result<Arc<EagerRuntime>, EagerRuntimeError> {
        Err(ExecutionContextEntryError::NoContext)
    }

    /// Refusing stand-in for the global `mat_mul`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn mat_mul<T: BlasMul>(
        _a: &Matrix<T>,
        _b: &Matrix<T>,
    ) -> Result<Matrix<T>, MatrixMulError> {
        Err(MatrixMulError::from(refusal()))
    }

    /// Refusing stand-in for the global `mat_mul_owned`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn mat_mul_owned<T: BlasMul>(
        _a: Matrix<T>,
        _b: Matrix<T>,
    ) -> Result<Matrix<T>, MatrixMulError> {
        Err(MatrixMulError::from(refusal()))
    }

    /// Refusing stand-in for the global `batched_mat_mul_same_shape_owned`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn batched_mat_mul_same_shape_owned<T: TensorScalar + Copy>(
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        _a: Vec<T>,
        _b: Vec<T>,
    ) -> Result<Vec<T>, MatrixMulError> {
        Err(MatrixMulError::from(refusal()))
    }

    /// Refusing stand-in for the global `solve_matrix`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn solve_matrix<T: MatrixSolveScalar>(
        _a: &Matrix<T>,
        _b: &Matrix<T>,
    ) -> Result<Matrix<T>, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `solve_matrix_owned`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn solve_matrix_owned<T: MatrixSolveScalar>(
        _a: Matrix<T>,
        _b: Matrix<T>,
    ) -> Result<Matrix<T>, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `triangular_solve_matrix_owned`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn triangular_solve_matrix_owned<T: MatrixTriangularSolveScalar>(
        _a: Matrix<T>,
        _b: Matrix<T>,
        _left_side: bool,
        _lower: bool,
        _transpose_a: bool,
        _unit_diagonal: bool,
    ) -> Result<Matrix<T>, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `svd_backend`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn svd_backend<T: BackendLinalgScalar>(
        _a: &TypedTensor<T>,
    ) -> Result<SvdResult<T>, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `qr_backend`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn qr_backend<T: BackendLinalgScalar>(
        _a: TypedTensor<T>,
    ) -> Result<(TypedTensor<T>, TypedTensor<T>), BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `src_error_estimate`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn src_error_estimate<T: MatrixTriangularSolveScalar + ComplexFloat>(
        _r: &Matrix<T>,
    ) -> Result<SrcErrorEstimate, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `src_error_estimate_general`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn src_error_estimate_general<T: MatrixSolveScalar + ComplexFloat>(
        _factor: &Matrix<T>,
    ) -> Result<SrcErrorEstimate, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `einsum_native_tensors`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn einsum_native_tensors(
        _operands: &[(&NativeTensor, &[usize])],
        _output_ids: &[usize],
    ) -> anyhow::Result<NativeTensor> {
        Err(refusal())
    }

    /// Refusing stand-in for the global `einsum_native_tensors_owned`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn einsum_native_tensors_owned(
        _operands: Vec<(NativeTensor, Vec<usize>)>,
        _output_ids: &[usize],
    ) -> anyhow::Result<NativeTensor> {
        Err(refusal())
    }

    /// Refusing stand-in for the global `einsum_native_tensor_reads`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn einsum_native_tensor_reads(
        _operands: &[(&NativeTensorReadInput<'_>, &[usize])],
        _output_ids: &[usize],
    ) -> anyhow::Result<NativeTensor> {
        Err(refusal())
    }

    /// Refusing stand-in for the global `einsum_native_subscripts`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn einsum_native_subscripts(
        _subscripts: &str,
        _operands: &[&NativeTensor],
    ) -> anyhow::Result<NativeTensor> {
        Err(refusal())
    }

    /// Refusing stand-in for the global `FullPivLuScalar::solve_right_full_piv_lu`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn solve_right_full_piv_lu<T: FullPivLuScalar>(
        _lhs_values: &[T],
        _lhs_rows: usize,
        _lhs_cols: usize,
        _pivot_values: &[T],
        _pivot_rows: usize,
        _pivot_cols: usize,
    ) -> Result<Vec<T>, BackendLinalgError> {
        Err(BackendLinalgError::from(refusal()))
    }

    /// Refusing stand-in for the global `NativeEinsumPlan::execute_reads`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn native_einsum_plan_execute_reads(
        _plan: &NativeEinsumPlan,
        _inputs: &[TensorRead<'_>],
    ) -> anyhow::Result<NativeTensor> {
        Err(refusal())
    }

    /// Refusing stand-in for the global `native_tensor_primal_to_diag`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn native_tensor_primal_to_diag<T: TensorElement>(
        _tensor: &NativeTensor,
    ) -> Result<Vec<T>, BridgeError> {
        Err(BridgeError::from(refusal()))
    }

    /// Refusing stand-in for the global `hermitian_eigendecomposition`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn hermitian_eigendecomposition<T: HermitianEigenScalar>(
        _matrix: &Matrix<T>,
        _hermitian_tol: f64,
    ) -> Result<HermitianEigendecomposition<T>, HermitianEigenError> {
        Err(eigen_refusal())
    }

    /// Refusing stand-in for the global `lowest_hermitian_eigenpair`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn lowest_hermitian_eigenpair<T: HermitianEigenScalar>(
        _matrix: &Matrix<T>,
        _hermitian_tol: f64,
    ) -> Result<HermitianEigenpair<T>, HermitianEigenError> {
        Err(eigen_refusal())
    }

    /// Refusing stand-in for the global `hermitian_exponential_first_column`.
    ///
    /// # Errors
    ///
    /// Always refuses with `ExecutionContextEntryError::NoContext`, wrapped in the
    /// returned error type: there is no process-global context.
    pub fn hermitian_exponential_first_column<T: HermitianEigenScalar>(
        _matrix: &Matrix<T>,
        _exponent: Complex64,
        _hermitian_tol: f64,
    ) -> Result<Vec<Complex64>, HermitianEigenError> {
        Err(eigen_refusal())
    }
}

#[cfg(feature = "global-defaults")]
pub use global::*;
#[cfg(not(feature = "global-defaults"))]
pub use refuse::*;

#[cfg(test)]
mod tests;
