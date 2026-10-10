//! Opt-in explicit/concrete execution frontend.
//!
//! This is the second, opt-in frontend of
//! [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859): a
//! caller-supplied [`CpuExecutionContext`] plus one concrete session that a whole
//! stage or batch reuses, instead of the process-global convenience surface in
//! the compatibility entry point. The compatibility frontend keeps its own signatures,
//! defaults and behaviour; nothing here reroutes it.
//!
//! Properties of this frontend, from the coexistence record
//! (`docs/design/859-dual-frontend-coexistence.md`):
//!
//! - **One explicit entry.** [`CpuExecutionContext::with_concrete_session`] opens
//!   the session for the whole closure, so a stage pays one entry rather than one
//!   per operation, and the engine's prepared plans and buffer pool stay warm.
//! - **No process-global selector.** No route here selects an execution context or
//!   backend from a default context, a thread-local override or an environment
//!   variable. (Upstream profiling hooks may still read their own environment flags;
//!   those do not choose where the work runs.)
//! - **No eager, trace or AD records.** These routes reach only concrete
//!   `Tensor`/`BackendSession` operations; they never construct `EagerTensor`,
//!   semantic nodes or gradient slots, and they take no eager owner lock. Tracked
//!   values are not accepted at all, so no detach can happen implicitly.
//! - **No implicit fallback.** An unsupported dtype or shape returns the
//!   backend's typed error. Nothing here retries on the global or CPU-eager route.
//! - **Explicit conversion at the boundary.** Crossing between this frontend and
//!   the compatibility frontend is [`LogicalTensor`](crate::LogicalTensor), which
//!   carries dtype, shape and column-major data and no backend identity.
//!
//! The session is a [`tenferro_tensor::BackendSession`] scope, not a value
//! representation: the concrete values stay the ordinary tenferro tensors.
//!
//! # Examples
//!
//! ```
//! use tensor4all_tensorbackend::{CpuExecutionContext, explicit::Session};
//! use tenferro::Tensor;
//! use tenferro_cpu::CpuBackend;
//!
//! let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
//! let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
//!
//! let product = context.with_concrete_session(|session: &mut Session<'_>| {
//!     session.qr(&x).map(|(q, _r)| q)
//! })?;
//! assert_eq!(product.expect("QR succeeds").shape(), &[2, 2]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use tenferro::Tensor as NativeTensor;
use tenferro::TensorSessionOpsExt;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};
use tenferro_linalg::TensorLinalgExt;

use tenferro_tensor::BackendSession;

use crate::context::{CpuExecutionContext, CpuExecutionContextError};
use crate::einsum_ids::{
    build_binary_einsum_ids, checked_native_einsum_labels, common_dtype, convert_native_tensor_in,
};

/// One explicit concrete session over a [`CpuExecutionContext`].
///
/// The session borrows the context's backend for the whole
/// [`CpuExecutionContext::with_concrete_session`] closure. Every route on it runs
/// through that one session, so a stage does not pay a session entry per
/// operation.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro::Tensor;
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let b = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0])?;
/// let (ab, a_again) = context.with_concrete_session(
///     |session| -> Result<_, Box<dyn std::error::Error + Send + Sync>> {
///         let ab = session.contraction(&a, &[1], &b, &[0])?;
///         // One session, two operations.
///         let a_again = session.permute(&a, &[1, 0])?;
///         Ok((ab, a_again))
///     },
/// )??;
/// // `a` is `[[1, 3], [2, 4]]` column-major and `b` swaps the axes, so `a * b`
/// // is `[[3, 1], [4, 2]]`.
/// assert_eq!(ab.as_slice::<f64>()?, &[3.0, 4.0, 1.0, 2.0]);
/// assert_eq!(a_again.as_slice::<f64>()?, &[1.0, 3.0, 2.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub struct Session<'session> {
    session: &'session mut dyn BackendSession,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("explicit::Session")
            .finish_non_exhaustive()
    }
}

/// One prepared N-ary einsum, reusable across operations of a session.
///
/// Preparing validates the labels and plans the contraction once; every
/// [`PreparedEinsum::execute`] and [`PreparedEinsum::execute_into`] reuses that plan, so
/// a stage keeps one plan across as many evaluations as it needs instead of preparing
/// per call. The plan borrows its operands, so it cannot outlive them, and it holds no
/// execution state: the session stays the caller's.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{explicit::PreparedEinsum, CpuExecutionContext};
/// use tenferro::Tensor;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{TensorRead, TensorWrite};
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0])?;
/// let reads = [TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)];
/// let plan = PreparedEinsum::prepare(&reads, &[&[0, 1], &[1, 2]], &[0, 2])?;
///
/// // Reused across a stage, including into a caller-provided destination.
/// let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 0.0])?;
/// context.with_concrete_session(|session| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     let product = plan.execute(session)?;
///     assert_eq!(product.as_slice::<f64>()?, &[23.0, 34.0]);
///     plan.execute_into(session, TensorWrite::Tensor(&mut out))?;
///     Ok(())
/// })??;
/// assert_eq!(out.as_slice::<f64>()?, &[23.0, 34.0]);
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub struct PreparedEinsum<'operands> {
    plan: ConcreteEinsumPlan,
    operands: Vec<tenferro_tensor::TensorRead<'operands>>,
}

impl std::fmt::Debug for PreparedEinsum<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("explicit::PreparedEinsum")
            .field("operands", &self.operands.len())
            .finish_non_exhaustive()
    }
}

impl<'operands> PreparedEinsum<'operands> {
    /// Plan an N-ary einsum over borrowed operands.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] for malformed labels, a
    /// label list whose length does not match the operand's rank, or a label-list count
    /// that does not match the operand count, and [`tenferro_einsum::Error`] from the
    /// contraction planner when the expression cannot be planned.
    pub fn prepare(
        operands: &[tenferro_tensor::TensorRead<'operands>],
        input_ids: &[&[usize]],
        output_ids: &[usize],
    ) -> tenferro_einsum::Result<Self> {
        if operands.len() != input_ids.len() {
            return Err(tenferro_einsum::Error::InvalidSubscripts {
                message: format!(
                    "einsum needs one label list per operand: {} operands, {} label lists",
                    operands.len(),
                    input_ids.len()
                ),
            });
        }
        let invalid = |error: anyhow::Error| tenferro_einsum::Error::InvalidSubscripts {
            message: format!("{error}"),
        };
        let inputs = input_ids
            .iter()
            .map(|ids| checked_native_einsum_labels(ids))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(invalid)?;
        let output = checked_native_einsum_labels(output_ids).map_err(invalid)?;
        let subscripts = EinsumSubscripts { inputs, output };
        let plan = ConcreteEinsumPlan::prepare_read_subscripts(operands, &subscripts)?;
        Ok(Self {
            plan,
            operands: operands.to_vec(),
        })
    }

    /// Evaluate the prepared plan on `session`, allocating the result.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::Tensor`] when the session rejects the operands
    /// or the contraction itself fails, and
    /// [`tenferro_einsum::Error::Validation`] when the operands no longer match the
    /// planned expression.
    pub fn execute(&self, session: &mut Session<'_>) -> tenferro_einsum::Result<NativeTensor> {
        self.plan
            .execute_read(self.operands.as_slice(), session.backend())
    }

    /// Evaluate the prepared plan into a caller-provided destination.
    ///
    /// The destination is fully written and is never zero-filled first. Operands and
    /// destination must already share one dtype; this route does not convert into a
    /// caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::Tensor`] when the session rejects the operands,
    /// the destination or the contraction, and
    /// [`tenferro_einsum::Error::Validation`] when the operands no longer match the
    /// planned expression.
    pub fn execute_into(
        &self,
        session: &mut Session<'_>,
        out: tenferro_tensor::TensorWrite<'_>,
    ) -> tenferro_einsum::Result<()> {
        self.plan
            .execute_read_into(self.operands.as_slice(), session.backend(), out)
    }
}

/// Promote a heterogeneous operand set to the dtype both frontends contract in.
///
/// Returns an empty vector when every operand already has the common dtype, so the
/// common case allocates nothing.
///
/// # Errors
///
/// Returns [`tenferro_einsum::Error::Tensor`] when an operand cannot be converted.
fn promote_operands(
    session: &mut dyn BackendSession,
    operands: &[&NativeTensor],
) -> tenferro_einsum::Result<Vec<NativeTensor>> {
    let dtypes = operands
        .iter()
        .map(|tensor| tensor.dtype())
        .collect::<Vec<_>>();
    let target = common_dtype(&dtypes);
    if operands.iter().all(|tensor| tensor.dtype() == target) {
        return Ok(Vec::new());
    }
    operands
        .iter()
        .map(|tensor| {
            convert_native_tensor_in(session, tensor, target)
                .map_err(tenferro_einsum::Error::Tensor)
        })
        .collect()
}

impl<'session> Session<'session> {
    pub(crate) fn new(session: &'session mut dyn BackendSession) -> Self {
        Self { session }
    }

    /// The concrete session behind this explicit session.
    ///
    /// Exposed so a prepared plan can evaluate itself on the caller's session; it is
    /// not a second entry point, and it never consults another backend.
    pub fn backend(&mut self) -> &mut dyn BackendSession {
        self.session
    }

    /// Reshape a concrete tensor without changing its column-major linearization.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the element counts
    /// differ or the requested layout is invalid,
    /// [`tenferro_tensor::Error::UnsupportedDType`] for a dtype the backend cannot
    /// reshape, and [`tenferro_tensor::Error::BackendSource`] when the operation
    /// itself fails. It never retries on another route.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let reshaped = context.with_concrete_session(|session| session.reshape(&x, &[2, 2]))??;
    /// assert_eq!(reshaped.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reshape(
        &mut self,
        tensor: &NativeTensor,
        shape: &[usize],
    ) -> tenferro_tensor::Result<NativeTensor> {
        tensor.reshape(shape, self.session)
    }

    /// Permute the axes of a concrete tensor.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when `perm` is not a
    /// permutation of the tensor's axes, and
    /// [`tenferro_tensor::Error::BackendSource`] when the operation itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let transposed = context.with_concrete_session(|session| session.permute(&x, &[1, 0]))??;
    /// assert_eq!(transposed.as_slice::<f64>()?, &[1.0, 3.0, 2.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn permute(
        &mut self,
        tensor: &NativeTensor,
        perm: &[usize],
    ) -> tenferro_tensor::Result<NativeTensor> {
        tensor.transpose(perm, self.session)
    }

    /// Binary contraction along the given axes, preserving the remaining axes in
    /// order.
    ///
    /// The operand identity and axis building is the same code path the
    /// compatibility frontend uses, so both frontends contract identically; only
    /// the session differs.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] when the axis lists are
    /// malformed or of different length, and [`tenferro_einsum::Error::Tensor`] when the
    /// backend rejects the operands, when an operand cannot be promoted, or when the
    /// contraction itself fails.
    ///
    /// Operands are promoted to a common dtype exactly as the compatibility frontend
    /// promotes them: a heterogeneous set contracts in the promoted dtype, and a set
    /// that already shares one dtype is used as is.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0])?;
    /// let product =
    ///     context.with_concrete_session(|session| session.contraction(&a, &[1], &b, &[0]))??;
    /// assert_eq!(product.as_slice::<f64>()?, &[23.0, 34.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contraction(
        &mut self,
        lhs: &NativeTensor,
        lhs_axes: &[usize],
        rhs: &NativeTensor,
        rhs_axes: &[usize],
    ) -> tenferro_einsum::Result<NativeTensor> {
        let (lhs_ids, rhs_ids, output_ids) =
            build_binary_einsum_ids(lhs.shape().len(), lhs_axes, rhs.shape().len(), rhs_axes)
                .map_err(|error| tenferro_einsum::Error::InvalidSubscripts {
                    message: format!("{error}"),
                })?;
        let subscripts = EinsumSubscripts {
            inputs: vec![lhs_ids, rhs_ids],
            output: output_ids,
        };
        let operands = [lhs, rhs];
        let promoted = promote_operands(self.session, &operands)?;
        let operands = if promoted.is_empty() {
            operands.to_vec()
        } else {
            promoted.iter().collect::<Vec<_>>()
        };
        operands.einsum_subscripts(&subscripts, self.session)
    }

    /// N-ary einsum from integer labels, evaluated on this session.
    ///
    /// The labels are validated by the same helper the compatibility frontend
    /// uses; the evaluation is tenferro-einsum's session-direct einsum, so it does
    /// not compile a semantic graph or start a runtime worker.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] for malformed labels, a
    /// label list whose length does not match the operand's rank, or a label-list
    /// count that does not match the operand count, and
    /// [`tenferro_einsum::Error::Tensor`] when the backend rejects the operands, when an
    /// operand cannot be promoted, or when the contraction itself fails.
    ///
    /// Operands are promoted to a common dtype exactly as the compatibility frontend
    /// promotes them: a heterogeneous set contracts in the promoted dtype, and a set
    /// that already shares one dtype is used as is.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
    /// let product = context.with_concrete_session(|session| {
    ///     session.einsum(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2])
    /// })??;
    /// assert_eq!(product.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn einsum(
        &mut self,
        operands: &[&NativeTensor],
        input_ids: &[&[usize]],
        output_ids: &[usize],
    ) -> tenferro_einsum::Result<NativeTensor> {
        if operands.len() != input_ids.len() {
            return Err(tenferro_einsum::Error::InvalidSubscripts {
                message: format!(
                    "einsum needs one label list per operand: {} operands, {} label lists",
                    operands.len(),
                    input_ids.len()
                ),
            });
        }
        let inputs = input_ids
            .iter()
            .map(|ids| checked_native_einsum_labels(ids))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|error| tenferro_einsum::Error::InvalidSubscripts {
                message: format!("{error}"),
            })?;
        let subscripts = EinsumSubscripts {
            inputs,
            output: checked_native_einsum_labels(output_ids).map_err(|error| {
                tenferro_einsum::Error::InvalidSubscripts {
                    message: format!("{error}"),
                }
            })?,
        };
        let promoted = promote_operands(self.session, operands)?;
        let operands: Vec<&NativeTensor> = if promoted.is_empty() {
            operands.to_vec()
        } else {
            promoted.iter().collect()
        };
        operands.einsum_subscripts(&subscripts, self.session)
    }

    /// Sum every element of a concrete tensor, returning the rank-0 result.
    ///
    /// The compatibility frontend wraps the same rank-0 tensor in a
    /// `BackendScalar`; this route stays on the concrete value so it works without
    /// the compatibility feature.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an unsupported reduction
    /// shape and [`tenferro_tensor::Error::BackendSource`] when the reduction itself
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let total = context.with_concrete_session(|session| session.sum(&x))??;
    /// assert_eq!(total.shape(), &[] as &[usize]);
    /// assert_eq!(total.as_slice::<f64>()?, &[10.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn sum(&mut self, tensor: &NativeTensor) -> tenferro_tensor::Result<NativeTensor> {
        if tensor.shape().is_empty() {
            return tensor.duplicate();
        }
        let axes = (0..tensor.shape().len()).collect::<Vec<_>>();
        tensor.reduce_sum(Some(&axes), self.session)
    }

    /// Conjugate a concrete tensor.
    ///
    /// A real tensor is returned unchanged; a complex tensor is conjugated
    /// elementwise.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for a dtype the backend cannot
    /// conjugate and [`tenferro_tensor::Error::BackendSource`] when the operation
    /// itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    /// use num_complex::Complex64;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let x = Tensor::from_vec_col_major(
    ///     vec![2],
    ///     vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    /// )?;
    /// let conjugated = context.with_concrete_session(|session| session.conj(&x))??;
    /// assert_eq!(
    ///     conjugated.as_slice::<Complex64>()?,
    ///     &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn conj(&mut self, tensor: &NativeTensor) -> tenferro_tensor::Result<NativeTensor> {
        tensor.conj(self.session)
    }

    /// N-ary einsum over borrowed read inputs, without materializing them.
    ///
    /// Same contract as [`Session::einsum`], except that the operands are
    /// [`TensorRead`](tenferro_tensor::TensorRead) views: a non-contiguous or lazily
    /// represented operand stays borrowed until the contract evaluates it.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] for malformed labels or a
    /// label list whose length does not match the operand's rank, and
    /// [`tenferro_einsum::Error::Tensor`] when the backend rejects the operands or the
    /// contraction itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::TensorRead;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0])?;
    /// let reads = [TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)];
    /// let product = context
    ///     .with_concrete_session(|session| session.einsum_reads(&reads, &[&[0, 1], &[1, 2]], &[0, 2]))??;
    /// assert_eq!(product.as_slice::<f64>()?, &[23.0, 34.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn einsum_reads(
        &mut self,
        operands: &[tenferro_tensor::TensorRead<'_>],
        input_ids: &[&[usize]],
        output_ids: &[usize],
    ) -> tenferro_einsum::Result<NativeTensor> {
        if operands.len() != input_ids.len() {
            return Err(tenferro_einsum::Error::InvalidSubscripts {
                message: format!(
                    "einsum needs one label list per operand: {} operands, {} label lists",
                    operands.len(),
                    input_ids.len()
                ),
            });
        }
        let invalid = |error: anyhow::Error| tenferro_einsum::Error::InvalidSubscripts {
            message: format!("{error}"),
        };
        let inputs = input_ids
            .iter()
            .map(|ids| checked_native_einsum_labels(ids))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(invalid)?;
        let output = checked_native_einsum_labels(output_ids).map_err(invalid)?;
        let subscripts = EinsumSubscripts { inputs, output };
        let plan = ConcreteEinsumPlan::prepare_read_subscripts(operands, &subscripts)?;
        plan.execute_read(operands, self.session)
    }

    /// Binary contraction into a caller-provided destination.
    ///
    /// The destination is fully written and is never zero-filled first. The operands
    /// and the destination must already share one dtype: this route does not convert
    /// into a caller-owned output, and a mismatch is rejected typed by the backend.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] when the axis lists are
    /// malformed or of different length, and [`tenferro_einsum::Error::Tensor`] when the
    /// backend rejects the operands, the destination or the contraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::TensorWrite;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0])?;
    /// let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 0.0])?;
    /// context.with_concrete_session(|session| {
    ///     session.contraction_into(&a, &[1], &b, &[0], TensorWrite::Tensor(&mut out))
    /// })??;
    /// assert_eq!(out.as_slice::<f64>()?, &[23.0, 34.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contraction_into(
        &mut self,
        lhs: &NativeTensor,
        lhs_axes: &[usize],
        rhs: &NativeTensor,
        rhs_axes: &[usize],
        out: tenferro_tensor::TensorWrite<'_>,
    ) -> tenferro_einsum::Result<()> {
        let (lhs_ids, rhs_ids, output_ids) =
            build_binary_einsum_ids(lhs.shape().len(), lhs_axes, rhs.shape().len(), rhs_axes)
                .map_err(|error| tenferro_einsum::Error::InvalidSubscripts {
                    message: format!("{error}"),
                })?;
        let subscripts = EinsumSubscripts {
            inputs: vec![lhs_ids, rhs_ids],
            output: output_ids,
        };
        let inputs = [
            tenferro_tensor::TensorRead::from_tensor(lhs),
            tenferro_tensor::TensorRead::from_tensor(rhs),
        ];
        let plan = ConcreteEinsumPlan::prepare_read_subscripts(&inputs[..], &subscripts)?;
        plan.execute_read_into(&inputs[..], self.session, out)
    }

    /// Matrix multiplication `A * B` on this session.
    ///
    /// The same kernel and shape validation as
    /// [`mat_mul`](crate::mat_mul), entered through this session instead of the
    /// process-global one, and returning the shared
    /// [`Matrix`](crate::Matrix) container.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixMulError`](crate::MatrixMulError) when the shapes disagree or
    /// the session operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuExecutionContext, Matrix};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);
    /// let b = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);
    /// let c = context.with_concrete_session(|session| session.mat_mul(&a, &b))??;
    /// assert_eq!(c.as_col_major_slice(), &[1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "global-defaults")]
    pub fn mat_mul<T: crate::matrix::BlasMul>(
        &mut self,
        a: &crate::matrix::Matrix<T>,
        b: &crate::matrix::Matrix<T>,
    ) -> Result<crate::matrix::Matrix<T>, crate::matrix::MatrixMulError> {
        crate::mat_mul_in(self.session, a, b)
    }

    /// Execute grouped GEMMs on this session.
    ///
    /// Same validation, job translation and provider rules as
    /// [`grouped_mat_mul_shared`](crate::grouped_mat_mul_shared), entered through this
    /// session instead of the process-global one. A job whose contracted extent is
    /// zero is a no-op segment, exactly as on the compatibility entry.
    ///
    /// # Errors
    ///
    /// Returns [`GroupedGemmError`](crate::GroupedGemmError) when the buffers and jobs
    /// disagree, or when the session rejects the request.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuExecutionContext, GroupedGemmJob, GroupedGemmOptions};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
    /// let mut output = [0.0_f64];
    /// context.with_concrete_session(|session| {
    ///     session.grouped_mat_mul_shared(
    ///         &[3.0],
    ///         &[4.0],
    ///         &mut output,
    ///         &jobs,
    ///         GroupedGemmOptions::default(),
    ///     )
    /// })??;
    /// assert_eq!(output, [12.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "global-defaults")]
    pub fn grouped_mat_mul_shared<T: crate::matrix::MatrixScalar + tenferro::TensorScalar>(
        &mut self,
        lhs: &[T],
        rhs: &[T],
        output: &mut [T],
        jobs: &[crate::matrix::GroupedGemmJob],
        options: crate::matrix::GroupedGemmOptions,
    ) -> Result<(), crate::matrix::GroupedGemmError> {
        crate::grouped_mat_mul_shared_in(self.session, lhs, rhs, output, jobs, options)
    }

    /// Thin/economy QR decomposition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an unsupported dtype or
    /// rank, and [`tenferro_tensor::Error::BackendSource`] when the factorization
    /// itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let (q, r) = context.with_concrete_session(|session| session.qr(&a))??;
    /// // `A = Q R` is reproduced to rounding, which is the correctness property
    /// // of a factorization.
    /// let reconstructed =
    ///     context.with_concrete_session(|session| session.contraction(&q, &[1], &r, &[0]))??;
    /// for (got, want) in reconstructed.as_slice::<f64>()?.iter().zip([1.0, 2.0, 3.0, 4.0]) {
    ///     assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn qr(
        &mut self,
        tensor: &NativeTensor,
    ) -> tenferro_tensor::Result<(NativeTensor, NativeTensor)> {
        tensor.qr(self.session)
    }

    /// Thin/economy SVD.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an unsupported dtype or
    /// rank, and [`tenferro_tensor::Error::BackendSource`] when the factorization
    /// itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 2.0])?;
    /// let (u, s, vt) = context.with_concrete_session(|session| session.svd(&a))??;
    /// // The singular values of diag(3, 2) are 3 and 2.
    /// assert_eq!(s.as_slice::<f64>()?, &[3.0, 2.0]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// assert_eq!(vt.shape(), &[2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn svd(
        &mut self,
        tensor: &NativeTensor,
    ) -> tenferro_tensor::Result<(NativeTensor, NativeTensor, NativeTensor)> {
        tensor.svd(self.session)
    }

    /// Solve `A X = B` for a square `A`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when `lhs` is not square, when
    /// the operands have mismatched shapes, or for an unsupported dtype, and
    /// [`tenferro_tensor::Error::BackendSource`] when the solve itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![6.0_f64, 8.0])?;
    /// let x = context.with_concrete_session(|session| session.solve(&a, &b))??;
    /// assert_eq!(x.as_slice::<f64>()?, &[3.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn solve(
        &mut self,
        lhs: &NativeTensor,
        rhs: &NativeTensor,
    ) -> tenferro_tensor::Result<NativeTensor> {
        lhs.solve(rhs, self.session)
    }

    /// Solve `A X = B` where `A` is triangular.
    ///
    /// - `left_side`: solve `A X = B` when true, `X A = B` when false;
    /// - `lower`: `A` is lower triangular;
    /// - `transpose_a`: solve with `A` transposed;
    /// - `unit_diagonal`: treat the diagonal of `A` as one, ignoring the stored
    ///   values.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for a shape mismatch or an
    /// unsupported dtype, and [`tenferro_tensor::Error::BackendSource`] when the solve
    /// itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// // Lower triangular, column-major.
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 4.0])?;
    /// let x = context.with_concrete_session(|session| {
    ///     session.triangular_solve(&a, &b, true, true, false, false)
    /// })??;
    /// assert_eq!(x.as_slice::<f64>()?, &[2.0, 0.5]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn triangular_solve(
        &mut self,
        lhs: &NativeTensor,
        rhs: &NativeTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<NativeTensor> {
        lhs.triangular_solve(
            rhs,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            self.session,
        )
    }

    /// Full-pivoting LU decomposition `P A Q = L U`, returning
    /// `(P, L, U, Q, parity)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for a non-square `A` or an
    /// unsupported dtype, and [`tenferro_tensor::Error::BackendSource`] when the
    /// factorization itself fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let (p, l, u, q, _parity) = context
    ///     .with_concrete_session(|session| session.full_piv_lu(&a))?
    ///     .expect("full-pivoting LU");
    /// for factor in [&p, &l, &u, &q] {
    ///     assert_eq!(factor.shape(), &[2, 2]);
    /// }
    /// // Column-major, so index 0 is (0, 0), index 1 is (1, 0) and index 2 is (0, 1).
    /// let l = l.as_slice::<f64>()?;
    /// let u = u.as_slice::<f64>()?;
    /// assert_eq!(l[2], 0.0, "L is lower triangular");
    /// assert_eq!(u[1], 0.0, "U is upper triangular");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn full_piv_lu(
        &mut self,
        tensor: &NativeTensor,
    ) -> tenferro_tensor::Result<(
        NativeTensor,
        NativeTensor,
        NativeTensor,
        NativeTensor,
        NativeTensor,
    )> {
        tensor.full_piv_lu(self.session)
    }
}

impl CpuExecutionContext {
    /// Run `f` inside one explicit concrete session of this context.
    ///
    /// The session is opened once for the whole closure, so a stage pays one entry
    /// and reuses the engine's prepared plans and buffer pool. All routes on the
    /// [`Session`] run on this context's backend only: nothing here consults the
    /// process-global default context, and nothing falls back to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::Tensor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
    /// let doubled = context.with_concrete_session(|session| session.reshape(&x, &[2]))??;
    /// assert_eq!(doubled.as_slice::<f64>()?, &[1.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::SessionEntry`] when the session cannot be
    /// opened, with [`tenferro_tensor::SessionEntryError`] as its typed cause:
    /// [`tenferro_tensor::SessionEntryError::Reentered`] for a nested session on this
    /// thread, [`tenferro_tensor::SessionEntryError::Contended`] for a caller inside the
    /// context's own pool or a busy resource the backend cannot wait for, and
    /// [`tenferro_tensor::SessionEntryError::ResourcePoisoned`] for poisoned admission
    /// state. It also returns the same error when the calling thread cannot restore its
    /// CPU affinity after `f` ran; that failure replaces the callback's value, exactly as
    /// the compatibility entry reports it.
    ///
    /// # Panics
    ///
    /// Never. A panic in `f` propagates after the session is released.
    ///
    pub fn with_concrete_session<R>(
        &self,
        f: impl FnOnce(&mut Session<'_>) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        // The compatibility frontend keeps its historical worker fallback, which
        // runs the session on an unrelated one-thread backend. This frontend promises
        // the caller's own backend, so a worker caller is rejected typed instead of
        // being rerouted.
        if rayon::current_thread_index().is_some() {
            return Err(CpuExecutionContextError::SessionEntry {
                source: std::sync::Arc::new(tenferro_tensor::SessionEntryError::Contended {
                    backend: "CpuExecutionContext",
                    message: "an explicit concrete session must be entered from outside the \
                              context's own pool; the compatibility frontend keeps its \
                              worker fallback"
                        .to_owned(),
                }),
            });
        }
        self.with_session(|session| f(&mut Session::new(session)))
    }
}

#[cfg(test)]
mod tests;
