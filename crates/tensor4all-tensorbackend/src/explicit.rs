//! Opt-in explicit/concrete execution frontend.
//!
//! This is the second, opt-in frontend of
//! [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859): a
//! caller-supplied [`CpuExecutionContext`] plus one concrete session that a whole
//! stage or batch reuses, instead of the process-global convenience surface in
//! [`crate::context`]. The compatibility frontend keeps its own signatures,
//! defaults and behaviour; nothing here reroutes it.
//!
//! Properties of this frontend, from the coexistence record
//! (`docs/design/859-dual-frontend-coexistence.md`):
//!
//! - **One explicit entry.** [`CpuExecutionContext::with_concrete_session`] opens
//!   the session for the whole closure, so a stage pays one entry rather than one
//!   per operation, and the engine's prepared plans and buffer pool stay warm.
//! - **No process-global selector.** No route here reads a default context, an
//!   environment variable or a thread-local override.
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
//! The session is a [`BackendSession`] scope, not a value representation: the
//! concrete values stay the ordinary tenferro tensors.
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
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_linalg::TensorLinalgExt;

use crate::context::{CpuExecutionContext, CpuExecutionContextError};
use crate::tenferro_bridge::{build_binary_einsum_ids, checked_native_einsum_labels};

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
    session: &'session mut dyn tenferro_tensor::BackendSession,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("explicit::Session")
            .finish_non_exhaustive()
    }
}

impl<'session> Session<'session> {
    pub(crate) fn new(session: &'session mut dyn tenferro_tensor::BackendSession) -> Self {
        Self { session }
    }

    /// Reshape a concrete tensor without changing its column-major linearization.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the element counts
    /// differ or the requested layout is invalid, and [`tenferro_tensor::Error::Tensor`]
    /// when the backend rejects the operation. It never retries on another route.
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
    /// permutation of the tensor's axes, and [`tenferro_tensor::Error::Tensor`] when
    /// the backend rejects the operation.
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
    /// malformed or of different length, [`tenferro_einsum::Error::Validation`] for a
    /// shape or dtype mismatch, and [`tenferro_einsum::Error::Numerical`] when the
    /// contraction itself fails.
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
        [lhs, rhs].einsum_subscripts(&subscripts, self.session)
    }

    /// N-ary einsum from integer labels, evaluated on this session.
    ///
    /// The labels are validated by the same helper the compatibility frontend
    /// uses; the evaluation is tenferro-einsum's session-direct einsum, so it does
    /// not compile a semantic graph or start a runtime worker.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_einsum::Error::InvalidSubscripts`] for malformed labels or a
    /// label list whose length does not match the operand's rank,
    /// [`tenferro_einsum::Error::Validation`] for an invalid contraction, and
    /// [`tenferro_einsum::Error::Numerical`] when the contraction itself fails.
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
        assert_eq!(
            operands.len(),
            input_ids.len(),
            "each einsum operand needs one label list"
        );
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
        operands.einsum_subscripts(&subscripts, self.session)
    }

    /// Thin/economy QR decomposition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an unsupported dtype or
    /// rank, and [`tenferro_tensor::Error::Tensor`] when the backend cannot complete
    /// the factorization.
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
    /// rank, and [`tenferro_tensor::Error::Tensor`] when the backend cannot complete
    /// the factorization.
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
    /// [`tenferro_tensor::Error::Tensor`] when the backend cannot complete the solve.
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
    /// Returns [`CpuExecutionContextError::SessionEntry`] when the context's
    /// backend refuses the session: [`tenferro_tensor::SessionEntryError`] keeps
    /// its own typed cause for a nested entry, a busy resource the backend cannot
    /// wait for, a mismatched execution scope or poisoned admission state.
    ///
    /// # Panics
    ///
    /// Never. A panic in `f` propagates after the session is released.
    ///
    /// The callback and its value must both be `Send`, because the backend may run
    /// the session on one of its own workers.
    pub fn with_concrete_session<R: Send>(
        &self,
        f: impl FnOnce(&mut Session<'_>) -> R + Send,
    ) -> Result<R, CpuExecutionContextError> {
        self.with_session(|session| f(&mut Session::new(session)))
    }
}

#[cfg(test)]
mod tests;
