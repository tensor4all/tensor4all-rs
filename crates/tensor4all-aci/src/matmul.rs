//! The ACI sweep's matrix products, on the caller's CPU execution context only.
//!
//! Every product of the sweep (frame updates, local input factors, local block
//! materialization, solution core updates and the environments of injected
//! global pivots) goes through these helpers. There is no context-free route:
//! the context-free public entry points resolve the process-global context once
//! and hand it in, so one code path runs on whichever engine the caller chose
//! (tensor4all-rs#1382). A failure is reported as [`AciError::Backend`] through
//! [`frame_matmul_error`], never as an initial-guess error.

use crate::scalar::AciScalar;
use crate::{AciError, Result};
use tensor4all_tensorbackend::{CpuExecutionContext, Matrix};

/// Report a failed matrix product of the sweep as a backend failure.
///
/// `operation` names the product (for example `"left frame update at input 0,
/// site 1"`); `source` is the backend, session or shape error that stopped it.
pub(crate) fn frame_matmul_error(
    operation: impl Into<String>,
    source: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> AciError {
    AciError::Backend {
        operation: operation.into(),
        source: source.into(),
    }
}

/// `left * right` on `context`'s engine.
///
/// # Errors
///
/// [`AciError::Backend`] when the shapes disagree, the context refuses the
/// session, or the engine rejects the product.
pub(crate) fn mat_mul_in<T: AciScalar>(
    context: &CpuExecutionContext,
    left: &Matrix<T>,
    right: &Matrix<T>,
    operation: impl FnOnce() -> String,
) -> Result<Matrix<T>> {
    if left.ncols() != right.nrows() {
        return Err(frame_matmul_error(
            operation(),
            format!(
                "incompatible matrix shapes: left ({}, {}), right ({}, {})",
                left.nrows(),
                left.ncols(),
                right.nrows(),
                right.ncols()
            ),
        ));
    }
    match context.with_concrete_session(|session| session.mat_mul(left, right)) {
        Ok(Ok(product)) => Ok(product),
        Ok(Err(error)) => Err(frame_matmul_error(operation(), error)),
        Err(error) => Err(frame_matmul_error(operation(), error)),
    }
}

/// `batch` same-shape products `A[p] * B[p]` (`m x k` times `k x n`, column
/// major, stored back to back) on `context`'s engine.
///
/// # Errors
///
/// [`AciError::Backend`] when a buffer length disagrees with the declared
/// shape, the context refuses the session, or the engine rejects the product.
pub(crate) fn batched_mat_mul_in<T: AciScalar>(
    context: &CpuExecutionContext,
    (batch, m, k, n): (usize, usize, usize, usize),
    a: Vec<T>,
    b: Vec<T>,
    operation: impl FnOnce() -> String,
) -> Result<Vec<T>> {
    match context.with_concrete_session(|session| {
        session.batched_mat_mul_same_shape_owned(batch, m, k, n, a, b)
    }) {
        Ok(Ok(values)) => Ok(values),
        Ok(Err(error)) => Err(frame_matmul_error(operation(), error)),
        Err(error) => Err(frame_matmul_error(operation(), error)),
    }
}

/// The CPU context a context-free public entry point runs on.
///
/// The context-free entries have no caller context, so they take the
/// process-global one here, once, and run the same context-taking code path as
/// the `*_in` entries.
///
/// # Errors
///
/// [`AciError::Backend`] in a build without the process-global route
/// (`tensor4all-core`'s `global-defaults` feature off): call an `*_in` entry
/// with an explicit context instead.
pub(crate) fn default_context() -> Result<std::sync::Arc<CpuExecutionContext>> {
    tensor4all_core::legacy::default_cpu_execution_context()
        .map_err(|error| frame_matmul_error("resolving the process-global CPU context", error))
}
