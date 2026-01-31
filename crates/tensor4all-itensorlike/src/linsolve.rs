//! Linear equation solver for tensor trains.
//!
//! This module provides the [`linsolve`] function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where `A` is an MPO (TensorTrain),
//! and `x`, `b` are MPS (TensorTrain).
//!
//! Internally delegates to [`tensor4all_treetn::square_linsolve`].

use tensor4all_treetn::{square_linsolve, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_truncation_params, LinsolveOptions};
use crate::tensortrain::{truncate_alg_to_form, TensorTrain};
use tensor4all_core::truncation::HasTruncationParams;

/// Solve `(a₀ + a₁ * A) * x = b` for `x`.
///
/// # Arguments
/// * `operator` - The operator `A` (MPO as TensorTrain)
/// * `rhs` - The right-hand side `b` (MPS as TensorTrain)
/// * `init` - Initial guess for `x` (MPS as TensorTrain, consumed)
/// * `options` - Solver options
///
/// # Returns
/// The solution `x` as a TensorTrain.
///
/// # Errors
/// Returns an error if:
/// - Any tensor train is empty
/// - `nhalfsweeps` is not a multiple of 2
/// - The solver fails internally
pub fn linsolve(
    operator: &TensorTrain,
    rhs: &TensorTrain,
    init: TensorTrain,
    options: &LinsolveOptions,
) -> Result<TensorTrain> {
    if operator.is_empty() || rhs.is_empty() || init.is_empty() {
        return Err(TensorTrainError::InvalidStructure {
            message: "Cannot linsolve with empty tensor trains".to_string(),
        });
    }

    if !options.nhalfsweeps().is_multiple_of(2) {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "nhalfsweeps must be a multiple of 2, got {}",
                options.nhalfsweeps()
            ),
        });
    }

    validate_truncation_params(options.truncation_params())?;

    if !options.krylov_tol().is_finite() || options.krylov_tol() <= 0.0 {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "krylov_tol must be finite and > 0, got {}",
                options.krylov_tol()
            ),
        });
    }

    if options.krylov_maxiter() == 0 {
        return Err(TensorTrainError::OperationError {
            message: "krylov_maxiter must be >= 1".to_string(),
        });
    }

    if options.krylov_dim() == 0 {
        return Err(TensorTrainError::OperationError {
            message: "krylov_dim must be >= 1".to_string(),
        });
    }

    if let Some(tol) = options.convergence_tol() {
        if !tol.is_finite() || tol < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!("convergence_tol must be finite and >= 0, got {}", tol),
            });
        }
    }

    // Convert LinsolveOptions → treetn::LinsolveOptions
    let form = truncate_alg_to_form(options.alg());
    let nfullsweeps = options.nhalfsweeps() / 2;

    let treetn_options = tensor4all_treetn::LinsolveOptions::new(nfullsweeps)
        .with_truncation(TruncationOptions::new().with_form(form))
        .with_krylov_tol(options.krylov_tol())
        .with_krylov_maxiter(options.krylov_maxiter())
        .with_krylov_dim(options.krylov_dim())
        .with_coefficients(options.coefficients().0, options.coefficients().1);

    let treetn_options = if let Some(rtol) = options.rtol() {
        treetn_options.with_rtol(rtol)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(max_rank) = options.max_rank() {
        treetn_options.with_max_rank(max_rank)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(tol) = options.convergence_tol() {
        treetn_options.with_convergence_tol(tol)
    } else {
        treetn_options
    };

    // Use the last site as the sweep center
    let center = init.len() - 1;

    let result = square_linsolve(
        operator.as_treetn(),
        rhs.as_treetn(),
        init.treetn,
        &center,
        treetn_options,
    )
    .map_err(|e| TensorTrainError::OperationError {
        message: format!("Linsolve failed: {}", e),
    })?;

    Ok(TensorTrain::from_inner(result.solution, Some(form)))
}

impl TensorTrain {
    /// Solve `(a₀ + a₁ * A) * x = b` for `x`.
    ///
    /// `self` is the operator `A`, `rhs` is `b`, `init` is the initial guess.
    ///
    /// See [`linsolve`] for details.
    pub fn linsolve(&self, rhs: &Self, init: Self, options: &LinsolveOptions) -> Result<Self> {
        linsolve(self, rhs, init, options)
    }
}
