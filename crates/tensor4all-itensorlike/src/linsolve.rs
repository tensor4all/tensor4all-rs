//! Linear equation solver for tensor trains.
//!
//! This module provides the [`linsolve`] function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where `A` is an MPO (TensorTrain),
//! and `x`, `b` are MPS (TensorTrain).
//!
//! Internally delegates to [`tensor4all_treetn::square_linsolve`].

use std::collections::HashMap;

use tensor4all_core::truncation::HasTruncationParams;
use tensor4all_core::{DynIndex, IndexLike};
use tensor4all_treetn::{square_linsolve, IndexMapping, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_truncation_params, LinsolveOptions};
use crate::tensortrain::{truncate_alg_to_form, TensorTrain};

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

    // Auto-infer index mappings when the operator (MPO) has distinct input/output
    // site indices. For each site, the MPO has 2 site indices while the MPS has 1.
    // We find the MPO index that shares an ID with init's site index (input) and
    // the remaining one (output).
    let (input_mapping, output_mapping) =
        infer_index_mappings(operator, &init).map_err(|e| TensorTrainError::OperationError {
            message: format!("Failed to infer index mappings: {}", e),
        })?;

    let result = square_linsolve(
        operator.as_treetn(),
        rhs.as_treetn(),
        init.treetn,
        &center,
        treetn_options,
        input_mapping,
        output_mapping,
    )
    .map_err(|e| TensorTrainError::OperationError {
        message: format!("Linsolve failed: {}", e),
    })?;

    TensorTrain::from_inner(result.solution, Some(form))
}

type SiteMappings = (
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
);

/// Infer index mappings from the MPO/MPS structure.
///
/// For each site where the operator has exactly 2 site indices and init has 1,
/// finds the operator index sharing an ID with init's index (input) and the
/// remaining one (output). Returns `(None, None)` when no mappings are needed
/// (operator and init share all site indices).
fn infer_index_mappings(
    operator: &TensorTrain,
    init: &TensorTrain,
) -> std::result::Result<SiteMappings, String> {
    let op_treetn = operator.as_treetn();
    let init_treetn = init.as_treetn();
    let nsites = init.len();

    let mut needs_mapping = false;

    // First pass: check if any site needs mappings
    for site in 0..nsites {
        let op_site = op_treetn.site_space(&site);
        let init_site = init_treetn.site_space(&site);

        if let (Some(op_indices), Some(init_indices)) = (op_site, init_site) {
            if op_indices.len() == 2 && init_indices.len() == 1 {
                // MPO-like site: check if we need mappings
                let init_idx = init_indices.iter().next().unwrap();
                let has_shared = op_indices.iter().any(|idx| idx.same_id(init_idx));
                if has_shared {
                    // The shared index exists but may differ by plev → need mapping
                    let input_idx = op_indices.iter().find(|idx| idx.same_id(init_idx));
                    if let Some(input_idx) = input_idx {
                        if input_idx != init_idx {
                            needs_mapping = true;
                        }
                    }
                    // Check if output index differs from init
                    let output_idx = op_indices.iter().find(|idx| !idx.same_id(init_idx));
                    if output_idx.is_some() {
                        needs_mapping = true;
                    }
                } else {
                    // No shared ID between operator and init site indices.
                    // Fall through to (None, None) and let square_linsolve
                    // auto-infer via LinearOperator::from_mpo_and_state.
                    return Ok((None, None));
                }
            }
        }
    }

    if !needs_mapping {
        return Ok((None, None));
    }

    // Second pass: build mappings
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for site in 0..nsites {
        let op_site = op_treetn.site_space(&site);
        let init_site = init_treetn.site_space(&site);

        if let (Some(op_indices), Some(init_indices)) = (op_site, init_site) {
            if op_indices.len() == 2 && init_indices.len() == 1 {
                let init_idx = init_indices.iter().next().unwrap();

                let op_input = op_indices.iter().find(|idx| idx.same_id(init_idx)).unwrap();
                let op_output = op_indices
                    .iter()
                    .find(|idx| !idx.same_id(init_idx))
                    .unwrap();

                input_mapping.insert(
                    site,
                    IndexMapping {
                        true_index: init_idx.clone(),
                        internal_index: op_input.clone(),
                    },
                );
                output_mapping.insert(
                    site,
                    IndexMapping {
                        true_index: init_idx.clone(),
                        internal_index: op_output.clone(),
                    },
                );
            }
        }
    }

    Ok((Some(input_mapping), Some(output_mapping)))
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
