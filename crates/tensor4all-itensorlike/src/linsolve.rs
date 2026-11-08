//! Linear equation solver for tensor trains.
//!
//! This module provides the [`linsolve`] function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where `A` is an MPO (TensorTrain),
//! and `x`, `b` are MPS (TensorTrain).
//!
//! Internally delegates to [`tensor4all_treetn::square_linsolve`].

use std::collections::HashMap;

use tensor4all_core::DynIndex;
use tensor4all_treetn::{square_linsolve, IndexMapping, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_svd_truncation_options, CanonicalForm, LinsolveOptions};
use crate::tensortrain::TensorTrain;

/// Solve `(a₀ + a₁ * A) * x = b` for `x`.
///
/// The default two-site MALS updates can adapt bond dimensions. Use
/// [`LinsolveOptions::with_nsite(1)`](LinsolveOptions::with_nsite) for one-site
/// ALS without bond growth or SVD truncation. One-site mode also supports
/// single-tensor trains. Choose sufficient initial ranks for the solution.
/// Select GMRES (default), CG, or BiCGSTAB with
/// [`LinsolveOptions::with_local_solver`]. CG requires a Hermitian
/// positive-definite effective operator `a0 I + a1 A`.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_itensorlike::{linsolve, LinsolveOptions, TensorTrain};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s = DynIndex::new_dyn(2);
/// // Column-major diagonal operator diag(2, 3).
/// let a = TensorTrain::new(vec![IdxTensor::from_dense(
///     vec![s.prime(), s.clone()], vec![2.0_f64, 0.0, 0.0, 3.0],
/// )?])?;
/// let b = TensorTrain::new(vec![IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 9.0])?])?;
/// let x0 = TensorTrain::new(vec![IdxTensor::from_dense(vec![s], vec![0.0_f64, 0.0])?])?;
/// let x = linsolve(&a, &b, x0, &LinsolveOptions::new(1).with_nsite(1))?;
/// let values = x.to_dense()?.to_vec::<f64>()?;
/// assert!((values[0] - 2.0).abs() < 1e-10);
/// assert!((values[1] - 3.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
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
///
/// Returns [`TensorTrainError::InvalidStructure`] when any input train is
/// empty or when their lengths differ, [`TensorTrainError::OperationError`]
/// when a sweep or SVD option is invalid or when the operator and initial guess
/// cannot be index-mapped, and [`TensorTrainError::TreeTn`] when the update mode
/// or the local solver options are invalid (see
/// [`tensor4all_treetn::LinsolveOptions::validate_local_solver`]) or the
/// underlying TreeTN solve fails. Invalid native CG/BiCGSTAB options, CG
/// non-Hermitian or non-positive-curvature errors and BiCGSTAB breakdown carry a
/// [`KrylovError`](tensor4all_core::krylov::KrylovError) in the source chain.
///
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
    if operator.len() != rhs.len() || operator.len() != init.len() {
        return Err(TensorTrainError::InvalidStructure {
            message: format!(
                "operator, rhs, and init must have equal lengths; got {}, {}, {}",
                operator.len(),
                rhs.len(),
                init.len()
            ),
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

    validate_svd_truncation_options(options.max_bond_dim(), options.svd_policy())?;

    if let Some(tol) = options.convergence_tol() {
        if !tol.is_finite() || tol < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!("convergence_tol must be finite and >= 0, got {}", tol),
            });
        }
    }

    // Convert LinsolveOptions → treetn::LinsolveOptions
    let nfullsweeps = options.nhalfsweeps() / 2;

    let (a0, a1) = options.coefficients();
    let treetn_options = tensor4all_treetn::LinsolveOptions::new(nfullsweeps)
        .with_nsite(options.nsite())
        .with_local_solver(options.local_solver().clone())
        .with_truncation(TruncationOptions::new())
        .with_gmres_tol(options.gmres_tol())
        .with_gmres_max_restarts(options.gmres_max_restarts())
        .with_gmres_restart_dim(options.gmres_restart_dim())
        .with_coefficients(a0, a1);

    let treetn_options = if let Some(policy) = options.svd_policy() {
        treetn_options.with_svd_policy(policy)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(max_bond_dim) = options.max_bond_dim() {
        treetn_options.with_max_bond_dim(max_bond_dim)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(tol) = options.convergence_tol() {
        treetn_options.with_convergence_tol(tol)
    } else {
        treetn_options
    };
    let treetn_options = treetn_options.with_residual_check(options.check_residual());
    treetn_options
        .validate_local_solver()
        .map_err(|source| TensorTrainError::tree_tn("Invalid local solver options", source))?;

    // Use the last site as the sweep center
    let center = init.len() - 1;

    // Auto-infer index mappings when the operator (MPO) has distinct input/output
    // site indices. For each site, the MPO has 2 site indices while the MPS has 1.
    // We find the MPO index that shares an ID with init's site index (input) and
    // the remaining one (output).
    let (input_mapping, output_mapping) = infer_index_mappings(operator, &init)?;

    let result = square_linsolve(
        operator.as_treetn(),
        rhs.as_treetn(),
        init.treetn,
        &center,
        treetn_options,
        input_mapping,
        output_mapping,
    )
    .map_err(|source| TensorTrainError::tree_tn("Linsolve failed", source))?;

    TensorTrain::from_inner(result.solution, Some(CanonicalForm::Unitary))
}

type SiteMappings = (
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
);

/// Infer index mappings from the MPO/MPS structure.
///
/// For each site where the operator has exactly 2 site indices and init has 1,
/// finds the operator index exactly matching init's index (input) and the
/// remaining full index (output). Returns `(None, None)` when no mappings are
/// needed (operator and init share all site indices).
fn infer_index_mappings(operator: &TensorTrain, init: &TensorTrain) -> Result<SiteMappings> {
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
                // MPO-like site: exact input match plus one distinct output leg.
                let init_idx =
                    init_indices
                        .iter()
                        .next()
                        .ok_or_else(|| TensorTrainError::OperationError {
                            message: format!("Site {site}: init site space is empty"),
                        })?;
                let op_input = op_indices
                    .iter()
                    .find(|idx| *idx == init_idx)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {}: operator has 2 site indices but none exactly match init's \
                             site index {:?}. Cannot auto-infer index mappings. \
                             Use the treetn-level API with explicit IndexMapping.",
                            site, init_idx
                        ),
                    })?;
                op_indices
                    .iter()
                    .find(|idx| *idx != op_input)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no output index distinct from init index {:?}",
                            init_idx
                        ),
                    })?;
                needs_mapping = true;
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
                let init_idx =
                    init_indices
                        .iter()
                        .next()
                        .ok_or_else(|| TensorTrainError::OperationError {
                            message: format!("Site {site}: init site space is empty"),
                        })?;

                let op_input = op_indices
                    .iter()
                    .find(|idx| *idx == init_idx)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no input index exactly matching init index {:?}",
                            init_idx
                        ),
                    })?;
                let op_output = op_indices
                    .iter()
                    .find(|idx| *idx != op_input)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no output index distinct from init index {:?}",
                            init_idx
                        ),
                    })?;

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
    ///
    /// # Errors
    ///
    /// The same as [`linsolve`]: [`TensorTrainError::InvalidStructure`] when any
    /// input train is empty or when their lengths differ,
    /// [`TensorTrainError::OperationError`] when a sweep or SVD option is
    /// invalid or when the operator and initial guess cannot be index-mapped, and
    /// [`TensorTrainError::TreeTn`] when the update mode or the local solver
    /// options are invalid or the underlying TreeTN solve fails.
    ///
    pub fn linsolve(&self, rhs: &Self, init: Self, options: &LinsolveOptions) -> Result<Self> {
        linsolve(self, rhs, init, options)
    }
}

#[cfg(test)]
mod tests;
