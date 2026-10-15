//! Contraction operations for tensor trains.
//!
//! This module provides both a free function [`contract`] and an impl method
//! [`TensorTrain::contract`] for contracting two tensor trains.

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_tensorbackend::ExecutionContext;
use tensor4all_treetn::contraction::{
    contract as treetn_contract, contract_src_with_rng_in, ContractionMethod,
    ContractionOptions as TreeTNContractionOptions,
};
use tensor4all_treetn::CanonicalForm;

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_svd_truncation_options, ContractMethod, ContractOptions};
use crate::tensortrain::TensorTrain;

/// Contract two tensor trains, returning a new tensor train.
///
/// This performs element-wise contraction of corresponding sites,
/// similar to MPO-MPO contraction in ITensor.
///
/// # Arguments
/// * `a` - The first tensor train
/// * `b` - The second tensor train
/// * `options` - Contraction options (method, max_bond_dim, rtol, nhalfsweeps)
///
/// # Returns
/// A new tensor train resulting from the contraction.
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// index mismatch, or a backend failure).
///
/// Shared entry validation for every contraction route. Both [`contract`] and
/// [`contract_in`] run exactly this, in this order, so the routes cannot drift
/// on empty inputs, unequal lengths, invalid SVD options or a non-multiple-of-2
/// Fit sweep count.
fn validate_contraction_inputs(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
) -> Result<()> {
    if a.is_empty() || b.is_empty() {
        return Err(TensorTrainError::InvalidStructure {
            message: "Cannot contract empty tensor trains".to_string(),
        });
    }
    if a.len() != b.len() {
        return Err(TensorTrainError::InvalidStructure {
            message: format!(
                "Tensor trains must have the same length for contraction: {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }
    validate_svd_truncation_options(options.max_bond_dim(), options.svd_policy())?;
    if matches!(options.method(), ContractMethod::Fit) && !options.nhalfsweeps().is_multiple_of(2) {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "nhalfsweeps must be a multiple of 2 for Fit method, got {}",
                options.nhalfsweeps()
            ),
        });
    }
    Ok(())
}

/// The method dispatch shared by every contraction route. Callers run
/// [`validate_contraction_inputs`] first, so the dispatch may assume the
/// validated contract and does not re-check it.
fn contract_dispatch(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
) -> Result<TensorTrain> {
    let treetn_options = to_treetn_options(options);

    // Use the last site as the canonical center (consistent with existing behavior)
    let center = a.len() - 1;
    let result_inner = if matches!(options.method(), ContractMethod::Zipup) {
        a.as_treetn()
            .contract_zipup_with(
                b.as_treetn(),
                &center,
                CanonicalForm::Unitary,
                options.svd_policy(),
                options.max_bond_dim(),
            )
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Zip-up contraction failed: {}", e),
            })?
    } else {
        treetn_contract(a.as_treetn(), b.as_treetn(), &center, treetn_options).map_err(|e| {
            TensorTrainError::InvalidStructure {
                message: format!("TreeTN contraction failed: {}", e),
            }
        })?
    };

    TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary))
}

/// Contract two tensor trains, returning a new tensor train.
///
/// This performs element-wise contraction of corresponding sites,
/// similar to MPO-MPO contraction in ITensor.
///
/// # Arguments
/// * `a` - The first tensor train
/// * `b` - The second tensor train
/// * `options` - Contraction options (method, max_bond_dim, rtol, nhalfsweeps)
///
/// # Returns
/// A new tensor train resulting from the contraction.
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// index mismatch, or a backend failure).
pub fn contract(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
) -> Result<TensorTrain> {
    validate_contraction_inputs(a, b, options)?;
    contract_dispatch(a, b, options)
}

/// Convert [`ContractOptions`] to the caller-independent TreeTN contraction
/// options, sharing the method, SVD policy, bond cap and dense-reference limit
/// across every dispatch route.
fn to_treetn_options(options: &ContractOptions) -> TreeTNContractionOptions {
    // Convert nhalfsweeps to nfullsweeps (nhalfsweeps / 2)
    let nfullsweeps = options.nhalfsweeps() / 2;
    let mut treetn_options = TreeTNContractionOptions::new(match options.method() {
        ContractMethod::Zipup => ContractionMethod::Zipup,
        ContractMethod::Fit => ContractionMethod::Fit,
        ContractMethod::Naive => ContractionMethod::Naive,
        ContractMethod::Src => ContractionMethod::Src,
    })
    .with_nfullsweeps(nfullsweeps)
    .with_src_options(options.src_options().clone());
    if let Some(max_bond_dim) = options.max_bond_dim() {
        treetn_options = treetn_options.with_max_bond_dim(max_bond_dim);
    }
    if let Some(policy) = options.svd_policy() {
        treetn_options = treetn_options.with_svd_policy(policy);
    }
    if let Some(limit) = options.dense_reference_limit() {
        treetn_options = treetn_options.with_dense_reference_limit(limit);
    }
    treetn_options
}

/// Contract two tensor trains, running the SRC method in a caller-owned
/// execution context.
///
/// The same algorithm, options and validation as [`contract`]; what differs is
/// the resource the method's numerical work runs in. With a context, SRC runs
/// through treetn's `contract_src_with_rng_in` and Zip-up through treetn's
/// `contract_zipup_with_in` (both context-scoped). Fit and Naive are refused
/// with a typed error rather than silently run on the process-global context:
/// treetn has no session-taking twin for their sweep / dense-reference paths
/// yet, which is recorded in the F7 inventory
/// (`docs/design/859-f7-consumer-inventory.md`).
///
/// # Arguments
/// * `a` - The first tensor train
/// * `b` - The second tensor train
/// * `options` - Contraction options (method, max_bond_dim, rtol, nhalfsweeps)
/// * `context` - The caller-owned execution context the SRC method runs in
///
/// # Returns
/// A new tensor train resulting from the contraction.
///
/// # Errors
///
/// Returns the same errors as [`contract`], plus a structural error when the
/// caller-owned context refuses a Zip-up contraction (including a provenance
/// or factorization refusal), and a typed
/// [`TensorTrainError::OperationError`] for the Fit and Naive methods (the
/// caller-owned route cannot honor the supplied context for those methods
/// yet).
pub fn contract_in(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
    context: &ExecutionContext,
) -> Result<TensorTrain> {
    validate_contraction_inputs(a, b, options)?;

    if matches!(options.method(), ContractMethod::Zipup) {
        // `validate_contraction_inputs` rejects empty trains, so `len() >= 1`
        // and the canonical center below cannot underflow.
        // Use the last site as the canonical center (consistent with existing behavior).
        let center = a.len() - 1;
        // Zip-up runs in the caller-owned context through treetn's twin
        // (`contract_zipup_with_in`): the per-site contractions and
        // factorizations use the context-scoped core entries.
        let result_inner = a
            .as_treetn()
            .contract_zipup_with_in(
                b.as_treetn(),
                &center,
                CanonicalForm::Unitary,
                options.svd_policy(),
                options.max_bond_dim(),
                context,
            )
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Zip-up contraction failed: {}", e),
            })?;
        return TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary));
    }

    if matches!(options.method(), ContractMethod::Fit) {
        // `validate_contraction_inputs` rejects empty trains, so `len() >= 1`.
        let center = a.len() - 1;
        // Fit runs in the caller-owned context through treetn's dispatch
        // (`contract_in`), which routes it to `contract_fit_in`: the initial
        // zip-up, the environment contractions and the per-sweep
        // factorizations use the context-scoped core entries.
        let result_inner = tensor4all_treetn::contract_in(
            a.as_treetn(),
            b.as_treetn(),
            &center,
            to_treetn_options(options),
            context,
        )
        .map_err(|e| TensorTrainError::InvalidStructure {
            message: format!("TreeTN Fit contraction failed: {e}"),
        })?;
        return TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary));
    }

    if matches!(options.method(), ContractMethod::Naive) {
        // The Naive dense reference runs in the caller-owned context through
        // treetn's dispatch (`contract_in` -> `contract_naive_to_treetn_with_leaves`):
        // its dense product, pairwise product and decomposition factorizations
        // use the session leaves.
        // `validate_contraction_inputs` rejects empty trains, so `len() >= 1`.
        let center = a.len() - 1;
        let result_inner = tensor4all_treetn::contract_in(
            a.as_treetn(),
            b.as_treetn(),
            &center,
            to_treetn_options(options),
            context,
        )
        .map_err(|e| TensorTrainError::InvalidStructure {
            message: format!("TreeTN Naive contraction failed: {}", e),
        })?;
        return TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(options.src_options().seed);
    // `validate_contraction_inputs` rejects empty trains, so `len() >= 1`.
    // Use the last site as the canonical center (consistent with existing behavior).
    let center = a.len() - 1;
    let result_inner = contract_src_with_rng_in(
        a.as_treetn(),
        b.as_treetn(),
        &center,
        to_treetn_options(options),
        &mut rng,
        context,
    )
    .map_err(|e| TensorTrainError::InvalidStructure {
        message: format!("TreeTN SRC contraction failed: {}", e),
    })?;
    TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary))
}

impl TensorTrain {
    /// Contract two tensor trains, returning a new tensor train.
    ///
    /// This performs element-wise contraction of corresponding sites,
    /// similar to MPO-MPO contraction in ITensor.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to contract with
    /// * `options` - Contraction options (method, max_bond_dim, rtol, nhalfsweeps)
    ///
    /// # Returns
    /// A new tensor train resulting from the contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the contraction or operation fails (a shape or
    /// index mismatch, or a backend failure).
    ///
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        contract(self, other, options)
    }

    /// Contract two tensor trains with explicit contraction options.
    ///
    /// This is an alias for [`TensorTrain::contract`]. It exists for callers
    /// that use `contract_pair` to mean pairwise tensor-train contraction with
    /// compression, rather than the dense tensor-level pair contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the pair is incompatible (a shape or index mismatch)
    /// or the contraction fails (a backend failure).
    ///
    pub fn contract_pair(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        self.contract(other, options)
    }

    /// Contract two tensor trains, running the SRC method in a caller-owned
    /// execution context.
    ///
    /// Same contract as [`crate::contract_in`]: SRC and Zip-up run in the
    /// caller-owned context; Fit and Naive are refused with a typed error. See
    /// that function for the inventory record.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to contract with
    /// * `options` - Contraction options (method, max_bond_dim, rtol, nhalfsweeps)
    /// * `context` - The caller-owned execution context the SRC method runs in
    ///
    /// # Returns
    /// A new tensor train resulting from the contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the pair is incompatible (a shape or index
    /// mismatch), the contraction fails (a backend failure), or the
    /// caller-owned context refuses the SRC projection.
    pub fn contract_in(
        &self,
        other: &Self,
        options: &ContractOptions,
        context: &ExecutionContext,
    ) -> Result<Self> {
        contract_in(self, other, options, context)
    }
}

#[cfg(test)]
mod tests;
