//! Algorithm dispatch for MPO contraction
//!
//! Provides a unified `contract` function that dispatches to the appropriate
//! algorithm based on [`ContractionAlgorithm`].

/// Algorithm for MPO contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionAlgorithm {
    /// Naive contraction (exact but memory-intensive)
    #[default]
    Naive,
    /// Zip-up contraction with on-the-fly compression
    ZipUp,
}

use super::contract_naive::contract_naive;
use super::contract_zipup::contract_zipup;
use super::contraction::ContractionOptions;
use super::error::Result;
use super::factorize::SVDScalar;
use super::mpo::MPO;
use crate::einsum_helper::EinsumScalar;

/// Unified contraction function with algorithm dispatch
///
/// Contracts two MPOs using the specified algorithm.
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `algorithm` - Which algorithm to use
/// * `options` - Contraction options (tolerance, max_bond_dim, etc.)
///
/// # Returns
/// The contracted MPO
///
/// # Errors
///
/// Returns [`MPOError::LengthMismatch`](super::MPOError::LengthMismatch) when
/// the two MPOs have different lengths,
/// [`MPOError::SharedDimensionMismatch`](super::MPOError::SharedDimensionMismatch)
/// when a shared physical dimension disagrees, or
/// [`MPOError::InvalidOperation`](super::MPOError::InvalidOperation) /
/// [`MPOError::FactorizationError`](super::MPOError::FactorizationError) when a
/// site contraction or the selected algorithm's compression fails.
///
/// # Example
///
/// ```
/// use tensor4all_simplett::mpo::{contract, MPO, ContractionOptions, ContractionAlgorithm};
///
/// let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
/// let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
/// let options = ContractionOptions::default();
///
/// // Use naive algorithm
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
/// assert_eq!(result.site_dims(), vec![(2, 2), (2, 2)]);
///
/// // Use zip-up algorithm for memory efficiency
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
/// assert_eq!(result.len(), 2);
/// ```
pub fn contract<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    algorithm: ContractionAlgorithm,
    options: &ContractionOptions,
) -> Result<MPO<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    match algorithm {
        ContractionAlgorithm::Naive => contract_naive(mpo_a, mpo_b, Some(options.clone())),
        ContractionAlgorithm::ZipUp => contract_zipup(mpo_a, mpo_b, options),
    }
}

#[cfg(test)]
mod tests;
