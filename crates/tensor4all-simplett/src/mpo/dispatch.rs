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
    /// Variational fitting algorithm
    Fit,
}

use super::contract_fit::{contract_fit, FitOptions};
use super::contract_naive::contract_naive;
use super::contract_zipup::contract_zipup;
use super::contraction::ContractionOptions;
use super::error::Result;
use super::factorize::SVDScalar;
use super::mpo::MPO;

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
/// # Example
///
/// ```ignore
/// use tensor4all_simplett::mpo::{contract, MPO, ContractionOptions, ContractionAlgorithm};
///
/// let mpo_a = MPO::identity(&[2, 2])?;
/// let mpo_b = MPO::identity(&[2, 2])?;
/// let options = ContractionOptions::default();
///
/// // Use naive algorithm
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options)?;
///
/// // Use zip-up algorithm for memory efficiency
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options)?;
///
/// // Use variational fitting for controlled bond dimension
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options)?;
/// ```
pub fn contract<T: SVDScalar>(
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
        ContractionAlgorithm::Fit => {
            let fit_options = FitOptions {
                tolerance: options.tolerance,
                max_bond_dim: options.max_bond_dim,
                factorize_method: options.factorize_method.clone(),
                ..Default::default()
            };
            contract_fit(mpo_a, mpo_b, &fit_options, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_dispatch_naive() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
        let options = ContractionOptions::default();

        let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_contract_dispatch_zipup() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
        let options = ContractionOptions::default();

        let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_contract_dispatch_fit() {
        let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
        let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
        let options = ContractionOptions::default();

        let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_contract_algorithms_consistent() {
        // All algorithms should give the same result for simple case
        // constant takes &[(s1, s2)] - tuple per site
        let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
        let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);
        let options = ContractionOptions::default();

        let result_naive = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
        let result_zipup = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
        let result_fit = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options).unwrap();

        // Check all algorithms give the same result
        // evaluate takes &[LocalIndex] - flat list, 2 per site
        let val_naive = result_naive.evaluate(&[0, 0]).unwrap();
        let val_zipup = result_zipup.evaluate(&[0, 0]).unwrap();
        let val_fit = result_fit.evaluate(&[0, 0]).unwrap();

        // All algorithms should give the same result
        assert!((val_naive - val_zipup).abs() < 1e-10);
        assert!((val_naive - val_fit).abs() < 1e-10);
    }
}
