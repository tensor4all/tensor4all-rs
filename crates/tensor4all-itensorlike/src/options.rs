//! Configuration options for tensor train operations.

use std::ops::Range;
use tensor4all_core::truncation::{HasTruncationParams, TruncationParams};

// Re-export CanonicalForm from treetn for convenience
pub use tensor4all_treetn::algorithm::CanonicalForm;

// Re-export DecompositionAlg for convenience
pub use tensor4all_core::truncation::DecompositionAlg;

/// Truncation algorithm.
///
/// This specifies which algorithm to use for truncating bond dimensions.
/// This is an alias for [`DecompositionAlg`] for backwards compatibility.
pub type TruncateAlg = DecompositionAlg;

/// Options for tensor train truncation.
///
/// Inspired by ITensorMPS.jl's truncation interface, but using tensor4all-rs
/// naming conventions (`rtol` instead of `cutoff`, `max_rank` instead of `maxdim`).
///
/// # Difference from ITensorMPS.jl
///
/// This crate uses **relative tolerance** (`rtol`) semantics:
/// - Singular values are truncated when `σ_i / σ_max < rtol`
///
/// ITensorMPS.jl uses **cutoff** semantics:
/// - Singular values are truncated when `σ_i² < cutoff`
///
/// **Conversion**: For normalized tensors (where `σ_max = 1`):
/// - ITensorMPS.jl's `cutoff` = tensor4all-rs's `rtol²`
/// - To match ITensorMPS.jl behavior: use `rtol = sqrt(cutoff)`
/// - Example: ITensorMPS.jl `cutoff=1e-10` ↔ tensor4all-rs `rtol=1e-5`
///
/// # Example
///
/// ```
/// use tensor4all_itensorlike::TruncateOptions;
///
/// // SVD with relative tolerance
/// let opts = TruncateOptions::svd().with_rtol(1e-10);
///
/// // LU with max rank
/// let opts = TruncateOptions::lu().with_max_rank(50);
///
/// // CI with both constraints
/// let opts = TruncateOptions::ci()
///     .with_rtol(1e-8)
///     .with_max_rank(100);
/// ```
#[derive(Debug, Clone)]
pub struct TruncateOptions {
    /// Algorithm to use for truncation.
    pub alg: TruncateAlg,

    /// Truncation parameters (rtol, max_rank).
    pub truncation: TruncationParams,

    /// Range of sites to truncate (0-indexed, exclusive end).
    ///
    /// If `None`, all bonds are truncated.
    pub site_range: Option<Range<usize>>,
}

impl Default for TruncateOptions {
    fn default() -> Self {
        Self {
            alg: TruncateAlg::SVD,
            truncation: TruncationParams::default(),
            site_range: None,
        }
    }
}

impl HasTruncationParams for TruncateOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

impl TruncateOptions {
    /// Create options for SVD-based truncation.
    pub fn svd() -> Self {
        Self {
            alg: TruncateAlg::SVD,
            ..Default::default()
        }
    }

    /// Create options for LU-based truncation.
    pub fn lu() -> Self {
        Self {
            alg: TruncateAlg::LU,
            ..Default::default()
        }
    }

    /// Create options for CI-based truncation.
    pub fn ci() -> Self {
        Self {
            alg: TruncateAlg::CI,
            ..Default::default()
        }
    }

    /// Set the relative tolerance for truncation.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self
    }

    /// Set the maximum rank (bond dimension).
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.truncation.max_rank = Some(max_rank);
        self
    }

    /// Set the site range for truncation.
    ///
    /// The range is 0-indexed with exclusive end.
    /// For example, `0..5` truncates bonds between sites 0-1, 1-2, 2-3, 3-4.
    pub fn with_site_range(mut self, range: Range<usize>) -> Self {
        self.site_range = Some(range);
        self
    }

    /// Get rtol (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank (for backwards compatibility).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}

/// Contraction method for tensor train operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractMethod {
    /// Zip-up contraction (faster, one-pass).
    #[default]
    Zipup,
    /// Fit/variational contraction (iterative optimization).
    Fit,
    /// Naive contraction: contract to full tensor, then decompose back.
    /// Useful for debugging and testing, but O(exp(n)) in memory.
    Naive,
}

/// Options for tensor train contraction.
///
/// # Example
///
/// ```
/// use tensor4all_itensorlike::ContractOptions;
///
/// // Zipup with max rank
/// let opts = ContractOptions::zipup().with_max_rank(50);
///
/// // Fit with relative tolerance
/// let opts = ContractOptions::fit()
///     .with_rtol(1e-10)
///     .with_nhalfsweeps(10);  // 10 half-sweeps = 5 full sweeps
/// ```
#[derive(Debug, Clone)]
pub struct ContractOptions {
    /// Contraction method to use.
    pub method: ContractMethod,
    /// Truncation parameters (rtol, max_rank).
    pub truncation: TruncationParams,
    /// Number of half-sweeps for Fit method.
    /// 
    /// A half-sweep visits edges in one direction only (forward or backward).
    /// This must be a multiple of 2 (each full sweep consists of 2 half-sweeps).
    pub nhalfsweeps: usize,
}

impl Default for ContractOptions {
    fn default() -> Self {
        Self {
            method: ContractMethod::default(),
            truncation: TruncationParams::default(),
            nhalfsweeps: 2,
        }
    }
}

impl HasTruncationParams for ContractOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

impl ContractOptions {
    /// Create options for zipup contraction.
    pub fn zipup() -> Self {
        Self {
            method: ContractMethod::Zipup,
            ..Default::default()
        }
    }

    /// Create options for fit contraction.
    pub fn fit() -> Self {
        Self {
            method: ContractMethod::Fit,
            ..Default::default()
        }
    }

    /// Create options for naive contraction.
    ///
    /// Note: Naive contraction is O(exp(n)) in memory and is primarily
    /// useful for debugging and testing.
    pub fn naive() -> Self {
        Self {
            method: ContractMethod::Naive,
            ..Default::default()
        }
    }

    /// Set maximum bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.truncation.max_rank = Some(max_rank);
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self
    }

    /// Set number of half-sweeps for Fit method.
    /// 
    /// # Arguments
    /// * `nhalfsweeps` - Number of half-sweeps (must be a multiple of 2)
    /// 
    /// # Panics
    /// Panics if `nhalfsweeps` is not a multiple of 2.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        if nhalfsweeps % 2 != 0 {
            panic!(
                "nhalfsweeps must be a multiple of 2, got {}",
                nhalfsweeps
            );
        }
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Get rtol (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank (for backwards compatibility).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_options_builder() {
        let opts = TruncateOptions::svd().with_rtol(1e-10).with_max_rank(50);
        assert_eq!(opts.alg, TruncateAlg::SVD);
        assert_eq!(opts.rtol(), Some(1e-10));
        assert_eq!(opts.max_rank(), Some(50));
    }

    #[test]
    fn test_truncate_options_algorithms() {
        assert_eq!(TruncateOptions::svd().alg, TruncateAlg::SVD);
        assert_eq!(TruncateOptions::lu().alg, TruncateAlg::LU);
        assert_eq!(TruncateOptions::ci().alg, TruncateAlg::CI);
    }

    #[test]
    fn test_truncate_options_site_range() {
        let opts = TruncateOptions::svd().with_site_range(0..5);
        assert_eq!(opts.site_range, Some(0..5));
    }

    #[test]
    fn test_truncate_options_default() {
        let opts = TruncateOptions::default();
        assert_eq!(opts.alg, TruncateAlg::SVD);
        assert_eq!(opts.rtol(), None);
        assert_eq!(opts.max_rank(), None);
        assert!(opts.site_range.is_none());
    }

    #[test]
    fn test_truncate_options_has_truncation_params() {
        let mut opts = TruncateOptions::svd();
        assert!(opts.truncation_params().rtol.is_none());
        opts.truncation_params_mut().rtol = Some(1e-8);
        assert_eq!(opts.truncation_params().rtol, Some(1e-8));
    }

    #[test]
    fn test_contract_options_builder() {
        let opts = ContractOptions::zipup()
            .with_max_rank(100)
            .with_rtol(1e-12)
            .with_nhalfsweeps(6);  // 6 half-sweeps = 3 full sweeps
        assert_eq!(opts.method, ContractMethod::Zipup);
        assert_eq!(opts.max_rank(), Some(100));
        assert_eq!(opts.rtol(), Some(1e-12));
        assert_eq!(opts.nhalfsweeps, 6);
    }

    #[test]
    fn test_contract_options_methods() {
        assert_eq!(ContractOptions::zipup().method, ContractMethod::Zipup);
        assert_eq!(ContractOptions::fit().method, ContractMethod::Fit);
        assert_eq!(ContractOptions::naive().method, ContractMethod::Naive);
    }

    #[test]
    fn test_contract_options_default() {
        let opts = ContractOptions::default();
        assert_eq!(opts.method, ContractMethod::Zipup);
        assert_eq!(opts.nhalfsweeps, 2);
        assert_eq!(opts.rtol(), None);
        assert_eq!(opts.max_rank(), None);
    }

    #[test]
    fn test_contract_options_has_truncation_params() {
        let mut opts = ContractOptions::zipup();
        assert!(opts.truncation_params().rtol.is_none());
        opts.truncation_params_mut().max_rank = Some(50);
        assert_eq!(opts.truncation_params().max_rank, Some(50));
    }

    #[test]
    fn test_contract_method_default() {
        let method: ContractMethod = Default::default();
        assert_eq!(method, ContractMethod::Zipup);
    }
}
