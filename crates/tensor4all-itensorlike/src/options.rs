//! Configuration options for tensor train operations.

use std::ops::Range;

// Re-export CanonicalForm from core-common for convenience
pub use tensor4all_core_common::CanonicalForm;

/// Truncation algorithm.
///
/// This specifies which algorithm to use for truncating bond dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncateAlg {
    /// Singular Value Decomposition (optimal truncation).
    #[default]
    SVD,
    /// Rank-revealing LU decomposition.
    LU,
    /// Cross Interpolation.
    CI,
}

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

    /// Relative tolerance for truncation.
    ///
    /// Singular values satisfying `σ_i / σ_max < rtol` are truncated,
    /// where `σ_max` is the largest singular value.
    ///
    /// **Note**: ITensorMPS.jl's `cutoff` = `rtol²` (for normalized tensors).
    /// Use `rtol = sqrt(cutoff)` to match ITensorMPS.jl behavior.
    ///
    /// If `None`, no tolerance-based truncation is applied.
    pub rtol: Option<f64>,

    /// Maximum bond dimension (rank).
    ///
    /// If `None`, no rank limit is applied.
    pub max_rank: Option<usize>,

    /// Range of sites to truncate (0-indexed, exclusive end).
    ///
    /// If `None`, all bonds are truncated.
    pub site_range: Option<Range<usize>>,
}

impl Default for TruncateOptions {
    fn default() -> Self {
        Self {
            alg: TruncateAlg::SVD,
            rtol: None,
            max_rank: None,
            site_range: None,
        }
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
        self.rtol = Some(rtol);
        self
    }

    /// Set the maximum rank (bond dimension).
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
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
///     .with_nsweeps(5);
/// ```
#[derive(Debug, Clone)]
pub struct ContractOptions {
    /// Contraction method to use.
    pub method: ContractMethod,
    /// Maximum bond dimension.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation.
    pub rtol: Option<f64>,
    /// Number of sweeps for Fit method.
    pub nsweeps: usize,
}

impl Default for ContractOptions {
    fn default() -> Self {
        Self {
            method: ContractMethod::default(),
            max_rank: None,
            rtol: None,
            nsweeps: 2,
        }
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
        self.max_rank = Some(max_rank);
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set number of sweeps for Fit method.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nsweeps = nsweeps;
        self
    }
}
