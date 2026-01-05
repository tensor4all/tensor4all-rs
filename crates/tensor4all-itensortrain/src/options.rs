//! Configuration options for tensor train operations.

use std::ops::Range;

/// Canonicalization algorithm.
///
/// This specifies which algorithm to use for orthogonalizing tensors
/// during canonicalization sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CanonicalMethod {
    /// Singular Value Decomposition (most accurate, but slowest).
    #[default]
    SVD,
    /// Rank-revealing LU decomposition (fast, good accuracy).
    LU,
    /// Cross Interpolation (fastest, may be less accurate).
    CI,
}

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
/// use tensor4all_itensortrain::TruncateOptions;
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
