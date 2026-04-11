//! Common truncation options and traits.
//!
//! This module provides shared types and traits for truncation parameters
//! used across tensor operations like SVD, QR, and tensor train compression.

/// Decomposition/factorization algorithm.
///
/// This enum unifies the algorithm choices across different crates
/// (tensor4all-core, tensor4all-treetn, etc.).
///
/// # Examples
///
/// ```
/// use tensor4all_core::DecompositionAlg;
///
/// assert_eq!(DecompositionAlg::default(), DecompositionAlg::SVD);
/// assert!(DecompositionAlg::SVD.is_svd_based());
/// assert!(DecompositionAlg::SVD.is_orthogonal());
/// assert!(!DecompositionAlg::LU.is_orthogonal());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionAlg {
    /// Singular Value Decomposition (optimal truncation).
    #[default]
    SVD,
    /// Randomized SVD (faster for large matrices).
    RSVD,
    /// QR decomposition.
    QR,
    /// Rank-revealing LU decomposition.
    LU,
    /// Cross Interpolation.
    CI,
}

impl DecompositionAlg {
    /// Check if this algorithm is SVD-based (SVD or RSVD).
    #[must_use]
    pub fn is_svd_based(&self) -> bool {
        matches!(self, Self::SVD | Self::RSVD)
    }

    /// Check if this algorithm provides orthogonal factors.
    #[must_use]
    pub fn is_orthogonal(&self) -> bool {
        matches!(self, Self::SVD | Self::RSVD | Self::QR)
    }
}

/// Common truncation parameters.
///
/// This struct contains the core parameters used for rank truncation
/// across various tensor decomposition and compression operations.
///
/// # Semantics
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
/// # Examples
///
/// ```
/// use tensor4all_core::TruncationParams;
///
/// // Builder pattern
/// let params = TruncationParams::new()
///     .with_rtol(1e-8)
///     .with_max_rank(50);
/// assert_eq!(params.rtol, Some(1e-8));
/// assert_eq!(params.max_rank, Some(50));
///
/// // ITensorMPS.jl compatibility
/// let params = TruncationParams::new().with_cutoff(1e-10);
/// let rtol = params.rtol.unwrap();
/// assert!((rtol - 1e-5).abs() < 1e-10);  // sqrt(1e-10) = 1e-5
///
/// // Effective values with defaults
/// let params = TruncationParams::new();
/// assert_eq!(params.effective_rtol(1e-12), 1e-12);  // uses default
/// assert_eq!(params.effective_max_rank(), usize::MAX);  // no limit
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct TruncationParams {
    /// Relative tolerance for truncation.
    ///
    /// Singular values satisfying `σ_i / σ_max < rtol` are truncated,
    /// where `σ_max` is the largest singular value.
    ///
    /// If `None`, uses the algorithm's default tolerance.
    pub rtol: Option<f64>,

    /// Maximum rank (bond dimension).
    ///
    /// If `None`, no rank limit is applied.
    pub max_rank: Option<usize>,

    /// Cutoff value (ITensorMPS.jl convention).
    ///
    /// When set via [`with_cutoff`](TruncationParams::with_cutoff), `rtol` is
    /// automatically set to `√cutoff`. This field tracks the original cutoff
    /// value for inspection; `rtol` is always the authoritative tolerance.
    ///
    /// If `None`, cutoff was not used.
    pub cutoff: Option<f64>,
}

impl TruncationParams {
    /// Create new truncation parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the relative tolerance.
    ///
    /// Clears any previously set cutoff origin.
    #[must_use]
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self.cutoff = None;
        self
    }

    /// Set the maximum rank.
    #[must_use]
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set cutoff (ITensorMPS.jl convention).
    ///
    /// Internally converted to `rtol = √cutoff`. Clears any previously set
    /// `rtol` origin so `cutoff` becomes the authoritative tolerance source.
    #[must_use]
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Some(cutoff);
        self.rtol = Some(cutoff.sqrt());
        self
    }

    /// Set maxdim (alias for [`with_max_rank`](Self::with_max_rank)).
    ///
    /// This is provided for ITensorMPS.jl compatibility.
    #[must_use]
    pub fn with_maxdim(mut self, maxdim: usize) -> Self {
        self.max_rank = Some(maxdim);
        self
    }

    /// Get the effective rtol, using the provided default if not set.
    #[must_use]
    pub fn effective_rtol(&self, default: f64) -> f64 {
        self.rtol.unwrap_or(default)
    }

    /// Get the effective max_rank, using usize::MAX if not set.
    #[must_use]
    pub fn effective_max_rank(&self) -> usize {
        self.max_rank.unwrap_or(usize::MAX)
    }

    /// Merge with another set of parameters, preferring self's values.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            rtol: self.rtol.or(other.rtol),
            max_rank: self.max_rank.or(other.max_rank),
            cutoff: self.cutoff.or(other.cutoff),
        }
    }
}

/// Trait for types that contain truncation parameters.
///
/// This trait provides a common interface for accessing and modifying
/// truncation parameters in various options structs (e.g., [`SvdOptions`](crate::SvdOptions),
/// [`QrOptions`](crate::QrOptions)).
///
/// # Examples
///
/// ```
/// use tensor4all_core::svd::SvdOptions;
/// use tensor4all_core::HasTruncationParams;
///
/// let opts = SvdOptions::with_rtol(1e-6).with_max_rank(20);
/// assert_eq!(opts.rtol(), Some(1e-6));
/// assert_eq!(opts.max_rank(), Some(20));
/// ```
pub trait HasTruncationParams {
    /// Get a reference to the truncation parameters.
    fn truncation_params(&self) -> &TruncationParams;

    /// Get a mutable reference to the truncation parameters.
    fn truncation_params_mut(&mut self) -> &mut TruncationParams;

    /// Get the rtol value.
    fn rtol(&self) -> Option<f64> {
        self.truncation_params().rtol
    }

    /// Get the max_rank value.
    fn max_rank(&self) -> Option<usize> {
        self.truncation_params().max_rank
    }

    /// Set the rtol value (builder pattern).
    ///
    /// Clears any previously set cutoff origin.
    fn with_rtol(mut self, rtol: f64) -> Self
    where
        Self: Sized,
    {
        let p = self.truncation_params_mut();
        p.rtol = Some(rtol);
        p.cutoff = None;
        self
    }

    /// Set the max_rank value (builder pattern).
    fn with_max_rank(mut self, max_rank: usize) -> Self
    where
        Self: Sized,
    {
        self.truncation_params_mut().max_rank = Some(max_rank);
        self
    }

    /// Set cutoff (ITensorMPS.jl convention, builder pattern).
    ///
    /// Internally converted to `rtol = √cutoff`.
    fn with_cutoff(mut self, cutoff: f64) -> Self
    where
        Self: Sized,
    {
        let p = self.truncation_params_mut();
        p.cutoff = Some(cutoff);
        p.rtol = Some(cutoff.sqrt());
        self
    }

    /// Set maxdim (alias for max_rank, builder pattern).
    fn with_maxdim(mut self, maxdim: usize) -> Self
    where
        Self: Sized,
    {
        self.truncation_params_mut().max_rank = Some(maxdim);
        self
    }

    /// Set cutoff via mutable reference.
    fn set_cutoff(&mut self, cutoff: f64) {
        let p = self.truncation_params_mut();
        p.cutoff = Some(cutoff);
        p.rtol = Some(cutoff.sqrt());
    }

    /// Set maxdim via mutable reference (alias for max_rank).
    fn set_maxdim(&mut self, maxdim: usize) {
        self.truncation_params_mut().max_rank = Some(maxdim);
    }
}

// Implement HasTruncationParams for TruncationParams itself
impl HasTruncationParams for TruncationParams {
    fn truncation_params(&self) -> &TruncationParams {
        self
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        self
    }
}

#[cfg(test)]
mod tests;
