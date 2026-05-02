//! Configuration options for tensor train operations.

use std::ops::Range;

use tensor4all_core::SvdTruncationPolicy;

use crate::error::{Result, TensorTrainError};

// Re-export CanonicalForm from treetn for convenience.
pub use tensor4all_treetn::algorithm::CanonicalForm;

pub(crate) fn validate_svd_truncation_options(
    max_rank: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
) -> Result<()> {
    if let Some(policy) = svd_policy {
        if !policy.threshold.is_finite() || policy.threshold < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!(
                    "svd_policy.threshold must be finite and >= 0, got {}",
                    policy.threshold
                ),
            });
        }
    }

    if let Some(max_rank) = max_rank {
        if max_rank == 0 {
            return Err(TensorTrainError::OperationError {
                message: "max_rank/maxdim must be >= 1".to_string(),
            });
        }
    }

    Ok(())
}

/// Options for tensor train truncation.
///
/// Truncation is explicitly SVD-based. Canonicalization remains the API for
/// LU/CI-style forms; truncate itself only accepts SVD truncation controls.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::TruncateOptions;
///
/// let opts = TruncateOptions::svd()
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
///     .with_max_rank(20)
///     .with_site_range(0..4);
///
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// assert_eq!(opts.max_rank(), Some(20));
/// assert_eq!(opts.site_range(), Some(0..4));
/// ```
#[derive(Debug, Clone, Default)]
pub struct TruncateOptions {
    max_rank: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    site_range: Option<Range<usize>>,
}

impl TruncateOptions {
    /// Create options for SVD-based truncation.
    pub fn svd() -> Self {
        Self::default()
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set the site range for truncation.
    ///
    /// The range is 0-indexed with exclusive end.
    pub fn with_site_range(mut self, range: Range<usize>) -> Self {
        self.site_range = Some(range);
        self
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_rank(&self) -> Option<usize> {
        self.max_rank
    }

    /// Get the site range for truncation.
    #[inline]
    pub fn site_range(&self) -> Option<Range<usize>> {
        self.site_range.clone()
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
    /// Dense/reference contraction: contract to a full tensor, then decompose back.
    /// Useful only for small debugging and testing cases; memory scales as the
    /// product of external dimensions.
    Naive,
}

/// Options for tensor train contraction.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::ContractOptions;
///
/// let opts = ContractOptions::fit()
///     .with_svd_policy(SvdTruncationPolicy::new(1e-8))
///     .with_max_rank(50)
///     .with_nsweeps(3);
///
/// assert_eq!(opts.max_rank(), Some(50));
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-8)));
/// assert_eq!(opts.nhalfsweeps(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct ContractOptions {
    method: ContractMethod,
    max_rank: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    nhalfsweeps: usize,
}

impl Default for ContractOptions {
    fn default() -> Self {
        Self {
            method: ContractMethod::default(),
            max_rank: None,
            svd_policy: None,
            nhalfsweeps: 2,
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
    pub fn naive() -> Self {
        Self {
            method: ContractMethod::Naive,
            ..Default::default()
        }
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set number of half-sweeps for fit contraction.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps.
    ///
    /// A full sweep is two half-sweeps.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nhalfsweeps = nsweeps * 2;
        self
    }

    /// Get the contraction method.
    #[inline]
    pub fn method(&self) -> ContractMethod {
        self.method
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_rank(&self) -> Option<usize> {
        self.max_rank
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
    }
}

/// Options for the linear solver.
///
/// Solves `(a₀ + a₁ * A) * x = b` using DMRG-like sweeps with local GMRES.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::LinsolveOptions;
///
/// let opts = LinsolveOptions::new(5)
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
///     .with_max_rank(64)
///     .with_krylov_tol(1e-8)
///     .with_coefficients(1.0, -1.0);
///
/// assert_eq!(opts.max_rank(), Some(64));
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// assert_eq!(opts.nhalfsweeps(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    nhalfsweeps: usize,
    max_rank: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    krylov_tol: f64,
    krylov_maxiter: usize,
    krylov_dim: usize,
    a0: f64,
    a1: f64,
    convergence_tol: Option<f64>,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nhalfsweeps: 10,
            max_rank: None,
            svd_policy: None,
            krylov_tol: 1e-10,
            krylov_maxiter: 100,
            krylov_dim: 30,
            a0: 0.0,
            a1: 1.0,
            convergence_tol: None,
        }
    }
}

impl LinsolveOptions {
    /// Create options with the specified number of full sweeps.
    pub fn new(nsweeps: usize) -> Self {
        Self {
            nhalfsweeps: nsweeps * 2,
            ..Default::default()
        }
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set number of half-sweeps.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nhalfsweeps = nsweeps * 2;
        self
    }

    /// Set GMRES tolerance.
    pub fn with_krylov_tol(mut self, tol: f64) -> Self {
        self.krylov_tol = tol;
        self
    }

    /// Set maximum GMRES iterations per local solve.
    pub fn with_krylov_maxiter(mut self, maxiter: usize) -> Self {
        self.krylov_maxiter = maxiter;
        self
    }

    /// Set Krylov subspace dimension (restart parameter).
    pub fn with_krylov_dim(mut self, dim: usize) -> Self {
        self.krylov_dim = dim;
        self
    }

    /// Set coefficients `a₀` and `a₁` in `(a₀ + a₁ * A) * x = b`.
    pub fn with_coefficients(mut self, a0: f64, a1: f64) -> Self {
        self.a0 = a0;
        self.a1 = a1;
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_rank(&self) -> Option<usize> {
        self.max_rank
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
    }

    /// Get GMRES tolerance.
    #[inline]
    pub fn krylov_tol(&self) -> f64 {
        self.krylov_tol
    }

    /// Get maximum GMRES iterations per local solve.
    #[inline]
    pub fn krylov_maxiter(&self) -> usize {
        self.krylov_maxiter
    }

    /// Get Krylov subspace dimension.
    #[inline]
    pub fn krylov_dim(&self) -> usize {
        self.krylov_dim
    }

    /// Get coefficients `(a0, a1)`.
    #[inline]
    pub fn coefficients(&self) -> (f64, f64) {
        (self.a0, self.a1)
    }

    /// Get convergence tolerance.
    #[inline]
    pub fn convergence_tol(&self) -> Option<f64> {
        self.convergence_tol
    }
}

#[cfg(test)]
mod tests;
