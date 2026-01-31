//! Configuration options for tensor train operations.

use std::ops::Range;
use tensor4all_core::truncation::{HasTruncationParams, TruncationParams};

use crate::error::{Result, TensorTrainError};

// Re-export CanonicalForm from treetn for convenience
pub use tensor4all_treetn::algorithm::CanonicalForm;

// Re-export DecompositionAlg for convenience
pub use tensor4all_core::truncation::DecompositionAlg;

pub(crate) fn validate_truncation_params(p: &TruncationParams) -> Result<()> {
    if let Some(cutoff) = p.cutoff {
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!("cutoff must be finite and >= 0, got {}", cutoff),
            });
        }
    }

    if let Some(rtol) = p.rtol {
        if !rtol.is_finite() || rtol < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!("rtol must be finite and >= 0, got {}", rtol),
            });
        }
    }

    if let Some(max_rank) = p.max_rank {
        if max_rank == 0 {
            return Err(TensorTrainError::OperationError {
                message: "max_rank/maxdim must be >= 1".to_string(),
            });
        }
    }

    Ok(())
}

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
    alg: TruncateAlg,

    /// Truncation parameters (rtol, max_rank).
    truncation: TruncationParams,

    /// Range of sites to truncate (0-indexed, exclusive end).
    ///
    /// If `None`, all bonds are truncated.
    site_range: Option<Range<usize>>,
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
    ///
    /// Clears any previously set cutoff origin.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self.truncation.cutoff = None;
        self
    }

    /// Set the maximum rank (bond dimension).
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.truncation.max_rank = Some(max_rank);
        self
    }

    /// Set cutoff (ITensorMPS.jl convention). Converted to `rtol = √cutoff`.
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.truncation.cutoff = Some(cutoff);
        self.truncation.rtol = Some(cutoff.sqrt());
        self
    }

    /// Set maxdim (alias for [`with_max_rank`](Self::with_max_rank)).
    pub fn with_maxdim(mut self, maxdim: usize) -> Self {
        self.truncation.max_rank = Some(maxdim);
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

    /// Get the truncation algorithm.
    #[inline]
    pub fn alg(&self) -> TruncateAlg {
        self.alg
    }

    /// Get truncation parameters.
    ///
    /// This returns a copy because [`TruncationParams`] is cheap and `Copy`.
    #[inline]
    pub fn truncation(&self) -> TruncationParams {
        self.truncation
    }

    /// Get the site range for truncation.
    #[inline]
    pub fn site_range(&self) -> Option<Range<usize>> {
        self.site_range.clone()
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
    method: ContractMethod,
    /// Truncation parameters (rtol, max_rank).
    truncation: TruncationParams,
    /// Number of half-sweeps for Fit method.
    ///
    /// A half-sweep visits edges in one direction only (forward or backward).
    /// This must be a multiple of 2 (each full sweep consists of 2 half-sweeps).
    nhalfsweeps: usize,
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
    ///
    /// Clears any previously set cutoff origin.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self.truncation.cutoff = None;
        self
    }

    /// Set cutoff (ITensorMPS.jl convention). Converted to `rtol = √cutoff`.
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.truncation.cutoff = Some(cutoff);
        self.truncation.rtol = Some(cutoff.sqrt());
        self
    }

    /// Set maxdim (alias for [`with_max_rank`](Self::with_max_rank)).
    pub fn with_maxdim(mut self, maxdim: usize) -> Self {
        self.truncation.max_rank = Some(maxdim);
        self
    }

    /// Set number of half-sweeps for Fit method.
    ///
    /// # Arguments
    /// * `nhalfsweeps` - Number of half-sweeps (must be a multiple of 2)
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps (ITensorMPS.jl convention).
    ///
    /// A full sweep = 2 half-sweeps (forward + backward).
    /// Equivalent to `with_nhalfsweeps(nsweeps * 2)`.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nhalfsweeps = nsweeps * 2;
        self
    }

    /// Get the contraction method.
    #[inline]
    pub fn method(&self) -> ContractMethod {
        self.method
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
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

/// Options for the linear solver.
///
/// Solves `(a₀ + a₁ * A) * x = b` using DMRG-like sweeps with local GMRES.
///
/// # Example
///
/// ```
/// use tensor4all_itensorlike::LinsolveOptions;
///
/// let opts = LinsolveOptions::default()
///     .with_nsweeps(10)
///     .with_cutoff(1e-10)
///     .with_maxdim(100)
///     .with_krylov_tol(1e-8)
///     .with_coefficients(1.0, -1.0);
/// ```
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    /// Number of half-sweeps (must be a multiple of 2).
    ///
    /// Default: 10 (= 5 full sweeps).
    nhalfsweeps: usize,
    /// Truncation parameters (rtol/cutoff, max_rank/maxdim).
    truncation: TruncationParams,
    /// Algorithm for truncation.
    alg: TruncateAlg,
    /// GMRES tolerance.
    krylov_tol: f64,
    /// Maximum GMRES iterations per local solve.
    krylov_maxiter: usize,
    /// Krylov subspace dimension (restart parameter).
    krylov_dim: usize,
    /// Coefficient a₀ in (a₀ + a₁ * A) * x = b.
    a0: f64,
    /// Coefficient a₁ in (a₀ + a₁ * A) * x = b.
    a1: f64,
    /// Convergence tolerance for early termination.
    ///
    /// If `Some(tol)`, stop when relative residual < tol.
    convergence_tol: Option<f64>,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nhalfsweeps: 10, // 5 full sweeps
            truncation: TruncationParams::default(),
            alg: TruncateAlg::SVD,
            krylov_tol: 1e-10,
            krylov_maxiter: 100,
            krylov_dim: 30,
            a0: 0.0,
            a1: 1.0,
            convergence_tol: None,
        }
    }
}

impl HasTruncationParams for LinsolveOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

impl LinsolveOptions {
    /// Create options with specified number of full sweeps.
    pub fn new(nsweeps: usize) -> Self {
        Self {
            nhalfsweeps: nsweeps * 2,
            ..Default::default()
        }
    }

    /// Set the relative tolerance for truncation.
    ///
    /// Clears any previously set cutoff origin.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self.truncation.cutoff = None;
        self
    }

    /// Set the maximum rank (bond dimension).
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.truncation.max_rank = Some(max_rank);
        self
    }

    /// Set cutoff (ITensorMPS.jl convention). Converted to `rtol = √cutoff`.
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.truncation.cutoff = Some(cutoff);
        self.truncation.rtol = Some(cutoff.sqrt());
        self
    }

    /// Set maxdim (alias for [`with_max_rank`](Self::with_max_rank)).
    pub fn with_maxdim(mut self, maxdim: usize) -> Self {
        self.truncation.max_rank = Some(maxdim);
        self
    }

    /// Set the truncation algorithm.
    pub fn with_alg(mut self, alg: TruncateAlg) -> Self {
        self.alg = alg;
        self
    }

    /// Set number of half-sweeps.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps (ITensorMPS.jl convention).
    ///
    /// A full sweep = 2 half-sweeps (forward + backward).
    /// Equivalent to `with_nhalfsweeps(nsweeps * 2)`.
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

    /// Set coefficients a₀ and a₁ in (a₀ + a₁ * A) * x = b.
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

    /// Get rtol (for convenience).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank (for convenience).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
    }

    /// Get truncation algorithm.
    #[inline]
    pub fn alg(&self) -> TruncateAlg {
        self.alg
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

    /// Get coefficients (a0, a1).
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
            .with_nhalfsweeps(6); // 6 half-sweeps = 3 full sweeps
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
    fn test_contract_options_nhalfsweeps_must_be_multiple_of_2() {
        // builder should not panic; validation happens at operation entry
        let opts = ContractOptions::fit().with_nhalfsweeps(1);
        assert_eq!(opts.nhalfsweeps(), 1);
    }

    #[test]
    fn test_contract_method_default() {
        let method: ContractMethod = Default::default();
        assert_eq!(method, ContractMethod::Zipup);
    }

    #[test]
    fn test_truncate_options_cutoff() {
        let opts = TruncateOptions::svd().with_cutoff(1e-10);
        assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
        assert_eq!(opts.truncation.cutoff, Some(1e-10));
    }

    #[test]
    fn test_truncate_options_maxdim() {
        let opts = TruncateOptions::svd().with_maxdim(42);
        assert_eq!(opts.max_rank(), Some(42));
    }

    #[test]
    fn test_contract_options_cutoff() {
        let opts = ContractOptions::zipup().with_cutoff(1e-8);
        assert!((opts.rtol().unwrap() - 1e-4).abs() < 1e-15);
        assert_eq!(opts.truncation.cutoff, Some(1e-8));
    }

    #[test]
    fn test_contract_options_maxdim() {
        let opts = ContractOptions::fit().with_maxdim(30);
        assert_eq!(opts.max_rank(), Some(30));
    }

    #[test]
    fn test_contract_options_nsweeps() {
        let opts = ContractOptions::fit().with_nsweeps(5);
        assert_eq!(opts.nhalfsweeps, 10);
    }

    #[test]
    fn test_cutoff_rtol_last_wins_truncate() {
        // cutoff then rtol: rtol wins
        let opts = TruncateOptions::svd().with_cutoff(1e-10).with_rtol(1e-3);
        assert_eq!(opts.rtol(), Some(1e-3));
        assert_eq!(opts.truncation.cutoff, None);

        // rtol then cutoff: cutoff wins
        let opts = TruncateOptions::svd().with_rtol(1e-3).with_cutoff(1e-10);
        assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
        assert_eq!(opts.truncation.cutoff, Some(1e-10));
    }

    #[test]
    fn test_linsolve_options_default() {
        let opts = LinsolveOptions::default();
        assert_eq!(opts.nhalfsweeps, 10);
        assert_eq!(opts.a0, 0.0);
        assert_eq!(opts.a1, 1.0);
        assert_eq!(opts.krylov_tol, 1e-10);
        assert_eq!(opts.krylov_maxiter, 100);
        assert_eq!(opts.krylov_dim, 30);
        assert!(opts.convergence_tol.is_none());
        assert_eq!(opts.rtol(), None);
        assert_eq!(opts.max_rank(), None);
    }

    #[test]
    fn test_linsolve_options_new() {
        let opts = LinsolveOptions::new(5);
        assert_eq!(opts.nhalfsweeps(), 10); // 5 full sweeps = 10 half-sweeps
    }

    #[test]
    fn test_linsolve_options_builder() {
        let opts = LinsolveOptions::default()
            .with_nsweeps(10)
            .with_cutoff(1e-10)
            .with_maxdim(100)
            .with_krylov_tol(1e-8)
            .with_krylov_maxiter(200)
            .with_krylov_dim(50)
            .with_coefficients(1.0, -1.0)
            .with_convergence_tol(1e-6);

        assert_eq!(opts.nhalfsweeps(), 20);
        assert!((opts.rtol().unwrap() - 1e-5).abs() < 1e-15);
        assert_eq!(opts.truncation_params().cutoff, Some(1e-10));
        assert_eq!(opts.max_rank(), Some(100));
        assert_eq!(opts.krylov_tol(), 1e-8);
        assert_eq!(opts.krylov_maxiter(), 200);
        assert_eq!(opts.krylov_dim(), 50);
        assert_eq!(opts.coefficients(), (1.0, -1.0));
        assert_eq!(opts.convergence_tol(), Some(1e-6));
    }

    #[test]
    fn test_linsolve_options_nsweeps() {
        let opts = LinsolveOptions::default().with_nsweeps(3);
        assert_eq!(opts.nhalfsweeps(), 6);
    }

    #[test]
    fn test_linsolve_options_odd_nhalfsweeps_allowed_in_builder() {
        // builder should not panic; validation happens at operation entry
        let opts = LinsolveOptions::default().with_nhalfsweeps(3);
        assert_eq!(opts.nhalfsweeps(), 3);
    }
}
