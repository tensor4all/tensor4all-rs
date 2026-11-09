//! Common options for linsolve algorithms.

use crate::TruncationOptions;
use tensor4all_core::{AnyScalar, SvdTruncationPolicy};

/// Residual tolerance convention for local GMRES solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmresToleranceMode {
    /// Stop when `||b - A*x|| / ||b|| < gmres_tol`.
    Relative,
    /// Stop when `||b - A*x|| < gmres_tol`.
    Absolute,
}

/// Residual-based bond enrichment for one-site ALS.
///
/// After each one-site solve on an edge `(src, dst)`, the two-site projected
/// residual `b - (a₀ + a₁ A) x` on `{src, dst}` is formed. Its component
/// outside the current `src` bond space is factorized by SVD, and up to
/// [`max_added`](Self::max_added) of its leading left singular vectors are
/// appended to the `src`–`dst` bond. The `dst` tensor receives zero
/// coefficients along the new directions, so the represented state is
/// unchanged by enrichment; the following local solve at `dst` uses them.
///
/// This lets one-site ALS reach solutions whose ranks exceed those of the
/// initial guess, at the cost of one two-site operator application and one
/// SVD per step. Two-site MALS ([`LinsolveOptions::nsite`] `= 2`) adapts ranks
/// by itself and does not accept enrichment.
///
/// Enriched bonds are never truncated, so the final state can carry
/// directions with small or zero coefficients. Truncate the result afterwards
/// when a compact representation is needed.
///
/// # Examples
/// ```
/// use tensor4all_treetn::{LinsolveOptions, ResidualEnrichment};
/// let enrichment = ResidualEnrichment::default().with_max_bond_dim(16);
/// let options = LinsolveOptions::new(4).with_nsite(1).with_enrichment(enrichment);
/// let enrichment = options.enrichment.unwrap();
/// assert_eq!(enrichment.max_added, 4);
/// assert_eq!(enrichment.cutoff, 1e-12);
/// assert_eq!(enrichment.max_bond_dim, Some(16));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualEnrichment {
    /// Maximum number of directions added to one bond per step. Must be at
    /// least 1. Default: `4`. Larger values grow ranks in fewer sweeps but
    /// enlarge every later local problem.
    pub max_added: usize,
    /// Residual directions whose singular value is at most
    /// `cutoff * ||b_local||` are not added, where `b_local` is the two-site
    /// projected right-hand side. Must be finite and nonnegative.
    /// Default: `1e-12`, which only discards numerically zero directions.
    pub cutoff: f64,
    /// Bond dimension that enrichment never exceeds. `None` (the default)
    /// limits a bond only by the dimension of the space on the `src` side.
    /// Bonds already larger than this value are left unchanged.
    pub max_bond_dim: Option<usize>,
}

impl Default for ResidualEnrichment {
    fn default() -> Self {
        Self {
            max_added: 4,
            cutoff: 1e-12,
            max_bond_dim: None,
        }
    }
}

impl ResidualEnrichment {
    /// Set the maximum number of directions added to one bond per step.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::ResidualEnrichment;
    /// assert_eq!(ResidualEnrichment::default().with_max_added(2).max_added, 2);
    /// ```
    pub fn with_max_added(mut self, max_added: usize) -> Self {
        self.max_added = max_added;
        self
    }

    /// Set the singular-value cutoff relative to the two-site local RHS norm.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::ResidualEnrichment;
    /// assert_eq!(ResidualEnrichment::default().with_cutoff(1e-8).cutoff, 1e-8);
    /// ```
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Set the bond dimension that enrichment never exceeds.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::ResidualEnrichment;
    /// assert_eq!(ResidualEnrichment::default().with_max_bond_dim(8).max_bond_dim, Some(8));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.max_added >= 1,
            "ResidualEnrichment::max_added must be at least 1"
        );
        anyhow::ensure!(
            self.cutoff.is_finite() && self.cutoff >= 0.0,
            "ResidualEnrichment::cutoff must be finite and nonnegative, got {}",
            self.cutoff
        );
        anyhow::ensure!(
            self.max_bond_dim != Some(0),
            "ResidualEnrichment::max_bond_dim must be at least 1"
        );
        Ok(())
    }
}

/// Options for one-site ALS or two-site MALS linear solves.
///
/// Two-site updates can adapt bond dimensions. One-site updates keep the
/// initial bond spaces unless [`ResidualEnrichment`] is enabled.
///
/// # Examples
/// ```
/// use tensor4all_treetn::LinsolveOptions;
/// let options = LinsolveOptions::new(3).with_nsite(1);
/// assert_eq!(options.nsite, 1);
/// assert_eq!(options.nfullsweeps, 3);
/// ```
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    /// Number of full sweeps to perform.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Number of tensors optimized at once: `1` for ALS, `2` for MALS (default).
    ///
    /// One-site ALS grows bonds only with [`Self::enrichment`]. Exact QR may
    /// remove structurally redundant dimensions. Use two-site updates or
    /// enrichment when the required ranks are unknown.
    pub nsite: usize,
    /// Residual-based bond enrichment for one-site ALS. Default: `None`.
    ///
    /// Valid only with `nsite = 1`.
    pub enrichment: Option<ResidualEnrichment>,
    /// Truncation options for two-site factorization.
    ///
    /// Explicit bond caps or SVD policies are invalid with `nsite = 1`.
    pub truncation: TruncationOptions,
    /// Tolerance for GMRES convergence.
    pub gmres_tol: f64,
    /// Residual tolerance convention for GMRES convergence.
    pub gmres_tolerance_mode: GmresToleranceMode,
    /// Maximum number of GMRES restart cycles per local solve.
    ///
    /// This matches KrylovKit's `maxiter` convention. The maximum number of
    /// operator expansion steps is roughly `gmres_max_restarts * gmres_restart_dim`.
    pub gmres_max_restarts: usize,
    /// GMRES restart cycle length.
    pub gmres_restart_dim: usize,
    /// Coefficient a₀ in (a₀ + a₁ * A) * x = b.
    pub a0: AnyScalar,
    /// Coefficient a₁ in (a₀ + a₁ * A) * x = b.
    pub a1: AnyScalar,
    /// Convergence tolerance for early termination.
    /// If Some(tol), stop when relative residual < tol.
    pub convergence_tol: Option<f64>,
    /// Whether to compute and return a final true residual after the sweep.
    ///
    /// Disabling this skips an extra operator application after the requested sweeps. A residual
    /// is still computed when `convergence_tol` is set because early stopping depends on it.
    pub check_residual: bool,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 5,
            nsite: 2,
            enrichment: None,
            truncation: TruncationOptions::default(),
            gmres_tol: 1e-10,
            gmres_tolerance_mode: GmresToleranceMode::Relative,
            gmres_max_restarts: 100,
            gmres_restart_dim: 30,
            a0: AnyScalar::new_real(0.0),
            a1: AnyScalar::new_real(1.0),
            convergence_tol: None,
            check_residual: true,
        }
    }
}

impl LinsolveOptions {
    /// Create new options with specified number of full sweeps.
    pub fn new(nfullsweeps: usize) -> Self {
        Self {
            nfullsweeps,
            ..Default::default()
        }
    }

    /// Set number of full sweeps.
    pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self {
        self.nfullsweeps = nfullsweeps;
        self
    }

    /// Choose one-site ALS (`1`) or two-site MALS (`2`, the default).
    ///
    /// One-site ALS uses the initial bond spaces without SVD truncation; it can
    /// fail to converge if those spaces are too small, unless
    /// [`Self::with_enrichment`] lets it grow them. QR can
    /// remove structurally redundant dimensions. Invalid values and one-site
    /// truncation controls are rejected by [`crate::square_linsolve`].
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::LinsolveOptions;
    /// assert_eq!(LinsolveOptions::default().with_nsite(1).nsite, 1);
    /// ```
    pub fn with_nsite(mut self, nsite: usize) -> Self {
        self.nsite = nsite;
        self
    }

    /// Enable residual-based bond enrichment for one-site ALS.
    ///
    /// Requires `nsite = 1`; [`crate::square_linsolve`] rejects enrichment
    /// with two-site updates.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::{LinsolveOptions, ResidualEnrichment};
    /// let options = LinsolveOptions::new(2)
    ///     .with_nsite(1)
    ///     .with_enrichment(ResidualEnrichment::default());
    /// assert_eq!(options.enrichment, Some(ResidualEnrichment::default()));
    /// ```
    pub fn with_enrichment(mut self, enrichment: ResidualEnrichment) -> Self {
        self.enrichment = Some(enrichment);
        self
    }

    pub(crate) fn validate_update_mode(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.nsite, 1 | 2),
            "linsolve supports nsite=1 or nsite=2, got nsite={}",
            self.nsite
        );
        anyhow::ensure!(
            self.nsite != 1
                || (self.truncation.max_bond_dim().is_none()
                    && self.truncation.svd_policy().is_none()),
            "one-site ALS does not truncate bonds; use nsite=2 for max_bond_dim or svd_policy"
        );
        if let Some(enrichment) = &self.enrichment {
            anyhow::ensure!(
                self.nsite == 1,
                "residual enrichment applies only to one-site ALS (nsite=1); \
                 two-site MALS adapts bond dimensions by itself"
            );
            enrichment.validate()?;
        }
        Ok(())
    }

    /// Set truncation options.
    pub fn with_truncation(mut self, truncation: TruncationOptions) -> Self {
        self.truncation = truncation;
        self
    }

    /// Set maximum bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.truncation = self.truncation.with_max_bond_dim(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy used by TreeTN truncation steps.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.truncation = self.truncation.with_svd_policy(policy);
        self
    }

    /// Set GMRES tolerance.
    pub fn with_gmres_tol(mut self, tol: f64) -> Self {
        self.gmres_tol = tol;
        self
    }

    /// Set the residual tolerance convention for GMRES.
    pub fn with_gmres_tolerance_mode(mut self, mode: GmresToleranceMode) -> Self {
        self.gmres_tolerance_mode = mode;
        self
    }

    /// Use relative residual convergence for GMRES.
    pub fn with_gmres_relative_tolerance(mut self) -> Self {
        self.gmres_tolerance_mode = GmresToleranceMode::Relative;
        self
    }

    /// Use absolute residual convergence for GMRES.
    pub fn with_gmres_absolute_tolerance(mut self) -> Self {
        self.gmres_tolerance_mode = GmresToleranceMode::Absolute;
        self
    }

    /// Set maximum number of GMRES restart cycles.
    ///
    /// This follows KrylovKit's `maxiter` convention.
    pub fn with_gmres_max_restarts(mut self, max_restarts: usize) -> Self {
        self.gmres_max_restarts = max_restarts;
        self
    }

    /// Set GMRES restart cycle length.
    pub fn with_gmres_restart_dim(mut self, dim: usize) -> Self {
        self.gmres_restart_dim = dim;
        self
    }

    /// Set coefficients a₀ and a₁.
    pub fn with_coefficients<A0, A1>(mut self, a0: A0, a1: A1) -> Self
    where
        A0: Into<AnyScalar>,
        A1: Into<AnyScalar>,
    {
        self.a0 = a0.into();
        self.a1 = a1.into();
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Set whether `square_linsolve` computes a final true residual.
    ///
    /// Use `false` when the caller only needs the swept solution and wants to avoid the extra
    /// post-solve operator application. The residual is still evaluated if `convergence_tol` is
    /// enabled, because it is required for early stopping.
    pub fn with_residual_check(mut self, check_residual: bool) -> Self {
        self.check_residual = check_residual;
        self
    }
}

#[cfg(test)]
mod tests;
