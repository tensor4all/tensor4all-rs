//! Common options for linsolve algorithms.

use super::LocalSolver;
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

/// Options for one-site ALS or two-site MALS linear solves.
///
/// Two-site updates can adapt bond dimensions; one-site updates keep the initial
/// bond spaces and cannot recover a solution requiring larger ranks.
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
    /// One-site ALS does not grow bonds. Exact QR may remove structurally
    /// redundant dimensions. Use two-site updates when the required ranks are unknown.
    pub nsite: usize,
    /// Truncation options for two-site factorization.
    ///
    /// Explicit bond caps or SVD policies are invalid with `nsite = 1`.
    pub truncation: TruncationOptions,
    /// Local Krylov solver; GMRES by default. CG requires the effective
    /// operator `a0 I + a1 A` to be Hermitian positive-definite.
    pub local_solver: LocalSolver,
    /// Tolerance for GMRES convergence; ignored when another solver is selected.
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
            truncation: TruncationOptions::default(),
            local_solver: LocalSolver::default(),
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
    /// One-site ALS uses the initial bond spaces without SVD truncation or bond
    /// growth; it can fail to converge if those spaces are too small. QR can
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

    /// Select the local solver and its native options; GMRES is the default.
    ///
    /// CG requires a Hermitian positive-definite effective operator; BiCGSTAB
    /// and GMRES accept general systems. Native CG/BiCGSTAB options determine
    /// tolerances and budgets; the `gmres_*` settings apply only to GMRES.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::krylov::BiCgStabOptions;
    /// use tensor4all_treetn::{LinsolveOptions, LocalSolver};
    /// let options = LinsolveOptions::new(3)
    ///     .with_local_solver(LocalSolver::BiCgStab(BiCgStabOptions::default()));
    /// assert!(matches!(options.local_solver, LocalSolver::BiCgStab(_)));
    /// ```
    pub fn with_local_solver(mut self, local_solver: LocalSolver) -> Self {
        self.local_solver = local_solver;
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
