//! Options for linsolve algorithm.

use crate::TruncationOptions;

/// Options for the linsolve algorithm.
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    /// Number of full sweeps to perform.
    /// 
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Truncation options for factorization.
    pub truncation: TruncationOptions,
    /// Tolerance for GMRES convergence.
    pub krylov_tol: f64,
    /// Maximum GMRES iterations per local solve.
    pub krylov_maxiter: usize,
    /// Krylov subspace dimension (restart parameter).
    pub krylov_dim: usize,
    /// Coefficient a₀ in (a₀ + a₁ * A) * x = b.
    pub a0: f64,
    /// Coefficient a₁ in (a₀ + a₁ * A) * x = b.
    pub a1: f64,
    /// Convergence tolerance for early termination.
    /// If Some(tol), stop when relative residual < tol.
    pub convergence_tol: Option<f64>,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 5,
            truncation: TruncationOptions::default(),
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

    /// Set truncation options.
    pub fn with_truncation(mut self, truncation: TruncationOptions) -> Self {
        self.truncation = truncation;
        self
    }

    /// Set maximum bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.truncation = self.truncation.with_max_rank(max_rank);
        self
    }

    /// Set relative tolerance for truncation.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation = self.truncation.with_rtol(rtol);
        self
    }

    /// Set GMRES tolerance.
    pub fn with_krylov_tol(mut self, tol: f64) -> Self {
        self.krylov_tol = tol;
        self
    }

    /// Set maximum GMRES iterations.
    pub fn with_krylov_maxiter(mut self, maxiter: usize) -> Self {
        self.krylov_maxiter = maxiter;
        self
    }

    /// Set Krylov subspace dimension.
    pub fn with_krylov_dim(mut self, dim: usize) -> Self {
        self.krylov_dim = dim;
        self
    }

    /// Set coefficients a₀ and a₁.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = LinsolveOptions::default();
        assert_eq!(opts.nfullsweeps, 5);
        assert_eq!(opts.a0, 0.0);
        assert_eq!(opts.a1, 1.0);
        assert!(opts.convergence_tol.is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let opts = LinsolveOptions::new(5)
            .with_max_rank(100)
            .with_krylov_tol(1e-8)
            .with_coefficients(1.0, -1.0)
            .with_convergence_tol(1e-6);

        assert_eq!(opts.nfullsweeps, 5);
        assert_eq!(opts.truncation.max_rank(), Some(100));
        assert_eq!(opts.krylov_tol, 1e-8);
        assert_eq!(opts.a0, 1.0);
        assert_eq!(opts.a1, -1.0);
        assert_eq!(opts.convergence_tol, Some(1e-6));
    }
}
