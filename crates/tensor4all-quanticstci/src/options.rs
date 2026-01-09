//! Options for Quantics TCI interpolation.

use quanticsgrids::UnfoldingScheme;
use tensor4all_tensorci::{PivotSearchStrategy, TCI2Options};

/// Options for Quantics TCI interpolation.
///
/// This combines TCI algorithm options with quantics-specific settings.
#[derive(Debug, Clone)]
pub struct QtciOptions {
    /// Tolerance for convergence (relative)
    pub tolerance: f64,
    /// Maximum bond dimension (None = unlimited)
    pub maxbonddim: Option<usize>,
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Number of random initial pivots to add
    pub nrandominitpivot: usize,
    /// Unfolding scheme for tensor train structure
    pub unfoldingscheme: UnfoldingScheme,
    /// Whether to normalize error by max sample value
    pub normalize_error: bool,
    /// Verbosity level (0 = silent)
    pub verbosity: usize,
    /// Number of global pivots to search per iteration
    pub nsearchglobalpivot: usize,
    /// Number of random searches for global pivots
    pub nsearch: usize,
    /// Pivot search strategy
    pub pivot_search: PivotSearchStrategy,
}

impl Default for QtciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            maxbonddim: None,
            maxiter: 200,
            nrandominitpivot: 5,
            unfoldingscheme: UnfoldingScheme::Interleaved,
            normalize_error: true,
            verbosity: 0,
            nsearchglobalpivot: 5,
            nsearch: 100,
            pivot_search: PivotSearchStrategy::Full,
        }
    }
}

impl QtciOptions {
    /// Create new options with specified tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum bond dimension.
    pub fn with_maxbonddim(mut self, maxbonddim: usize) -> Self {
        self.maxbonddim = Some(maxbonddim);
        self
    }

    /// Set maximum number of iterations.
    pub fn with_maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;
        self
    }

    /// Set number of random initial pivots.
    pub fn with_nrandominitpivot(mut self, n: usize) -> Self {
        self.nrandominitpivot = n;
        self
    }

    /// Set unfolding scheme.
    pub fn with_unfoldingscheme(mut self, scheme: UnfoldingScheme) -> Self {
        self.unfoldingscheme = scheme;
        self
    }

    /// Set verbosity level.
    pub fn with_verbosity(mut self, verbosity: usize) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Set number of global pivots to search per iteration.
    pub fn with_nsearchglobalpivot(mut self, n: usize) -> Self {
        self.nsearchglobalpivot = n;
        self
    }

    /// Set number of random searches for global pivots.
    pub fn with_nsearch(mut self, n: usize) -> Self {
        self.nsearch = n;
        self
    }

    /// Set pivot search strategy.
    pub fn with_pivot_search(mut self, strategy: PivotSearchStrategy) -> Self {
        self.pivot_search = strategy;
        self
    }

    /// Convert to TCI2Options for the underlying algorithm.
    pub fn to_tci2_options(&self) -> TCI2Options {
        TCI2Options {
            tolerance: self.tolerance,
            max_iter: self.maxiter,
            max_bond_dim: self.maxbonddim.unwrap_or(usize::MAX),
            normalize_error: self.normalize_error,
            verbosity: self.verbosity,
            max_nglobal_pivot: self.nsearchglobalpivot,
            nsearch: self.nsearch,
            pivot_search: self.pivot_search,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = QtciOptions::default();
        assert!((opts.tolerance - 1e-8).abs() < 1e-15);
        assert!(opts.maxbonddim.is_none());
        assert_eq!(opts.maxiter, 200);
        assert_eq!(opts.nrandominitpivot, 5);
        assert_eq!(opts.unfoldingscheme, UnfoldingScheme::Interleaved);
        assert_eq!(opts.nsearchglobalpivot, 5);
        assert_eq!(opts.nsearch, 100);
        assert_eq!(opts.pivot_search, PivotSearchStrategy::Full);
    }

    #[test]
    fn test_builder_pattern() {
        let opts = QtciOptions::default()
            .with_tolerance(1e-6)
            .with_maxbonddim(100)
            .with_maxiter(50)
            .with_nsearchglobalpivot(10)
            .with_nsearch(200)
            .with_pivot_search(PivotSearchStrategy::Rook);

        assert!((opts.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(opts.maxbonddim, Some(100));
        assert_eq!(opts.maxiter, 50);
        assert_eq!(opts.nsearchglobalpivot, 10);
        assert_eq!(opts.nsearch, 200);
        assert_eq!(opts.pivot_search, PivotSearchStrategy::Rook);
    }

    #[test]
    fn test_to_tci2_options() {
        let opts = QtciOptions::default()
            .with_tolerance(1e-6)
            .with_maxbonddim(100)
            .with_nsearchglobalpivot(10)
            .with_nsearch(200);

        let tci_opts = opts.to_tci2_options();
        assert!((tci_opts.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(tci_opts.max_bond_dim, 100);
        assert_eq!(tci_opts.max_nglobal_pivot, 10);
        assert_eq!(tci_opts.nsearch, 200);
    }
}
