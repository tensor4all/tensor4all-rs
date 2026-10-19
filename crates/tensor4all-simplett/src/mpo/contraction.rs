//! Options shared by the MPO contraction algorithms

/// Options for contraction operations
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Relative truncation tolerance forwarded to
    /// [`FactorizeOptions::tolerance`](super::factorize::FactorizeOptions::tolerance).
    ///
    /// Singular values smaller than `tolerance * sigma_max` are discarded at
    /// each bond, where `sigma_max` is the largest singular value of that
    /// bond's matrix. The threshold is invariant under a global rescaling of
    /// the operators being contracted.
    pub tolerance: f64,
    /// Maximum bond dimension after contraction. `None` means no limit.
    pub max_bond_dim: Option<usize>,
    /// Factorization method for compression
    pub factorize_method: super::factorize::FactorizeMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: None,
            factorize_method: super::factorize::FactorizeMethod::SVD,
        }
    }
}
