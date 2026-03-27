//! TensorCI2 - Two-site Tensor Cross Interpolation algorithm
//!
//! This implements the TCI2 algorithm which uses two-site updates for
//! more efficient convergence. Unlike TCI1, it supports batch evaluation
//! of function values through an explicit batch function parameter.

use crate::error::{Result, TCIError};
use crate::globalpivot::{DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput};
use rand::SeedableRng;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use tensor4all_simplett::{tensor3_zeros, TTScalar, Tensor3, Tensor3Ops, TensorTrain};
use tensor4all_tcicore::matrix::zeros;
use tensor4all_tcicore::MultiIndex;
use tensor4all_tcicore::Scalar;
use tensor4all_tcicore::{
    rrlu, AbstractMatrixCI, CrossFactors, DenseFaerLuKernel, DenseMatrixSource,
    LazyBlockRookKernel, LazyMatrixSource, MatrixLUCI, PivotKernel, PivotKernelOptions,
    RrLUOptions,
};

/// Options for TCI2 algorithm
#[derive(Debug, Clone)]
pub struct TCI2Options {
    /// Tolerance for convergence.
    ///
    /// When `normalize_error` is enabled, this is applied to the normalized
    /// bond error. `Full` search normalizes by the maximum value seen while
    /// materializing the full candidate matrix. `Rook` search normalizes by the
    /// maximum value observed through the lazily requested matrix entries so the
    /// block-rook path stays lazy.
    pub tolerance: f64,
    /// Maximum number of iterations (half-sweeps)
    pub max_iter: usize,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Pivot search strategy
    pub pivot_search: PivotSearchStrategy,
    /// Whether to normalize error by the maximum observed sample value.
    ///
    /// `Full` and `Rook` search share the same formula but not the same
    /// observation set: `Rook` only sees entries requested by lazy block-rook
    /// search and factor reconstruction.
    pub normalize_error: bool,
    /// Verbosity level
    pub verbosity: usize,
    /// Maximum number of global pivots to add per iteration
    pub max_nglobal_pivot: usize,
    /// Number of random searches for global pivots
    pub nsearch: usize,
    /// Sweep strategy for 2-site sweeps
    pub sweep_strategy: Sweep2Strategy,
    /// Number of iterations to check for convergence history
    pub ncheck_history: usize,
    /// Whether to use strictly nested index sets
    pub strictly_nested: bool,
    /// Tolerance margin for global pivot search
    pub tol_margin_global_search: f64,
    /// Random seed (None = use thread_rng)
    pub seed: Option<u64>,
}

impl Default for TCI2Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: usize::MAX,
            pivot_search: PivotSearchStrategy::Full,
            normalize_error: true,
            verbosity: 0,
            max_nglobal_pivot: 5,
            nsearch: 5,
            sweep_strategy: Sweep2Strategy::BackAndForth,
            ncheck_history: 3,
            strictly_nested: false,
            tol_margin_global_search: 10.0,
            seed: None,
        }
    }
}

/// Pivot search strategy for TCI2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PivotSearchStrategy {
    /// Full search: evaluate entire Pi matrix
    #[default]
    Full,
    /// Rook search: use lazy block-rook pivoting over partial matrix blocks.
    ///
    /// This avoids materializing the full candidate matrix, so error
    /// normalization uses the maximum sample value observed through the lazy
    /// requests instead of a full-grid scan.
    Rook,
}

/// Sweep strategy for 2-site sweeps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sweep2Strategy {
    /// Forward sweep only
    Forward,
    /// Backward sweep only
    Backward,
    /// Alternate between forward and backward sweeps
    #[default]
    BackAndForth,
}

/// TensorCI2 - Two-site Tensor Cross Interpolation
///
/// This structure represents a tensor train constructed using the TCI2 algorithm.
/// TCI2 uses two-site updates which can be more efficient than TCI1 for some functions.
#[derive(Debug, Clone)]
pub struct TensorCI2<T: Scalar + TTScalar> {
    /// Index sets I for each site
    i_set: Vec<Vec<MultiIndex>>,
    /// Index sets J for each site
    j_set: Vec<Vec<MultiIndex>>,
    /// Local dimensions
    local_dims: Vec<usize>,
    /// Site tensors (3-leg tensors)
    site_tensors: Vec<Tensor3<T>>,
    /// Pivot errors during back-truncation
    pivot_errors: Vec<f64>,
    /// Bond errors from 2-site sweep
    bond_errors: Vec<f64>,
    /// Maximum observed sample value found during function evaluation
    max_sample_value: f64,
    /// History of I-sets for non-strictly-nested mode
    i_set_history: Vec<Vec<Vec<MultiIndex>>>,
    /// History of J-sets for non-strictly-nested mode
    j_set_history: Vec<Vec<Vec<MultiIndex>>>,
}

impl<T> TensorCI2<T>
where
    T: Scalar + TTScalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    DenseFaerLuKernel: PivotKernel<T>,
    LazyBlockRookKernel: PivotKernel<T>,
{
    /// Create a new empty TensorCI2
    pub fn new(local_dims: Vec<usize>) -> Result<Self> {
        if local_dims.len() < 2 {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims should have at least 2 elements".to_string(),
            });
        }

        let n = local_dims.len();
        Ok(Self {
            i_set: (0..n).map(|_| Vec::new()).collect(),
            j_set: (0..n).map(|_| Vec::new()).collect(),
            local_dims: local_dims.clone(),
            site_tensors: local_dims.iter().map(|&d| tensor3_zeros(0, d, 0)).collect(),
            pivot_errors: Vec::new(),
            bond_errors: vec![0.0; n.saturating_sub(1)],
            max_sample_value: 0.0,
            i_set_history: Vec::new(),
            j_set_history: Vec::new(),
        })
    }

    /// Number of sites
    pub fn len(&self) -> usize {
        self.local_dims.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.local_dims.is_empty()
    }

    /// Get local dimensions
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Get current rank (maximum bond dimension)
    pub fn rank(&self) -> usize {
        if self.len() <= 1 {
            return if self.i_set.is_empty() || self.i_set[0].is_empty() {
                0
            } else {
                1
            };
        }
        self.i_set
            .iter()
            .skip(1)
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
    }

    /// Get bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        self.i_set.iter().skip(1).map(|s| s.len()).collect()
    }

    /// Get the maximum observed sample value seen so far
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Get maximum bond error
    pub fn max_bond_error(&self) -> f64 {
        self.bond_errors.iter().cloned().fold(0.0, f64::max)
    }

    /// Get pivot errors from back-truncation
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }

    /// Check if site tensors are available
    pub fn is_site_tensors_available(&self) -> bool {
        self.site_tensors
            .iter()
            .all(|t| t.left_dim() > 0 || t.right_dim() > 0)
    }

    /// Get site tensor at position p
    pub fn site_tensor(&self, p: usize) -> &Tensor3<T> {
        &self.site_tensors[p]
    }

    /// Convert to TensorTrain
    pub fn to_tensor_train(&self) -> Result<TensorTrain<T>> {
        let tensors = self.site_tensors.clone();
        TensorTrain::new(tensors).map_err(TCIError::TensorTrainError)
    }

    /// Add global pivots to the TCI
    pub fn add_global_pivots(&mut self, pivots: &[MultiIndex]) -> Result<()> {
        for pivot in pivots {
            if pivot.len() != self.len() {
                return Err(TCIError::DimensionMismatch {
                    message: format!(
                        "Pivot length ({}) must match number of sites ({})",
                        pivot.len(),
                        self.len()
                    ),
                });
            }

            // Add to I and J sets
            for p in 0..self.len() {
                let i_indices: MultiIndex = pivot[0..p].to_vec();
                let j_indices: MultiIndex = pivot[p + 1..].to_vec();

                if !self.i_set[p].contains(&i_indices) {
                    self.i_set[p].push(i_indices);
                }
                if !self.j_set[p].contains(&j_indices) {
                    self.j_set[p].push(j_indices);
                }
            }
        }

        // Invalidate site tensors after adding pivots
        self.invalidate_site_tensors();

        Ok(())
    }

    /// Get I-set at site p
    pub fn i_set(&self, p: usize) -> &[MultiIndex] {
        &self.i_set[p]
    }

    /// Get J-set at site p
    pub fn j_set(&self, p: usize) -> &[MultiIndex] {
        &self.j_set[p]
    }

    /// Invalidate all site tensors
    pub fn invalidate_site_tensors(&mut self) {
        for p in 0..self.len() {
            self.site_tensors[p] = tensor3_zeros(0, self.local_dims[p], 0);
        }
    }

    /// Flush pivot errors (reset to empty)
    pub fn flush_pivot_errors(&mut self) {
        self.pivot_errors.clear();
    }

    /// Perform one 2-site sweep.
    ///
    /// This is a public wrapper around the internal `update_pivots` logic,
    /// suitable for calling from C-API.
    pub fn sweep2site<F, B>(
        &mut self,
        f: &F,
        batched_f: &Option<B>,
        forward: bool,
        options: &TCI2Options,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
        B: Fn(&[MultiIndex]) -> Vec<T>,
    {
        let n = self.len();
        self.invalidate_site_tensors();
        self.flush_pivot_errors();

        let empty: Vec<MultiIndex> = Vec::new();
        if forward {
            for b in 0..n - 1 {
                update_pivots(self, b, f, batched_f, true, options, &empty, &empty)?;
            }
        } else {
            for b in (0..n - 1).rev() {
                update_pivots(self, b, f, batched_f, false, options, &empty, &empty)?;
            }
        }

        // Fill site tensors after sweep
        self.fill_site_tensors(f)?;
        Ok(())
    }

    /// Update pivot errors with element-wise max
    fn update_pivot_errors(&mut self, errors: &[f64]) {
        if self.pivot_errors.len() < errors.len() {
            self.pivot_errors.resize(errors.len(), 0.0);
        }
        for (i, &e) in errors.iter().enumerate() {
            self.pivot_errors[i] = self.pivot_errors[i].max(e);
        }
    }

    /// Evaluate function at all combinations of left indices × local index × right indices.
    ///
    /// Returns a 3D array of shape (len(i_indices), local_dim, len(j_indices)).
    fn fill_tensor<F>(
        &self,
        f: &F,
        i_indices: &[MultiIndex],
        j_indices: &[MultiIndex],
        local_dim: usize,
        site: usize,
    ) -> Tensor3<T>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let ni = i_indices.len();
        let nj = j_indices.len();
        let mut tensor = tensor3_zeros(ni, local_dim, nj);
        for (ii, i_multi) in i_indices.iter().enumerate() {
            for s in 0..local_dim {
                for (jj, j_multi) in j_indices.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.push(s);
                    full_idx.extend(j_multi.iter().cloned());
                    debug_assert_eq!(
                        full_idx.len(),
                        self.local_dims.len(),
                        "fill_tensor: full_idx length {} != n_sites {} at site {}",
                        full_idx.len(),
                        self.local_dims.len(),
                        site
                    );
                    let val = f(&full_idx);
                    tensor.set3(ii, s, jj, val);
                }
            }
        }
        tensor
    }

    /// Perform a 1-site sweep, updating I/J sets and optionally site tensors.
    ///
    /// This is used for cleanup after adding global pivots, and for computing
    /// canonical site tensors.
    ///
    /// Port of Julia's `sweep1site!` from `tensorci2.jl`.
    pub fn sweep1site<F>(
        &mut self,
        f: &F,
        forward: bool,
        rel_tol: f64,
        abs_tol: f64,
        max_bond_dim: usize,
        update_tensors: bool,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.flush_pivot_errors();
        self.invalidate_site_tensors();

        let n = self.len();

        if forward {
            for b in 0..n - 1 {
                self.sweep1site_at_bond(
                    f,
                    b,
                    true,
                    rel_tol,
                    abs_tol,
                    max_bond_dim,
                    update_tensors,
                )?;
            }
        } else {
            for b in (1..n).rev() {
                self.sweep1site_at_bond(
                    f,
                    b,
                    false,
                    rel_tol,
                    abs_tol,
                    max_bond_dim,
                    update_tensors,
                )?;
            }
        }

        // Update last tensor according to last index set
        if update_tensors {
            let last_idx = if forward { n - 1 } else { 0 };
            let tensor = self.fill_tensor(
                f,
                &self.i_set[last_idx].clone(),
                &self.j_set[last_idx].clone(),
                self.local_dims[last_idx],
                last_idx,
            );
            self.site_tensors[last_idx] = tensor;
        }

        Ok(())
    }

    /// Process one bond during 1-site sweep.
    fn sweep1site_at_bond<F>(
        &mut self,
        f: &F,
        b: usize,
        forward: bool,
        rel_tol: f64,
        abs_tol: f64,
        max_bond_dim: usize,
        update_tensors: bool,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        // Build combined indices: for forward, Kronecker(I_b, local_b) × J_b
        //                         for backward, I_b × Kronecker(local_b, J_b)
        let (is, js) = if forward {
            (self.kronecker_i(b), self.j_set[b].clone())
        } else {
            (self.i_set[b].clone(), self.kronecker_j(b))
        };

        if is.is_empty() || js.is_empty() {
            return Ok(());
        }

        // Build Pi matrix by evaluating function at all (I, J) combinations
        let ni = is.len();
        let nj = js.len();
        let mut pi = zeros(ni, nj);
        for (i, i_multi) in is.iter().enumerate() {
            for (j, j_multi) in js.iter().enumerate() {
                let mut full_idx = i_multi.clone();
                full_idx.extend(j_multi.iter().cloned());
                let val = f(&full_idx);
                pi[[i, j]] = val;
                let abs_val = f64::sqrt(Scalar::abs_sq(val));
                if abs_val > self.max_sample_value {
                    self.max_sample_value = abs_val;
                }
            }
        }

        // LU-based cross interpolation
        let lu_options = RrLUOptions {
            max_rank: max_bond_dim,
            rel_tol,
            abs_tol,
            left_orthogonal: forward,
        };
        let luci = MatrixLUCI::from_matrix(&pi, Some(lu_options))?;

        let row_indices = luci.row_indices();
        let col_indices = luci.col_indices();

        // Update I/J sets
        if forward {
            self.i_set[b + 1] = row_indices.iter().map(|&i| is[i].clone()).collect();
            self.j_set[b] = col_indices.iter().map(|&j| js[j].clone()).collect();
        } else {
            self.i_set[b] = row_indices.iter().map(|&i| is[i].clone()).collect();
            self.j_set[b - 1] = col_indices.iter().map(|&j| js[j].clone()).collect();
        }

        // Update site tensor
        if update_tensors {
            let mat = if forward { luci.left() } else { luci.right() };
            let local_dim = self.local_dims[b];
            if forward {
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let right_dim = luci.rank().max(1);
                let mut tensor = tensor3_zeros(left_dim, local_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..local_dim {
                        for r in 0..right_dim {
                            let row = l * local_dim + s;
                            if row < mat.nrows() && r < mat.ncols() {
                                tensor.set3(l, s, r, mat[[row, r]]);
                            }
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            } else {
                let left_dim = luci.rank().max(1);
                let right_dim = if b == self.len() - 1 {
                    1
                } else {
                    self.j_set[b].len()
                };
                let mut tensor = tensor3_zeros(left_dim, local_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..local_dim {
                        for r in 0..right_dim {
                            let col = s * right_dim + r;
                            if l < mat.nrows() && col < mat.ncols() {
                                tensor.set3(l, s, r, mat[[l, col]]);
                            }
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            }
        }

        // Update errors
        let errors = luci.pivot_errors();
        if !errors.is_empty() {
            let bond_idx = if forward { b } else { b - 1 };
            self.bond_errors[bond_idx] = *errors.last().unwrap_or(&0.0);
        }
        self.update_pivot_errors(&errors);

        Ok(())
    }

    /// Fill all site tensors using 1-site LU decomposition at each bond.
    ///
    /// For each site b (except the last), computes the Pi matrix
    /// (Kronecker(I_b, d_b) × J_b) and the pivot matrix P (I_{b+1} × J_b),
    /// then solves P^T \ Pi^T to get the site tensor T_b = Pi * P^{-1}.
    /// The last site tensor is set by direct evaluation.
    ///
    /// Port of Julia's `fillsitetensors!` / `setsitetensor!`.
    pub fn fill_site_tensors<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let n = self.len();
        for b in 0..n {
            let i_kron = self.kronecker_i(b);
            let j_set_b = self.j_set[b].clone();

            if i_kron.is_empty() || j_set_b.is_empty() {
                continue;
            }

            // Pi1: evaluate f at (Kronecker(I_b, d_b), J_b)
            let ni = i_kron.len();
            let nj = j_set_b.len();
            let mut pi1 = zeros(ni, nj);
            for (i, i_multi) in i_kron.iter().enumerate() {
                for (j, j_multi) in j_set_b.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    pi1[[i, j]] = f(&full_idx);
                }
            }

            if b == n - 1 {
                // Last site: store Pi1 directly
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let site_dim = self.local_dims[b];
                let right_dim = 1; // last site
                let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        let row = l * site_dim + s;
                        if row < ni {
                            tensor.set3(l, s, 0, pi1[[row, 0]]);
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            } else {
                // Non-last site: solve P^T \ Pi1^T to get Tmat = Pi1 * P^{-1}
                // P = pivot matrix (I_{b+1} × J_b)
                let i_set_bp1 = self.i_set[b + 1].clone();
                let np = i_set_bp1.len();

                let mut p_mat = zeros(np, nj);
                for (i, i_multi) in i_set_bp1.iter().enumerate() {
                    for (j, j_multi) in j_set_b.iter().enumerate() {
                        let mut full_idx = i_multi.clone();
                        full_idx.extend(j_multi.iter().cloned());
                        p_mat[[i, j]] = f(&full_idx);
                    }
                }

                // Solve P * X^T = Pi1^T via LU factorization
                // First transpose P and Pi1
                let mut p_t = zeros(nj, np);
                for i in 0..np {
                    for j in 0..nj {
                        p_t[[j, i]] = p_mat[[i, j]];
                    }
                }
                let mut pi1_t = zeros(nj, ni);
                for i in 0..ni {
                    for j in 0..nj {
                        pi1_t[[j, i]] = pi1[[i, j]];
                    }
                }

                // LU factorize P^T with full pivoting
                let lu = rrlu(&p_t, None)?;
                let l_mat = lu.left(true);
                let u_mat = lu.right(true);

                // Solve L*U * X_t = Pi1^T
                let x_t = tensor4all_tcicore::matrixlu::solve_lu(&l_mat, &u_mat, &pi1_t)?;

                // X = X_t^T → shape (ni, np) = (|I_b|*d_b, |I_{b+1}|)
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let site_dim = self.local_dims[b];
                let right_dim = np; // = |I_{b+1}|
                let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            let row = l * site_dim + s;
                            tensor.set3(l, s, r, x_t[[r, row]]);
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            }
        }
        Ok(())
    }

    /// Make the TCI canonical by performing 3 one-site sweeps.
    ///
    /// 1. Forward sweep (exact, no truncation)
    /// 2. Backward sweep (with truncation)
    /// 3. Forward sweep (with truncation + update tensors)
    ///
    /// Port of Julia's `makecanonical!`.
    pub fn make_canonical<F>(
        &mut self,
        f: &F,
        rel_tol: f64,
        abs_tol: f64,
        max_bond_dim: usize,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        // First half-sweep: exact, no truncation
        self.sweep1site(f, true, 0.0, 0.0, usize::MAX, false)?;
        // Second half-sweep: backward with truncation
        self.sweep1site(f, false, rel_tol, abs_tol, max_bond_dim, false)?;
        // Third half-sweep: forward with truncation and tensor updates
        self.sweep1site(f, true, rel_tol, abs_tol, max_bond_dim, true)?;
        Ok(())
    }

    /// Expand indices by Kronecker product with local dimension
    fn kronecker_i(&self, p: usize) -> Vec<MultiIndex> {
        let mut result = Vec::new();
        for i_multi in &self.i_set[p] {
            for local_idx in 0..self.local_dims[p] {
                let mut new_idx = i_multi.clone();
                new_idx.push(local_idx);
                result.push(new_idx);
            }
        }
        result
    }

    fn kronecker_j(&self, p: usize) -> Vec<MultiIndex> {
        let mut result = Vec::new();
        for local_idx in 0..self.local_dims[p] {
            for j_multi in &self.j_set[p] {
                let mut new_idx = vec![local_idx];
                new_idx.extend(j_multi.iter().cloned());
                result.push(new_idx);
            }
        }
        result
    }
}

/// Check convergence based on history of ranks, errors, and global pivots.
///
/// Port of Julia's `convergencecriterion`.
fn convergence_criterion(
    ranks: &[usize],
    errors: &[f64],
    nglobal_pivots: &[usize],
    tolerance: f64,
    max_bond_dim: usize,
    ncheck_history: usize,
) -> bool {
    if errors.len() < ncheck_history {
        return false;
    }

    let n = errors.len();
    let last_errors = &errors[n - ncheck_history..];
    let last_ranks = &ranks[n - ncheck_history..];
    let last_ngp = &nglobal_pivots[n - ncheck_history..];

    let errors_converged = last_errors.iter().all(|&e| e < tolerance);
    let no_global_pivots = last_ngp.iter().all(|&n| n == 0);
    let rank_stable =
        last_ranks.iter().min().copied().unwrap_or(0) == last_ranks.last().copied().unwrap_or(0);
    let at_max_bond = last_ranks.iter().all(|&r| r >= max_bond_dim);

    (errors_converged && no_global_pivots && rank_stable) || at_max_bond
}

/// Cross interpolate a function using TCI2 algorithm
///
/// This matches Julia's `crossinterpolate2` / `optimize!` behavior:
/// - 2-site sweeps with global pivot search
/// - History-based convergence criterion
/// - Final 1-site sweep for cleanup
///
/// # Arguments
/// * `f` - Function to interpolate, takes a multi-index and returns a value
/// * `batched_f` - Optional batch evaluation function for efficiency
/// * `local_dims` - Local dimensions for each site
/// * `initial_pivots` - Initial pivot points
/// * `options` - Algorithm options
///
/// # Returns
/// * `TensorCI2` - The constructed tensor cross interpolation
/// * `Vec<usize>` - Ranks at each iteration
/// * `Vec<f64>` - Errors at each iteration
pub fn crossinterpolate2<T, F, B>(
    f: F,
    batched_f: Option<B>,
    local_dims: Vec<usize>,
    initial_pivots: Vec<MultiIndex>,
    options: TCI2Options,
) -> Result<(TensorCI2<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    DenseFaerLuKernel: PivotKernel<T>,
    LazyBlockRookKernel: PivotKernel<T>,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    if local_dims.len() < 2 {
        return Err(TCIError::DimensionMismatch {
            message: "local_dims should have at least 2 elements".to_string(),
        });
    }

    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = TensorCI2::new(local_dims)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value
    for pivot in &pivots {
        let value = f(pivot);
        let abs_val = f64::sqrt(Scalar::abs_sq(value));
        if abs_val > tci.max_sample_value {
            tci.max_sample_value = abs_val;
        }
    }

    if tci.max_sample_value < 1e-30 {
        return Err(TCIError::InvalidPivot {
            message: "Initial pivots have zero function values".to_string(),
        });
    }

    let n = tci.len();
    let mut errors = Vec::new();
    let mut ranks = Vec::new();
    let mut nglobal_pivots_history: Vec<usize> = Vec::new();

    // Create RNG
    let mut rng = if let Some(seed) = options.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_os_rng()
    };

    // Create global pivot finder
    let finder = DefaultGlobalPivotFinder::new(
        options.nsearch,
        options.max_nglobal_pivot,
        options.tol_margin_global_search,
    );

    // Main optimization loop
    for iter in 0..options.max_iter {
        let error_normalization = if options.normalize_error && tci.max_sample_value > 0.0 {
            tci.max_sample_value
        } else {
            1.0
        };
        let abs_tol = options.tolerance * error_normalization;

        // Determine sweep direction
        let is_forward = match options.sweep_strategy {
            Sweep2Strategy::Forward => true,
            Sweep2Strategy::Backward => false,
            Sweep2Strategy::BackAndForth => iter % 2 == 0,
        };

        // Get extra index sets from history for non-strictly-nested mode
        let (extra_i_set, extra_j_set) =
            if !options.strictly_nested && !tci.i_set_history.is_empty() {
                let last = tci.i_set_history.len() - 1;
                (
                    tci.i_set_history[last].clone(),
                    tci.j_set_history[last].clone(),
                )
            } else {
                let empty: Vec<Vec<MultiIndex>> = (0..n).map(|_| Vec::new()).collect();
                (empty.clone(), empty)
            };

        // Save current sets to history
        tci.i_set_history.push(tci.i_set.clone());
        tci.j_set_history.push(tci.j_set.clone());

        // 2-site sweep
        tci.invalidate_site_tensors();
        tci.flush_pivot_errors();

        if is_forward {
            for b in 0..n - 1 {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    &batched_f,
                    true,
                    &options,
                    &extra_i_set[b + 1],
                    &extra_j_set[b],
                )?;
            }
        } else {
            for b in (0..n - 1).rev() {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    &batched_f,
                    false,
                    &options,
                    &extra_i_set[b + 1],
                    &extra_j_set[b],
                )?;
            }
        }

        // Fill site tensors after sweep
        tci.fill_site_tensors(&f)?;

        // Record error
        let error = tci.max_bond_error();
        let error_normalized = error / error_normalization;
        errors.push(error_normalized);

        // Global pivot search
        let tt = tci.to_tensor_train()?;
        let input = GlobalPivotSearchInput {
            local_dims: tci.local_dims.clone(),
            current_tt: tt,
            max_sample_value: tci.max_sample_value,
            i_set: tci.i_set.clone(),
            j_set: tci.j_set.clone(),
        };

        let global_pivots = finder.find_global_pivots(&input, &f, abs_tol, &mut rng);
        let n_global = global_pivots.len();
        tci.add_global_pivots(&global_pivots)?;
        nglobal_pivots_history.push(n_global);

        ranks.push(tci.rank());

        if options.verbosity > 0 {
            println!(
                "iteration = {}, rank = {}, error = {:.2e}, maxsamplevalue = {:.2e}, nglobalpivot = {}",
                iter + 1,
                tci.rank(),
                error_normalized,
                tci.max_sample_value,
                n_global
            );
        }

        // Check convergence
        if convergence_criterion(
            &ranks,
            &errors,
            &nglobal_pivots_history,
            abs_tol,
            options.max_bond_dim,
            options.ncheck_history,
        ) {
            break;
        }
    }

    // Final 1-site sweep to:
    // 1. Remove unnecessary pivots added by global pivots
    // 2. Compute site tensors
    let error_normalization = if options.normalize_error && tci.max_sample_value > 0.0 {
        tci.max_sample_value
    } else {
        1.0
    };
    let abs_tol = options.tolerance * error_normalization;
    tci.sweep1site(&f, true, 1e-14, abs_tol, options.max_bond_dim, true)?;

    // Normalize errors for return
    let normalized_errors: Vec<f64> = errors.iter().copied().collect();

    Ok((tci, ranks, normalized_errors))
}

/// Update pivots at bond b using LU-based cross interpolation
fn update_pivots<T, F, B>(
    tci: &mut TensorCI2<T>,
    b: usize,
    f: &F,
    batched_f: &Option<B>,
    left_orthogonal: bool,
    options: &TCI2Options,
    extra_i_set: &[MultiIndex],
    extra_j_set: &[MultiIndex],
) -> Result<()>
where
    T: Scalar + TTScalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    DenseFaerLuKernel: PivotKernel<T>,
    LazyBlockRookKernel: PivotKernel<T>,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    // Build combined index sets, including extra sets from history
    let mut i_combined = tci.kronecker_i(b);
    let mut j_combined = tci.kronecker_j(b + 1);

    // Union with extra sets (for non-strictly-nested mode)
    for extra in extra_i_set {
        if !i_combined.contains(extra) {
            i_combined.push(extra.clone());
        }
    }
    for extra in extra_j_set {
        if !j_combined.contains(extra) {
            j_combined.push(extra.clone());
        }
    }

    if i_combined.is_empty() || j_combined.is_empty() {
        return Ok(());
    }

    // Apply LU-based cross interpolation
    let lu_options = PivotKernelOptions {
        max_rank: options.max_bond_dim,
        rel_tol: options.tolerance,
        abs_tol: 0.0,
        left_orthogonal,
    };

    let selection;
    let factors;
    if options.pivot_search == PivotSearchStrategy::Full {
        let mut pi = zeros(i_combined.len(), j_combined.len());

        if let Some(ref batch_fn) = batched_f {
            let mut all_indices: Vec<MultiIndex> =
                Vec::with_capacity(i_combined.len() * j_combined.len());
            for i_multi in &i_combined {
                for j_multi in &j_combined {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    all_indices.push(full_idx);
                }
            }

            let values = batch_fn(&all_indices);
            if values.len() != all_indices.len() {
                return Err(callback_length_mismatch(values.len(), all_indices.len()));
            }
            let mut idx = 0;
            for i in 0..i_combined.len() {
                for j in 0..j_combined.len() {
                    pi[[i, j]] = values[idx];
                    update_max_sample_value(tci, values[idx]);
                    idx += 1;
                }
            }
        } else {
            for (i, i_multi) in i_combined.iter().enumerate() {
                for (j, j_multi) in j_combined.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    let value = f(&full_idx);
                    pi[[i, j]] = value;
                    update_max_sample_value(tci, value);
                }
            }
        }

        let mut data = Vec::with_capacity(pi.nrows() * pi.ncols());
        for col in 0..pi.ncols() {
            for row in 0..pi.nrows() {
                data.push(pi[[row, col]]);
            }
        }
        let source = DenseMatrixSource::from_column_major(&data, pi.nrows(), pi.ncols());
        selection = DenseFaerLuKernel.factorize(&source, &lu_options)?;
        factors = CrossFactors::from_source(&source, &selection)?;
    } else {
        let evaluator =
            LazyPiEvaluator::new(&i_combined, &j_combined, f, batched_f, tci.max_sample_value);
        let source = LazyMatrixSource::new(
            i_combined.len(),
            j_combined.len(),
            |rows, cols, out: &mut [T]| {
                evaluator.fill_block(rows, cols, out);
            },
        );
        let selection_result = LazyBlockRookKernel.factorize(&source, &lu_options);
        if let Some(err) = evaluator.take_error() {
            return Err(err);
        }
        selection = selection_result?;

        let factors_result = CrossFactors::from_source(&source, &selection);
        if let Some(err) = evaluator.take_error() {
            return Err(err);
        }
        factors = factors_result?;
        tci.max_sample_value = evaluator.sampled_max();
    }

    // Update I and J sets
    let row_indices = &selection.row_indices;
    let col_indices = &selection.col_indices;

    tci.i_set[b + 1] = row_indices.iter().map(|&i| i_combined[i].clone()).collect();
    tci.j_set[b] = col_indices.iter().map(|&j| j_combined[j].clone()).collect();

    // Skip site tensor update if extra sets were used (tensors will be
    // filled separately by fill_site_tensors after the sweep).
    if !extra_i_set.is_empty() || !extra_j_set.is_empty() {
        // Update bond error only
        let errors = &selection.pivot_errors;
        if !errors.is_empty() {
            tci.bond_errors[b] = *errors.last().unwrap_or(&0.0);
        }
        return Ok(());
    }

    // Update site tensors
    let left = if left_orthogonal {
        factors.cols_times_pivot_inv()?
    } else {
        factors.pivot_cols.clone()
    };
    let right = if left_orthogonal {
        factors.pivot_rows.clone()
    } else {
        factors.pivot_inv_times_rows()?
    };

    // Convert left matrix to tensor at site b
    let left_dim = if b == 0 { 1 } else { tci.i_set[b].len() };
    let site_dim_b = tci.local_dims[b];
    let new_bond_dim = selection.rank.max(1);

    let mut tensor_b = tensor3_zeros(left_dim, site_dim_b, new_bond_dim);
    for l in 0..left_dim {
        for s in 0..site_dim_b {
            for r in 0..new_bond_dim {
                let row = l * site_dim_b + s;
                if row < left.nrows() && r < left.ncols() {
                    tensor_b.set3(l, s, r, left[[row, r]]);
                }
            }
        }
    }
    tci.site_tensors[b] = tensor_b;

    // Convert right matrix to tensor at site b+1
    let site_dim_bp1 = tci.local_dims[b + 1];
    let right_dim = if b + 1 == tci.len() - 1 {
        1
    } else {
        tci.j_set[b + 1].len()
    };

    let mut tensor_bp1 = tensor3_zeros(new_bond_dim, site_dim_bp1, right_dim);
    for l in 0..new_bond_dim {
        for s in 0..site_dim_bp1 {
            for r in 0..right_dim {
                let col = s * right_dim + r;
                if l < right.nrows() && col < right.ncols() {
                    tensor_bp1.set3(l, s, r, right[[l, col]]);
                }
            }
        }
    }
    tci.site_tensors[b + 1] = tensor_bp1;

    // Update bond error
    if !selection.pivot_errors.is_empty() {
        tci.bond_errors[b] = *selection.pivot_errors.last().unwrap_or(&0.0);
    }

    Ok(())
}

fn update_max_sample_value<T: Scalar + TTScalar>(tci: &mut TensorCI2<T>, value: T) {
    let abs_val = f64::sqrt(Scalar::abs_sq(value));
    if abs_val > tci.max_sample_value {
        tci.max_sample_value = abs_val;
    }
}

fn build_full_index(
    i_combined: &[MultiIndex],
    j_combined: &[MultiIndex],
    row: usize,
    col: usize,
) -> MultiIndex {
    let mut full_idx = i_combined[row].clone();
    full_idx.extend(j_combined[col].iter().cloned());
    full_idx
}

fn callback_length_mismatch(actual: usize, expected: usize) -> TCIError {
    TCIError::InvalidOperation {
        message: format!(
            "batch callback returned {actual} values for {expected} requested entries"
        ),
    }
}

struct LazyPiEvaluator<'a, T, F, B>
where
    T: Scalar + TTScalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    i_combined: &'a [MultiIndex],
    j_combined: &'a [MultiIndex],
    f: &'a F,
    batched_f: &'a Option<B>,
    cache: RefCell<HashMap<(usize, usize), T>>,
    pending_error: RefCell<Option<TCIError>>,
    sampled_max: Cell<f64>,
}

impl<'a, T, F, B> LazyPiEvaluator<'a, T, F, B>
where
    T: Scalar + TTScalar + Default + tensor4all_tcicore::MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    fn new(
        i_combined: &'a [MultiIndex],
        j_combined: &'a [MultiIndex],
        f: &'a F,
        batched_f: &'a Option<B>,
        initial_max: f64,
    ) -> Self {
        Self {
            i_combined,
            j_combined,
            f,
            batched_f,
            cache: RefCell::new(HashMap::new()),
            pending_error: RefCell::new(None),
            sampled_max: Cell::new(initial_max),
        }
    }

    fn fill_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        if self.pending_error.borrow().is_some() {
            out.fill(T::zero());
            return;
        }

        let mut missing_entries = Vec::new();
        let mut missing_indices = Vec::new();

        {
            let cache_ref = self.cache.borrow();
            for (j_pos, &col) in cols.iter().enumerate() {
                for (i_pos, &row) in rows.iter().enumerate() {
                    let out_idx = i_pos + rows.len() * j_pos;
                    if let Some(&value) = cache_ref.get(&(row, col)) {
                        out[out_idx] = value;
                    } else {
                        missing_entries.push((out_idx, row, col));
                        missing_indices.push(build_full_index(
                            self.i_combined,
                            self.j_combined,
                            row,
                            col,
                        ));
                    }
                }
            }
        }

        if missing_entries.is_empty() {
            return;
        }

        let values = if let Some(batch_fn) = self.batched_f {
            batch_fn(&missing_indices)
        } else {
            missing_indices.iter().map(self.f).collect()
        };
        if values.len() != missing_entries.len() {
            *self.pending_error.borrow_mut() = Some(callback_length_mismatch(
                values.len(),
                missing_entries.len(),
            ));
            for (out_idx, _, _) in missing_entries {
                out[out_idx] = T::zero();
            }
            return;
        }

        let mut cache_ref = self.cache.borrow_mut();
        for ((out_idx, row, col), value) in missing_entries.into_iter().zip(values.into_iter()) {
            out[out_idx] = value;
            cache_ref.insert((row, col), value);

            let abs_val = f64::sqrt(Scalar::abs_sq(value));
            if abs_val > self.sampled_max.get() {
                self.sampled_max.set(abs_val);
            }
        }
    }

    fn sampled_max(&self) -> f64 {
        self.sampled_max.get()
    }

    fn take_error(&self) -> Option<TCIError> {
        self.pending_error.borrow_mut().take()
    }
}

#[cfg(test)]
mod tests;
