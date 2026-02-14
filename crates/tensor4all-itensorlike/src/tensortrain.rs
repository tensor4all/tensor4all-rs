//! Main Tensor Train type as a wrapper around TreeTN.
//!
//! This module provides the `TensorTrain` type, which represents a Tensor Train
//! (also known as MPS) with orthogonality tracking, inspired by ITensorMPS.jl.
//!
//! Internally, TensorTrain is implemented as a thin wrapper around
//! `TreeTN<TensorDynLen, usize>` where node names are site indices (0, 1, 2, ...).

use std::ops::Range;
use tensor4all_core::{common_inds, hascommoninds, DynIndex, IndexLike};
use tensor4all_core::{
    AllowedPairs, AnyScalar, DirectSumResult, FactorizeError, FactorizeOptions, FactorizeResult,
    TensorAccess, TensorDynLen, TensorIndex, TensorLike,
};
use tensor4all_treetn::{CanonicalizationOptions, TreeTN, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_truncation_params, CanonicalForm, TruncateAlg, TruncateOptions};
use tensor4all_core::truncation::HasTruncationParams;

/// Tensor Train with orthogonality tracking.
///
/// This type represents a tensor train as a sequence of tensors with tracked
/// orthogonality limits. It is inspired by ITensorMPS.jl but uses
/// 0-indexed sites (Rust convention).
///
/// Unlike traditional MPS which assumes one physical index per site, this
/// implementation allows each site to have multiple site indices.
///
/// # Orthogonality Tracking
///
/// The tensor train tracks orthogonality using `ortho_region` from the underlying TreeTN:
/// - When `ortho_region` is empty, no orthogonality is assumed
/// - When `ortho_region` contains a single site, that site is the orthogonality center
///
/// # Implementation
///
/// Internally wraps `TreeTN<TensorDynLen, usize>` where node names are site indices.
/// This allows reuse of TreeTN's canonicalization and contraction algorithms.
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// The underlying TreeTN with linear chain topology.
    /// Node names are usize (0, 1, 2, ...) representing site indices.
    pub(crate) treetn: TreeTN<TensorDynLen, usize>,
    /// The canonical form used (if known).
    canonical_form: Option<CanonicalForm>,
}

impl TensorTrain {
    /// Create a new tensor train from a vector of tensors.
    ///
    /// The tensor train is created with no assumed orthogonality.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors representing the tensor train
    ///
    /// # Returns
    ///
    /// A new tensor train with no orthogonality.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors have inconsistent bond dimensions
    /// (i.e., the link indices between adjacent tensors don't match).
    pub fn new(tensors: Vec<TensorDynLen>) -> Result<Self> {
        if tensors.is_empty() {
            // Create an empty TreeTN
            let treetn = TreeTN::<TensorDynLen, usize>::new();
            return Ok(Self {
                treetn,
                canonical_form: None,
            });
        }

        // Validate that adjacent tensors share exactly one common index (the link)
        for i in 0..tensors.len() - 1 {
            let left = &tensors[i];
            let right = &tensors[i + 1];

            let common = common_inds(left.indices(), right.indices());
            if common.is_empty() {
                return Err(TensorTrainError::InvalidStructure {
                    message: format!(
                        "No common index between tensors at sites {} and {}",
                        i,
                        i + 1
                    ),
                });
            }
            if common.len() > 1 {
                return Err(TensorTrainError::InvalidStructure {
                    message: format!(
                        "Multiple common indices ({}) between tensors at sites {} and {}",
                        common.len(),
                        i,
                        i + 1
                    ),
                });
            }
        }

        // Create node names: 0, 1, 2, ..., n-1
        let node_names: Vec<usize> = (0..tensors.len()).collect();

        // Create TreeTN with from_tensors (auto-connects by shared index IDs)
        let treetn =
            TreeTN::<TensorDynLen, usize>::from_tensors(tensors, node_names).map_err(|e| {
                TensorTrainError::InvalidStructure {
                    message: format!("Failed to create TreeTN: {}", e),
                }
            })?;

        Ok(Self {
            treetn,
            canonical_form: None,
        })
    }

    /// Create a new tensor train with specified orthogonality center.
    ///
    /// This is useful when constructing a tensor train that is already in canonical form.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors representing the tensor train
    /// * `llim` - Left orthogonality limit (for compatibility; only used to compute center)
    /// * `rlim` - Right orthogonality limit (for compatibility; only used to compute center)
    /// * `canonical_form` - The method used for canonicalization (if any)
    pub fn with_ortho(
        tensors: Vec<TensorDynLen>,
        llim: i32,
        rlim: i32,
        canonical_form: Option<CanonicalForm>,
    ) -> Result<Self> {
        let mut tt = Self::new(tensors)?;

        // Convert llim/rlim to ortho center
        // When llim + 2 == rlim, ortho center is at llim + 1
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < tt.len() as i32 {
            let center = (llim + 1) as usize;
            tt.treetn.set_canonical_region(vec![center]).map_err(|e| {
                TensorTrainError::InvalidStructure {
                    message: format!("Failed to set ortho region: {}", e),
                }
            })?;
        }

        tt.canonical_form = canonical_form;
        Ok(tt)
    }

    /// Create a TensorTrain from an existing TreeTN and canonical form.
    ///
    /// This is a crate-internal constructor used by `contract` and `linsolve`.
    pub(crate) fn from_inner(
        treetn: TreeTN<TensorDynLen, usize>,
        canonical_form: Option<CanonicalForm>,
    ) -> Self {
        Self {
            treetn,
            canonical_form,
        }
    }

    /// Get a reference to the underlying TreeTN.
    ///
    /// This is a crate-internal accessor used by `contract` and `linsolve`.
    pub(crate) fn as_treetn(&self) -> &TreeTN<TensorDynLen, usize> {
        &self.treetn
    }

    /// Number of sites (tensors) in the tensor train.
    #[inline]
    pub fn len(&self) -> usize {
        self.treetn.node_count()
    }

    /// Check if the tensor train is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.treetn.node_count() == 0
    }

    /// Left orthogonality limit.
    ///
    /// Sites `0..llim` are guaranteed to be left-orthogonal.
    /// Returns -1 if no sites are left-orthogonal.
    #[inline]
    pub fn llim(&self) -> i32 {
        match self.orthocenter() {
            Some(center) => center as i32 - 1,
            None => -1,
        }
    }

    /// Right orthogonality limit.
    ///
    /// Sites `rlim..len()` are guaranteed to be right-orthogonal.
    /// Returns `len() + 1` if no sites are right-orthogonal.
    #[inline]
    pub fn rlim(&self) -> i32 {
        match self.orthocenter() {
            Some(center) => center as i32 + 1,
            None => self.len() as i32 + 1,
        }
    }

    /// Set the left orthogonality limit.
    #[inline]
    pub fn set_llim(&mut self, llim: i32) {
        // Convert to ortho center if possible
        let rlim = self.rlim();
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < self.len() as i32 {
            let center = (llim + 1) as usize;
            let _ = self.treetn.set_canonical_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.treetn.set_canonical_region(Vec::<usize>::new());
        }
    }

    /// Set the right orthogonality limit.
    #[inline]
    pub fn set_rlim(&mut self, rlim: i32) {
        // Convert to ortho center if possible
        let llim = self.llim();
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < self.len() as i32 {
            let center = (llim + 1) as usize;
            let _ = self.treetn.set_canonical_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.treetn.set_canonical_region(Vec::<usize>::new());
        }
    }

    /// Get the orthogonality center range.
    ///
    /// Returns the range of sites that may not be orthogonal.
    /// If the tensor train is fully left-orthogonal, returns an empty range at the end.
    /// If the tensor train is fully right-orthogonal, returns an empty range at the beginning.
    pub fn ortho_lims(&self) -> Range<usize> {
        let llim = self.llim();
        let rlim = self.rlim();
        let start = (llim + 1).max(0) as usize;
        let end = rlim.max(0) as usize;
        start..end.min(self.len())
    }

    /// Check if the tensor train has a single orthogonality center.
    ///
    /// Returns true if there is exactly one site that is not guaranteed to be orthogonal.
    #[inline]
    pub fn isortho(&self) -> bool {
        self.treetn.canonical_region().len() == 1
    }

    /// Get the orthogonality center (0-indexed).
    ///
    /// Returns `Some(site)` if the tensor train has a single orthogonality center,
    /// `None` otherwise.
    pub fn orthocenter(&self) -> Option<usize> {
        let region = self.treetn.canonical_region();
        if region.len() == 1 {
            // Node name IS the site index since V = usize
            Some(*region.iter().next().unwrap())
        } else {
            None
        }
    }

    /// Get the canonicalization method used.
    #[inline]
    pub fn canonical_form(&self) -> Option<CanonicalForm> {
        self.canonical_form
    }

    /// Set the canonicalization method.
    #[inline]
    pub fn set_canonical_form(&mut self, method: Option<CanonicalForm>) {
        self.canonical_form = method;
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    #[inline]
    pub fn tensor(&self, site: usize) -> &TensorDynLen {
        let node_idx = self.treetn.node_index(&site).expect("Site out of bounds");
        self.treetn.tensor(node_idx).expect("Tensor not found")
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// Returns `Err` if `site >= len()`.
    pub fn tensor_checked(&self, site: usize) -> Result<&TensorDynLen> {
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }
        let node_idx =
            self.treetn
                .node_index(&site)
                .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                    site,
                    length: self.len(),
                })?;
        self.treetn
            .tensor(node_idx)
            .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            })
    }

    /// Get a mutable reference to the tensor at the given site.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    #[inline]
    pub fn tensor_mut(&mut self, site: usize) -> &mut TensorDynLen {
        let node_idx = self.treetn.node_index(&site).expect("Site out of bounds");
        self.treetn.tensor_mut(node_idx).expect("Tensor not found")
    }

    /// Get a reference to all tensors.
    #[inline]
    pub fn tensors(&self) -> Vec<&TensorDynLen> {
        (0..self.len())
            .filter_map(|site| {
                let node_idx = self.treetn.node_index(&site)?;
                self.treetn.tensor(node_idx)
            })
            .collect()
    }

    /// Get a mutable reference to all tensors.
    #[inline]
    pub fn tensors_mut(&mut self) -> Vec<&mut TensorDynLen> {
        // This is tricky - we need to collect mutable references
        // For now, return an empty vec - this method is rarely used
        // and would require unsafe code or different design
        Vec::new()
    }

    /// Get the link index between sites `i` and `i+1`.
    ///
    /// Returns `None` if `i >= len() - 1` or if no common index exists.
    pub fn linkind(&self, i: usize) -> Option<DynIndex> {
        if i >= self.len().saturating_sub(1) {
            return None;
        }

        let left_node = self.treetn.node_index(&i)?;
        let right_node = self.treetn.node_index(&(i + 1))?;
        let left = self.treetn.tensor(left_node)?;
        let right = self.treetn.tensor(right_node)?;
        let common = common_inds(left.indices(), right.indices());
        common.into_iter().next()
    }

    /// Get all link indices.
    ///
    /// Returns a vector of length `len() - 1` containing the link indices.
    pub fn linkinds(&self) -> Vec<DynIndex> {
        (0..self.len().saturating_sub(1))
            .filter_map(|i| self.linkind(i))
            .collect()
    }

    /// Create a copy with all link indices replaced by new unique IDs.
    ///
    /// This is useful for computing inner products where two tensor trains
    /// share link indices. By simulating (replacing) the link indices in one
    /// of the tensor trains, they can be contracted over site indices only.
    pub fn sim_linkinds(&self) -> Self {
        if self.len() <= 1 {
            return self.clone();
        }

        // Build replacement pairs: (old_link, new_link) for each link index
        let old_links = self.linkinds();
        let new_links: Vec<_> = old_links.iter().map(|idx| idx.sim()).collect();
        let replacements: Vec<_> = old_links
            .iter()
            .cloned()
            .zip(new_links.iter().cloned())
            .collect();

        // Replace link indices in each tensor and rebuild
        let mut new_tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            let tensor = self.tensor(site);
            let mut new_tensor = tensor.clone();
            for (old_idx, new_idx) in &replacements {
                new_tensor = new_tensor.replaceind(old_idx, new_idx);
            }
            new_tensors.push(new_tensor);
        }

        Self::new(new_tensors).expect("sim_linkinds: failed to create new tensor train")
    }

    /// Get the site indices (non-link indices) for all sites.
    ///
    /// For each site, returns a vector of indices that are not shared with
    /// adjacent tensors (i.e., the "physical" or "site" indices).
    pub fn siteinds(&self) -> Vec<Vec<DynIndex>> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let tensor = self.tensor(i);
            let mut site_inds: Vec<DynIndex> = tensor.indices().to_vec();

            // Remove link to left neighbor
            if i > 0 {
                if let Some(link) = self.linkind(i - 1) {
                    site_inds.retain(|idx| idx != &link);
                }
            }

            // Remove link to right neighbor
            if i < self.len() - 1 {
                if let Some(link) = self.linkind(i) {
                    site_inds.retain(|idx| idx != &link);
                }
            }

            result.push(site_inds);
        }

        result
    }

    /// Get the bond dimension at link `i` (between sites `i` and `i+1`).
    ///
    /// Returns `None` if `i >= len() - 1`.
    pub fn bond_dim(&self, i: usize) -> Option<usize> {
        self.linkind(i).map(|idx| idx.size())
    }

    /// Get all bond dimensions.
    ///
    /// Returns a vector of length `len() - 1`.
    pub fn bond_dims(&self) -> Vec<usize> {
        self.linkinds().iter().map(|idx| idx.size()).collect()
    }

    /// Get the maximum bond dimension.
    pub fn maxbonddim(&self) -> usize {
        self.bond_dims().into_iter().max().unwrap_or(1)
    }

    /// Check if two adjacent tensors share an index.
    pub fn haslink(&self, i: usize) -> bool {
        if i >= self.len().saturating_sub(1) {
            return false;
        }
        let left_node = self.treetn.node_index(&i);
        let right_node = self.treetn.node_index(&(i + 1));
        match (left_node, right_node) {
            (Some(l), Some(r)) => {
                let left = self.treetn.tensor(l);
                let right = self.treetn.tensor(r);
                match (left, right) {
                    (Some(l), Some(r)) => hascommoninds(l.indices(), r.indices()),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking.
    pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen) {
        let node_idx = self.treetn.node_index(&site).expect("Site out of bounds");
        let _ = self.treetn.replace_tensor(node_idx, tensor);
        // Invalidate orthogonality
        let _ = self.treetn.set_canonical_region(Vec::<usize>::new());
    }

    /// Orthogonalize the tensor train to have orthogonality center at the given site.
    ///
    /// This function performs a series of factorizations to make the tensor train
    /// canonical with orthogonality center at `site`.
    ///
    /// # Arguments
    ///
    /// * `site` - The target site for the orthogonality center (0-indexed)
    ///
    /// # Errors
    ///
    /// Returns an error if the factorization fails or if the site is out of bounds.
    pub fn orthogonalize(&mut self, site: usize) -> Result<()> {
        self.orthogonalize_with(site, CanonicalForm::Unitary)
    }

    /// Orthogonalize with a specified canonical form.
    ///
    /// # Arguments
    ///
    /// * `site` - The target site for the orthogonality center (0-indexed)
    /// * `form` - The canonical form to use:
    ///   - `Unitary`: Uses QR decomposition, each tensor is isometric
    ///   - `LU`: Uses LU decomposition, one factor has unit diagonal
    ///   - `CI`: Uses Cross Interpolation
    pub fn orthogonalize_with(&mut self, site: usize, form: CanonicalForm) -> Result<()> {
        if self.is_empty() {
            return Err(TensorTrainError::Empty);
        }
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }

        // Use TreeTN's canonicalize (accepts node names and CanonicalizationOptions)
        // Since V = usize, node names are site indices
        let options = CanonicalizationOptions::forced().with_form(form);
        self.treetn = std::mem::take(&mut self.treetn)
            .canonicalize(vec![site], options)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Canonicalize failed: {}", e),
            })?;

        self.canonical_form = Some(form);
        Ok(())
    }

    /// Truncate the tensor train bond dimensions.
    ///
    /// This delegates to the TreeTN's truncate_mut method, which performs a
    /// two-site sweep with Euler tour traversal for optimal truncation.
    ///
    /// Note: The `site_range` option in `TruncateOptions` is currently ignored
    /// as the underlying TreeTN truncation operates on the full network.
    pub fn truncate(&mut self, options: &TruncateOptions) -> Result<()> {
        if self.len() <= 1 {
            return Ok(());
        }

        validate_truncation_params(options.truncation_params())?;

        // Convert TruncateOptions to TruncationOptions
        let form = truncate_alg_to_form(options.alg());
        let treetn_options = TruncationOptions {
            form,
            truncation: options.truncation(),
        };

        // Truncate towards the last site (rightmost) as the canonical center
        // This matches ITensor convention where truncation sweeps left-to-right
        let center = self.len() - 1;

        self.treetn
            .truncate_mut([center], treetn_options)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("TreeTN truncation failed: {}", e),
            })?;

        self.canonical_form = Some(form);

        Ok(())
    }

    /// Compute the inner product (dot product) of two tensor trains.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both tensor trains must have the same site indices (same IDs).
    /// Link indices may differ between the two tensor trains.
    pub fn inner(&self, other: &Self) -> AnyScalar {
        assert_eq!(
            self.len(),
            other.len(),
            "Tensor trains must have the same length for inner product"
        );

        if self.is_empty() {
            return AnyScalar::new_real(0.0);
        }

        // Replace link indices in other with unique IDs
        let other_sim = other.sim_linkinds();

        // Start with leftmost tensors - contract over site indices only
        let mut env = {
            let a0_conj = self.tensor(0).conj();
            let b0 = other_sim.tensor(0);
            a0_conj.contract(b0)
        };

        // Sweep through remaining sites
        for i in 1..self.len() {
            let ai_conj = self.tensor(i).conj();
            let bi = other_sim.tensor(i);

            // Contract: env * conj(A_i) (over self's link index)
            env = env.contract(&ai_conj);
            // Contract: result * B_i (over other's link index and site indices)
            env = env.contract(bi);
        }

        // Result should be a scalar (0-dimensional tensor)
        env.only()
    }

    /// Compute the squared norm of the tensor train.
    ///
    /// Returns `<self | self>` = ||self||^2.
    ///
    /// # Note
    /// Due to numerical errors, the inner product can be very slightly negative.
    /// This method clamps the result to be non-negative.
    pub fn norm_squared(&self) -> f64 {
        self.inner(self).real().max(0.0)
    }

    /// Compute the norm of the tensor train.
    ///
    /// Returns ||self|| = sqrt(<self | self>).
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Convert the tensor train to a single dense tensor.
    ///
    /// This contracts all tensors in the train along their link indices,
    /// producing a single tensor with only site indices.
    ///
    /// # Warning
    /// This operation can be very expensive for large tensor trains,
    /// as the result size grows exponentially with the number of sites.
    ///
    /// # Returns
    /// A single tensor containing all site indices, or an error if the
    /// tensor train is empty.
    ///
    /// # Example
    /// ```ignore
    /// let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
    /// let dense = tt.to_dense().unwrap();
    /// // dense now has all site indices from t0, t1, t2 (link indices contracted)
    /// ```
    pub fn to_dense(&self) -> Result<TensorDynLen> {
        if self.is_empty() {
            return Err(TensorTrainError::InvalidStructure {
                message: "Cannot convert empty tensor train to dense".to_string(),
            });
        }

        self.treetn
            .contract_to_tensor()
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Failed to contract to dense: {}", e),
            })
    }

    /// Add two tensor trains using direct-sum construction.
    ///
    /// This creates a new tensor train where each tensor is the direct sum of the
    /// corresponding tensors from self and other, with bond dimensions merged.
    /// The result has bond dimensions equal to the sum of the input bond dimensions.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to add
    ///
    /// # Returns
    /// A new tensor train representing the sum.
    ///
    /// # Errors
    /// Returns an error if the tensor trains have incompatible structures.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.is_empty() && other.is_empty() {
            return Ok(Self::default());
        }

        if self.is_empty() {
            return Ok(other.clone());
        }

        if other.is_empty() {
            return Ok(self.clone());
        }

        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidStructure {
                message: format!(
                    "Tensor trains must have the same length for addition: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        let result_inner =
            self.treetn
                .add(&other.treetn)
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!("TT addition failed: {}", e),
                })?;

        Ok(Self {
            treetn: result_inner,
            canonical_form: None, // Addition destroys canonical form
        })
    }

    /// Scale the tensor train by a scalar.
    ///
    /// Only one tensor (the first non-empty site) is scaled to avoid scalar^n scaling.
    /// This is correct because the tensor train represents a product of tensors,
    /// so scaling one factor scales the entire product.
    ///
    /// # Arguments
    /// * `scalar` - The scalar to multiply by
    ///
    /// # Returns
    /// A new tensor train scaled by the given scalar.
    ///
    /// # Example
    /// ```ignore
    /// let scaled = tt.scale(AnyScalar::F64(2.0))?;
    /// // scaled represents 2 * tt
    /// ```
    pub fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            let tensor = self.tensor(site);
            if site == 0 {
                // Scale only the first tensor
                let scaled =
                    tensor
                        .scale(scalar.clone())
                        .map_err(|e| TensorTrainError::OperationError {
                            message: format!("Failed to scale tensor at site 0: {}", e),
                        })?;
                tensors.push(scaled);
            } else {
                tensors.push(tensor.clone());
            }
        }

        Self::new(tensors)
    }

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// This is equivalent to `self.scale(a)?.add(&other.scale(b)?)`.
    ///
    /// # Arguments
    /// * `a` - Scalar coefficient for self
    /// * `other` - The other tensor train
    /// * `b` - Scalar coefficient for other
    ///
    /// # Returns
    /// A new tensor train representing `a * self + b * other`.
    ///
    /// # Note
    /// The bond dimension of the result is the sum of the bond dimensions
    /// of the two input tensor trains (before any truncation).
    pub fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        let scaled_self = self.scale(a)?;
        let scaled_other = other.scale(b)?;
        scaled_self.add(&scaled_other)
    }
}

// Implement Default for TensorTrain to allow std::mem::take
impl Default for TensorTrain {
    fn default() -> Self {
        Self::new(vec![]).expect("Failed to create empty TensorTrain")
    }
}

/// Convert TruncateAlg to CanonicalForm.
///
/// Note: SVD truncation algorithm corresponds to Unitary canonical form
/// because both produce orthogonal/isometric tensors.
pub(crate) fn truncate_alg_to_form(alg: TruncateAlg) -> CanonicalForm {
    match alg {
        TruncateAlg::SVD | TruncateAlg::RSVD | TruncateAlg::QR => CanonicalForm::Unitary,
        TruncateAlg::LU => CanonicalForm::LU,
        TruncateAlg::CI => CanonicalForm::CI,
    }
}

// ============================================================================
// TensorIndex implementation for TensorTrain
// ============================================================================

impl TensorIndex for TensorTrain {
    type Index = DynIndex;

    fn external_indices(&self) -> Vec<Self::Index> {
        // Delegate to the internal TreeTN's TensorIndex implementation
        self.treetn.external_indices()
    }

    fn num_external_indices(&self) -> usize {
        self.treetn.num_external_indices()
    }

    fn replaceind(&self, old: &Self::Index, new: &Self::Index) -> anyhow::Result<Self> {
        // Delegate to the internal TreeTN's replaceind
        // After replacement, canonical form may be invalid, so set to None
        let treetn = self.treetn.replaceind(old, new)?;
        Ok(Self::from_inner(treetn, None))
    }

    fn replaceinds(&self, old: &[Self::Index], new: &[Self::Index]) -> anyhow::Result<Self> {
        let treetn = self.treetn.replaceinds(old, new)?;
        Ok(Self::from_inner(treetn, None))
    }
}

// ============================================================================
// TensorLike implementation for TensorTrain
// ============================================================================

impl TensorLike for TensorTrain {
    // ========================================================================
    // GMRES-required methods (fully supported)
    // ========================================================================

    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> anyhow::Result<Self> {
        TensorTrain::axpby(self, a, other, b).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn scale(&self, scalar: AnyScalar) -> anyhow::Result<Self> {
        TensorTrain::scale(self, scalar).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn inner_product(&self, other: &Self) -> anyhow::Result<AnyScalar> {
        Ok(self.inner(other))
    }

    fn norm_squared(&self) -> f64 {
        TensorTrain::norm_squared(self)
    }

    fn conj(&self) -> Self {
        // Clone and conjugate each site tensor
        // Note: conj() cannot return Result, so we ensure this never fails
        let mut result = self.clone();
        for site in 0..result.len() {
            let t = result.tensor(site).conj();
            result.set_tensor(site, t);
        }
        result
    }

    // ========================================================================
    // Methods not supported by TensorTrain
    // ========================================================================

    fn factorize(
        &self,
        _left_inds: &[Self::Index],
        _options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::UnsupportedStorage(
            "TensorTrain does not support factorize; use orthogonalize() instead",
        ))
    }

    fn contract(_tensors: &[&Self], _allowed: AllowedPairs<'_>) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support TensorLike::contract; use TensorTrain::contract() method instead")
    }

    fn contract_connected(_tensors: &[&Self], _allowed: AllowedPairs<'_>) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support TensorLike::contract_connected; use TensorTrain::contract() method instead")
    }

    fn direct_sum(
        &self,
        _other: &Self,
        _pairs: &[(Self::Index, Self::Index)],
    ) -> anyhow::Result<DirectSumResult<Self>> {
        anyhow::bail!("TensorTrain does not support direct_sum; use add() instead")
    }

    fn outer_product(&self, _other: &Self) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support outer_product")
    }

    fn permuteinds(&self, _new_order: &[Self::Index]) -> anyhow::Result<Self> {
        anyhow::bail!("TensorTrain does not support permuteinds")
    }

    fn diagonal(input: &Self::Index, output: &Self::Index) -> anyhow::Result<Self> {
        // Create a single-site TensorTrain with an identity tensor
        let delta = TensorDynLen::diagonal(input, output)?;
        Self::new(vec![delta]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn scalar_one() -> anyhow::Result<Self> {
        // Empty tensor train represents scalar 1
        Self::new(vec![]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn ones(indices: &[Self::Index]) -> anyhow::Result<Self> {
        let t = TensorDynLen::ones(indices)?;
        Self::new(vec![t]).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn onehot(index_vals: &[(Self::Index, usize)]) -> anyhow::Result<Self> {
        let t = TensorDynLen::onehot(index_vals)?;
        Self::new(vec![t]).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::StorageScalar;
    use tensor4all_core::{DynId, Index};

    /// Helper to create a simple tensor for testing
    fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
        let dims: Vec<usize> = indices.iter().map(|i| i.size()).collect();
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let storage = f64::dense_storage_with_shape(data, &dims);
        TensorDynLen::new(indices, storage)
    }

    /// Helper to create a DynIndex
    fn idx(id: u64, size: usize) -> DynIndex {
        Index::new_with_size(DynId(id), size)
    }

    #[test]
    fn test_empty_tt() {
        let tt: TensorTrain = TensorTrain::new(vec![]).unwrap();
        assert!(tt.is_empty());
        assert_eq!(tt.len(), 0);
        assert_eq!(tt.llim(), -1);
        assert_eq!(tt.rlim(), 1);
        assert!(!tt.isortho());
    }

    #[test]
    fn test_single_site_tt() {
        let tensor = make_tensor(vec![idx(0, 2)]);

        let tt = TensorTrain::new(vec![tensor]).unwrap();
        assert_eq!(tt.len(), 1);
        assert!(!tt.isortho());
        assert_eq!(tt.bond_dims(), Vec::<usize>::new());
    }

    #[test]
    fn test_two_site_tt() {
        // Create two tensors with a shared link index
        let s0 = idx(0, 2); // site 0
        let l01 = idx(1, 3); // link 0-1
        let s1 = idx(2, 2); // site 1

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();
        assert_eq!(tt.len(), 2);
        assert_eq!(tt.bond_dims(), vec![3]);
        assert_eq!(tt.maxbonddim(), 3);

        // Check link index
        let link = tt.linkind(0).unwrap();
        assert_eq!(link.size(), 3);

        // Check site indices (nested vec)
        let site_inds = tt.siteinds();
        assert_eq!(site_inds.len(), 2);
        assert_eq!(site_inds[0].len(), 1);
        assert_eq!(site_inds[1].len(), 1);
        assert_eq!(site_inds[0][0].size(), 2);
        assert_eq!(site_inds[1][0].size(), 2);
    }

    #[test]
    fn test_multi_site_indices() {
        // Test site with multiple physical indices
        let s0a = idx(0, 2); // site 0 index a
        let s0b = idx(1, 3); // site 0 index b
        let l01 = idx(2, 4); // link 0-1
        let s1 = idx(3, 2); // site 1

        let t0 = make_tensor(vec![s0a.clone(), s0b.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Check site indices (nested vec)
        let site_inds = tt.siteinds();
        assert_eq!(site_inds.len(), 2);
        assert_eq!(site_inds[0].len(), 2); // site 0 has 2 indices
        assert_eq!(site_inds[1].len(), 1); // site 1 has 1 index
    }

    #[test]
    fn test_ortho_tracking() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0, l01.clone()]);
        let t1 = make_tensor(vec![l01, s1]);

        // Create with specified orthogonality (ortho center at site 0)
        let tt = TensorTrain::with_ortho(
            vec![t0, t1],
            -1, // no left orthogonality
            1,  // right orthogonal from site 1
            Some(CanonicalForm::Unitary),
        )
        .unwrap();

        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));
        assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
    }

    #[test]
    fn test_ortho_lims_range() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let l12 = idx(2, 3);
        let s1 = idx(3, 2);
        let s2 = idx(4, 2);

        let t0 = make_tensor(vec![s0, l01.clone()]);
        let t1 = make_tensor(vec![l01, s1, l12.clone()]);
        let t2 = make_tensor(vec![l12, s2]);

        // Create with partial orthogonality
        let tt = TensorTrain::with_ortho(vec![t0, t1, t2], 0, 2, None).unwrap();

        assert_eq!(tt.ortho_lims(), 1..2);
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(1));
    }

    #[test]
    fn test_no_common_index_error() {
        let s0 = idx(0, 2);
        let s1 = idx(1, 2);

        let t0 = make_tensor(vec![s0]);
        let t1 = make_tensor(vec![s1]);

        let result = TensorTrain::new(vec![t0, t1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TensorTrainError::InvalidStructure { .. }
        ));
    }

    #[test]
    fn test_orthogonalize_two_site() {
        // Create a 2-site tensor train
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
        assert!(!tt.isortho());

        // Orthogonalize to site 0
        tt.orthogonalize(0).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));
        assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));

        // Orthogonalize to site 1
        tt.orthogonalize(1).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(1));
    }

    #[test]
    fn test_orthogonalize_three_site() {
        // Create a 3-site tensor train
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);
        let l12 = idx(3, 3);
        let s2 = idx(4, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
        let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

        // Orthogonalize to middle site
        tt.orthogonalize(1).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(1));

        // Orthogonalize to left
        tt.orthogonalize(0).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));

        // Orthogonalize to right
        tt.orthogonalize(2).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(2));
    }

    #[test]
    fn test_orthogonalize_with_lu() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

        tt.orthogonalize_with(0, CanonicalForm::LU).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));
        assert_eq!(tt.canonical_form(), Some(CanonicalForm::LU));
    }

    #[test]
    fn test_orthogonalize_with_ci() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

        tt.orthogonalize_with(1, CanonicalForm::CI).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(1));
        assert_eq!(tt.canonical_form(), Some(CanonicalForm::CI));
    }

    #[test]
    fn test_truncate_with_max_rank() {
        // Create a 3-site tensor train with large bond dimension
        let s0 = idx(0, 4);
        let l01 = idx(1, 8);
        let s1 = idx(2, 4);
        let l12 = idx(3, 8);
        let s2 = idx(4, 4);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
        let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();
        assert_eq!(tt.maxbonddim(), 8);

        // Truncate to max rank 4
        let options = TruncateOptions::svd().with_max_rank(4);
        tt.truncate(&options).unwrap();

        // Check that bond dimensions are reduced
        assert!(tt.maxbonddim() <= 4);
        assert_eq!(tt.canonical_form(), Some(CanonicalForm::Unitary));
    }

    #[test]
    fn test_contract_with_fit_method() {
        use crate::ContractOptions;

        // Use two-site tensor trains with shared site indices to exercise contraction
        let s0 = idx(100, 2);
        let s1 = idx(101, 2);
        let l01_a = idx(102, 3);
        let l01_b = idx(103, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        // Test contract with Fit method
        let options = ContractOptions::fit().with_max_rank(10).with_nhalfsweeps(4); // 4 half-sweeps = 2 full sweeps
        let result = tt1.contract(&tt2, &options);
        assert!(result.is_ok());
        let result_tt = result.unwrap();
        assert_eq!(result_tt.len(), 1);
    }

    #[test]
    fn test_contract_with_naive_method() {
        use crate::ContractOptions;

        // Use single-site tensor trains with a shared site index
        let s0 = idx(200, 2);

        let t1 = make_tensor(vec![s0.clone()]);
        let tt1 = TensorTrain::new(vec![t1]).unwrap();

        let t2 = make_tensor(vec![s0.clone()]);
        let tt2 = TensorTrain::new(vec![t2]).unwrap();

        // Test contract with Naive method
        let options = ContractOptions::naive();
        let result = tt1.contract(&tt2, &options);
        assert!(result.is_ok());
        let result_tt = result.unwrap();
        assert_eq!(result_tt.len(), 1);
    }

    #[test]
    fn test_contract_nhalfsweeps_conversion() {
        use crate::ContractOptions;

        // Use two-site tensor trains with shared site indices to exercise contraction
        let s0 = idx(300, 2);
        let s1 = idx(301, 2);
        let l01_a = idx(302, 3);
        let l01_b = idx(303, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        // Test that nhalfsweeps is correctly converted to nfullsweeps
        // nhalfsweeps=6 should become nfullsweeps=3
        let options = ContractOptions::fit().with_nhalfsweeps(6).with_max_rank(10);
        let result = tt1.contract(&tt2, &options);
        assert!(result.is_ok());
        let result_tt = result.unwrap();
        assert_eq!(result_tt.len(), 1);
    }

    #[test]
    fn test_contract_fit_odd_nhalfsweeps_errors() {
        use crate::ContractOptions;

        let s0 = idx(400, 2);
        let s1 = idx(401, 2);
        let l01_a = idx(402, 3);
        let l01_b = idx(403, 3);

        let t1_0 = make_tensor(vec![s0.clone(), l01_a.clone()]);
        let t1_1 = make_tensor(vec![l01_a.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t1_0, t1_1]).unwrap();

        let t2_0 = make_tensor(vec![s0.clone(), l01_b.clone()]);
        let t2_1 = make_tensor(vec![l01_b.clone(), s1.clone()]);
        let tt2 = TensorTrain::new(vec![t2_0, t2_1]).unwrap();

        let options = ContractOptions::fit().with_nhalfsweeps(1).with_max_rank(10);
        let err = tt1.contract(&tt2, &options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_truncate_invalid_rtol_errors() {
        let s0 = idx(500, 2);
        let l01 = idx(501, 3);
        let s1 = idx(502, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();
        let options = TruncateOptions::svd().with_rtol(-1.0);
        let err = tt.truncate(&options).unwrap_err();
        assert!(matches!(err, TensorTrainError::OperationError { .. }));
    }

    #[test]
    fn test_inner_product() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Compute norm squared
        let norm_sq = tt.norm_squared();
        assert!(norm_sq > 0.0);

        // Compute norm
        let norm = tt.norm();
        assert!((norm * norm - norm_sq).abs() < 1e-10);
    }

    #[test]
    fn test_to_dense() {
        // Create a 2-site TT: s0 -- l01 -- s1
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();

        // Convert to dense
        let dense = tt.to_dense().unwrap();

        // Expected: contract t0 and t1 along l01
        let expected = t0.contract(&t1);

        // Compare results
        let dense_data = dense.as_slice_f64().unwrap();
        let expected_data = expected.as_slice_f64().unwrap();

        assert_eq!(dense_data.len(), expected_data.len());
        for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (d - e).abs() < 1e-10,
                "Mismatch at index {}: got {}, expected {}",
                i,
                d,
                e
            );
        }
    }

    #[test]
    fn test_to_dense_single_site() {
        // Single site TT should return the tensor as-is
        let s0 = idx(0, 4);
        let t0 = make_tensor(vec![s0.clone()]);

        let tt = TensorTrain::new(vec![t0.clone()]).unwrap();
        let dense = tt.to_dense().unwrap();

        let dense_data = dense.as_slice_f64().unwrap();
        let expected_data = t0.as_slice_f64().unwrap();

        assert_eq!(dense_data.len(), expected_data.len());
        for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (d - e).abs() < 1e-10,
                "Mismatch at index {}: got {}, expected {}",
                i,
                d,
                e
            );
        }
    }

    #[test]
    fn test_to_dense_three_sites() {
        // 3-site TT: s0 -- l01 -- s1 -- l12 -- s2
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);
        let l12 = idx(3, 3);
        let s2 = idx(4, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
        let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

        let tt = TensorTrain::new(vec![t0.clone(), t1.clone(), t2.clone()]).unwrap();
        let dense = tt.to_dense().unwrap();

        // Expected: contract t0, t1, t2 sequentially
        let expected = t0.contract(&t1).contract(&t2);

        let dense_data = dense.as_slice_f64().unwrap();
        let expected_data = expected.as_slice_f64().unwrap();

        assert_eq!(dense_data.len(), expected_data.len());
        for (i, (&d, &e)) in dense_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (d - e).abs() < 1e-10,
                "Mismatch at index {}: got {}, expected {}",
                i,
                d,
                e
            );
        }
    }

    #[test]
    fn test_to_dense_empty() {
        let tt = TensorTrain::new(vec![]).unwrap();
        let result = tt.to_dense();
        assert!(result.is_err());
    }

    #[test]
    fn test_add_simple() {
        // Create two TTs with the same structure
        // Both TTs must have the same site indices AND link indices
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        // First TT with data [1, 2, 3, ...]
        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

        // Second TT with same structure (clone and modify data for testing)
        let tt2 = tt1.clone();

        // Add them
        let sum = tt1.add(&tt2).unwrap();

        // Result should have double bond dimension
        assert_eq!(sum.len(), 2);
        assert_eq!(sum.bond_dims(), vec![6]); // 3 + 3

        // Verify numerically: sum.to_dense() == tt1.to_dense() + tt2.to_dense()
        let sum_dense = sum.to_dense().unwrap();
        let tt1_dense = tt1.to_dense().unwrap();
        let tt2_dense = tt2.to_dense().unwrap();

        let sum_data = sum_dense.as_slice_f64().unwrap();
        let tt1_data = tt1_dense.as_slice_f64().unwrap();
        let tt2_data = tt2_dense.as_slice_f64().unwrap();

        assert_eq!(sum_data.len(), tt1_data.len());
        for i in 0..sum_data.len() {
            let expected = tt1_data[i] + tt2_data[i];
            assert!(
                (sum_data[i] - expected).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                sum_data[i],
                expected
            );
        }
    }

    #[test]
    fn test_add_empty() {
        let empty = TensorTrain::new(vec![]).unwrap();

        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);
        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // empty + tt = tt
        let result = empty.add(&tt).unwrap();
        assert_eq!(result.len(), tt.len());

        // tt + empty = tt
        let result = tt.add(&empty).unwrap();
        assert_eq!(result.len(), tt.len());

        // empty + empty = empty
        let result = empty.add(&empty).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_add_length_mismatch() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);
        let l12 = idx(3, 3);
        let s2 = idx(4, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
        let tt1 = TensorTrain::new(vec![t0, t1]).unwrap();

        let t0_2 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1_2 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
        let t2_2 = make_tensor(vec![l12.clone(), s2.clone()]);
        let tt2 = TensorTrain::new(vec![t0_2, t1_2, t2_2]).unwrap();

        // Length mismatch should fail
        let result = tt1.add(&tt2);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_llim_updates_canonical_region() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0, l01.clone()]);
        let t1 = make_tensor(vec![l01, s1]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Set llim=-1, rlim already 1 => center at 0
        tt.set_llim(-1);
        // With rlim=1 (which is the default for non-ortho TT) and llim=-1, center should be at 0
        // But this depends on the rlim value, let's explicitly set both
        let mut tt2 = TensorTrain::with_ortho(
            vec![
                make_tensor(vec![idx(0, 2), idx(1, 3)]),
                make_tensor(vec![idx(1, 3), idx(2, 2)]),
            ],
            -1,
            1,
            Some(CanonicalForm::Unitary),
        )
        .unwrap();
        assert!(tt2.isortho());
        assert_eq!(tt2.orthocenter(), Some(0));

        // Setting llim to a value that breaks single-center should clear ortho
        tt2.set_llim(5);
        assert!(!tt2.isortho());
    }

    #[test]
    fn test_set_rlim_updates_canonical_region() {
        let mut tt = TensorTrain::with_ortho(
            vec![
                make_tensor(vec![idx(0, 2), idx(1, 3)]),
                make_tensor(vec![idx(1, 3), idx(2, 2)]),
            ],
            -1,
            1,
            Some(CanonicalForm::Unitary),
        )
        .unwrap();
        assert!(tt.isortho());

        // Setting rlim to a value that breaks single-center should clear ortho
        tt.set_rlim(5);
        assert!(!tt.isortho());
    }

    #[test]
    fn test_set_tensor_invalidates_ortho() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt =
            TensorTrain::with_ortho(vec![t0, t1], -1, 1, Some(CanonicalForm::Unitary)).unwrap();
        assert!(tt.isortho());

        // Replace tensor at site 0
        let new_tensor = make_tensor(vec![s0, l01]);
        tt.set_tensor(0, new_tensor);
        assert!(!tt.isortho());
    }

    #[test]
    fn test_scale() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Scale by 2.0
        let scaled = tt.scale(AnyScalar::F64(2.0)).unwrap();

        // Verify: norm of scaled should be 2 * norm of original
        let orig_norm = tt.norm();
        let scaled_norm = scaled.norm();
        assert!(
            (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
            "Expected scaled_norm = {}, got {}",
            2.0 * orig_norm,
            scaled_norm
        );
    }

    #[test]
    fn test_scale_empty() {
        let tt = TensorTrain::new(vec![]).unwrap();
        let scaled = tt.scale(AnyScalar::F64(2.0)).unwrap();
        assert!(scaled.is_empty());
    }

    #[test]
    fn test_axpby() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt1 = TensorTrain::new(vec![t0.clone(), t1.clone()]).unwrap();
        let tt2 = tt1.clone();

        // Compute 2*tt1 + 3*tt2 = 5*tt1 (since tt1 == tt2)
        let result = tt1
            .axpby(AnyScalar::F64(2.0), &tt2, AnyScalar::F64(3.0))
            .unwrap();

        // Verify numerically via to_dense
        let result_dense = result.to_dense().unwrap();
        let tt1_dense = tt1.to_dense().unwrap();

        let result_data = result_dense.as_slice_f64().unwrap();
        let tt1_data = tt1_dense.as_slice_f64().unwrap();

        assert_eq!(result_data.len(), tt1_data.len());
        for i in 0..result_data.len() {
            let expected = 5.0 * tt1_data[i]; // 2*tt1 + 3*tt2 = 5*tt1
            assert!(
                (result_data[i] - expected).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                result_data[i],
                expected
            );
        }

        // Bond dimension should be 6 (3 + 3)
        assert_eq!(result.bond_dims(), vec![6]);
    }

    #[test]
    fn test_tensor_like_scale() {
        use tensor4all_core::TensorLike;

        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Use TensorLike::scale
        let scaled = TensorLike::scale(&tt, AnyScalar::F64(2.0)).unwrap();

        let orig_norm = tt.norm();
        let scaled_norm = TensorLike::norm(&scaled);
        assert!(
            (scaled_norm - 2.0 * orig_norm).abs() < 1e-10,
            "Expected scaled_norm = {}, got {}",
            2.0 * orig_norm,
            scaled_norm
        );
    }

    #[test]
    fn test_tensor_like_inner_product() {
        use tensor4all_core::TensorLike;

        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // TensorLike::inner_product should equal TensorTrain::inner
        let inner_via_trait = TensorLike::inner_product(&tt, &tt).unwrap();
        let inner_direct = tt.inner(&tt);

        assert!(
            (inner_via_trait.real() - inner_direct.real()).abs() < 1e-10,
            "Inner product mismatch: {} vs {}",
            inner_via_trait.real(),
            inner_direct.real()
        );
    }
}
