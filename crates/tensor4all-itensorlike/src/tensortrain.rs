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
    TensorDynLen, TensorIndex, TensorLike,
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

        let mut tt = Self {
            treetn,
            canonical_form: None,
        };
        tt.normalize_site_tensor_orders()?;
        Ok(tt)
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
    ) -> Result<Self> {
        let mut node_names = treetn.node_names();
        node_names.sort_unstable();
        let mut tt = Self {
            treetn,
            canonical_form,
        };
        for (site, old_name) in node_names.into_iter().enumerate() {
            if old_name != site {
                tt.treetn.rename_node(&old_name, site).map_err(|e| {
                    TensorTrainError::InvalidStructure {
                        message: format!("Failed to renumber TensorTrain sites: {}", e),
                    }
                })?;
            }
        }
        if tt.has_simple_linear_links() {
            tt.normalize_site_tensor_orders()?;
        }
        Ok(tt)
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
        let node_indices: Vec<_> = (0..self.len())
            .map(|site| self.treetn.node_index(&site).expect("Site out of bounds"))
            .collect();
        let mut tensor_ptrs = Vec::with_capacity(node_indices.len());
        for node_idx in node_indices {
            let tensor = self.treetn.tensor_mut(node_idx).expect("Tensor not found");
            tensor_ptrs.push(tensor as *mut TensorDynLen);
        }

        // SAFETY: TensorTrain site names are unique, so each site resolves to a
        // distinct TreeTN node. We collect at most one pointer per node and do
        // not mutate the network structure before converting those pointers back
        // into mutable references.
        unsafe { tensor_ptrs.into_iter().map(|tensor| &mut *tensor).collect() }
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

    fn normalize_site_tensor_orders(&mut self) -> Result<()> {
        for site in 0..self.len() {
            self.normalize_site_tensor_order(site)?;
        }
        Ok(())
    }

    fn has_simple_linear_links(&self) -> bool {
        if self.len() <= 1 {
            return true;
        }

        (0..self.len() - 1).all(|site| {
            let left = self.tensor(site);
            let right = self.tensor(site + 1);
            common_inds(left.indices(), right.indices()).len() <= 1
        })
    }

    fn can_normalize_site_tensor_order(&self, site: usize) -> bool {
        let left_ok = if site > 0 {
            common_inds(self.tensor(site - 1).indices(), self.tensor(site).indices()).len() <= 1
        } else {
            true
        };
        let right_ok = if site + 1 < self.len() {
            common_inds(self.tensor(site).indices(), self.tensor(site + 1).indices()).len() <= 1
        } else {
            true
        };
        left_ok && right_ok
    }

    fn normalize_site_tensor_order(&mut self, site: usize) -> Result<()> {
        if !self.can_normalize_site_tensor_order(site) {
            return Ok(());
        }

        let tensor = self.tensor_checked(site)?.clone();
        let current = tensor.indices().to_vec();
        let left = if site > 0 {
            self.linkind(site - 1)
        } else {
            None
        };
        let right = if site + 1 < self.len() {
            self.linkind(site)
        } else {
            None
        };

        let mut desired = Vec::with_capacity(current.len());
        if let Some(ref left_link) = left {
            desired.push(left_link.clone());
        }
        desired.extend(
            current
                .iter()
                .filter(|idx| Some(*idx) != left.as_ref() && Some(*idx) != right.as_ref())
                .cloned(),
        );
        if let Some(ref right_link) = right {
            desired.push(right_link.clone());
        }

        if desired == current {
            return Ok(());
        }

        let normalized =
            tensor
                .permuteinds(&desired)
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!(
                        "Failed to normalize site tensor index order at site {}: {}",
                        site, e
                    ),
                })?;
        self.set_tensor_raw(site, normalized)
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
    fn set_tensor_raw(&mut self, site: usize, tensor: TensorDynLen) -> Result<()> {
        let node_idx = self.treetn.node_index(&site).expect("Site out of bounds");
        self.treetn.replace_tensor(node_idx, tensor).map_err(|e| {
            TensorTrainError::InvalidStructure {
                message: format!("Failed to replace tensor at site {}: {}", site, e),
            }
        })?;
        Ok(())
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking.
    pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen) {
        self.set_tensor_raw(site, tensor)
            .and_then(|()| self.normalize_site_tensor_order(site))
            .unwrap_or_else(|e| panic!("TensorTrain::set_tensor failed: {}", e));
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

        self.treetn.inner(&other.treetn).unwrap_or_else(|e| {
            panic!("TensorTrain::inner failed while delegating to TreeTN::inner: {e}")
        })
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

        Self::from_inner(result_inner, None)
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
    /// let scaled = tt.scale(AnyScalar::new_real(2.0))?;
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
        Self::from_inner(treetn, None).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn replaceinds(&self, old: &[Self::Index], new: &[Self::Index]) -> anyhow::Result<Self> {
        let treetn = self.treetn.replaceinds(old, new)?;
        Self::from_inner(treetn, None).map_err(|e| anyhow::anyhow!("{}", e))
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

    fn maxabs(&self) -> f64 {
        self.to_dense().map(|t| t.maxabs()).unwrap_or(0.0)
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
mod tests;
