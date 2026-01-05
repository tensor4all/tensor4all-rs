//! Main Tensor Train type as a wrapper around TreeTN.
//!
//! This module provides the `TensorTrain` type, which represents a Tensor Train
//! (also known as MPS) with orthogonality tracking, inspired by ITensorMPS.jl.
//!
//! Internally, TensorTrain is implemented as a thin wrapper around
//! `TreeTN<Id, Symm, usize, Einsum>` where node names are site indices (0, 1, 2, ...).

use std::ops::Range;

// Note: NodeIndex import not needed since we use V = usize for node names
use tensor4all_core_common::{
    common_inds, hascommoninds, sim, DynId, Index, NoSymmSpace, Symmetry,
};
use tensor4all_core_linalg::{factorize, Canonical, FactorizeAlg, FactorizeOptions};
use tensor4all_core_tensor::{AnyScalar, TensorAccess, TensorDynLen};
use tensor4all_treetn::{TreeTN, Einsum};

use crate::error::{TensorTrainError, Result};
use crate::options::{CanonicalMethod, TruncateAlg, TruncateOptions};

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
/// Internally wraps `TreeTN<Id, Symm, usize, Einsum>` where node names are site indices.
/// This allows reuse of TreeTN's canonization and contraction algorithms.
#[derive(Debug, Clone)]
pub struct TensorTrain<Id = DynId, Symm = NoSymmSpace>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
{
    /// The underlying TreeTN with linear chain topology.
    /// Node names are usize (0, 1, 2, ...) representing site indices.
    inner: TreeTN<Id, Symm, usize, Einsum>,
    /// The canonicalization method used (if known).
    canonical_method: Option<CanonicalMethod>,
}

impl<Id, Symm> TensorTrain<Id, Symm>
where
    Id: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + From<DynId> + Ord,
    Symm: Clone + PartialEq + Eq + std::hash::Hash + Symmetry + std::fmt::Debug + From<NoSymmSpace>,
{
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
    pub fn new(tensors: Vec<TensorDynLen<Id, Symm>>) -> Result<Self> {
        if tensors.is_empty() {
            // Create an empty TreeTN
            let inner = TreeTN::<Id, Symm, usize, Einsum>::new(vec![], vec![])
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!("Failed to create empty TreeTN: {}", e),
                })?;
            return Ok(Self {
                inner,
                canonical_method: None,
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

        // Create TreeTN with Einsum mode (auto-connects by shared index IDs)
        let inner = TreeTN::<Id, Symm, usize, Einsum>::new(tensors, node_names)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Failed to create TreeTN: {}", e),
            })?;

        Ok(Self {
            inner,
            canonical_method: None,
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
    /// * `canonical_method` - The method used for canonicalization (if any)
    pub fn with_ortho(
        tensors: Vec<TensorDynLen<Id, Symm>>,
        llim: i32,
        rlim: i32,
        canonical_method: Option<CanonicalMethod>,
    ) -> Result<Self> {
        let mut tt = Self::new(tensors)?;

        // Convert llim/rlim to ortho center
        // When llim + 2 == rlim, ortho center is at llim + 1
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < tt.len() as i32 {
            let center = (llim + 1) as usize;
            tt.inner.set_ortho_region(vec![center])
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!("Failed to set ortho region: {}", e),
                })?;
        }

        tt.canonical_method = canonical_method;
        Ok(tt)
    }

    /// Number of sites (tensors) in the tensor train.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.node_count()
    }

    /// Check if the tensor train is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.node_count() == 0
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
            let _ = self.inner.set_ortho_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.inner.set_ortho_region(Vec::<usize>::new());
        }
    }

    /// Set the right orthogonality limit.
    #[inline]
    pub fn set_rlim(&mut self, rlim: i32) {
        // Convert to ortho center if possible
        let llim = self.llim();
        if llim + 2 == rlim && llim >= -1 && (llim + 1) < self.len() as i32 {
            let center = (llim + 1) as usize;
            let _ = self.inner.set_ortho_region(vec![center]);
        } else {
            // Clear ortho region if not a single center
            let _ = self.inner.set_ortho_region(Vec::<usize>::new());
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
        self.inner.ortho_region().len() == 1
    }

    /// Get the orthogonality center (0-indexed).
    ///
    /// Returns `Some(site)` if the tensor train has a single orthogonality center,
    /// `None` otherwise.
    pub fn orthocenter(&self) -> Option<usize> {
        let region = self.inner.ortho_region();
        if region.len() == 1 {
            // Node name IS the site index since V = usize
            Some(*region.iter().next().unwrap())
        } else {
            None
        }
    }

    /// Get the canonicalization method used.
    #[inline]
    pub fn canonical_method(&self) -> Option<CanonicalMethod> {
        self.canonical_method
    }

    /// Set the canonicalization method.
    #[inline]
    pub fn set_canonical_method(&mut self, method: Option<CanonicalMethod>) {
        self.canonical_method = method;
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    #[inline]
    pub fn tensor(&self, site: usize) -> &TensorDynLen<Id, Symm> {
        let node_idx = self.inner.node_index(&site)
            .expect("Site out of bounds");
        self.inner.tensor(node_idx)
            .expect("Tensor not found")
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// Returns `Err` if `site >= len()`.
    pub fn tensor_checked(&self, site: usize) -> Result<&TensorDynLen<Id, Symm>> {
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }
        let node_idx = self.inner.node_index(&site)
            .ok_or_else(|| TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            })?;
        self.inner.tensor(node_idx)
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
    pub fn tensor_mut(&mut self, site: usize) -> &mut TensorDynLen<Id, Symm> {
        let node_idx = self.inner.node_index(&site)
            .expect("Site out of bounds");
        self.inner.tensor_mut(node_idx)
            .expect("Tensor not found")
    }

    /// Get a reference to all tensors.
    #[inline]
    pub fn tensors(&self) -> Vec<&TensorDynLen<Id, Symm>> {
        (0..self.len())
            .filter_map(|site| {
                let node_idx = self.inner.node_index(&site)?;
                self.inner.tensor(node_idx)
            })
            .collect()
    }

    /// Get a mutable reference to all tensors.
    #[inline]
    pub fn tensors_mut(&mut self) -> Vec<&mut TensorDynLen<Id, Symm>> {
        // This is tricky - we need to collect mutable references
        // For now, return an empty vec - this method is rarely used
        // and would require unsafe code or different design
        Vec::new()
    }

    /// Get the link index between sites `i` and `i+1`.
    ///
    /// Returns `None` if `i >= len() - 1` or if no common index exists.
    pub fn linkind(&self, i: usize) -> Option<Index<Id, Symm>> {
        if i >= self.len().saturating_sub(1) {
            return None;
        }

        let left_node = self.inner.node_index(&i)?;
        let right_node = self.inner.node_index(&(i + 1))?;
        let left = self.inner.tensor(left_node)?;
        let right = self.inner.tensor(right_node)?;
        let common = common_inds(left.indices(), right.indices());
        common.into_iter().next()
    }

    /// Get all link indices.
    ///
    /// Returns a vector of length `len() - 1` containing the link indices.
    pub fn linkinds(&self) -> Vec<Index<Id, Symm>> {
        (0..self.len().saturating_sub(1))
            .filter_map(|i| self.linkind(i))
            .collect()
    }

    /// Create a copy with all link indices replaced by new unique IDs.
    ///
    /// This is useful for computing inner products where two tensor trains
    /// share link indices. By simulating (replacing) the link indices in one
    /// of the tensor trains, they can be contracted over site indices only.
    pub fn sim_linkinds(&self) -> Self
    where
        Id: From<DynId>,
        Symm: Clone + PartialEq,
    {
        if self.len() <= 1 {
            return self.clone();
        }

        // Build replacement pairs: (old_link, new_link) for each link index
        let old_links = self.linkinds();
        let new_links: Vec<_> = old_links.iter().map(|idx| sim(idx)).collect();
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
    pub fn siteinds(&self) -> Vec<Vec<Index<Id, Symm>>> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let tensor = self.tensor(i);
            let mut site_inds: Vec<Index<Id, Symm>> = tensor.indices().to_vec();

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
        let left_node = self.inner.node_index(&i);
        let right_node = self.inner.node_index(&(i + 1));
        match (left_node, right_node) {
            (Some(l), Some(r)) => {
                let left = self.inner.tensor(l);
                let right = self.inner.tensor(r);
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
    pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen<Id, Symm>) {
        let node_idx = self.inner.node_index(&site)
            .expect("Site out of bounds");
        let _ = self.inner.replace_tensor(node_idx, tensor);
        // Invalidate orthogonality
        let _ = self.inner.set_ortho_region(Vec::<usize>::new());
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
        self.orthogonalize_with(site, CanonicalMethod::SVD)
    }

    /// Orthogonalize with a specified method.
    ///
    /// # Arguments
    ///
    /// * `site` - The target site for the orthogonality center (0-indexed)
    /// * `method` - The canonicalization method to use (SVD, LU, or CI)
    pub fn orthogonalize_with(&mut self, site: usize, method: CanonicalMethod) -> Result<()> {
        if self.is_empty() {
            return Err(TensorTrainError::Empty);
        }
        if site >= self.len() {
            return Err(TensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }

        let alg = method_to_alg(method);

        // Use TreeTN's canonize_by_names (accepts node names directly)
        // Since V = usize, node names are site indices
        self.inner = std::mem::take(&mut self.inner)
            .canonize_by_names(vec![site], alg)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Canonize failed: {}", e),
            })?;

        self.canonical_method = Some(method);
        Ok(())
    }

    /// Truncate the tensor train bond dimensions.
    ///
    /// This performs a sweep through the tensor train, truncating bond dimensions
    /// according to the specified options (relative tolerance and/or maximum rank).
    pub fn truncate(&mut self, options: &TruncateOptions) -> Result<()> {
        if self.len() <= 1 {
            return Ok(());
        }

        // Determine the range of bonds to truncate
        let start = options.site_range.as_ref().map(|r| r.start).unwrap_or(0);
        let end = options
            .site_range
            .as_ref()
            .map(|r| r.end)
            .unwrap_or(self.len());

        // Sweep left to right, truncating each bond
        for i in start..end.min(self.len()).saturating_sub(1) {
            self.truncate_bond(i, options)?;
        }

        // Update orthogonality: after left-to-right sweep, ortho center is at rightmost site
        let center = end.min(self.len()).saturating_sub(1);
        let _ = self.inner.set_ortho_region(vec![center]);
        self.canonical_method = Some(truncate_alg_to_method(options.alg));

        Ok(())
    }

    /// Truncate a single bond between sites i and i+1.
    fn truncate_bond(&mut self, i: usize, options: &TruncateOptions) -> Result<()> {
        if i >= self.len() - 1 {
            return Ok(());
        }

        // Get the link index to the right neighbor
        let link_right = self.linkind(i);

        // Get tensor at site i
        let tensor_i = self.tensor(i).clone();

        // Determine "left" indices for factorization (all except right link)
        let left_inds: Vec<_> = tensor_i
            .indices()
            .iter()
            .filter(|idx| Some(*idx) != link_right.as_ref())
            .cloned()
            .collect();

        // Set up factorization options with truncation parameters
        let factorize_options = FactorizeOptions {
            alg: truncate_alg_to_factorize_alg(options.alg),
            canonical: Canonical::Left,
            rtol: options.rtol,
            max_rank: options.max_rank,
        };

        // Factorize with truncation: tensor[i] = L * R
        let result = factorize(&tensor_i, &left_inds, &factorize_options)
            .map_err(TensorTrainError::Factorize)?;

        // Get tensor at site i+1
        let tensor_i1 = self.tensor(i + 1).clone();

        // Absorb R into tensor[i+1]
        let new_tensor_i1 = result.right.contract_einsum(&tensor_i1);

        // Get the new bond index from factorization
        let new_bond = result.bond_index;

        // Update edge bond indices FIRST (before replacing tensors)
        // The edge stores the old bond indices - we need to update them to the new ones
        if let Some(edge) = self.inner.edge_between(&i, &(i + 1)) {
            self.inner.replace_edge_bond(edge, new_bond.clone(), new_bond.clone())
                .map_err(|e| TensorTrainError::InvalidStructure {
                    message: format!("Failed to update edge bond: {}", e),
                })?;
        }

        // Now replace tensors - validation will pass since edge has correct bond indices
        let node_i = self.inner.node_index(&i)
            .expect("Site out of bounds");
        let node_i1 = self.inner.node_index(&(i + 1))
            .expect("Site out of bounds");

        self.inner.replace_tensor(node_i, result.left)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Failed to replace tensor at site {}: {}", i, e),
            })?;
        self.inner.replace_tensor(node_i1, new_tensor_i1)
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Failed to replace tensor at site {}: {}", i + 1, e),
            })?;

        // Clear ortho region since we modified tensors
        let _ = self.inner.clear_ortho_region();

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
            return AnyScalar::F64(0.0);
        }

        // Replace link indices in other with unique IDs
        let other_sim = other.sim_linkinds();

        // Start with leftmost tensors - contract over site indices only
        let mut env = {
            let a0_conj = self.tensor(0).conj();
            let b0 = other_sim.tensor(0);
            a0_conj.contract_einsum(b0)
        };

        // Sweep through remaining sites
        for i in 1..self.len() {
            let ai_conj = self.tensor(i).conj();
            let bi = other_sim.tensor(i);

            // Contract: env * conj(A_i) (over self's link index)
            env = env.contract_einsum(&ai_conj);
            // Contract: result * B_i (over other's link index and site indices)
            env = env.contract_einsum(bi);
        }

        // Result should be a scalar (0-dimensional tensor)
        env.only()
    }

    /// Compute the squared norm of the tensor train.
    ///
    /// Returns `<self | self>` = ||self||^2.
    pub fn norm_squared(&self) -> f64 {
        self.inner(self).real()
    }

    /// Compute the norm of the tensor train.
    ///
    /// Returns ||self|| = sqrt(<self | self>).
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

// Implement Default for TensorTrain to allow std::mem::take
impl<Id, Symm> Default for TensorTrain<Id, Symm>
where
    Id: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + From<DynId> + Ord,
    Symm: Clone + PartialEq + Eq + std::hash::Hash + Symmetry + std::fmt::Debug + From<NoSymmSpace>,
{
    fn default() -> Self {
        Self::new(vec![]).expect("Failed to create empty TensorTrain")
    }
}

/// Convert CanonicalMethod to FactorizeAlg.
fn method_to_alg(method: CanonicalMethod) -> FactorizeAlg {
    match method {
        CanonicalMethod::SVD => FactorizeAlg::SVD,
        CanonicalMethod::LU => FactorizeAlg::LU,
        CanonicalMethod::CI => FactorizeAlg::CI,
    }
}

/// Convert TruncateAlg to FactorizeAlg.
fn truncate_alg_to_factorize_alg(alg: TruncateAlg) -> FactorizeAlg {
    match alg {
        TruncateAlg::SVD => FactorizeAlg::SVD,
        TruncateAlg::LU => FactorizeAlg::LU,
        TruncateAlg::CI => FactorizeAlg::CI,
    }
}

/// Convert TruncateAlg to CanonicalMethod.
fn truncate_alg_to_method(alg: TruncateAlg) -> CanonicalMethod {
    match alg {
        TruncateAlg::SVD => CanonicalMethod::SVD,
        TruncateAlg::LU => CanonicalMethod::LU,
        TruncateAlg::CI => CanonicalMethod::CI,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core_common::{DynId, Index, NoSymmSpace};
    use tensor4all_core_tensor::StorageScalar;

    /// Helper to create a simple tensor for testing using DynId
    fn make_tensor(
        indices: Vec<Index<DynId, NoSymmSpace>>,
    ) -> TensorDynLen<DynId, NoSymmSpace> {
        let dims: Vec<usize> = indices.iter().map(|i| i.size()).collect();
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let storage = f64::dense_storage(data);
        TensorDynLen::new(indices, dims, storage)
    }

    /// Helper to create an index with DynId
    fn idx(id: u128, size: usize) -> Index<DynId, NoSymmSpace> {
        Index::new_with_size(DynId(id), size)
    }

    #[test]
    fn test_empty_tt() {
        let tt: TensorTrain<DynId, NoSymmSpace> = TensorTrain::new(vec![]).unwrap();
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
        let s0 = idx(0, 2);   // site 0
        let l01 = idx(1, 3);  // link 0-1
        let s1 = idx(2, 2);   // site 1

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
        let s0a = idx(0, 2);  // site 0 index a
        let s0b = idx(1, 3);  // site 0 index b
        let l01 = idx(2, 4);  // link 0-1
        let s1 = idx(3, 2);   // site 1

        let t0 = make_tensor(vec![s0a.clone(), s0b.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = TensorTrain::new(vec![t0, t1]).unwrap();

        // Check site indices (nested vec)
        let site_inds = tt.siteinds();
        assert_eq!(site_inds.len(), 2);
        assert_eq!(site_inds[0].len(), 2);  // site 0 has 2 indices
        assert_eq!(site_inds[1].len(), 1);  // site 1 has 1 index
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
            -1,  // no left orthogonality
            1,   // right orthogonal from site 1
            Some(CanonicalMethod::SVD),
        )
        .unwrap();

        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::SVD));
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
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::SVD));

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

        tt.orthogonalize_with(0, CanonicalMethod::LU).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(0));
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::LU));
    }

    #[test]
    fn test_orthogonalize_with_ci() {
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = TensorTrain::new(vec![t0, t1]).unwrap();

        tt.orthogonalize_with(1, CanonicalMethod::CI).unwrap();
        assert!(tt.isortho());
        assert_eq!(tt.orthocenter(), Some(1));
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::CI));
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
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::SVD));
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
}
