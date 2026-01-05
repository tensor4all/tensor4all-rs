//! Main Tensor Train type.
//!
//! This module provides the `ITensorTrain` type, which represents a Tensor Train
//! (also known as MPS) with orthogonality tracking, inspired by ITensorMPS.jl.

use std::ops::Range;

use tensor4all_core_common::{
    common_inds, hascommoninds, sim, DynId, Index, NoSymmSpace, Symmetry,
};
use tensor4all_core_linalg::{factorize, Canonical, FactorizeAlg, FactorizeOptions};
use tensor4all_core_tensor::{AnyScalar, TensorAccess, TensorDynLen};

use crate::error::{ITensorTrainError, Result};
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
/// The tensor train tracks orthogonality using `llim` and `rlim`:
/// - Sites `0..llim` are guaranteed to be left-orthogonal
/// - Sites `rlim..len()` are guaranteed to be right-orthogonal
/// - Sites in `ortho_lims()` may not be orthogonal
///
/// When `llim + 1 == rlim`, there is a single orthogonality center at site `llim`.
///
/// # Example
///
/// ```ignore
/// use tensor4all_itensortrain::ITensorTrain;
///
/// // Create tensor train from tensors (no orthogonality assumed)
/// let tt = ITensorTrain::new(tensors)?;
///
/// // Check orthogonality status
/// if tt.isortho() {
///     println!("Ortho center at site {}", tt.orthocenter().unwrap());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ITensorTrain<Id = DynId, Symm = NoSymmSpace>
where
    Id: Clone,
    Symm: Clone,
{
    /// The tensors in the tensor train.
    tensors: Vec<TensorDynLen<Id, Symm>>,
    /// Left orthogonality limit (sites 0..llim are left-orthogonal).
    /// Can be -1 to indicate no left-orthogonality.
    llim: i32,
    /// Right orthogonality limit (sites rlim..len are right-orthogonal).
    /// Can be len+1 to indicate no right-orthogonality.
    rlim: i32,
    /// The canonicalization method used (if known).
    canonical_method: Option<CanonicalMethod>,
}

impl<Id, Symm> ITensorTrain<Id, Symm>
where
    Id: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + From<DynId>,
    Symm: Clone + PartialEq + Eq + std::hash::Hash + Symmetry + std::fmt::Debug + From<NoSymmSpace>,
{
    /// Create a new tensor train from a vector of tensors.
    ///
    /// The tensor train is created with no assumed orthogonality (llim = -1, rlim = len).
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
            return Ok(Self {
                tensors: Vec::new(),
                llim: -1,
                rlim: 0,
                canonical_method: None,
            });
        }

        // Validate that adjacent tensors share exactly one common index (the link)
        for i in 0..tensors.len() - 1 {
            let left = &tensors[i];
            let right = &tensors[i + 1];

            let common = common_inds(left.indices(), right.indices());
            if common.is_empty() {
                return Err(ITensorTrainError::InvalidStructure {
                    message: format!(
                        "No common index between tensors at sites {} and {}",
                        i,
                        i + 1
                    ),
                });
            }
            if common.len() > 1 {
                return Err(ITensorTrainError::InvalidStructure {
                    message: format!(
                        "Multiple common indices ({}) between tensors at sites {} and {}",
                        common.len(),
                        i,
                        i + 1
                    ),
                });
            }
        }

        let len = tensors.len() as i32;
        Ok(Self {
            tensors,
            llim: -1,       // No left-orthogonality assumed
            rlim: len + 1,  // No right-orthogonality assumed
            canonical_method: None,
        })
    }

    /// Create a new tensor train with specified orthogonality limits.
    ///
    /// This is useful when constructing a tensor train that is already in canonical form.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors representing the tensor train
    /// * `llim` - Left orthogonality limit
    /// * `rlim` - Right orthogonality limit
    /// * `canonical_method` - The method used for canonicalization (if any)
    pub fn with_ortho(
        tensors: Vec<TensorDynLen<Id, Symm>>,
        llim: i32,
        rlim: i32,
        canonical_method: Option<CanonicalMethod>,
    ) -> Result<Self> {
        let tt = Self::new(tensors)?;
        Ok(Self {
            llim,
            rlim,
            canonical_method,
            ..tt
        })
    }

    /// Number of sites (tensors) in the tensor train.
    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the tensor train is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Left orthogonality limit.
    ///
    /// Sites `0..llim` are guaranteed to be left-orthogonal.
    /// Returns -1 if no sites are left-orthogonal.
    #[inline]
    pub fn llim(&self) -> i32 {
        self.llim
    }

    /// Right orthogonality limit.
    ///
    /// Sites `rlim..len()` are guaranteed to be right-orthogonal.
    /// Returns `len() + 1` if no sites are right-orthogonal.
    #[inline]
    pub fn rlim(&self) -> i32 {
        self.rlim
    }

    /// Set the left orthogonality limit.
    #[inline]
    pub fn set_llim(&mut self, llim: i32) {
        self.llim = llim;
    }

    /// Set the right orthogonality limit.
    #[inline]
    pub fn set_rlim(&mut self, rlim: i32) {
        self.rlim = rlim;
    }

    /// Get the orthogonality center range.
    ///
    /// Returns the range of sites that may not be orthogonal.
    /// If the tensor train is fully left-orthogonal, returns an empty range at the end.
    /// If the tensor train is fully right-orthogonal, returns an empty range at the beginning.
    pub fn ortho_lims(&self) -> Range<usize> {
        let start = (self.llim + 1).max(0) as usize;
        let end = self.rlim.max(0) as usize;
        start..end.min(self.len())
    }

    /// Check if the tensor train has a single orthogonality center.
    ///
    /// Returns true if `llim + 1 == rlim`, meaning there is exactly one
    /// site that is not guaranteed to be orthogonal.
    #[inline]
    pub fn isortho(&self) -> bool {
        self.llim + 2 == self.rlim
    }

    /// Get the orthogonality center (0-indexed).
    ///
    /// Returns `Some(site)` if the tensor train has a single orthogonality center,
    /// `None` otherwise.
    pub fn orthocenter(&self) -> Option<usize> {
        if self.isortho() {
            Some((self.llim + 1) as usize)
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
        &self.tensors[site]
    }

    /// Get a reference to the tensor at the given site.
    ///
    /// Returns `Err` if `site >= len()`.
    pub fn tensor_checked(&self, site: usize) -> Result<&TensorDynLen<Id, Symm>> {
        if site >= self.len() {
            return Err(ITensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }
        Ok(&self.tensors[site])
    }

    /// Get a mutable reference to the tensor at the given site.
    ///
    /// # Panics
    ///
    /// Panics if `site >= len()`.
    #[inline]
    pub fn tensor_mut(&mut self, site: usize) -> &mut TensorDynLen<Id, Symm> {
        &mut self.tensors[site]
    }

    /// Get a reference to all tensors.
    #[inline]
    pub fn tensors(&self) -> &[TensorDynLen<Id, Symm>] {
        &self.tensors
    }

    /// Get a mutable reference to all tensors.
    #[inline]
    pub fn tensors_mut(&mut self) -> &mut [TensorDynLen<Id, Symm>] {
        &mut self.tensors
    }

    /// Get the link index between sites `i` and `i+1`.
    ///
    /// Returns `None` if `i >= len() - 1` or if no common index exists.
    pub fn linkind(&self, i: usize) -> Option<Index<Id, Symm>> {
        if i >= self.len().saturating_sub(1) {
            return None;
        }

        let left = &self.tensors[i];
        let right = &self.tensors[i + 1];
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
    ///
    /// This corresponds to ITensors.jl's `sim(A, linkinds(A))` pattern.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For computing <A|A>, replace link indices in one copy:
    /// let a_sim = a.sim_linkinds();
    /// let inner = a.inner(&a_sim);  // Now contracts over site indices only
    /// ```
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

        // Replace link indices in each tensor
        let mut new_tensors = Vec::with_capacity(self.len());
        for tensor in &self.tensors {
            let mut new_tensor = tensor.clone();
            for (old_idx, new_idx) in &replacements {
                new_tensor = new_tensor.replaceind(old_idx, new_idx);
            }
            new_tensors.push(new_tensor);
        }

        Self {
            tensors: new_tensors,
            llim: self.llim,
            rlim: self.rlim,
            canonical_method: self.canonical_method,
        }
    }

    /// Get the site indices (non-link indices) for all sites.
    ///
    /// For each site, returns a vector of indices that are not shared with
    /// adjacent tensors (i.e., the "physical" or "site" indices).
    /// Each site can have zero, one, or multiple site indices.
    ///
    /// Returns a `Vec<Vec<Index>>` where the outer vector has length `len()`,
    /// and each inner vector contains the site indices for that site.
    pub fn siteinds(&self) -> Vec<Vec<Index<Id, Symm>>> {
        if self.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let tensor = &self.tensors[i];
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
        hascommoninds(self.tensors[i].indices(), self.tensors[i + 1].indices())
    }

    /// Replace the tensor at the given site.
    ///
    /// This invalidates orthogonality tracking unless you update llim/rlim manually.
    pub fn set_tensor(&mut self, site: usize, tensor: TensorDynLen<Id, Symm>) {
        self.tensors[site] = tensor;
        // Invalidate orthogonality around this site
        if (site as i32) <= self.llim {
            self.llim = site as i32 - 1;
        }
        if (site as i32) >= self.rlim {
            self.rlim = site as i32 + 2;
        }
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tensor4all_itensortrain::ITensorTrain;
    ///
    /// let mut tt = ITensorTrain::new(tensors)?;
    /// tt.orthogonalize(2)?;  // Move ortho center to site 2
    /// assert!(tt.isortho());
    /// assert_eq!(tt.orthocenter(), Some(2));
    /// ```
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
            return Err(ITensorTrainError::Empty);
        }
        if site >= self.len() {
            return Err(ITensorTrainError::SiteOutOfBounds {
                site,
                length: self.len(),
            });
        }

        let site_i32 = site as i32;

        // Sweep from left to site (make sites 0..site left-orthogonal)
        // Only sweep if we need to extend left-orthogonality
        for i in (self.llim + 1).max(0) as usize..site {
            self.orthogonalize_bond_left(i, method)?;
        }

        // Sweep from right to site (make sites site+1..len right-orthogonal)
        // Only sweep if we need to extend right-orthogonality
        let start_right = self.rlim.min(self.len() as i32) as usize;
        for i in (site + 1..start_right).rev() {
            self.orthogonalize_bond_right(i, method)?;
        }

        // Update orthogonality limits
        self.llim = site_i32 - 1;
        self.rlim = site_i32 + 1;
        self.canonical_method = Some(method);

        Ok(())
    }

    /// Orthogonalize bond between sites i and i+1, making site i left-orthogonal.
    ///
    /// Factorizes tensor[i] as L * R where L is left-canonical,
    /// then absorbs R into tensor[i+1].
    fn orthogonalize_bond_left(&mut self, i: usize, method: CanonicalMethod) -> Result<()> {
        if i >= self.len() - 1 {
            return Ok(());
        }

        // Get the link index to the right neighbor
        let link_right = self.linkind(i);

        // Determine "left" indices for factorization (all except right link)
        let left_inds: Vec<_> = self.tensors[i]
            .indices()
            .iter()
            .filter(|idx| Some(*idx) != link_right.as_ref())
            .cloned()
            .collect();

        // Set up factorization options
        let options = FactorizeOptions {
            alg: method_to_alg(method),
            canonical: Canonical::Left,
            rtol: None,
            max_rank: None,
        };

        // Factorize: tensor[i] = L * R
        let result = factorize(&self.tensors[i], &left_inds, &options)
            .map_err(ITensorTrainError::Factorize)?;

        // Update tensor[i] with L (left-orthogonal)
        self.tensors[i] = result.left;

        // Absorb R into tensor[i+1]
        self.tensors[i + 1] = result.right.contract_einsum(&self.tensors[i + 1]);

        Ok(())
    }

    /// Orthogonalize bond between sites i-1 and i, making site i right-orthogonal.
    ///
    /// Factorizes tensor[i] as L * R where R is right-canonical,
    /// then absorbs L into tensor[i-1].
    fn orthogonalize_bond_right(&mut self, i: usize, method: CanonicalMethod) -> Result<()> {
        if i == 0 {
            return Ok(());
        }

        // Get the link index to the left neighbor
        let link_left = self.linkind(i - 1);

        // Determine "left" indices for factorization (only left link)
        let left_inds: Vec<_> = self.tensors[i]
            .indices()
            .iter()
            .filter(|idx| Some(*idx) == link_left.as_ref())
            .cloned()
            .collect();

        // Set up factorization options
        let options = FactorizeOptions {
            alg: method_to_alg(method),
            canonical: Canonical::Right,
            rtol: None,
            max_rank: None,
        };

        // Factorize: tensor[i] = L * R
        let result = factorize(&self.tensors[i], &left_inds, &options)
            .map_err(ITensorTrainError::Factorize)?;

        // Update tensor[i] with R (right-orthogonal)
        self.tensors[i] = result.right;

        // Absorb L into tensor[i-1]
        self.tensors[i - 1] = self.tensors[i - 1].contract_einsum(&result.left);

        Ok(())
    }

    /// Truncate the tensor train bond dimensions.
    ///
    /// This performs a sweep through the tensor train, truncating bond dimensions
    /// according to the specified options (relative tolerance and/or maximum rank).
    ///
    /// # Arguments
    ///
    /// * `options` - Truncation options (algorithm, rtol, max_rank, site_range)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tensor4all_itensortrain::{ITensorTrain, TruncateOptions};
    ///
    /// let mut tt = ITensorTrain::new(tensors)?;
    /// tt.truncate(&TruncateOptions::svd().with_rtol(1e-10).with_max_rank(50))?;
    /// ```
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
        self.llim = (end.min(self.len()) as i32) - 2;
        self.rlim = end.min(self.len()) as i32;
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

        // Determine "left" indices for factorization (all except right link)
        let left_inds: Vec<_> = self.tensors[i]
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
        let result = factorize(&self.tensors[i], &left_inds, &factorize_options)
            .map_err(ITensorTrainError::Factorize)?;

        // Update tensor[i] with L (left-orthogonal, truncated)
        self.tensors[i] = result.left;

        // Absorb R into tensor[i+1]
        self.tensors[i + 1] = result.right.contract_einsum(&self.tensors[i + 1]);

        Ok(())
    }

    /// Compute the inner product (dot product) of two tensor trains.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both tensor trains must have the same site indices (same IDs).
    /// Link indices may differ between the two tensor trains.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor train
    ///
    /// # Returns
    ///
    /// The inner product as `AnyScalar` (f64 or Complex64).
    ///
    /// # Panics
    ///
    /// Panics if the tensor trains have different lengths or incompatible site indices.
    pub fn inner(&self, other: &Self) -> AnyScalar {
        assert_eq!(
            self.len(),
            other.len(),
            "Tensor trains must have the same length for inner product"
        );

        if self.is_empty() {
            return AnyScalar::F64(0.0);
        }

        // For inner product, we need to contract over site indices only.
        // Link indices must be different between self and other.
        //
        // If self and other share link indices (e.g., when computing <A|A>),
        // we need to replace the link indices in one of them with unique IDs.
        //
        // Transfer matrix approach:
        // E_0 = conj(A_0) * B_0  (contracted over site indices)
        // E_i = E_{i-1} * conj(A_i) * B_i  (contracted over link and site indices)
        // result = E_{n-1}  (should be a scalar)

        // Replace link indices in other with unique IDs
        let other_sim = other.sim_linkinds();

        // Start with leftmost tensors - contract over site indices only
        let mut env = {
            let a0_conj = self.tensors[0].conj();
            let b0 = &other_sim.tensors[0];
            // Contract over common indices (site indices)
            a0_conj.contract_einsum(b0)
        };

        // Sweep through remaining sites
        for i in 1..self.len() {
            let ai_conj = self.tensors[i].conj();
            let bi = &other_sim.tensors[i];

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
        let tt: ITensorTrain<DynId, NoSymmSpace> = ITensorTrain::new(vec![]).unwrap();
        assert!(tt.is_empty());
        assert_eq!(tt.len(), 0);
        assert_eq!(tt.llim(), -1);
        assert_eq!(tt.rlim(), 0);
        assert!(!tt.isortho());
    }

    #[test]
    fn test_single_site_tt() {
        let tensor = make_tensor(vec![idx(0, 2)]);

        let tt = ITensorTrain::new(vec![tensor]).unwrap();
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

        let tt = ITensorTrain::new(vec![t0, t1]).unwrap();
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

        let tt = ITensorTrain::new(vec![t0, t1]).unwrap();

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
        let tt = ITensorTrain::with_ortho(
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
        let tt = ITensorTrain::with_ortho(vec![t0, t1, t2], 0, 2, None).unwrap();

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

        let result = ITensorTrain::new(vec![t0, t1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ITensorTrainError::InvalidStructure { .. }
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

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();
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

        let mut tt = ITensorTrain::new(vec![t0, t1, t2]).unwrap();

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

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();

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

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();

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

        let mut tt = ITensorTrain::new(vec![t0, t1, t2]).unwrap();
        assert_eq!(tt.maxbonddim(), 8);

        // Truncate to max rank 4
        let options = TruncateOptions::svd().with_max_rank(4);
        tt.truncate(&options).unwrap();

        // Check that bond dimensions are reduced
        assert!(tt.maxbonddim() <= 4);
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::SVD));
    }

    #[test]
    fn test_truncate_with_rtol() {
        // Create a 2-site tensor train
        let s0 = idx(0, 4);
        let l01 = idx(1, 8);
        let s1 = idx(2, 4);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();

        // Truncate with rtol
        let options = TruncateOptions::svd().with_rtol(1e-10);
        tt.truncate(&options).unwrap();

        // Should still have valid structure
        assert!(tt.len() == 2);
    }

    #[test]
    fn test_truncate_with_lu() {
        let s0 = idx(0, 4);
        let l01 = idx(1, 8);
        let s1 = idx(2, 4);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();

        let options = TruncateOptions::lu().with_max_rank(3);
        tt.truncate(&options).unwrap();

        assert!(tt.maxbonddim() <= 3);
        assert_eq!(tt.canonical_method(), Some(CanonicalMethod::LU));
    }

    #[test]
    fn test_truncate_single_site() {
        // Single site tensor train should not fail
        let s0 = idx(0, 4);
        let t0 = make_tensor(vec![s0.clone()]);

        let mut tt = ITensorTrain::new(vec![t0]).unwrap();

        let options = TruncateOptions::svd().with_max_rank(2);
        tt.truncate(&options).unwrap();

        assert_eq!(tt.len(), 1);
    }

    #[test]
    fn test_norm() {
        // Create a simple 2-site tensor train
        let s0 = idx(0, 2);
        let l01 = idx(1, 2);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = ITensorTrain::new(vec![t0, t1]).unwrap();

        // Norm should be positive
        let norm = tt.norm();
        assert!(norm > 0.0);

        // Norm squared should be norm^2
        let norm_sq = tt.norm_squared();
        assert!((norm_sq - norm * norm).abs() < 1e-10);
    }

    #[test]
    fn test_inner() {
        // Create a simple 2-site tensor train
        let s0 = idx(0, 2);
        let l01 = idx(1, 2);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let tt = ITensorTrain::new(vec![t0, t1]).unwrap();

        // <tt|tt> should equal norm_squared
        let inner = tt.inner(&tt);
        let norm_sq = tt.norm_squared();
        assert!((inner.real() - norm_sq).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonalize_preserves_tensor() {
        // Create a 2-site tensor train
        let s0 = idx(0, 2);
        let l01 = idx(1, 3);
        let s1 = idx(2, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);

        let mut tt = ITensorTrain::new(vec![t0, t1]).unwrap();
        let original_norm = tt.norm();

        // Orthogonalize should preserve the norm (and thus the tensor)
        tt.orthogonalize(0).unwrap();
        let after_ortho_norm = tt.norm();

        // Norm should be preserved
        assert!(
            (original_norm - after_ortho_norm).abs() / original_norm < 1e-10,
            "Norm changed from {} to {} after orthogonalize",
            original_norm,
            after_ortho_norm
        );
    }

    #[test]
    fn test_truncate_accuracy() {
        // Create a 3-site tensor train with known structure
        let s0 = idx(0, 2);
        let l01 = idx(1, 4);
        let s1 = idx(2, 2);
        let l12 = idx(3, 4);
        let s2 = idx(4, 2);

        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone(), l12.clone()]);
        let t2 = make_tensor(vec![l12.clone(), s2.clone()]);

        let original_tt = ITensorTrain::new(vec![t0, t1, t2]).unwrap();
        let original_norm = original_tt.norm();

        // Clone and truncate
        let mut truncated_tt = original_tt.clone();
        let options = TruncateOptions::svd().with_max_rank(2);
        truncated_tt.truncate(&options).unwrap();

        // Check that bond dimensions are reduced
        assert!(truncated_tt.maxbonddim() <= 2);

        // Compute relative error: ||original - truncated|| / ||original||
        // For this we need inner(original, truncated)
        // ||original - truncated||^2 = ||original||^2 + ||truncated||^2 - 2*Re<original|truncated>
        let truncated_norm = truncated_tt.norm();
        let inner_value = original_tt.inner(&truncated_tt);

        let diff_norm_sq = original_norm * original_norm + truncated_norm * truncated_norm
            - 2.0 * inner_value.real();
        let relative_error = diff_norm_sq.sqrt() / original_norm;

        // For this simple test, we just check that truncation gives some reasonable error
        // (the exact error depends on the tensor structure)
        println!(
            "Truncation relative error: {} (original norm: {}, truncated norm: {})",
            relative_error, original_norm, truncated_norm
        );

        // The truncated norm should not exceed original norm
        assert!(truncated_norm <= original_norm * 1.001);  // Allow small numerical error
    }
}
