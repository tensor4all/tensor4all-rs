//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.

use petgraph::stable_graph::NodeIndex;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all::CanonicalForm;

use super::TreeTN;
use crate::options::TruncationOptions;

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    // ========================================================================
    // Consolidated Truncation API
    // ========================================================================

    /// Truncate the network towards the specified center using options.
    ///
    /// This is the recommended unified API for truncation. It accepts:
    /// - Center nodes specified by their node names (V)
    /// - [`TruncationOptions`] to control the form, rtol, and max_rank
    ///
    /// # Example
    /// ```ignore
    /// use tensor4all_treetn::TruncationOptions;
    ///
    /// // Truncate with max rank of 50
    /// let ttn = ttn.truncate_opt(
    ///     ["center"],
    ///     TruncationOptions::default().with_max_rank(50)
    /// )?;
    ///
    /// // Truncate with relative tolerance
    /// let ttn = ttn.truncate_opt(
    ///     ["center"],
    ///     TruncationOptions::default().with_rtol(1e-10)
    /// )?;
    /// ```
    pub fn truncate_opt(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.truncate_impl(
            canonical_center,
            options.form,
            options.rtol,
            options.max_rank,
            "truncate_opt",
        )?;
        Ok(self)
    }

    /// Truncate the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`truncate_opt`].
    pub fn truncate_opt_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        match taken.truncate_opt(canonical_center, options) {
            Ok(result) => {
                *self = result;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    // ========================================================================
    // Legacy Truncation methods (NodeIndex-based)
    // ========================================================================

    /// Truncate the tree tensor network towards the specified canonical_center with SVD.
    ///
    /// This performs a sweep from leaves to the canonical_center, truncating bond dimensions
    /// at each edge using SVD with the specified tolerance and/or maximum rank.
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation (singular values with σ_i/σ_max < rtol are dropped)
    /// * `max_rank` - Optional maximum bond dimension
    ///
    /// # Returns
    /// A new truncated TreeTN, or an error if validation fails or factorization fails.
    pub fn truncate(
        self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        #[allow(deprecated)]
        self.truncate_with(canonical_center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Truncate the tree tensor network towards the specified canonical_center with a specified algorithm.
    ///
    /// This performs a sweep from leaves to the canonical_center, truncating bond dimensions
    /// at each edge using the specified algorithm (SVD, LU, or CI).
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use (Unitary=SVD, LU, CI)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    ///
    /// # Returns
    /// A new truncated TreeTN, or an error if validation fails or factorization fails.
    #[deprecated(since = "0.2.0", note = "Use truncate_opt instead")]
    pub fn truncate_with(
        mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        // Convert NodeIndex to V and delegate to the new implementation
        let canonical_center_v: Vec<V> = canonical_center.into_iter()
            .map(V::from)
            .collect();
        self.truncate_impl(canonical_center_v, form, rtol, max_rank, "truncate_with")?;
        Ok(self)
    }

    /// Truncate the tree tensor network in-place towards the specified canonical_center.
    ///
    /// This is the `&mut self` version of `truncate`.
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        self.truncate_with_mut(canonical_center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Truncate the tree tensor network in-place with a specified algorithm.
    ///
    /// This is the `&mut self` version of `truncate_with`.
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_with_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        #[allow(deprecated)]
        match taken.truncate_with(canonical_center, form, rtol, max_rank) {
            Ok(truncated) => {
                *self = truncated;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    // ========================================================================
    // Truncation methods (by node name V)
    // ========================================================================

    /// Truncate the tree tensor network towards the specified center nodes (by name).
    ///
    /// Uses SVD (Unitary form) for truncation by default.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_by_names(
        self,
        canonical_center: impl IntoIterator<Item = V>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        self.truncate_by_names_with(canonical_center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Truncate the tree tensor network with a specified algorithm (by node name).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_by_names_with(
        self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        // Convert node names to NodeIndex
        let center_indices: Vec<NodeIndex> = canonical_center
            .into_iter()
            .filter_map(|name| self.node_index(&name))
            .collect();

        if center_indices.is_empty() {
            return Err(anyhow::anyhow!("No valid center nodes found"))
                .context("truncate_by_names_with: all specified node names must exist in the network");
        }

        // Delegate to the NodeIndex-based implementation
        #[allow(deprecated)]
        self.truncate_with(center_indices, form, rtol, max_rank)
    }

    /// Truncate the tree tensor network in-place (by node name).
    ///
    /// Uses SVD (Unitary form) for truncation by default.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_by_names_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        self.truncate_by_names_with_mut(canonical_center, CanonicalForm::Unitary, rtol, max_rank)
    }

    /// Truncate the tree tensor network in-place with a specified algorithm (by node name).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    pub fn truncate_by_names_with_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        // Convert node names to NodeIndex
        let center_indices: Vec<NodeIndex> = canonical_center
            .into_iter()
            .filter_map(|name| self.node_index(&name))
            .collect();

        if center_indices.is_empty() {
            return Err(anyhow::anyhow!("No valid center nodes found"))
                .context("truncate_by_names_with_mut: all specified node names must exist in the network");
        }

        // Delegate to the owned truncate_with (using mem::take pattern)
        let taken = std::mem::take(self);
        #[allow(deprecated)]
        match taken.truncate_with(center_indices, form, rtol, max_rank) {
            Ok(truncated) => {
                *self = truncated;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}
