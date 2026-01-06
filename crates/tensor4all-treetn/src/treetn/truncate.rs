//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.

use std::hash::Hash;

use anyhow::Result;

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
    {
        self.truncate_impl(
            canonical_center,
            options.form,
            options.rtol,
            options.max_rank,
            "truncate_opt_mut",
        )
    }

    /// Truncate the tree tensor network towards the specified center nodes (by name).
    ///
    /// Uses SVD (Unitary form) for truncation by default.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    #[deprecated(since = "0.2.0", note = "Use truncate_opt instead")]
    pub fn truncate_by_names(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.truncate_impl(
            canonical_center,
            CanonicalForm::Unitary,
            rtol,
            max_rank,
            "truncate_by_names",
        )?;
        Ok(self)
    }

    /// Truncate the tree tensor network with a specified algorithm (by node name).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    #[deprecated(since = "0.2.0", note = "Use truncate_opt instead")]
    pub fn truncate_by_names_with(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.truncate_impl(
            canonical_center,
            form,
            rtol,
            max_rank,
            "truncate_by_names_with",
        )?;
        Ok(self)
    }

    /// Truncate the tree tensor network in-place (by node name).
    ///
    /// Uses SVD (Unitary form) for truncation by default.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    #[deprecated(since = "0.2.0", note = "Use truncate_opt_mut instead")]
    pub fn truncate_by_names_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        rtol: Option<f64>,
        max_rank: Option<usize>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.truncate_impl(
            canonical_center,
            CanonicalForm::Unitary,
            rtol,
            max_rank,
            "truncate_by_names_mut",
        )
    }

    /// Truncate the tree tensor network in-place with a specified algorithm (by node name).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as the truncation center(s)
    /// * `form` - The canonical form / algorithm to use
    /// * `rtol` - Optional relative tolerance for truncation
    /// * `max_rank` - Optional maximum bond dimension
    #[deprecated(since = "0.2.0", note = "Use truncate_opt_mut instead")]
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
    {
        self.truncate_impl(
            canonical_center,
            form,
            rtol,
            max_rank,
            "truncate_by_names_with_mut",
        )
    }
}
