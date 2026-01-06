//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all::index::{DynId, NoSymmSpace, Symmetry};

use super::TreeTN;
use crate::options::TruncationOptions;

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
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
}
