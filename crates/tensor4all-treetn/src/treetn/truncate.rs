//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.

use std::hash::Hash;

use anyhow::Result;

use tensor4all::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all::{Canonical, CanonicalForm, FactorizeAlg, FactorizeOptions};

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

    /// Internal implementation for truncation.
    ///
    /// This is the core truncation logic that public methods delegate to.
    pub(crate) fn truncate_impl(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
        context_name: &str,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        // Determine algorithm from form (use SVD for Unitary in truncation, not QR)
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::SVD,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        // Prepare sweep context
        let sweep_ctx = self.prepare_sweep_to_center(canonical_center, context_name)?;

        // If no centers (empty), nothing to do
        let sweep_ctx = match sweep_ctx {
            Some(ctx) => ctx,
            None => return Ok(()),
        };

        // Set up factorization options WITH truncation parameters
        let factorize_options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            rtol,
            max_rank,
        };

        // Process edges in order (leaves towards center)
        for (src, dst) in &sweep_ctx.edges {
            self.sweep_edge(*src, *dst, &factorize_options, context_name)?;
        }

        // Set the canonical form
        self.canonical_form = Some(form);

        Ok(())
    }
}
