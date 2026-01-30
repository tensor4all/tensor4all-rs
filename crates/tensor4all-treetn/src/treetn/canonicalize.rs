//! Canonicalization methods for TreeTN.
//!
//! This module provides methods for canonicalizing tree tensor networks.

use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use tensor4all_core::{Canonical, FactorizeAlg, FactorizeOptions, TensorLike};

use super::TreeTN;
use crate::options::CanonicalizationOptions;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Canonicalize the network towards the specified center using options.
    ///
    /// This is the recommended unified API for canonicalization. It accepts:
    /// - Center nodes specified by their node names (V)
    /// - [`CanonicalizationOptions`] to control the form and force behavior
    ///
    /// # Behavior
    /// - If `options.force` is false (default):
    ///   - Already at target with same form: returns unchanged (no-op)
    ///   - Different form: returns an error (use `options.force()` to override)
    /// - If `options.force` is true:
    ///   - Always performs full canonicalization
    ///
    /// # Example
    /// ```ignore
    /// use tensor4all_treetn::CanonicalizationOptions;
    ///
    /// // Default canonicalization (Unitary form, smart behavior)
    /// let ttn = ttn.canonicalize(["A"], CanonicalizationOptions::default())?;
    ///
    /// // Force re-canonicalization with LU form
    /// let ttn = ttn.canonicalize(
    ///     ["B"],
    ///     CanonicalizationOptions::forced().with_form(CanonicalForm::LU)
    /// )?;
    /// ```
    pub fn canonicalize(
        mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: CanonicalizationOptions,
    ) -> Result<Self> {
        let center_v: HashSet<V> = canonical_region.into_iter().collect();

        // Smart behavior when not forced
        if !options.force {
            // Check if already canonicalized with a different form
            if let Some(current_form) = self.canonical_form {
                if current_form != options.form {
                    return Err(anyhow::anyhow!(
                        "Cannot move ortho center: current form is {:?} but {:?} was requested. \
                         Use CanonicalizationOptions::forced() to re-canonicalize with a different form.",
                        current_form,
                        options.form
                    ))
                    .context("canonicalize: form mismatch");
                }
            }

            // Check if already at target
            if self.canonical_region == center_v && self.canonical_form == Some(options.form) {
                return Ok(self);
            }
        }

        // Perform canonicalization
        self.canonicalize_impl(center_v, options.form, "canonicalize")?;
        Ok(self)
    }

    /// Canonicalize the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`Self::canonicalize`].
    pub fn canonicalize_mut(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: CanonicalizationOptions,
    ) -> Result<()>
    where
        Self: Default,
    {
        let taken = std::mem::take(self);
        match taken.canonicalize(canonical_region, options) {
            Ok(result) => {
                *self = result;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Internal implementation for canonicalization.
    ///
    /// This is the core canonicalization logic that public methods delegate to.
    pub(crate) fn canonicalize_impl(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        context_name: &str,
    ) -> Result<()> {
        // Determine algorithm from form
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::QR,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        // Prepare sweep context
        let sweep_ctx = self.prepare_sweep_to_center(canonical_region, context_name)?;

        // If no centers (empty), nothing to do
        let sweep_ctx = match sweep_ctx {
            Some(ctx) => ctx,
            None => return Ok(()),
        };

        // Set up factorization options (no truncation for canonicalization)
        let factorize_options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            rtol: None,
            max_rank: None,
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
