//! Canonicalization methods for TreeTN.
//!
//! This module provides methods for canonicalizing tree tensor networks.

use petgraph::stable_graph::NodeIndex;
use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all::{Canonical, CanonicalForm, FactorizeAlg, FactorizeOptions};

use super::TreeTN;
use crate::options::CanonicalizationOptions;

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    // ========================================================================
    // Consolidated Canonicalization API
    // ========================================================================

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
    /// let ttn = ttn.canonicalize_opt(["A"], CanonicalizationOptions::default())?;
    ///
    /// // Force re-canonicalization with LU form
    /// let ttn = ttn.canonicalize_opt(
    ///     ["B"],
    ///     CanonicalizationOptions::forced().with_form(CanonicalForm::LU)
    /// )?;
    /// ```
    pub fn canonicalize_opt(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: CanonicalizationOptions,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        let center_v: HashSet<V> = canonical_center.into_iter().collect();

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
                    .context("canonicalize_opt: form mismatch");
                }
            }

            // Check if already at target
            if self.canonical_center == center_v && self.canonical_form == Some(options.form) {
                return Ok(self);
            }
        }

        // Perform canonicalization
        self.canonicalize_impl(center_v, options.form, "canonicalize_opt")?;
        Ok(self)
    }

    /// Canonicalize the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`canonicalize_opt`].
    pub fn canonicalize_opt_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: CanonicalizationOptions,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        match taken.canonicalize_opt(canonical_center, options) {
            Ok(result) => {
                *self = result;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    // ========================================================================
    // Legacy Canonicalization API (for backward compatibility)
    // ========================================================================

    /// Canonicalize the network towards the specified canonical_center.
    ///
    /// This is a smart canonicalization that checks the current state:
    /// - If already canonicalized to the same canonical_center with the same form, returns unchanged
    /// - Otherwise, performs full canonicalization
    ///
    /// Uses the default canonical form (Unitary).
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    #[deprecated(since = "0.2.0", note = "Use canonicalize_opt instead")]
    pub fn canonicalize(
        self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        self.canonicalize_with(canonical_center, CanonicalForm::Unitary)
    }

    /// Canonicalize the network towards the specified canonical_center using a specified canonical form.
    ///
    /// This is a smart canonicalization that:
    /// - If not canonicalized: performs full canonicalization
    /// - If already at target with same form: returns unchanged (no-op)
    /// - If already canonicalized but at different position: moves ortho center efficiently
    /// - If already canonicalized with different form: returns an error
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if:
    /// - Validation fails
    /// - Factorization fails
    /// - Already canonicalized with a different form (use `force_canonicalize_with` to change form)
    pub fn canonicalize_with(
        self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        let target_indices: Vec<NodeIndex> = canonical_center.into_iter().collect();

        // Check if already canonicalized with a different form
        if let Some(current_form) = self.canonical_form {
            if current_form != form {
                return Err(anyhow::anyhow!(
                    "Cannot move ortho center: current form is {:?} but {:?} was requested. \
                     Use force_canonicalize_with() to re-canonicalize with a different form.",
                    current_form,
                    form
                ))
                .context("canonicalize_with: form mismatch");
            }
        }

        // Convert target to V for comparison
        let target_v: HashSet<V> = target_indices.iter()
            .map(|&idx| V::from(idx))
            .collect();

        // Check if already at target
        if self.canonical_center == target_v && self.canonical_form == Some(form) {
            return Ok(self);
        }

        // For single target (most common case), use edges_to_canonicalize
        if target_indices.len() == 1 {
            let target = target_indices[0];

            // Get current region as NodeIndex set
            let current_region: Option<HashSet<NodeIndex>> = if self.is_canonicalized() {
                Some(self.canonical_center.iter()
                    .filter_map(|v| self.graph.node_index(v))
                    .collect())
            } else {
                None
            };

            // Compute edges to process
            let edges = self.site_index_network.edges_to_canonicalize(
                current_region.as_ref(),
                target,
            );

            if edges.is_empty() {
                return Ok(self);
            }

            // Process edges via force_canonicalize_with for now
            #[allow(deprecated)]
            self.force_canonicalize_with(target_indices, form)
        } else {
            // Multiple targets: use force_canonicalize_with
            #[allow(deprecated)]
            self.force_canonicalize_with(target_indices, form)
        }
    }

    /// Force canonicalize the network towards the specified canonical_center.
    ///
    /// This method always performs full canonicalization, ignoring the current state.
    /// Uses the default canonical form (Unitary).
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    pub fn force_canonicalize(
        self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        #[allow(deprecated)]
        self.force_canonicalize_with(canonical_center, CanonicalForm::Unitary)
    }

    /// Force canonicalize the network towards the specified canonical_center using a specified canonical form.
    ///
    /// This method always performs full canonicalization, ignoring the current state.
    /// The algorithm:
    /// 1. Validates that the graph is a tree
    /// 2. Sets the canonical_center and validates connectivity
    /// 3. Computes distances from canonical_center using BFS
    /// 4. Processes nodes in order of decreasing distance (farthest first)
    /// 5. For each node, performs factorization on edges pointing towards canonical_center
    /// 6. Absorbs the right factor into parent nodes using tensordot
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    /// * `form` - The canonical form to use:
    ///   - `Unitary`: Uses QR decomposition, each tensor is isometric
    ///   - `LU`: Uses LU decomposition, one factor has unit diagonal
    ///   - `CI`: Uses Cross Interpolation
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The graph is not a tree
    /// - canonical_center are not connected
    /// - Factorization fails
    #[deprecated(since = "0.2.0", note = "Use canonicalize_opt instead")]
    pub fn force_canonicalize_with(
        mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
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
        self.canonicalize_impl(canonical_center_v, form, "force_canonicalize_with")?;
        Ok(self)
    }

    /// Canonicalize the network towards the specified canonical_center using node names directly.
    ///
    /// This is a smart canonicalization that checks the current state:
    /// - If already canonicalized to the same canonical_center, returns unchanged
    /// - Otherwise, performs full canonicalization via `force_canonicalize_by_names`
    ///
    /// This is a variant of `canonicalize` that accepts node names (V) directly,
    /// rather than requiring conversion from `NodeIndex`. This is useful when
    /// `V` does not implement `From<NodeIndex>` (e.g., when `V = usize`).
    ///
    /// Uses the default canonical form (`CanonicalForm::Unitary`).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    pub fn canonicalize_by_names(
        self,
        canonical_center: impl IntoIterator<Item = V>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        self.canonicalize_by_names_with(canonical_center, CanonicalForm::Unitary)
    }

    /// Canonicalize the network towards the specified canonical_center using node names directly
    /// with a specified canonical form.
    ///
    /// This is a smart canonicalization that:
    /// - If not canonicalized: performs full canonicalization
    /// - If already at target with same form: returns unchanged (no-op)
    /// - If already canonicalized but at different position: moves ortho center efficiently
    /// - If already canonicalized with different form: returns an error
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if:
    /// - Validation fails
    /// - Factorization fails
    /// - Already canonicalized with a different form (use `force_canonicalize_by_names_with` to change form)
    pub fn canonicalize_by_names_with(
        self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        let canonical_center_v: HashSet<V> = canonical_center.into_iter().collect();

        // Check if already canonicalized with a different form
        if let Some(current_form) = self.canonical_form {
            if current_form != form {
                return Err(anyhow::anyhow!(
                    "Cannot move ortho center: current form is {:?} but {:?} was requested. \
                     Use force_canonicalize_by_names_with() to re-canonicalize with a different form.",
                    current_form,
                    form
                ))
                .context("canonicalize_by_names_with: form mismatch");
            }
        }

        // Check if already at target
        if self.canonical_center == canonical_center_v && self.canonical_form == Some(form) {
            return Ok(self);
        }

        // Use force_canonicalize for now
        #[allow(deprecated)]
        self.force_canonicalize_by_names_with(canonical_center_v, form)
    }

    /// Force canonicalize the network towards the specified canonical_center using node names directly.
    ///
    /// This method always performs full canonicalization, ignoring the current state.
    /// Uses the default canonical form (`CanonicalForm::Unitary`).
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    pub fn force_canonicalize_by_names(
        self,
        canonical_center: impl IntoIterator<Item = V>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        #[allow(deprecated)]
        self.force_canonicalize_by_names_with(canonical_center, CanonicalForm::Unitary)
    }

    /// Force canonicalize the network towards the specified canonical_center using node names directly
    /// with a specified canonical form.
    ///
    /// This method always performs full canonicalization, ignoring the current state.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    /// * `form` - The canonical form to use:
    ///   - `Unitary`: Uses QR decomposition, each tensor is isometric
    ///   - `LU`: Uses LU decomposition, one factor has unit diagonal
    ///   - `CI`: Uses Cross Interpolation
    ///
    /// # Returns
    /// A new canonicalized TreeTN, or an error if validation fails or factorization fails.
    #[deprecated(since = "0.2.0", note = "Use canonicalize_opt instead")]
    pub fn force_canonicalize_by_names_with(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        // Delegate to the new implementation
        self.canonicalize_impl(canonical_center, form, "force_canonicalize_by_names_with")?;
        Ok(self)
    }

    /// Canonicalize the network in-place towards the specified canonical_center using node names.
    ///
    /// This is the `&mut self` version of `canonicalize_by_names_with`.
    /// Useful when you need to keep using the same variable after canonicalization.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    ///
    /// # Example
    /// ```ignore
    /// tn.canonicalize_by_names_mut(std::iter::once("A".to_string()), CanonicalForm::Unitary)?;
    /// // tn is now canonicalized and can be used directly
    /// ```
    pub fn canonicalize_by_names_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        Self: Default,
    {
        // Take self, canonicalize, and put it back
        let taken = std::mem::take(self);
        match taken.canonicalize_by_names_with(canonical_center, form) {
            Ok(canonicalized) => {
                *self = canonicalized;
                Ok(())
            }
            Err(e) => {
                // On error, self is left in default state
                // This is a limitation of this pattern
                Err(e)
            }
        }
    }

    /// Force canonicalize the network in-place using node names.
    ///
    /// This is the `&mut self` version of `force_canonicalize_by_names_with`.
    ///
    /// # Arguments
    /// * `canonical_center` - The node names that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    pub fn force_canonicalize_by_names_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        #[allow(deprecated)]
        match taken.force_canonicalize_by_names_with(canonical_center, form) {
            Ok(canonicalized) => {
                *self = canonicalized;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Canonicalize the network in-place towards the specified canonical_center.
    ///
    /// This is the `&mut self` version of `canonicalize`.
    /// Uses the default canonical form (Unitary).
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    ///
    /// # Example
    /// ```ignore
    /// tn.canonicalize_mut(std::iter::once(n1))?;
    /// // tn is now canonicalized and can be used directly
    /// ```
    pub fn canonicalize_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        self.canonicalize_with_mut(canonical_center, CanonicalForm::Unitary)
    }

    /// Canonicalize the network in-place towards the specified canonical_center using a specified canonical form.
    ///
    /// This is the `&mut self` version of `canonicalize_with`.
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    ///
    /// # Example
    /// ```ignore
    /// tn.canonicalize_with_mut(std::iter::once(n1), CanonicalForm::Unitary)?;
    /// // tn is now canonicalized and can be used directly
    /// ```
    pub fn canonicalize_with_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        match taken.canonicalize_with(canonical_center, form) {
            Ok(canonicalized) => {
                *self = canonicalized;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Force canonicalize the network in-place towards the specified canonical_center.
    ///
    /// This is the `&mut self` version of `force_canonicalize_with`.
    ///
    /// # Arguments
    /// * `canonical_center` - The nodes that will serve as canonicalization centers
    /// * `form` - The canonical form to use
    ///
    /// # Example
    /// ```ignore
    /// tn.force_canonicalize_with_mut(std::iter::once(n1), CanonicalForm::Unitary)?;
    /// // tn is now canonicalized and can be used directly
    /// ```
    pub fn force_canonicalize_with_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = NodeIndex>,
        form: CanonicalForm,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
        Self: Default,
    {
        let taken = std::mem::take(self);
        #[allow(deprecated)]
        match taken.force_canonicalize_with(canonical_center, form) {
            Ok(canonicalized) => {
                *self = canonicalized;
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
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        context_name: &str,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        // Determine algorithm from form
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::QR,
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
