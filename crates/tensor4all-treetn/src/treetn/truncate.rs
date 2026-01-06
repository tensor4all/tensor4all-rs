//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.
//!
//! The truncation algorithm uses a two-site sweep (nsite=2) based on Euler tour traversal:
//! 1. First, canonicalize the network towards the specified center
//! 2. Generate a sweep plan that visits each edge twice (forward and backward)
//! 3. For each step: extract two adjacent nodes, perform SVD-based truncation, replace
//! 4. The canonical center moves along the sweep path
//!
//! This ensures all bonds are optimally truncated in both directions.

use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::CanonicalForm;

use super::localupdate::{apply_local_update_sweep, LocalUpdateSweepPlan, TruncateUpdater};
use super::TreeTN;
use crate::options::{CanonicalizationOptions, TruncationOptions};

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
    /// # Algorithm
    /// 1. Canonicalize the network towards the center (required for truncation)
    /// 2. Generate a two-site sweep plan using Euler tour traversal
    /// 3. Apply SVD-based truncation at each step, visiting each edge twice
    ///
    /// # Example
    /// ```ignore
    /// use tensor4all_treetn::TruncationOptions;
    ///
    /// // Truncate with max rank of 50
    /// let ttn = ttn.truncate(
    ///     ["center"],
    ///     TruncationOptions::default().with_max_rank(50)
    /// )?;
    ///
    /// // Truncate with relative tolerance
    /// let ttn = ttn.truncate(
    ///     ["center"],
    ///     TruncationOptions::default().with_rtol(1e-10)
    /// )?;
    /// ```
    pub fn truncate(
        mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + PartialEq + std::fmt::Debug + From<NoSymmSpace>,
        V: Ord,
    {
        self.truncate_impl(
            canonical_center,
            options.form,
            options.rtol,
            options.max_rank,
            "truncate",
        )?;
        Ok(self)
    }

    /// Truncate the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`truncate`].
    pub fn truncate_mut(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + PartialEq + std::fmt::Debug + From<NoSymmSpace>,
        V: Ord,
    {
        self.truncate_impl(
            canonical_center,
            options.form,
            options.rtol,
            options.max_rank,
            "truncate_mut",
        )
    }

    /// Internal implementation for truncation.
    ///
    /// Uses LocalUpdateSweepPlan with TruncateUpdater for full two-site sweeps.
    pub(crate) fn truncate_impl(
        &mut self,
        canonical_center: impl IntoIterator<Item = V>,
        form: CanonicalForm,
        rtol: Option<f64>,
        max_rank: Option<usize>,
        context_name: &str,
    ) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + PartialEq + std::fmt::Debug + From<NoSymmSpace>,
        V: Ord,
    {
        // Collect center nodes
        let center_nodes: HashSet<V> = canonical_center.into_iter().collect();

        if center_nodes.is_empty() {
            return Ok(()); // Nothing to do
        }

        // Currently only single-node center is supported for truncation
        if center_nodes.len() != 1 {
            return Err(anyhow::anyhow!(
                "truncate currently requires a single-node center, got {} nodes",
                center_nodes.len()
            ))
            .context(format!("{}: multi-node center not supported", context_name));
        }

        let center_node = center_nodes.iter().next().unwrap().clone();

        // Step 1: Canonicalize towards the center (required before truncation sweep)
        let canonicalize_options = CanonicalizationOptions::default().with_form(form);
        self.canonicalize_impl(
            [center_node.clone()],
            canonicalize_options.form,
            &format!("{}: pre-canonicalize", context_name),
        )?;

        // Step 2: Generate sweep plan (nsite=2 for two-site truncation)
        let plan = LocalUpdateSweepPlan::from_treetn(self, &center_node, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan from center {:?}", center_node))
            .context(format!("{}: sweep plan creation failed", context_name))?;

        // If no steps (single node network), nothing more to do
        if plan.is_empty() {
            return Ok(());
        }

        // Step 3: Apply truncation sweep
        let mut updater = TruncateUpdater::new(max_rank, rtol);
        apply_local_update_sweep(self, &plan, &mut updater)
            .context(format!("{}: truncation sweep failed", context_name))?;

        // The canonical form is maintained by the sweep
        self.canonical_form = Some(form);

        Ok(())
    }
}
