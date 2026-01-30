//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - `log_norm` for computing the logarithm of the Frobenius norm

use std::hash::Hash;

use tensor4all_core::TensorLike;

use super::TreeTN;

// ============================================================================
// Default implementation
// ============================================================================

impl<T, V> Default for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clone implementation
// ============================================================================

impl<T, V> Clone for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            canonical_region: self.canonical_region.clone(),
            canonical_form: self.canonical_form,
            site_index_network: self.site_index_network.clone(),
            link_index_network: self.link_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

// ============================================================================
// Debug implementation
// ============================================================================

impl<T, V> std::fmt::Debug for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_region", &self.canonical_region)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use crate::CanonicalizationOptions;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Compute log(||TreeTN||_F), the log of the Frobenius norm.
    ///
    /// Uses canonicalization to avoid numerical overflow:
    /// when canonicalized to a single site with Unitary form,
    /// the Frobenius norm of the whole network equals the norm of the center tensor.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network
    /// to a single Unitary center. Use `log_norm` (without canonicalization) if you
    /// already have a properly canonicalized network.
    ///
    /// # Returns
    /// The natural logarithm of the Frobenius norm.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - Canonicalization fails
    pub fn log_norm(&mut self) -> Result<f64> {
        let n = self.node_count();
        if n == 0 {
            return Err(anyhow::anyhow!("Cannot compute log_norm of empty TreeTN"))
                .context("log_norm: network must have at least one node");
        }

        // Determine the single center site (by name)
        let center_name: V =
            if self.is_canonicalized() && self.canonical_form() == Some(CanonicalForm::Unitary) {
                if self.canonical_region.len() == 1 {
                    // Already Unitary canonicalized to single site - use it
                    self.canonical_region.iter().next().unwrap().clone()
                } else {
                    // Unitary canonicalized to multiple sites - canonicalize to min site
                    let min_center = self.canonical_region.iter().min().unwrap().clone();
                    self.canonicalize_mut(
                        std::iter::once(min_center.clone()),
                        CanonicalizationOptions::default(),
                    )
                    .context("log_norm: failed to canonicalize to single site")?;
                    min_center
                }
            } else {
                // Not canonicalized or not Unitary - canonicalize to min node name
                let min_node_name = self
                    .node_names()
                    .into_iter()
                    .min()
                    .ok_or_else(|| anyhow::anyhow!("No nodes in TreeTN"))
                    .context("log_norm: network must have nodes")?;
                self.canonicalize_mut(
                    std::iter::once(min_node_name.clone()),
                    CanonicalizationOptions::default(),
                )
                .context("log_norm: failed to canonicalize")?;
                min_node_name
            };

        // Get center node index and tensor
        let center_node = self
            .node_index(&center_name)
            .ok_or_else(|| anyhow::anyhow!("Center node not found"))
            .context("log_norm: center node must exist")?;

        let center_tensor = self
            .tensor(center_node)
            .ok_or_else(|| anyhow::anyhow!("Center tensor not found"))
            .context("log_norm: center tensor must exist")?;

        let norm_sq = center_tensor.norm_squared();
        let norm = norm_sq.sqrt();

        Ok(norm.ln())
    }
}
