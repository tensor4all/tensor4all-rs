//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - `Mul<f64>` and `Mul<Complex64>` for scalar multiplication
//! - `log_norm` for computing the logarithm of the Frobenius norm

use num_complex::Complex64;
use std::hash::Hash;
use std::ops::Mul;
use std::sync::Arc;

use anyhow::{Context, Result};

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::CanonicalForm;

use super::TreeTN;
use crate::options::CanonicalizationOptions;

// ============================================================================
// Default implementation
// ============================================================================

impl<Id, Symm, V> Default for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clone implementation
// ============================================================================

impl<Id, Symm, V> Clone for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            canonical_center: self.canonical_center.clone(),
            canonical_form: self.canonical_form,
            site_index_network: self.site_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

// ============================================================================
// Debug implementation
// ============================================================================

impl<Id, Symm, V> std::fmt::Debug for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_center", &self.canonical_center)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Scalar multiplication for TreeTN
// ============================================================================

impl<Id, Symm, V> Mul<f64> for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = Self;

    /// Multiply the TreeTN by a scalar with distributed scaling.
    ///
    /// # Distribution Rules
    ///
    /// - If `canonical_center` is **non-empty**: multiply each center node tensor by `a^{1/|C|}`.
    /// - If `canonical_center` is **empty**: multiply each tensor by `a^{1/N}`.
    /// - The product over all applied factors must equal `a`.
    ///
    /// # Negative Scalar Handling
    ///
    /// - Distribute the magnitude `|a|` using the rules above.
    /// - Apply the sign `-1` to exactly one representative node.
    /// - Representative node: `min(canonical_center)` if non-empty, else `min(all_nodes)`.
    ///
    /// # Zero Scalar Handling
    ///
    /// - Multiply only the representative node by `0.0`, leaving others unchanged.
    /// - The whole network then evaluates to `0`.
    fn mul(mut self, a: f64) -> Self::Output {
        let n = self.node_count();
        if n == 0 {
            return self;
        }

        // Determine representative node
        let representative_node_name: V = if !self.canonical_center.is_empty() {
            self.canonical_center.iter().min().cloned().unwrap()
        } else {
            // Get minimum name from all nodes
            self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .min()
                .unwrap()
        };
        let representative_node = self.graph.node_index(&representative_node_name).unwrap();

        // Handle zero scalar
        if a == 0.0 {
            // Multiply only the representative node by 0
            if let Some(tensor) = self.tensor_mut(representative_node) {
                let new_storage = tensor.storage.as_ref() * 0.0;
                tensor.storage = Arc::new(new_storage);
            }
            return self;
        }

        // Handle non-zero scalar
        let (magnitude, sign) = if a < 0.0 {
            (-a, -1.0_f64)
        } else {
            (a, 1.0_f64)
        };

        // Determine which nodes to scale and the per-node factor
        let (nodes_to_scale, per_node_factor): (Vec<_>, f64) = if !self.canonical_center.is_empty() {
            // Scale only canonical_center nodes
            let center_count = self.canonical_center.len();
            let factor = magnitude.powf(1.0 / center_count as f64);
            let nodes = self.canonical_center.iter()
                .filter_map(|v| self.graph.node_index(v))
                .collect();
            (nodes, factor)
        } else {
            // Scale all nodes
            let factor = magnitude.powf(1.0 / n as f64);
            let nodes = self.graph.graph().node_indices().collect();
            (nodes, factor)
        };

        // Apply scaling to the appropriate nodes
        for node_idx in nodes_to_scale {
            if let Some(tensor) = self.tensor_mut(node_idx) {
                let scale = if node_idx == representative_node {
                    per_node_factor * sign
                } else {
                    per_node_factor
                };
                let new_storage = tensor.storage.as_ref() * scale;
                tensor.storage = Arc::new(new_storage);
            }
        }

        self
    }
}

impl<Id, Symm, V> Mul<Complex64> for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = Self;

    /// Multiply the TreeTN by a complex scalar with distributed scaling.
    ///
    /// # Distribution Rules
    ///
    /// For complex scalars, we distribute the magnitude and apply the phase to one node.
    ///
    /// - If `canonical_center` is **non-empty**: multiply each center node tensor by `|a|^{1/|C|}`.
    /// - If `canonical_center` is **empty**: multiply each tensor by `|a|^{1/N}`.
    /// - The phase `a/|a|` is applied to the representative node.
    ///
    /// # Zero Scalar Handling
    ///
    /// - Multiply only the representative node by `0`, leaving others unchanged.
    fn mul(mut self, a: Complex64) -> Self::Output {
        let n = self.node_count();
        if n == 0 {
            return self;
        }

        // Determine representative node
        let representative_node_name: V = if !self.canonical_center.is_empty() {
            self.canonical_center.iter().min().cloned().unwrap()
        } else {
            self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .min()
                .unwrap()
        };
        let representative_node = self.graph.node_index(&representative_node_name).unwrap();

        // Handle zero scalar
        let magnitude = a.norm();
        if magnitude == 0.0 {
            // Multiply only the representative node by 0
            if let Some(tensor) = self.tensor_mut(representative_node) {
                let new_storage = tensor.storage.as_ref() * 0.0;
                tensor.storage = Arc::new(new_storage);
            }
            return self;
        }

        // Compute phase (unit complex number)
        let phase = a / magnitude;

        // Determine which nodes to scale and the per-node factor (real)
        let (nodes_to_scale, per_node_factor): (Vec<_>, f64) = if !self.canonical_center.is_empty() {
            let center_count = self.canonical_center.len();
            let factor = magnitude.powf(1.0 / center_count as f64);
            let nodes = self.canonical_center.iter()
                .filter_map(|v| self.graph.node_index(v))
                .collect();
            (nodes, factor)
        } else {
            let factor = magnitude.powf(1.0 / n as f64);
            let nodes = self.graph.graph().node_indices().collect();
            (nodes, factor)
        };

        // Apply scaling to the appropriate nodes
        for node_idx in nodes_to_scale {
            if let Some(tensor) = self.tensor_mut(node_idx) {
                if node_idx == representative_node {
                    // Apply both magnitude factor and phase to representative
                    let complex_scale = Complex64::new(per_node_factor, 0.0) * phase;
                    let new_storage = tensor.storage.as_ref() * complex_scale;
                    tensor.storage = Arc::new(new_storage);
                } else {
                    // Apply only magnitude factor to other nodes
                    let new_storage = tensor.storage.as_ref() * per_node_factor;
                    tensor.storage = Arc::new(new_storage);
                }
            }
        }

        self
    }
}

// Implement Mul with reference to avoid consuming TreeTN
impl<Id, Symm, V> Mul<f64> for &TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = TreeTN<Id, Symm, V>;

    fn mul(self, a: f64) -> Self::Output {
        self.clone() * a
    }
}

impl<Id, Symm, V> Mul<Complex64> for &TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = TreeTN<Id, Symm, V>;

    fn mul(self, a: Complex64) -> Self::Output {
        self.clone() * a
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
    Self: Default,
{
    /// Compute the natural logarithm of the Frobenius norm: ln(||TN||).
    ///
    /// **Warning**: This method may canonicalize the network if not already canonicalized
    /// to a single Unitary center. Use `log_norm` (without canonicalization) if you
    /// want to preserve the current canonicalization state.
    ///
    /// This is computed in a numerically stable way by:
    /// 1. Ensuring Unitary canonicalization to a single site
    /// 2. Computing ||center_tensor|| from that single center
    ///
    /// For Unitary canonical form, tensors outside the center satisfy Q†Q = I,
    /// so they don't contribute to the norm: ||TN|| = ||center_tensor||.
    ///
    /// # Algorithm
    /// - If already Unitary canonicalized to single site → use that site (no canonicalization)
    /// - If Unitary canonicalized to multiple sites → canonicalize to min of canonical_center first
    /// - Otherwise → Unitary canonicalize to min node name
    ///
    /// # Returns
    /// `ln(||TN||)` as f64
    ///
    /// # Note
    /// Unlike norm² which differs by factor 2 in log space from norm,
    /// this function returns `ln(||TN||)`, not `ln(||TN||²)`.
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
        let center_name: V = if self.is_canonicalized()
            && self.canonical_form() == Some(CanonicalForm::Unitary)
        {
            if self.canonical_center.len() == 1 {
                // Already Unitary canonicalized to single site - use it
                self.canonical_center.iter().next().unwrap().clone()
            } else {
                // Unitary canonicalized to multiple sites - canonicalize to min site
                let min_center = self.canonical_center.iter().min().unwrap().clone();
                self.canonicalize_mut(
                    std::iter::once(min_center.clone()),
                    CanonicalizationOptions::default(),
                ).context("log_norm: failed to canonicalize to single site")?;
                min_center
            }
        } else {
            // Not canonicalized or not Unitary - canonicalize to min node name
            let min_node_name = self.node_names().into_iter().min()
                .ok_or_else(|| anyhow::anyhow!("No nodes in TreeTN"))
                .context("log_norm: network must have nodes")?;
            self.canonicalize_mut(
                std::iter::once(min_node_name.clone()),
                CanonicalizationOptions::default(),
            ).context("log_norm: failed to canonicalize")?;
            min_node_name
        };

        // Get center node index and tensor
        let center_node = self.node_index(&center_name)
            .ok_or_else(|| anyhow::anyhow!("Center node not found"))
            .context("log_norm: center node must exist")?;

        let center_tensor = self.tensor(center_node)
            .ok_or_else(|| anyhow::anyhow!("Center tensor not found"))
            .context("log_norm: center tensor must exist")?;

        let norm_sq = center_tensor.norm_squared();
        let norm = norm_sq.sqrt();

        Ok(norm.ln())
    }
}
