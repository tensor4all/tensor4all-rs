//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - `log_norm` for computing the logarithm of the Frobenius norm
//! - `norm`, `norm_squared` for computing the Frobenius norm
//! - `inner` for computing inner products of two TreeTNs
//! - `to_dense` for contracting to a single tensor
//! - `evaluate` for evaluating at specific index values

use std::collections::HashMap;
use std::hash::Hash;

use tensor4all_core::{AnyScalar, IndexLike, TensorLike};

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

    /// Compute the Frobenius norm of the TreeTN.
    ///
    /// Uses `log_norm` internally: `norm = exp(log_norm)`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error if the network is empty or canonicalization fails.
    pub fn norm(&mut self) -> Result<f64> {
        let log_n = self
            .log_norm()
            .context("norm: failed to compute log_norm")?;
        Ok(log_n.exp())
    }

    /// Compute the squared Frobenius norm of the TreeTN.
    ///
    /// Returns `||self||^2 = norm()^2`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error if the network is empty or canonicalization fails.
    pub fn norm_squared(&mut self) -> Result<f64> {
        let n = self
            .norm()
            .context("norm_squared: failed to compute norm")?;
        Ok(n * n)
    }

    /// Compute the inner product of two TreeTNs.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both TreeTNs must have the same site indices (same IDs).
    /// Link indices may differ between the two TreeTNs.
    ///
    /// # Algorithm
    /// 1. Replace link indices in `other` with fresh IDs to avoid collision.
    /// 2. At each node, contract `conj(self_tensor) * other_tensor` pairwise.
    /// 3. Sweep from leaves to root, contracting the environment.
    ///
    /// This is equivalent to contracting the entire network
    /// `conj(self) * other` into a scalar.
    ///
    /// # Errors
    /// Returns an error if the networks have incompatible topologies.
    pub fn inner(&self, other: &Self) -> Result<AnyScalar>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if self.node_count() == 0 && other.node_count() == 0 {
            return Ok(AnyScalar::new_real(0.0));
        }

        // Use contract_naive which handles sim_internal_inds and full contraction.
        // conj(self) * other contracted over site indices gives the inner product.
        // Build conj(self) by conjugating all tensors.
        let mut conj_self = self.clone();
        for node_idx in conj_self.graph.graph().node_indices().collect::<Vec<_>>() {
            if let Some(tensor) = conj_self.graph.graph_mut().node_weight_mut(node_idx) {
                *tensor = tensor.conj();
            }
        }

        // Contract conj(self) with other using naive contraction
        let result_tensor = conj_self
            .contract_naive(other)
            .context("inner: failed to contract conj(self) with other")?;

        // The result should be a scalar (rank 0) since all site indices were contracted
        // (self and other share the same site indices, link indices were sim'd).
        // Extract the scalar value via inner_product with scalar_one.
        let scalar_one = T::scalar_one().context("inner: failed to create scalar_one")?;
        result_tensor
            .inner_product(&scalar_one)
            .context("inner: failed to extract scalar value")
    }

    /// Convert the TreeTN to a single dense tensor.
    ///
    /// This contracts all tensors in the network along their link/bond indices,
    /// producing a single tensor with only site (physical) indices.
    ///
    /// This is an alias for `contract_to_tensor()`.
    ///
    /// # Warning
    /// This operation can be very expensive for large networks,
    /// as the result size grows exponentially with the number of sites.
    ///
    /// # Errors
    /// Returns an error if the network is empty or contraction fails.
    pub fn to_dense(&self) -> Result<T> {
        self.contract_to_tensor()
            .context("to_dense: failed to contract network to tensor")
    }

    /// Evaluate the TreeTN at specific index values.
    ///
    /// For each node, selects the specified index value using one-hot vectors,
    /// effectively evaluating the tensor network function at a specific point.
    ///
    /// # Arguments
    /// * `index_values` - Map from node name to a vector of index values.
    ///   For each node, provides the values for the site (physical) indices
    ///   in the same order as they appear in the site space.
    ///
    /// # Returns
    /// The scalar value of the tensor network at the specified indices.
    ///
    /// # Errors
    /// Returns an error if:
    /// - A node name in `index_values` doesn't exist
    /// - Index values are out of bounds
    /// - Contraction fails
    pub fn evaluate(&self, index_values: &HashMap<V, Vec<usize>>) -> Result<AnyScalar>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot evaluate empty TreeTN"))
                .context("evaluate: network must have at least one node");
        }

        // For each node, contract the tensor with one-hot vectors for the site indices.
        // Then contract the resulting network along bond indices.

        // Build contracted tensors: for each node, contract with onehot for site indices
        let mut contracted_tensors: Vec<T> = Vec::new();
        let mut contracted_names: Vec<V> = Vec::new();

        for node_name in self.node_names() {
            let node_idx = self
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))
                .context("evaluate: node must exist")?;

            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("evaluate: tensor must exist")?;

            // Get site (physical) indices for this node
            let site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))
                .context("evaluate: site space must exist")?;

            if site_space.is_empty() {
                // No site indices - just use the tensor as is
                contracted_tensors.push(tensor.clone());
                contracted_names.push(node_name);
                continue;
            }

            // Get the index values for this node
            let values = index_values
                .get(&node_name)
                .ok_or_else(|| anyhow::anyhow!("No index values provided for node {:?}", node_name))
                .context(
                    "evaluate: index values must be provided for all nodes with site indices",
                )?;

            // Build one-hot index-value pairs
            let site_indices: Vec<T::Index> = site_space.iter().cloned().collect();
            if values.len() != site_indices.len() {
                return Err(anyhow::anyhow!(
                    "Expected {} index values for node {:?}, got {}",
                    site_indices.len(),
                    node_name,
                    values.len()
                ))
                .context("evaluate: index values count must match site indices count");
            }

            let index_vals: Vec<(T::Index, usize)> = site_indices
                .into_iter()
                .zip(values.iter().copied())
                .collect();

            let onehot =
                T::onehot(&index_vals).context("evaluate: failed to create one-hot tensor")?;

            // Contract tensor with one-hot vector
            let result = T::contract(&[tensor, &onehot], tensor4all_core::AllowedPairs::All)
                .context("evaluate: failed to contract tensor with one-hot")?;

            contracted_tensors.push(result);
            contracted_names.push(node_name);
        }

        // Build a temporary TreeTN from the contracted tensors and contract to scalar
        let temp_tn = TreeTN::<T, V>::from_tensors(contracted_tensors, contracted_names)
            .context("evaluate: failed to build temporary TreeTN")?;
        let result_tensor = temp_tn
            .contract_to_tensor()
            .context("evaluate: failed to contract to scalar")?;

        // The result should be a scalar (rank 0). Extract its value.
        let scalar_one = T::scalar_one().context("evaluate: failed to create scalar_one")?;
        result_tensor
            .inner_product(&scalar_one)
            .context("evaluate: failed to extract scalar value")
    }
}
