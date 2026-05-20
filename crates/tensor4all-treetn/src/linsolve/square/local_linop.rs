//! LocalLinOp: Linear operator wrapper for local GMRES solving (square case).
//!
//! This module provides a wrapper that applies the local projected operator
//! to tensors, enabling GMRES solving via `tensor4all_core::krylov::gmres`.

use std::hash::Hash;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use tensor4all_core::{IndexLike, TensorLike};

use crate::linsolve::common::ProjectedOperator;
use crate::treetn::TreeTN;

/// LocalLinOp: Wraps the projected operator for local GMRES solving.
///
/// This applies only the projected local operator `H * x`.
/// Affine coefficients such as `a₀ I + a₁ H` belong to the Krylov solver
/// so the Arnoldi basis is built from the same unshifted operator as KrylovKit.
///
/// This is the V_in = V_out specialized version that maintains a separate
/// reference state for stable environment computation.
pub struct LocalLinOp<'a, T, V>
where
    T: TensorLike + 'static,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// The projected operator (shared, mutable for environment caching)
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// The region being updated
    pub region: Vec<V>,
    /// Current state for ket in environment computation
    pub state: &'a TreeTN<T, V>,
    /// Reference state for bra in environment computation
    /// Uses separate bond indices to prevent unintended contractions
    pub reference_state: &'a TreeTN<T, V>,
}

impl<'a, T, V> LocalLinOp<'a, T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LocalLinOp with explicit reference state.
    ///
    /// The reference_state should have separate bond indices from state
    /// to prevent unintended bra↔ket contractions in environment computation.
    pub fn new(
        projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
        region: Vec<V>,
        state: &'a TreeTN<T, V>,
        reference_state: &'a TreeTN<T, V>,
    ) -> Self {
        Self {
            projected_operator,
            region,
            state,
            reference_state,
        }
    }

    fn local_input_indices(&self) -> Result<Vec<T::Index>> {
        let Some((first_node, rest_nodes)) = self.region.split_first() else {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply_projected: region must not be empty"
            ));
        };
        let first_idx = self.state.node_index(first_node).ok_or_else(|| {
            anyhow::anyhow!(
                "LocalLinOp::apply_projected: node {:?} not found in state",
                first_node
            )
        })?;
        let mut local = self
            .state
            .tensor(first_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "LocalLinOp::apply_projected: tensor for node {:?} not found in state",
                    first_node
                )
            })?
            .clone();

        for node in rest_nodes {
            let node_idx = self.state.node_index(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "LocalLinOp::apply_projected: node {:?} not found in state",
                    node
                )
            })?;
            let tensor = self.state.tensor(node_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "LocalLinOp::apply_projected: tensor for node {:?} not found in state",
                    node
                )
            })?;
            local = local.contract_pair(tensor)?;
        }

        Ok(local.external_indices())
    }

    fn same_index_set(left: &[T::Index], right: &[T::Index]) -> bool {
        let left_keys: std::collections::HashSet<_> =
            left.iter().map(|idx| (idx.clone(), idx.dim())).collect();
        let right_keys: std::collections::HashSet<_> =
            right.iter().map(|idx| (idx.clone(), idx.dim())).collect();
        left.len() == right.len() && left_keys == right_keys
    }

    /// Apply the projected local operator: `y = H * x`.
    pub fn apply_projected(&self, x: &T) -> Result<T> {
        let x_indices = x.external_indices();
        let expected_input_indices = self.local_input_indices()?;
        if !Self::same_index_set(&x_indices, &expected_input_indices) {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply_projected: index structure mismatch between input (x) and the local state vector space:\n  x has {} indices: {:?}\n  expected {} indices: {:?}",
                x_indices.len(),
                x_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                expected_input_indices.len(),
                expected_input_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
            ));
        }

        let mut proj_op = self
            .projected_operator
            .write()
            .map_err(|e| {
                anyhow::anyhow!("Failed to acquire write lock on projected_operator: {}", e)
            })
            .context("LocalLinOp::apply: lock poisoned")?;

        let hx = proj_op.apply(
            x,
            &self.region,
            self.state,
            self.reference_state,
            self.state.site_index_network(),
        )?;

        let hx_indices = hx.external_indices();
        let indices_match = x_indices.len() == hx_indices.len()
            && x_indices
                .iter()
                .zip(hx_indices.iter())
                .all(|(x_idx, hx_idx)| x_idx == hx_idx && x_idx.dim() == hx_idx.dim());
        if !indices_match {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply_projected: index structure mismatch between operator output (hx) and input (x):\n  x has {} indices: {:?}\n  hx has {} indices: {:?}\n\nProjectedOperator::apply is expected to return output in the same local vector space and index order as its input.",
                x_indices.len(),
                x_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                hx_indices.len(),
                hx_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
            ));
        }

        Ok(hx)
    }
}

#[cfg(test)]
mod tests;
