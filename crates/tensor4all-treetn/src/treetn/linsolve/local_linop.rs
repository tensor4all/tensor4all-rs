//! LocalLinOp: Linear operator wrapper for local GMRES solving.
//!
//! This module provides a wrapper that applies the local projected operator
//! to tensors, enabling GMRES solving via `tensor4all_core::krylov::gmres`.

use std::hash::Hash;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use tensor4all_core::any_scalar::AnyScalar;
use tensor4all_core::{IndexLike, TensorLike};

use super::projected_operator::ProjectedOperator;
use crate::treetn::TreeTN;

/// LocalLinOp: Wraps the projected operator for local GMRES solving.
///
/// This applies the local linear operator: `y = a₀ * x + a₁ * H * x`
/// where H is the projected operator.
pub struct LocalLinOp<T, V>
where
    T: TensorLike + 'static,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// The projected operator (shared, mutable for environment caching)
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// The region being updated
    pub region: Vec<V>,
    /// Current state for ket in environment computation (V_in space)
    pub state: TreeTN<T, V>,
    /// Reference state for bra in environment computation (V_out space)
    /// If None, uses `state` (same as ket) for V_in = V_out case
    pub bra_state: Option<TreeTN<T, V>>,
    /// Coefficient a₀ (can be real or complex)
    pub a0: AnyScalar,
    /// Coefficient a₁ (can be real or complex)
    pub a1: AnyScalar,
}

impl<T, V> LocalLinOp<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LocalLinOp for V_in = V_out case.
    pub fn new(
        projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
        region: Vec<V>,
        state: TreeTN<T, V>,
        a0: AnyScalar,
        a1: AnyScalar,
    ) -> Self {
        Self {
            projected_operator,
            region,
            state,
            bra_state: None, // Use state as bra (V_in = V_out)
            a0,
            a1,
        }
    }

    /// Create a new LocalLinOp for V_in ≠ V_out case with explicit bra_state.
    pub fn with_bra_state(
        projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
        region: Vec<V>,
        state: TreeTN<T, V>,
        bra_state: TreeTN<T, V>,
        a0: AnyScalar,
        a1: AnyScalar,
    ) -> Self {
        Self {
            projected_operator,
            region,
            state,
            bra_state: Some(bra_state),
            a0,
            a1,
        }
    }

    /// Get the bra state for environment computation.
    /// Returns bra_state if set, otherwise returns state (V_in = V_out case).
    fn get_bra_state(&self) -> &TreeTN<T, V> {
        self.bra_state.as_ref().unwrap_or(&self.state)
    }

    /// Apply the local linear operator: `y = a₀ * x + a₁ * H * x`
    ///
    /// This is used by `tensor4all_core::krylov::gmres` to solve the local problem.
    pub fn apply(&self, x: &T) -> Result<T> {
        // Apply operator: H * x
        // ProjectedOperator handles environment computation and index mappings
        let bra_state = self.get_bra_state();
        let mut proj_op = self.projected_operator.write().unwrap();

        let hx = proj_op.apply(
            x,
            &self.region,
            &self.state,
            bra_state,
            self.state.site_index_network(),
        )?;

        // The operator H may produce output indices that differ from x's indices.
        // To compute axpby, we need hx to have the same indices as x.
        // Replace hx's indices with x's indices (matched by position).
        let x_indices = x.external_indices();
        let hx_indices = hx.external_indices();

        let hx_aligned = if x_indices.len() == hx_indices.len() {
            // Check if indices already match
            let indices_match = x_indices
                .iter()
                .zip(hx_indices.iter())
                .all(|(xi, hi)| xi == hi);

            if indices_match {
                hx
            } else {
                // Replace hx indices with x indices
                hx.replaceinds(&hx_indices, &x_indices)?
            }
        } else {
            // Different number of indices - this shouldn't happen for a valid operator
            return Err(anyhow::anyhow!(
                "Index count mismatch in local operator: x has {} indices, Hx has {}",
                x_indices.len(),
                hx_indices.len()
            ));
        };

        // Compute y = a₀ * x + a₁ * H * x using TensorLike::axpby
        // y = a₀ * x + a₁ * hx_aligned
        x.axpby(self.a0.clone(), &hx_aligned, self.a1.clone())
    }
}
