//! LocalLinOp: Linear operator wrapper for local GMRES solving (square case).
//!
//! This module provides a wrapper that applies the local projected operator
//! to tensors, enabling GMRES solving via `tensor4all_core::krylov::gmres`.

use std::hash::Hash;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use tensor4all_core::any_scalar::AnyScalar;
use tensor4all_core::{IndexLike, TensorLike};

use crate::linsolve::common::ProjectedOperator;
use crate::treetn::TreeTN;

/// LocalLinOp: Wraps the projected operator for local GMRES solving.
///
/// This applies the local linear operator: `y = a₀ * x + a₁ * H * x`
/// where H is the projected operator.
///
/// This is the V_in = V_out specialized version that maintains a separate
/// reference state for stable environment computation.
pub struct LocalLinOp<T, V>
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
    pub state: TreeTN<T, V>,
    /// Reference state for bra in environment computation
    /// Uses separate bond indices to prevent unintended contractions
    pub reference_state: TreeTN<T, V>,
    /// Coefficient a₀ (can be real or complex)
    pub a0: AnyScalar,
    /// Coefficient a₁ (can be real or complex)
    pub a1: AnyScalar,
}

impl<T, V> LocalLinOp<T, V>
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
        state: TreeTN<T, V>,
        reference_state: TreeTN<T, V>,
        a0: AnyScalar,
        a1: AnyScalar,
    ) -> Self {
        Self {
            projected_operator,
            region,
            state,
            reference_state,
            a0,
            a1,
        }
    }

    /// Apply the local linear operator: `y = a₀ * x + a₁ * H * x`
    ///
    /// This is used by `tensor4all_core::krylov::gmres` to solve the local problem.
    pub fn apply(&self, x: &T) -> Result<T> {
        // Apply operator: H * x
        // ProjectedOperator handles environment computation and index mappings
        let mut proj_op = self
            .projected_operator
            .write()
            .map_err(|e| {
                anyhow::anyhow!("Failed to acquire write lock on projected_operator: {}", e)
            })
            .context("LocalLinOp::apply: lock poisoned")?;

        let mut hx = proj_op.apply(
            x,
            &self.region,
            &self.state,
            &self.reference_state,
            self.state.site_index_network(),
        )?;

        // Map output tensor's boundary bond indices back to ket space
        // The projected operator application produces output with bra-side boundary bonds
        for node in &self.region {
            for neighbor in self.state.site_index_network().neighbors(node) {
                if !self.region.contains(&neighbor) {
                    let ket_edge = match self.state.edge_between(node, &neighbor) {
                        Some(e) => e,
                        None => continue,
                    };
                    let bra_edge = match self.reference_state.edge_between(node, &neighbor) {
                        Some(e) => e,
                        None => continue,
                    };
                    let ket_bond = match self.state.bond_index(ket_edge) {
                        Some(b) => b,
                        None => continue,
                    };
                    let bra_bond = match self.reference_state.bond_index(bra_edge) {
                        Some(b) => b,
                        None => continue,
                    };

                    // Only replace if hx actually contains the bra bond
                    if hx
                        .external_indices()
                        .iter()
                        .any(|idx| idx.id() == bra_bond.id())
                    {
                        hx = hx.replaceind(bra_bond, ket_bond)?;
                    }
                }
            }
        }

        // When a0 = 0, just return a1 * H * x (avoids axpby which requires same indices)
        if self.a0.is_zero() {
            return hx.scale(self.a1.clone());
        }

        // Align hx indices to match x's index order for axpby
        let hx_aligned = hx.permuteinds(&x.external_indices())?;

        // Compute y = a₀ * x + a₁ * H * x
        x.axpby(self.a0.clone(), &hx_aligned, self.a1.clone())
    }
}
