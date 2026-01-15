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
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
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
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
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

        // When a0 = 0, just return a1 * H * x (avoids axpby which requires same indices)
        // This is important for space_in != space_out cases
        let a0_is_zero = match &self.a0 {
            AnyScalar::F64(x) => *x == 0.0,
            AnyScalar::C64(z) => z.re == 0.0 && z.im == 0.0,
        };
        if a0_is_zero {
            return hx.scale(self.a1.clone());
        }

        // Align hx indices to match x's index order for axpby
        let hx_aligned = hx.permuteinds(&x.external_indices())?;

        // Compute y = a₀ * x + a₁ * H * x
        x.axpby(self.a0.clone(), &hx_aligned, self.a1.clone())
    }
}
