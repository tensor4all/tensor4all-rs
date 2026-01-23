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
        // Debug: Log input tensor (only for first step with bond_dim=2: region=["site0", "site1"])
        // Check if this is the first step (site0, site1) by checking region content
        let region_str = format!("{:?}", self.region);
        let is_first_step =
            self.region.len() == 2 && region_str.contains("site0") && region_str.contains("site1");

        // Check if bond_dim=2 by checking tensor shape [2, 2, 2] for first step
        let x_shape: Vec<usize> = x.external_indices().iter().map(|i| i.dim()).collect();
        let is_bond_dim_2 = is_first_step && x_shape == vec![2, 2, 2];

        // Always log for bond_dim=2 first step to debug the issue
        static FIRST_BOND_DIM_2_CALL: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(true);
        let should_log_detailed = is_bond_dim_2
            && FIRST_BOND_DIM_2_CALL.swap(false, std::sync::atomic::Ordering::Relaxed);

        if should_log_detailed {
            let x_norm = x.norm();
            let x_shape: Vec<usize> = x.external_indices().iter().map(|i| i.dim()).collect();
            let x_indices: Vec<String> = x
                .external_indices()
                .iter()
                .map(|i| format!("{:?}", i.id()))
                .collect();
            eprintln!(
                "  [LocalLinOp::apply] region={:?}, input x: shape={:?}, indices={:?}, norm={:.6e}",
                self.region, x_shape, x_indices, x_norm
            );
        }

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

        if should_log_detailed {
            let hx_norm = hx.norm();
            let hx_shape: Vec<usize> = hx.external_indices().iter().map(|i| i.dim()).collect();
            let hx_indices: Vec<String> = hx
                .external_indices()
                .iter()
                .map(|i| format!("{:?}", i.id()))
                .collect();
            eprintln!(
                "  [LocalLinOp::apply] hx (before permute): shape={:?}, indices={:?}, norm={:.6e}",
                hx_shape, hx_indices, hx_norm
            );
        }

        // Align hx indices to match x's index order (needed for both axpby and a0=0 case)
        // Even when a0=0, we should align indices to match x's order for consistency
        // Check if indices match before permute
        let hx_indices: Vec<String> = hx
            .external_indices()
            .iter()
            .map(|i| format!("{:?}", i.id()))
            .collect();
        let x_indices: Vec<String> = x
            .external_indices()
            .iter()
            .map(|i| format!("{:?}", i.id()))
            .collect();
        let hx_indices_set: std::collections::HashSet<String> =
            hx_indices.iter().cloned().collect();
        let x_indices_set: std::collections::HashSet<String> = x_indices.iter().cloned().collect();

        if should_log_detailed {
            eprintln!("  [LocalLinOp::apply] hx indices: {:?}", hx_indices);
            eprintln!("  [LocalLinOp::apply] x indices: {:?}", x_indices);
            eprintln!(
                "  [LocalLinOp::apply] indices match (by ID): {}",
                hx_indices_set == x_indices_set
            );
        }

        // If indices don't match, permuteinds will fail or produce incorrect results
        if hx_indices_set != x_indices_set {
            if should_log_detailed {
                eprintln!(
                    "  [LocalLinOp::apply] WARNING: hx indices don't match x indices! permuteinds may fail or produce incorrect results"
                );
            }
        }

        let hx_aligned = hx.permuteinds(&x.external_indices())?;

        // Debug: Check permuteinds result for bond_dim=2 case
        if should_log_detailed {
            let hx_norm = hx.norm();
            let hx_aligned_norm = hx_aligned.norm();
            let x_norm = x.norm();
            let hx_indices: Vec<String> = hx
                .external_indices()
                .iter()
                .map(|i| format!("{:?}", i.id()))
                .collect();
            let hx_aligned_indices: Vec<String> = hx_aligned
                .external_indices()
                .iter()
                .map(|i| format!("{:?}", i.id()))
                .collect();
            let x_indices: Vec<String> = x
                .external_indices()
                .iter()
                .map(|i| format!("{:?}", i.id()))
                .collect();
            eprintln!("  [LocalLinOp::apply] After permuteinds: hx_norm={:.6e}, hx_aligned_norm={:.6e}, x_norm={:.6e}", hx_norm, hx_aligned_norm, x_norm);
            eprintln!("  [LocalLinOp::apply] hx indices: {:?}", hx_indices);
            eprintln!(
                "  [LocalLinOp::apply] hx_aligned indices: {:?}",
                hx_aligned_indices
            );
            eprintln!("  [LocalLinOp::apply] x indices: {:?}", x_indices);
            // Check if hx_aligned matches x (for A = I)
            let diff_norm = {
                let diff = x.axpby(AnyScalar::F64(1.0), &hx_aligned, AnyScalar::F64(-1.0))?;
                diff.norm()
            };
            eprintln!(
                "  [LocalLinOp::apply] ||x - hx_aligned||: {:.6e}",
                diff_norm
            );
        }

        // When a0 = 0, just return a1 * hx_aligned (after aligning indices)
        // This is important for space_in != space_out cases
        if self.a0.is_zero() {
            return Ok(hx_aligned.scale(self.a1.clone())?);
        }

        // Compute y = a₀ * x + a₁ * H * x
        // For a0=0, a1=1, this is: y = H * x
        // For A = I, we expect: y = x

        // Debug: Check values before axpby
        if should_log_detailed {
            let x_norm = x.norm();
            let hx_aligned_norm = hx_aligned.norm();
            eprintln!(
                "  [LocalLinOp::apply] Before axpby: x_norm={:.6e}, hx_aligned_norm={:.6e}",
                x_norm, hx_aligned_norm
            );
            eprintln!(
                "  [LocalLinOp::apply] a0={}, a1={}, computing: a0*x + a1*hx_aligned",
                self.a0, self.a1
            );
            // For a0=0, a1=1: result = hx_aligned, so result_norm should equal hx_aligned_norm
            if self.a0.is_zero() {
                eprintln!(
                    "  [LocalLinOp::apply] For a0=0: result = a1 * hx_aligned, expected result_norm = {} * {:.6e} = {:.6e}",
                    self.a1, hx_aligned_norm, hx_aligned_norm
                );
            }
        }

        let result = x.axpby(self.a0.clone(), &hx_aligned, self.a1.clone())?;

        if should_log_detailed {
            let result_norm = result.norm();
            let x_norm = x.norm();
            let hx_aligned_norm = hx_aligned.norm();
            let result_shape: Vec<usize> =
                result.external_indices().iter().map(|i| i.dim()).collect();
            eprintln!(
                "  [LocalLinOp::apply] After axpby: result_norm={:.6e}, x_norm={:.6e}, hx_aligned_norm={:.6e}",
                result_norm, x_norm, hx_aligned_norm
            );
            eprintln!("  [LocalLinOp::apply] result shape: {:?}", result_shape);

            // For a0=0, a1=1: result = hx_aligned, so result_norm should equal hx_aligned_norm
            if self.a0.is_zero() {
                eprintln!(
                    "  [LocalLinOp::apply] For a0=0: expected result_norm = hx_aligned_norm = {:.6e}, but got {:.6e}",
                    hx_aligned_norm, result_norm
                );
            }

            // For A = I, we expect: result ≈ x
            if self.a0.is_zero() {
                let diff_norm = {
                    let diff = x.axpby(AnyScalar::F64(1.0), &result, AnyScalar::F64(-1.0))?;
                    diff.norm()
                };
                eprintln!(
                    "  [LocalLinOp::apply] ||x - result|| (for a0=0): {:.6e}",
                    diff_norm
                );
                eprintln!(
                    "  [LocalLinOp::apply] For A=I, a0=0, a1=1: expected ||x - result|| ≈ 0, but got {:.6e}",
                    diff_norm
                );
            }
        }

        Ok(result)
    }
}
