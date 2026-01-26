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
        // Check that hx and x have the same index structure (by ID and count)
        let x_indices = x.external_indices();
        let hx_indices = hx.external_indices();
        let x_ids: std::collections::HashSet<_> = x_indices.iter().map(|i| i.id()).collect();
        let hx_ids: std::collections::HashSet<_> = hx_indices.iter().map(|i| i.id()).collect();

        let hx_aligned = if x_ids == hx_ids && x_indices.len() == hx_indices.len() {
            // Same index set and count - permute to match order
            hx.permuteinds(&x_indices)?
        } else {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply: index structure mismatch between operator output (hx) and input (x):\n  x has {} indices: {:?}\n  hx has {} indices: {:?}\n  x IDs: {:?}\n  hx IDs: {:?}\n\nThis suggests the projected operator application produced output with different index structure than expected.",
                x_indices.len(),
                x_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                hx_indices.len(),
                hx_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                x_ids.iter().collect::<Vec<_>>(),
                hx_ids.iter().collect::<Vec<_>>(),
            ));
        };

        // Compute y = a₀ * x + a₁ * H * x
        x.axpby(self.a0.clone(), &hx_aligned, self.a1.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use tensor4all_core::index::DynId;
    use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};

    use crate::operator::IndexMapping;
    use crate::treetn::TreeTN;

    use super::*;

    fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize) -> DynIndex {
        loop {
            let idx = DynIndex::new_dyn(dim);
            if used.insert(*idx.id()) {
                return idx;
            }
        }
    }

    #[test]
    fn test_local_linop_new() {
        use crate::linsolve::common::ProjectedOperator;

        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = DynIndex::new_dyn(2);
        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone()], vec![1.0, 2.0]);
        state.add_tensor("site0".to_string(), t0).unwrap();

        let reference_state = state.clone();
        let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

        let linop = LocalLinOp::new(
            projected_op,
            vec!["site0".to_string()],
            state,
            reference_state,
            AnyScalar::F64(1.0),
            AnyScalar::F64(0.0),
        );

        assert_eq!(linop.region.len(), 1);
        assert_eq!(linop.a0, AnyScalar::F64(1.0));
        assert_eq!(linop.a1, AnyScalar::F64(0.0));
    }

    /// Apply with a0=0 hits the early return path (scale only, no index alignment).
    #[test]
    fn test_local_linop_apply_a0_zero() {
        use crate::linsolve::common::ProjectedOperator;

        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = DynIndex::new_dyn(2);
        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone()], vec![1.0, 2.0]);
        state.add_tensor("site0".to_string(), t0).unwrap();

        let reference_state = state.clone();
        let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

        let linop = LocalLinOp::new(
            projected_op,
            vec!["site0".to_string()],
            state.clone(),
            reference_state,
            AnyScalar::F64(0.0),
            AnyScalar::F64(1.0),
        );

        let site0 = "site0".to_string();
        let x = state
            .tensor(state.node_index(&site0).unwrap())
            .unwrap()
            .clone();
        let y = linop.apply(&x).unwrap();
        assert_eq!(y.external_indices().len(), 0);
    }

    /// Apply with x whose index structure differs from operator output triggers index mismatch error.
    #[test]
    fn test_local_linop_apply_index_mismatch() {
        use crate::linsolve::common::ProjectedOperator;

        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = DynIndex::new_dyn(2);
        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone()], vec![1.0, 2.0]);
        state.add_tensor("site0".to_string(), t0).unwrap();

        let reference_state = state.clone();
        let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

        let linop = LocalLinOp::new(
            projected_op,
            vec!["site0".to_string()],
            state,
            reference_state,
            AnyScalar::F64(1.0),
            AnyScalar::F64(0.0),
        );

        let other = DynIndex::new_dyn(2);
        let x = TensorDynLen::from_dense_f64(vec![other], vec![1.0, 0.0]);
        let err = linop.apply(&x).unwrap_err();
        assert!(err.to_string().contains("index structure mismatch"));
    }

    /// Apply success with 1-node MPO-like state and identity operator (index mappings).
    #[test]
    fn test_local_linop_apply_success_mappings() {
        use crate::linsolve::common::ProjectedOperator;

        let phys_dim = 2usize;
        let ext_dim = 2usize;
        let mut used = HashSet::<DynId>::new();
        let contracted = unique_dyn_index(&mut used, phys_dim);
        let external = unique_dyn_index(&mut used, ext_dim);

        let mut state = TreeTN::<TensorDynLen, String>::new();
        let nelem = ext_dim * phys_dim;
        let t = TensorDynLen::from_dense_f64(
            vec![external.clone(), contracted.clone()],
            vec![1.0; nelem],
        );
        state.add_tensor("site0".to_string(), t).unwrap();

        let s_in = unique_dyn_index(&mut used, phys_dim);
        let s_out = unique_dyn_index(&mut used, phys_dim);
        let mut id_data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            id_data[k * phys_dim + k] = 1.0;
        }
        let op_t = TensorDynLen::from_dense_f64(vec![s_out.clone(), s_in.clone()], id_data);
        let mut op_tn = TreeTN::<TensorDynLen, String>::new();
        op_tn.add_tensor("site0".to_string(), op_t).unwrap();

        let mut im = HashMap::new();
        im.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: contracted.clone(),
                internal_index: s_in,
            },
        );
        let mut om = HashMap::new();
        om.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: contracted,
                internal_index: s_out,
            },
        );

        let projected_op = Arc::new(RwLock::new(ProjectedOperator::with_index_mappings(
            op_tn, im, om,
        )));
        let reference_state = state.clone();

        let linop = LocalLinOp::new(
            projected_op,
            vec!["site0".to_string()],
            state.clone(),
            reference_state,
            AnyScalar::F64(1.0),
            AnyScalar::F64(0.0),
        );

        let site0 = "site0".to_string();
        let x = state
            .tensor(state.node_index(&site0).unwrap())
            .unwrap()
            .clone();
        let y = linop.apply(&x).unwrap();
        let x_ids: HashSet<_> = x
            .external_indices()
            .iter()
            .map(|i: &DynIndex| *i.id())
            .collect();
        let y_ids: HashSet<_> = y
            .external_indices()
            .iter()
            .map(|i: &DynIndex| *i.id())
            .collect();
        assert_eq!(x_ids, y_ids);
    }
}
