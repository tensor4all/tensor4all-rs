//! LocalLinOp: LinOp wrapper for projected operator.
//!
//! This module provides a wrapper that implements kryst's `LinOp` trait
//! for our projected tensor network operator, enabling GMRES solving.

use std::any::Any;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use kryst::matrix::op::LinOp;
use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};
use tensor4all_core::storage::{DenseStorageF64, StorageScalar};
use tensor4all_core::{Storage, TensorDynLen};

use super::linear_operator::LinearOperator;
use super::projected_operator::ProjectedOperator;
use crate::treetn::TreeTN;

/// LocalLinOp: Wraps the projected operator for use with kryst GMRES.
///
/// This implements `LinOp` to compute `y = (a₀ + a₁ * H) * x`
/// where H is the projected operator and x, y are flattened tensor data.
///
/// All data is owned to satisfy kryst's `'static` requirement.
pub struct LocalLinOp<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// The projected operator (shared, mutable for environment caching)
    pub projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
    /// Linear operator with index mapping (optional, used when available)
    pub linear_operator: Option<Arc<LinearOperator<Id, Symm, V>>>,
    /// The region being updated
    pub region: Vec<V>,
    /// Current state for ket in environment computation (V_in space)
    pub state: TreeTN<Id, Symm, V>,
    /// Reference state for bra in environment computation (V_out space)
    /// If None, uses `state` (same as ket) for V_in = V_out case
    pub bra_state: Option<TreeTN<Id, Symm, V>>,
    /// Template tensor (stores index structure for reshaping)
    pub template: TensorDynLen<Id, Symm>,
    /// Coefficient a₀
    pub a0: f64,
    /// Coefficient a₁
    pub a1: f64,
    /// Dimension of the local Hilbert space
    pub dim: usize,
}

impl<Id, Symm, V> LocalLinOp<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new LocalLinOp for V_in = V_out case.
    pub fn new(
        projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
        region: Vec<V>,
        state: TreeTN<Id, Symm, V>,
        template: TensorDynLen<Id, Symm>,
        a0: f64,
        a1: f64,
    ) -> Self {
        // Compute dimension from template
        let dim: usize = template
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .product();

        Self {
            projected_operator,
            linear_operator: None,
            region,
            state,
            bra_state: None, // Use state as bra (V_in = V_out)
            template,
            a0,
            a1,
            dim,
        }
    }

    /// Create a new LocalLinOp for V_in ≠ V_out case with explicit bra_state.
    pub fn with_bra_state(
        projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
        region: Vec<V>,
        state: TreeTN<Id, Symm, V>,
        bra_state: TreeTN<Id, Symm, V>,
        template: TensorDynLen<Id, Symm>,
        a0: f64,
        a1: f64,
    ) -> Self {
        // Compute dimension from template
        let dim: usize = template
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .product();

        Self {
            projected_operator,
            linear_operator: None,
            region,
            state,
            bra_state: Some(bra_state),
            template,
            a0,
            a1,
            dim,
        }
    }

    /// Create a new LocalLinOp with a LinearOperator for index mapping.
    pub fn with_linear_operator(
        projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
        linear_operator: Arc<LinearOperator<Id, Symm, V>>,
        region: Vec<V>,
        state: TreeTN<Id, Symm, V>,
        bra_state: Option<TreeTN<Id, Symm, V>>,
        template: TensorDynLen<Id, Symm>,
        a0: f64,
        a1: f64,
    ) -> Self {
        // Compute dimension from template
        let dim: usize = template
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .product();

        Self {
            projected_operator,
            linear_operator: Some(linear_operator),
            region,
            state,
            bra_state,
            template,
            a0,
            a1,
            dim,
        }
    }

    /// Get the bra state for environment computation.
    /// Returns bra_state if set, otherwise returns state (V_in = V_out case).
    fn get_bra_state(&self) -> &TreeTN<Id, Symm, V> {
        self.bra_state.as_ref().unwrap_or(&self.state)
    }

    /// Convert flat array to tensor.
    fn array_to_tensor(&self, x: &[f64]) -> TensorDynLen<Id, Symm> {
        let dims: Vec<usize> = self
            .template
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .collect();
        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(x.to_vec())));
        TensorDynLen::new(self.template.indices.clone(), dims, storage)
    }

    /// Convert tensor to flat array.
    fn tensor_to_array(&self, tensor: &TensorDynLen<Id, Symm>) -> Vec<f64> {
        f64::extract_dense_view(tensor.storage.as_ref())
            .expect("Expected DenseF64 storage")
            .to_vec()
    }
}

impl<Id, Symm, V> LinOp for LocalLinOp<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm:
        Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        // Convert x to tensor
        let x_tensor = self.array_to_tensor(x);

        // Apply operator: H * x
        // Use LinearOperator if available (handles index mapping correctly)
        // Otherwise fall back to ProjectedOperator
        let hx = if let Some(ref linear_op) = self.linear_operator {
            // Use LinearOperator.apply_local for correct index handling
            linear_op
                .apply_local(&x_tensor, &self.region)
                .expect("Failed to apply linear operator")
        } else {
            // Fall back to ProjectedOperator
            // Pass both ket_state (self.state) and bra_state for environment computation
            let bra_state = self.get_bra_state();
            let mut proj_op = self.projected_operator.write().unwrap();
            proj_op
                .apply(
                    &x_tensor,
                    &self.region,
                    &self.state,
                    bra_state,
                    self.state.site_index_network(),
                )
                .expect("Failed to apply projected operator")
        };

        // Compute y = a₀ * x + a₁ * H * x
        let hx_data = self.tensor_to_array(&hx);

        for (i, yi) in y.iter_mut().enumerate() {
            *yi = self.a0 * x[i] + self.a1 * hx_data[i];
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
