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

use super::environment::NetworkTopology;
use super::projected_operator::ProjectedOperator;
use crate::treetn::TreeTN;

/// Simple topology stored as adjacency list.
#[derive(Clone, Debug)]
pub struct StaticTopology<V> {
    /// Adjacency list: node -> neighbors
    adjacency: std::collections::HashMap<V, Vec<V>>,
}

impl<V: Clone + Hash + Eq> StaticTopology<V> {
    /// Create from a TreeTN's topology.
    pub fn from_treetn<Id, Symm>(treetn: &TreeTN<Id, Symm, V>) -> Self
    where
        Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
        Symm: Clone + Symmetry + std::fmt::Debug,
        V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    {
        let mut adjacency = std::collections::HashMap::new();
        for node in treetn.site_index_network().node_names() {
            let neighbors: Vec<V> = treetn
                .site_index_network()
                .neighbors(node)
                .collect();
            adjacency.insert(node.clone(), neighbors);
        }
        Self { adjacency }
    }
}

impl<V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug> NetworkTopology<V> for StaticTopology<V> {
    type Neighbors<'a> = std::iter::Cloned<std::slice::Iter<'a, V>> where V: 'a;

    fn neighbors(&self, node: &V) -> Self::Neighbors<'_> {
        self.adjacency
            .get(node)
            .map(|v| v.iter().cloned())
            .unwrap_or_else(|| [].iter().cloned())
    }
}

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
    /// The region being updated
    pub region: Vec<V>,
    /// Current state for environment computation (cloned to own)
    pub state: TreeTN<Id, Symm, V>,
    /// Network topology (static/owned)
    pub topology: StaticTopology<V>,
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
    /// Create a new LocalLinOp.
    pub fn new(
        projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
        region: Vec<V>,
        state: TreeTN<Id, Symm, V>,
        topology: StaticTopology<V>,
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
            region,
            state,
            topology,
            template,
            a0,
            a1,
            dim,
        }
    }

    /// Convert flat array to tensor.
    fn array_to_tensor(&self, x: &[f64]) -> TensorDynLen<Id, Symm> {
        let dims: Vec<usize> = self.template.indices.iter().map(|idx| idx.symm.total_dim()).collect();
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
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    type S = f64;

    fn dims(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        // Convert x to tensor
        let x_tensor = self.array_to_tensor(x);

        // Apply projected operator: H * x
        let mut proj_op = self.projected_operator.write().unwrap();
        let hx = proj_op
            .apply(&x_tensor, &self.region, &self.state, &self.topology)
            .expect("Failed to apply projected operator");

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
