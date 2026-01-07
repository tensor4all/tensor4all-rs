//! LinsolveUpdater: Local update implementation for linsolve.
//!
//! Uses GMRES (via kryst) to solve the local linear problem at each sweep step.

use std::hash::Hash;
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::Result;

use kryst::context::ksp_context::Workspace;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::{GmresSolver, LinearSolver};
use kryst::utils::convergence::ConvergedReason;

use tensor4all_core::index::{DynId, Index, NoSymmSpace, Symmetry};
use tensor4all_core::storage::{DenseStorageF64, StorageScalar};
use tensor4all_core::{factorize, Canonical, FactorizeOptions, Storage, TensorDynLen};

use super::environment::NetworkTopology;
use super::local_linop::{LocalLinOp, StaticTopology};
use super::options::LinsolveOptions;
use super::projected_operator::ProjectedOperator;
use super::projected_state::ProjectedState;
use crate::treetn::localupdate::{LocalUpdateStep, LocalUpdater};
use crate::treetn::TreeTN;

/// Adapter to implement NetworkTopology for TreeTN's SiteIndexNetwork.
pub struct TreeTNTopology<'a, Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    treetn: &'a TreeTN<Id, Symm, V>,
}

impl<'a, Id, Symm, V> TreeTNTopology<'a, Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    pub fn new(treetn: &'a TreeTN<Id, Symm, V>) -> Self {
        Self { treetn }
    }
}

impl<'a, Id, Symm, V> NetworkTopology<V> for TreeTNTopology<'a, Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    type Neighbors<'b> = Box<dyn Iterator<Item = V> + 'b> where Self: 'b, V: 'b;

    fn neighbors(&self, node: &V) -> Self::Neighbors<'_> {
        Box::new(self.treetn.site_index_network().neighbors(node))
    }
}

/// LinsolveUpdater: Implements LocalUpdater for the linsolve algorithm.
///
/// At each sweep step:
/// 1. Compute local operator (from ProjectedOperator environments)
/// 2. Compute local RHS (from ProjectedState environments)
/// 3. Solve local linear system using GMRES
/// 4. Factorize the result and update the state
pub struct LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Projected operator (3-chain), wrapped in Arc<RwLock> for GMRES
    pub projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
    /// Projected state for RHS (2-chain)
    pub projected_state: ProjectedState<Id, Symm, V>,
    /// Solver options
    pub options: LinsolveOptions,
}

impl<Id, Symm, V> LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LinsolveUpdater.
    pub fn new(
        operator: TreeTN<Id, Symm, V>,
        rhs: TreeTN<Id, Symm, V>,
        options: LinsolveOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            projected_state: ProjectedState::new(rhs),
            options,
        }
    }

    /// Solve the local linear problem using GMRES.
    ///
    /// Solves: (a₀ + a₁ * H_local) |x_local⟩ = |b_local⟩
    fn solve_local(
        &mut self,
        region: &[V],
        init: &TensorDynLen<Id, Symm>,
        state: &TreeTN<Id, Symm, V>,
    ) -> Result<TensorDynLen<Id, Symm>> {
        let topology = TreeTNTopology::new(state);

        // Get local RHS: <b|_local
        let rhs_local = self.projected_state.local_constant_term(region, state, &topology)?;

        // Compute local dimension
        let dim: usize = init
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .product();

        // Convert RHS to flat array
        let b: Vec<f64> = f64::extract_dense_view(rhs_local.storage.as_ref())
            .map_err(|e| anyhow::anyhow!("RHS storage error: {}", e))?
            .to_vec();

        // Initial guess from input tensor
        let mut x: Vec<f64> = f64::extract_dense_view(init.storage.as_ref())
            .map_err(|e| anyhow::anyhow!("Init storage error: {}", e))?
            .to_vec();

        // Create static topology for ownership
        let static_topology = StaticTopology::from_treetn(state);

        // FIXME: state.clone() is inefficient for large states.
        // This is required because kryst's LinOp trait requires 'static lifetime,
        // so LocalLinOp must own all its data. Consider:
        // - Using Rc<RefCell<>> or Arc<RwLock<>> for shared state
        // - Restructuring to avoid repeated clones per sweep step
        // - Caching the LocalLinOp between calls if state hasn't changed
        let linop = LocalLinOp::new(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state.clone(),
            static_topology,
            init.clone(),
            self.options.a0,
            self.options.a1,
        );

        // Create GMRES solver
        let mut solver = GmresSolver::new(
            self.options.krylov_dim,
            self.options.krylov_tol,
            self.options.krylov_maxiter,
        );

        // Create workspace
        let mut workspace = Workspace::new(dim);

        // Create communicator (NoComm for single-process)
        let comm = UniverseComm::NoComm(NoComm);

        // Solve using GMRES
        let stats = solver.solve(
            &linop,
            None, // No preconditioner
            &b,
            &mut x,
            PcSide::Left, // Default preconditioner side (ignored without preconditioner)
            &comm,
            None, // No monitors
            Some(&mut workspace),
        )?;

        // Log convergence info if needed (using eprintln for now, could add tracing later)
        if stats.reason == ConvergedReason::DivergedMaxIts {
            eprintln!(
                "Warning: GMRES did not converge within {} iterations (residual: {})",
                stats.iterations,
                stats.final_residual
            );
        }

        // Convert solution back to tensor
        let dims: Vec<usize> = init.indices.iter().map(|idx| idx.symm.total_dim()).collect();
        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(x)));
        let result = TensorDynLen::new(init.indices.clone(), dims, storage);

        Ok(result)
    }
}

impl<Id, Symm, V> LocalUpdater<Id, Symm, V> for LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<Id, Symm, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<Id, Symm, V>,
    ) -> Result<TreeTN<Id, Symm, V>> {
        // LinsolveUpdater is designed for nsite=2
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "LinsolveUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            ));
        }

        let node_u = &step.nodes[0];
        let node_v = &step.nodes[1];

        // Get current local tensor (contracted 2-site tensor)
        let idx_u = subtree.node_index(node_u).unwrap();
        let idx_v = subtree.node_index(node_v).unwrap();
        let tensor_u = subtree.tensor(idx_u).unwrap();
        let tensor_v = subtree.tensor(idx_v).unwrap();

        // Contract u and v tensors
        let edge_uv = subtree.edge_between(node_u, node_v).unwrap();
        let bond_uv = subtree.bond_index(edge_uv).unwrap();

        let contract_inds: Vec<_> = tensor_u
            .indices
            .iter()
            .filter_map(|idx| {
                if idx.id == bond_uv.id {
                    tensor_v
                        .indices
                        .iter()
                        .find(|idx_v| idx_v.id == bond_uv.id)
                        .map(|idx_v| (idx.clone(), idx_v.clone()))
                } else {
                    None
                }
            })
            .collect();

        let init_local = tensor_u.tensordot(tensor_v, &contract_inds)?;

        // Solve local linear problem
        let solved_local = self.solve_local(&step.nodes, &init_local, full_treetn)?;

        // Factorize to get new tensors for u and v
        let site_c_u = full_treetn.site_space(node_u).cloned().unwrap_or_default();
        let left_inds: Vec<_> = solved_local
            .indices
            .iter()
            .filter(|idx| {
                // Keep site indices of u and link indices to u's other neighbors
                site_c_u.iter().any(|s| s.id == idx.id)
                    || full_treetn
                        .site_index_network()
                        .neighbors(node_u)
                        .filter(|n| n != node_v)
                        .any(|neighbor| {
                            full_treetn
                                .edge_between(node_u, &neighbor)
                                .and_then(|e| full_treetn.bond_index(e))
                                .map(|b| b.id == idx.id)
                                .unwrap_or(false)
                        })
            })
            .cloned()
            .collect();

        // Set up factorization options
        let mut options = FactorizeOptions::svd().with_canonical(Canonical::Left);

        if let Some(max_rank) = self.options.truncation.max_rank {
            options = options.with_max_rank(max_rank);
        }
        if let Some(rtol) = self.options.truncation.rtol {
            options = options.with_rtol(rtol);
        }

        // Factorize
        let factorize_result = factorize(&solved_local, &left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;

        let new_tensor_u = factorize_result.left;
        let new_tensor_v = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Preserve original bond index ID
        let old_bond = subtree.bond_index(edge_uv).unwrap().clone();
        let preserved_bond = Index::new_with_tags(
            old_bond.id.clone(),
            new_bond.symm.clone(),
            old_bond.tags.clone(),
        );

        let new_tensor_u = new_tensor_u.replaceind(&new_bond, &preserved_bond);
        let new_tensor_v = new_tensor_v.replaceind(&new_bond, &preserved_bond);

        // Update subtree
        let idx_u_sub = subtree.node_index(node_u).unwrap();
        let idx_v_sub = subtree.node_index(node_v).unwrap();

        subtree.replace_edge_bond(edge_uv, preserved_bond.clone())?;
        subtree.replace_tensor(idx_u_sub, new_tensor_u)?;
        subtree.replace_tensor(idx_v_sub, new_tensor_v)?;

        // Set ortho_towards
        subtree.set_ortho_towards(&preserved_bond.id, Some(step.new_center.clone()));
        subtree.set_canonical_center([step.new_center.clone()])?;

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<Id, Symm, V>,
    ) -> Result<()> {
        let topology = TreeTNTopology::new(full_treetn_after);

        // Invalidate all caches affected by the updated region
        {
            let mut proj_op = self.projected_operator.write().unwrap();
            proj_op.invalidate(&step.nodes, &topology);
        }
        self.projected_state.invalidate(&step.nodes, &topology);

        Ok(())
    }
}
