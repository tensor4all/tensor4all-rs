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
use tensor4all_core::{contract_multi, FactorizeAlg, Storage, TensorDynLen};

use super::linear_operator::LinearOperator;
use super::local_linop::LocalLinOp;
use super::options::LinsolveOptions;
use super::projected_operator::ProjectedOperator;
use super::projected_state::ProjectedState;
use crate::treetn::decompose::{factorize_tensor_to_treetn_with, TreeTopology};
use crate::treetn::localupdate::{LocalUpdateStep, LocalUpdater};
use crate::treetn::TreeTN;

/// Report from LinsolveUpdater::verify().
#[derive(Debug, Clone)]
pub struct LinsolveVerifyReport<V> {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// Errors that would prevent linsolve from working
    pub errors: Vec<String>,
    /// Warnings that might indicate issues
    pub warnings: Vec<String>,
    /// Per-node details
    pub node_details: Vec<NodeVerifyDetail<V>>,
}

impl<V> Default for LinsolveVerifyReport<V> {
    fn default() -> Self {
        Self {
            is_valid: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            node_details: Vec::new(),
        }
    }
}

/// Per-node verification details.
#[derive(Debug, Clone)]
pub struct NodeVerifyDetail<V> {
    /// Node name
    pub node: V,
    /// State's site space index IDs
    pub state_site_indices: Vec<String>,
    /// Operator's site space index IDs
    pub op_site_indices: Vec<String>,
    /// State tensor's all index IDs with dimensions
    pub state_tensor_indices: Vec<String>,
    /// Operator tensor's all index IDs with dimensions
    pub op_tensor_indices: Vec<String>,
    /// Number of common indices between state and operator
    pub common_index_count: usize,
}

impl<V: std::fmt::Debug> std::fmt::Display for LinsolveVerifyReport<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LinsolveVerifyReport:")?;
        writeln!(f, "  Valid: {}", self.is_valid)?;

        if !self.errors.is_empty() {
            writeln!(f, "  Errors:")?;
            for err in &self.errors {
                writeln!(f, "    - {}", err)?;
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warn in &self.warnings {
                writeln!(f, "    - {}", warn)?;
            }
        }

        if !self.node_details.is_empty() {
            writeln!(f, "  Node Details:")?;
            for detail in &self.node_details {
                writeln!(f, "    {:?}:", detail.node)?;
                writeln!(
                    f,
                    "      State site indices: {:?}",
                    detail.state_site_indices
                )?;
                writeln!(f, "      Op site indices: {:?}", detail.op_site_indices)?;
                writeln!(
                    f,
                    "      State tensor indices: {:?}",
                    detail.state_tensor_indices
                )?;
                writeln!(f, "      Op tensor indices: {:?}", detail.op_tensor_indices)?;
                writeln!(f, "      Common index count: {}", detail.common_index_count)?;
            }
        }

        Ok(())
    }
}

/// LinsolveUpdater: Implements LocalUpdater for the linsolve algorithm.
///
/// At each sweep step:
/// 1. Compute local operator (from ProjectedOperator environments)
/// 2. Compute local RHS (from ProjectedState environments)
/// 3. Solve local linear system using GMRES
/// 4. Factorize the result and update the state
///
/// # Input/Output Space Support
///
/// For operators where input space V_in ≠ output space V_out:
/// - Use `with_reference_state` constructor to specify a reference state in V_out
/// - The reference state is used as the "bra" in environment computations
/// - The current solution x (in V_in) is used as the "ket"
///
/// For V_in = V_out (default): The current solution x is used for both bra and ket.
pub struct LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Projected operator (3-chain), wrapped in Arc<RwLock> for GMRES
    pub projected_operator: Arc<RwLock<ProjectedOperator<Id, Symm, V>>>,
    /// Linear operator with index mapping (handles s_in/s_out correctly)
    pub linear_operator: Option<Arc<LinearOperator<Id, Symm, V>>>,
    /// Projected state for RHS (2-chain)
    pub projected_state: ProjectedState<Id, Symm, V>,
    /// Reference state for bra in environment computation (V_out space)
    /// If None, uses the current solution x (V_in = V_out case)
    pub reference_state_out: Option<TreeTN<Id, Symm, V>>,
    /// Solver options
    pub options: LinsolveOptions,
}

impl<Id, Symm, V> LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm:
        Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LinsolveUpdater for V_in = V_out case.
    pub fn new(
        operator: TreeTN<Id, Symm, V>,
        rhs: TreeTN<Id, Symm, V>,
        options: LinsolveOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            linear_operator: None,
            projected_state: ProjectedState::new(rhs),
            reference_state_out: None,
            options,
        }
    }

    /// Create a new LinsolveUpdater for V_in ≠ V_out case with explicit reference state.
    ///
    /// # Arguments
    /// * `operator` - The operator A mapping V_in → V_out
    /// * `rhs` - The RHS b in V_out
    /// * `reference_state_out` - Reference state in V_out for bra in environment computation
    /// * `options` - Solver options
    pub fn with_reference_state(
        operator: TreeTN<Id, Symm, V>,
        rhs: TreeTN<Id, Symm, V>,
        reference_state_out: TreeTN<Id, Symm, V>,
        options: LinsolveOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            linear_operator: None,
            projected_state: ProjectedState::new(rhs),
            reference_state_out: Some(reference_state_out),
            options,
        }
    }

    /// Create a new LinsolveUpdater with a LinearOperator for correct index handling.
    ///
    /// The LinearOperator wraps the MPO and handles the mapping between:
    /// - True site indices (from state x and b)
    /// - Internal MPO indices (s_in_tmp, s_out_tmp)
    ///
    /// This is required when the MPO uses independent indices for input/output
    /// (which is necessary because a tensor cannot have two indices with the same ID).
    pub fn with_linear_operator(
        linear_operator: LinearOperator<Id, Symm, V>,
        rhs: TreeTN<Id, Symm, V>,
        reference_state_out: Option<TreeTN<Id, Symm, V>>,
        options: LinsolveOptions,
    ) -> Self {
        let operator = linear_operator.mpo.clone();
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            linear_operator: Some(Arc::new(linear_operator)),
            projected_state: ProjectedState::new(rhs),
            reference_state_out,
            options,
        }
    }

    /// Get the bra state for environment computation.
    /// Returns reference_state_out if set, otherwise returns the ket_state (V_in = V_out case).
    pub fn get_bra_state<'a>(
        &'a self,
        ket_state: &'a TreeTN<Id, Symm, V>,
    ) -> &'a TreeTN<Id, Symm, V> {
        self.reference_state_out.as_ref().unwrap_or(ket_state)
    }

    /// Verify internal data consistency between operator, RHS, and state.
    ///
    /// This function checks that:
    /// 1. The operator's site space structure is compatible with the state
    /// 2. The operator's input indices can match the state's site indices
    /// 3. Environment computation requirements are satisfiable
    ///
    /// Returns a detailed report of any inconsistencies found.
    pub fn verify(&self, state: &TreeTN<Id, Symm, V>) -> Result<LinsolveVerifyReport<V>> {
        let mut report = LinsolveVerifyReport::default();

        let proj_op = self.projected_operator.read().unwrap();
        let operator = &proj_op.operator;
        let rhs = &self.projected_state.rhs;

        // Check node consistency
        let state_nodes: std::collections::BTreeSet<_> = state
            .site_index_network()
            .node_names()
            .into_iter()
            .collect();
        let op_nodes: std::collections::BTreeSet<_> = operator
            .site_index_network()
            .node_names()
            .into_iter()
            .collect();
        let rhs_nodes: std::collections::BTreeSet<_> =
            rhs.site_index_network().node_names().into_iter().collect();

        if state_nodes != op_nodes {
            report.errors.push(format!(
                "State and operator have different node sets. State: {:?}, Operator: {:?}",
                state_nodes, op_nodes
            ));
        }

        if state_nodes != rhs_nodes {
            report.errors.push(format!(
                "State and RHS have different node sets. State: {:?}, RHS: {:?}",
                state_nodes, rhs_nodes
            ));
        }

        // Check site index compatibility per node
        for node in &state_nodes {
            let state_site = state.site_space(node);
            let op_site = operator.site_space(node);
            let _rhs_site = rhs.site_space(node);

            // Get state tensor indices
            if let Some(state_idx) = state.node_index(node) {
                if let Some(state_tensor) = state.tensor(state_idx) {
                    let state_indices: Vec<_> = state_tensor
                        .indices
                        .iter()
                        .map(|idx| (idx.id.clone(), idx.symm.total_dim()))
                        .collect();

                    // Get operator tensor indices
                    if let Some(op_idx) = operator.node_index(node) {
                        if let Some(op_tensor) = operator.tensor(op_idx) {
                            let op_indices: Vec<_> = op_tensor
                                .indices
                                .iter()
                                .map(|idx| (idx.id.clone(), idx.symm.total_dim()))
                                .collect();

                            // Check for common indices (should have at least bond indices)
                            let common_count = state_indices
                                .iter()
                                .filter(|(id, _)| op_indices.iter().any(|(oid, _)| oid == id))
                                .count();

                            report.node_details.push(NodeVerifyDetail {
                                node: (*node).clone(),
                                state_site_indices: state_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id)).collect())
                                    .unwrap_or_default(),
                                op_site_indices: op_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id)).collect())
                                    .unwrap_or_default(),
                                state_tensor_indices: state_indices
                                    .iter()
                                    .map(|(id, dim)| format!("{:?}(dim={})", id, dim))
                                    .collect(),
                                op_tensor_indices: op_indices
                                    .iter()
                                    .map(|(id, dim)| format!("{:?}(dim={})", id, dim))
                                    .collect(),
                                common_index_count: common_count,
                            });

                            // Warn if no site indices in common (expected for MPO)
                            // In proper MPO structure, operator should have input indices
                            // that match state's site indices
                            if common_count == 0 {
                                report.warnings.push(format!(
                                    "Node {:?}: No common indices between state and operator tensors. \
                                     State has {:?}, operator has {:?}",
                                    node, state_indices, op_indices
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Final verdict
        report.is_valid = report.errors.is_empty();

        Ok(report)
    }

    /// Contract all tensors in the region into a single local tensor.
    fn contract_region(
        &self,
        subtree: &TreeTN<Id, Symm, V>,
        region: &[V],
    ) -> Result<TensorDynLen<Id, Symm>> {
        if region.is_empty() {
            return Err(anyhow::anyhow!("Region cannot be empty"));
        }

        // Collect all tensors in the region
        let tensors: Vec<TensorDynLen<Id, Symm>> = region
            .iter()
            .map(|node| {
                let idx = subtree
                    .node_index(node)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node))?;
                subtree
                    .tensor(idx)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))
                    .map(|t| t.clone())
            })
            .collect::<Result<_>>()?;

        // Use contract_multi for optimal contraction ordering
        contract_multi(&tensors)
    }

    /// Build TreeTopology for the subtree region from the solved tensor.
    ///
    /// Maps each node to the positions of its indices in the solved tensor.
    fn build_subtree_topology(
        &self,
        solved_tensor: &TensorDynLen<Id, Symm>,
        region: &[V],
        full_treetn: &TreeTN<Id, Symm, V>,
    ) -> Result<TreeTopology<V>> {
        use std::collections::HashMap;

        let mut nodes: HashMap<V, Vec<usize>> = HashMap::new();
        let mut edges: Vec<(V, V)> = Vec::new();

        // For each node in the region, find which indices belong to it
        for node in region {
            let mut positions = Vec::new();

            // Get site indices for this node
            if let Some(site_indices) = full_treetn.site_space(node) {
                for site_idx in site_indices {
                    // Find position in solved_tensor
                    if let Some(pos) = solved_tensor
                        .indices
                        .iter()
                        .position(|idx| idx.id == site_idx.id)
                    {
                        positions.push(pos);
                    }
                }
            }

            // Get bond indices to neighbors outside the region
            for neighbor in full_treetn.site_index_network().neighbors(node) {
                if !region.contains(&neighbor) {
                    // This is an external neighbor - the bond belongs to this node
                    if let Some(edge) = full_treetn.edge_between(node, &neighbor) {
                        if let Some(bond) = full_treetn.bond_index(edge) {
                            if let Some(pos) = solved_tensor
                                .indices
                                .iter()
                                .position(|idx| idx.id == bond.id)
                            {
                                positions.push(pos);
                            }
                        }
                    }
                }
            }

            nodes.insert(node.clone(), positions);
        }

        // Build edges between nodes in the region
        for (i, node_a) in region.iter().enumerate() {
            for node_b in region.iter().skip(i + 1) {
                if full_treetn.edge_between(node_a, node_b).is_some() {
                    edges.push((node_a.clone(), node_b.clone()));
                }
            }
        }

        Ok(TreeTopology::new(nodes, edges))
    }

    /// Copy decomposed tensors back to subtree, preserving original bond IDs.
    fn copy_decomposed_to_subtree(
        &self,
        subtree: &mut TreeTN<Id, Symm, V>,
        decomposed: &TreeTN<Id, Symm, V>,
        region: &[V],
        full_treetn: &TreeTN<Id, Symm, V>,
    ) -> Result<()> {
        // For each node in the region, update its tensor in subtree
        for node in region {
            let decomp_idx = decomposed
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in decomposed TreeTN", node))?;
            let mut new_tensor = decomposed.tensor(decomp_idx).unwrap().clone();

            // Replace bond indices to preserve original IDs
            for neighbor in full_treetn.site_index_network().neighbors(node) {
                if region.contains(&neighbor) {
                    // Internal bond - need to find and replace the new bond ID
                    if let Some(orig_edge) = subtree.edge_between(node, &neighbor) {
                        if let Some(orig_bond) = subtree.bond_index(orig_edge) {
                            // Find the decomposed edge bond
                            if let Some(decomp_edge) = decomposed.edge_between(node, &neighbor) {
                                if let Some(decomp_bond) = decomposed.bond_index(decomp_edge) {
                                    // Create preserved bond with original ID but new dimension
                                    let preserved_bond = Index::new_with_tags(
                                        orig_bond.id.clone(),
                                        decomp_bond.symm.clone(),
                                        orig_bond.tags.clone(),
                                    );
                                    new_tensor =
                                        new_tensor.replaceind(decomp_bond, &preserved_bond);

                                    // Update the edge bond in subtree (only once per edge)
                                    if node < &neighbor {
                                        subtree.replace_edge_bond(orig_edge, preserved_bond)?;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Update the tensor in subtree
            let subtree_idx = subtree.node_index(node).unwrap();
            subtree.replace_tensor(subtree_idx, new_tensor)?;
        }

        Ok(())
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
        // Use state's SiteIndexNetwork directly (implements NetworkTopology)
        let topology = state.site_index_network();

        // Get local RHS: <b|_local
        // For V_in ≠ V_out case, use reference_state_out for bra in environment computation
        let rhs_local = match &self.reference_state_out {
            Some(ref_out) => self
                .projected_state
                .local_constant_term_with_bra(region, state, ref_out, topology)?,
            None => self
                .projected_state
                .local_constant_term(region, state, topology)?,
        };

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

        // FIXME: state.clone() is inefficient for large states.
        // This is required because kryst's LinOp trait requires 'static lifetime,
        // so LocalLinOp must own all its data. Consider:
        // - Using Rc<RefCell<>> or Arc<RwLock<>> for shared state
        // - Restructuring to avoid repeated clones per sweep step
        // - Caching the LocalLinOp between calls if state hasn't changed
        //
        // For V_in ≠ V_out case, we use reference_state_out as the bra state.
        // For V_in = V_out case (reference_state_out is None), the state is used as both ket and bra.
        let bra_state = self.reference_state_out.clone();

        let linop = if let Some(ref linear_op) = self.linear_operator {
            // Use LinearOperator for correct index handling
            LocalLinOp::with_linear_operator(
                Arc::clone(&self.projected_operator),
                Arc::clone(linear_op),
                region.to_vec(),
                state.clone(),
                bra_state,
                init.clone(),
                self.options.a0,
                self.options.a1,
            )
        } else {
            // Legacy path without LinearOperator
            match bra_state {
                Some(bra) => LocalLinOp::with_bra_state(
                    Arc::clone(&self.projected_operator),
                    region.to_vec(),
                    state.clone(),
                    bra,
                    init.clone(),
                    self.options.a0,
                    self.options.a1,
                ),
                None => LocalLinOp::new(
                    Arc::clone(&self.projected_operator),
                    region.to_vec(),
                    state.clone(),
                    init.clone(),
                    self.options.a0,
                    self.options.a1,
                ),
            }
        };

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
                stats.iterations, stats.final_residual
            );
        }

        // Convert solution back to tensor
        let dims: Vec<usize> = init
            .indices
            .iter()
            .map(|idx| idx.symm.total_dim())
            .collect();
        let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(x)));
        let result = TensorDynLen::new(init.indices.clone(), dims, storage);

        Ok(result)
    }
}

impl<Id, Symm, V> LocalUpdater<Id, Symm, V> for LinsolveUpdater<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm:
        Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<Id, Symm, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<Id, Symm, V>,
    ) -> Result<TreeTN<Id, Symm, V>> {
        // Contract tensors in the region into a single local tensor
        let init_local = self.contract_region(&subtree, &step.nodes)?;

        // Solve local linear problem using GMRES
        let solved_local = self.solve_local(&step.nodes, &init_local, full_treetn)?;

        // Build TreeTopology for the subtree region
        let topology = self.build_subtree_topology(&solved_local, &step.nodes, full_treetn)?;

        // Decompose solved tensor back into TreeTN using factorize_tensor_to_treetn
        let decomposed =
            factorize_tensor_to_treetn_with(&solved_local, &topology, FactorizeAlg::SVD)?;

        // Copy decomposed tensors back to subtree, preserving original bond IDs
        self.copy_decomposed_to_subtree(&mut subtree, &decomposed, &step.nodes, full_treetn)?;

        // Set canonical center
        subtree.set_canonical_center([step.new_center.clone()])?;

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<Id, Symm, V>,
    ) -> Result<()> {
        // Use state's SiteIndexNetwork directly (implements NetworkTopology)
        let topology = full_treetn_after.site_index_network();

        // Invalidate all caches affected by the updated region
        {
            let mut proj_op = self.projected_operator.write().unwrap();
            proj_op.invalidate(&step.nodes, topology);
        }
        self.projected_state.invalidate(&step.nodes, topology);

        Ok(())
    }
}
