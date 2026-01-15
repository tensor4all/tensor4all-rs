//! LinsolveUpdater: Local update implementation for linsolve.
//!
//! Uses GMRES (via tensor4all_core::krylov) to solve the local linear problem at each sweep step.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::Result;

use tensor4all_core::any_scalar::AnyScalar;
use tensor4all_core::krylov::{gmres, GmresOptions};
use tensor4all_core::{AllowedPairs, FactorizeAlg, IndexLike, TensorLike};

use super::local_linop::LocalLinOp;
use super::options::LinsolveOptions;
use super::projected_operator::{IndexMapping, ProjectedOperator};
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
pub struct LinsolveUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Projected operator (3-chain), wrapped in Arc<RwLock> for GMRES
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// Projected state for RHS (2-chain)
    pub projected_state: ProjectedState<T, V>,
    /// Reference state for bra in environment computation (V_out space)
    /// If None, uses the current solution x (V_in = V_out case)
    pub reference_state_out: Option<TreeTN<T, V>>,
    /// Solver options
    pub options: LinsolveOptions,
}

impl<T, V> LinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LinsolveUpdater for V_in = V_out case.
    pub fn new(operator: TreeTN<T, V>, rhs: TreeTN<T, V>, options: LinsolveOptions) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
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
        operator: TreeTN<T, V>,
        rhs: TreeTN<T, V>,
        reference_state_out: TreeTN<T, V>,
        options: LinsolveOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            projected_state: ProjectedState::new(rhs),
            reference_state_out: Some(reference_state_out),
            options,
        }
    }

    /// Create a new LinsolveUpdater with index mappings for correct index handling.
    ///
    /// Use this when the MPO uses internal indices (s_in_tmp, s_out_tmp) that differ
    /// from the state's site indices. The mappings define how to translate between them.
    ///
    /// # Arguments
    /// * `operator` - The MPO with internal index IDs
    /// * `input_mapping` - Mapping from true input indices to MPO's internal indices
    /// * `output_mapping` - Mapping from true output indices to MPO's internal indices
    /// * `rhs` - The RHS b
    /// * `reference_state_out` - Optional reference state for V_in ≠ V_out case
    /// * `options` - Solver options
    pub fn with_index_mappings(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
        rhs: TreeTN<T, V>,
        reference_state_out: Option<TreeTN<T, V>>,
        options: LinsolveOptions,
    ) -> Self {
        let projected_operator =
            ProjectedOperator::with_index_mappings(operator, input_mapping, output_mapping);
        Self {
            projected_operator: Arc::new(RwLock::new(projected_operator)),
            projected_state: ProjectedState::new(rhs),
            reference_state_out,
            options,
        }
    }

    /// Get the bra state for environment computation.
    /// Returns reference_state_out if set, otherwise returns the ket_state (V_in = V_out case).
    pub fn get_bra_state<'a>(&'a self, ket_state: &'a TreeTN<T, V>) -> &'a TreeTN<T, V> {
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
    pub fn verify(&self, state: &TreeTN<T, V>) -> Result<LinsolveVerifyReport<V>> {
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
                    let state_indices_vec = state_tensor.external_indices();
                    let state_indices: Vec<_> = state_indices_vec
                        .iter()
                        .map(|idx| (idx.id().clone(), idx.dim()))
                        .collect();

                    // Get operator tensor indices
                    if let Some(op_idx) = operator.node_index(node) {
                        if let Some(op_tensor) = operator.tensor(op_idx) {
                            let op_indices_vec = op_tensor.external_indices();
                            let op_indices: Vec<_> = op_indices_vec
                                .iter()
                                .map(|idx| (idx.id().clone(), idx.dim()))
                                .collect();

                            // Check for common indices (should have at least bond indices)
                            let common_count = state_indices
                                .iter()
                                .filter(|(id, _)| op_indices.iter().any(|(oid, _)| oid == id))
                                .count();

                            report.node_details.push(NodeVerifyDetail {
                                node: (*node).clone(),
                                state_site_indices: state_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id())).collect())
                                    .unwrap_or_default(),
                                op_site_indices: op_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id())).collect())
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
    fn contract_region(&self, subtree: &TreeTN<T, V>, region: &[V]) -> Result<T> {
        if region.is_empty() {
            return Err(anyhow::anyhow!("Region cannot be empty"));
        }

        // Collect all tensors in the region
        let tensors: Vec<T> = region
            .iter()
            .map(|node| {
                let idx = subtree
                    .node_index(node)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node))?;
                let tensor = subtree
                    .tensor(idx)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?;
                Ok(tensor.clone())
            })
            .collect::<Result<_>>()?;

        // Use TensorLike::contract for contraction
        T::contract(&tensors, AllowedPairs::All)
    }

    /// Build TreeTopology for the subtree region from the solved tensor.
    ///
    /// Maps each node to the positions of its indices in the solved tensor.
    fn build_subtree_topology(
        &self,
        solved_tensor: &T,
        region: &[V],
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTopology<V>> {
        use std::collections::HashMap;

        let mut nodes: HashMap<V, Vec<usize>> = HashMap::new();
        let mut edges: Vec<(V, V)> = Vec::new();

        let solved_indices = solved_tensor.external_indices();

        // For each node in the region, find which indices belong to it
        for node in region {
            let mut positions = Vec::new();

            // Get site indices for this node
            if let Some(site_indices) = full_treetn.site_space(node) {
                for site_idx in site_indices {
                    // Find position in solved_tensor
                    if let Some(pos) = solved_indices.iter().position(|idx| idx == site_idx) {
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
                            if let Some(pos) = solved_indices.iter().position(|idx| idx == bond) {
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
        subtree: &mut TreeTN<T, V>,
        decomposed: &TreeTN<T, V>,
        region: &[V],
        full_treetn: &TreeTN<T, V>,
    ) -> Result<()> {
        use std::collections::HashMap;

        // Phase 1: Build a mapping from decomposed bond IDs to new bond indices
        // For internal bonds, we create a single new bond index that will be used
        // for both nodes sharing that edge
        let mut bond_mapping: HashMap<<T::Index as IndexLike>::Id, T::Index> = HashMap::new();

        for (i, node_a) in region.iter().enumerate() {
            for node_b in region.iter().skip(i + 1) {
                // Check if there's an edge between these nodes
                if let Some(decomp_edge) = decomposed.edge_between(node_a, node_b) {
                    if let Some(decomp_bond) = decomposed.bond_index(decomp_edge) {
                        // Create a new bond index with original ID structure
                        // Use sim() once for this edge
                        if let Some(orig_edge) = subtree.edge_between(node_a, node_b) {
                            if let Some(orig_bond) = subtree.bond_index(orig_edge) {
                                let new_bond = orig_bond.sim();
                                bond_mapping.insert(decomp_bond.id().clone(), new_bond.clone());

                                // Update the edge bond in subtree
                                subtree.replace_edge_bond(orig_edge, new_bond)?;
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: For each node in the region, update its tensor using the bond mapping
        for node in region {
            let decomp_idx = decomposed
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in decomposed TreeTN", node))?;
            let mut new_tensor = decomposed.tensor(decomp_idx).unwrap().clone();

            // Replace bond indices using the pre-computed mapping
            for neighbor in full_treetn.site_index_network().neighbors(node) {
                if region.contains(&neighbor) {
                    // Internal bond - use the mapped bond
                    if let Some(decomp_edge) = decomposed.edge_between(node, &neighbor) {
                        if let Some(decomp_bond) = decomposed.bond_index(decomp_edge) {
                            if let Some(new_bond) = bond_mapping.get(decomp_bond.id()) {
                                new_tensor = new_tensor.replaceind(decomp_bond, new_bond)?;
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
    fn solve_local(&mut self, region: &[V], init: &T, state: &TreeTN<T, V>) -> Result<T> {
        // Use state's SiteIndexNetwork directly (implements NetworkTopology)
        let topology = state.site_index_network();

        // Get local RHS: <b|_local
        // For V_in ≠ V_out case, use reference_state_out for bra in environment computation
        let rhs_local_raw = match &self.reference_state_out {
            Some(ref_out) => self
                .projected_state
                .local_constant_term_with_bra(region, state, ref_out, topology)?,
            None => self
                .projected_state
                .local_constant_term(region, state, topology)?,
        };

        // Align RHS indices with init indices.
        // The RHS may have indices from the `rhs` TreeTN, while init has indices from
        // the current `state`. For GMRES operations (like b - A*x), they must match.
        let init_indices = init.external_indices();
        let rhs_indices = rhs_local_raw.external_indices();

        let rhs_local = if init_indices.len() == rhs_indices.len() {
            let indices_match = init_indices
                .iter()
                .zip(rhs_indices.iter())
                .all(|(ii, ri)| ii == ri);
            if indices_match {
                rhs_local_raw
            } else {
                // Replace RHS indices with init indices
                rhs_local_raw.replaceinds(&rhs_indices, &init_indices)?
            }
        } else {
            return Err(anyhow::anyhow!(
                "Index count mismatch between init ({}) and RHS ({})",
                init_indices.len(),
                rhs_indices.len()
            ));
        };

        // For V_in ≠ V_out case, we use reference_state_out as the bra state.
        // For V_in = V_out case (reference_state_out is None), the state is used as both ket and bra.
        let bra_state = self.reference_state_out.clone();

        // Convert coefficients to AnyScalar
        let a0 = AnyScalar::F64(self.options.a0);
        let a1 = AnyScalar::F64(self.options.a1);

        // Create local linear operator
        // ProjectedOperator handles both environment computation and index mappings
        let linop = match bra_state {
            Some(bra) => LocalLinOp::with_bra_state(
                Arc::clone(&self.projected_operator),
                region.to_vec(),
                state.clone(),
                bra,
                a0,
                a1,
            ),
            None => LocalLinOp::new(
                Arc::clone(&self.projected_operator),
                region.to_vec(),
                state.clone(),
                a0,
                a1,
            ),
        };

        // Create closure for GMRES that applies the linear operator
        let apply_a = |x: &T| linop.apply(x);

        // Set up GMRES options
        let gmres_options = GmresOptions {
            max_iter: self.options.krylov_dim,
            rtol: self.options.krylov_tol,
            max_restarts: (self.options.krylov_maxiter / self.options.krylov_dim).max(1),
            verbose: false,
        };

        // Solve using GMRES (works directly with TensorDynLen)
        let result = gmres(apply_a, &rhs_local, init, &gmres_options)?;

        // Log convergence info if needed
        if !result.converged {
            eprintln!(
                "Warning: GMRES did not converge (iterations: {}, residual: {:.6e})",
                result.iterations, result.residual_norm
            );
        }

        Ok(result.solution)
    }
}

impl<T, V> LocalUpdater<T, V> for LinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>> {
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
        full_treetn_after: &TreeTN<T, V>,
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
