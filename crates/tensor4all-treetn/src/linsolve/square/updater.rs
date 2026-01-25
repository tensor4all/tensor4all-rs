//! SquareLinsolveUpdater: Local update implementation for square linsolve.
//!
//! Uses GMRES (via tensor4all_core::krylov) to solve the local linear problem at each sweep step.
//! This is the V_in = V_out specialized version.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::{Context, Result};

use tensor4all_core::any_scalar::AnyScalar;
use tensor4all_core::krylov::{gmres, GmresOptions};
use tensor4all_core::{AllowedPairs, FactorizeOptions, IndexLike, TensorLike};

use super::local_linop::LocalLinOp;
use super::projected_state::ProjectedState;
use crate::linsolve::common::{LinsolveOptions, ProjectedOperator};
use crate::operator::IndexMapping;
use crate::{
    factorize_tensor_to_treetn_with, get_boundary_edges, LocalUpdateStep, LocalUpdater, TreeTN,
    TreeTopology,
};

/// Report from SquareLinsolveUpdater::verify().
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

/// SquareLinsolveUpdater: Implements LocalUpdater for the square linsolve algorithm.
///
/// At each sweep step:
/// 1. Compute local operator (from ProjectedOperator environments)
/// 2. Compute local RHS (from ProjectedState environments)
/// 3. Solve local linear system using GMRES
/// 4. Factorize the result and update the state
///
/// This is the V_in = V_out specialized version. The current solution x is used
/// with a separate reference state (with different bond indices) for stable
/// environment computation.
pub struct SquareLinsolveUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Projected operator (3-chain), wrapped in Arc<RwLock> for GMRES
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// Projected state for RHS (2-chain)
    pub projected_state: ProjectedState<T, V>,
    /// Solver options
    pub options: LinsolveOptions,
    /// Reference state (separate from ket to avoid unintended contractions).
    /// Link indices are different from ket_state to prevent bra↔ket link contractions.
    /// Boundary bonds (region ↔ outside) maintain stable IDs for cache consistency.
    reference_state: TreeTN<T, V>,
    /// Mapping from boundary edge (node_in_region, neighbor_outside) to reference-side bond index.
    /// This ensures boundary bonds keep stable IDs across updates for environment cache reuse.
    boundary_bond_map: HashMap<(V, V), T::Index>,
}

impl<T, V> SquareLinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new SquareLinsolveUpdater.
    ///
    /// The reference_state will be initialized lazily on the first `before_step` call.
    pub fn new(operator: TreeTN<T, V>, rhs: TreeTN<T, V>, options: LinsolveOptions) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            projected_state: ProjectedState::new(rhs),
            options,
            reference_state: TreeTN::new(),
            boundary_bond_map: HashMap::new(),
        }
    }

    /// Create a new SquareLinsolveUpdater with index mappings for correct index handling.
    ///
    /// Use this when the MPO uses internal indices (s_in_tmp, s_out_tmp) that differ
    /// from the state's site indices. The mappings define how to translate between them.
    ///
    /// The reference_state will be initialized lazily on the first `before_step` call.
    ///
    /// # Arguments
    /// * `operator` - The MPO with internal index IDs
    /// * `input_mapping` - Mapping from true input indices to MPO's internal indices
    /// * `output_mapping` - Mapping from true output indices to MPO's internal indices
    /// * `rhs` - The RHS b
    /// * `options` - Solver options
    pub fn with_index_mappings(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
        rhs: TreeTN<T, V>,
        options: LinsolveOptions,
    ) -> Self {
        let projected_operator =
            ProjectedOperator::with_index_mappings(operator, input_mapping, output_mapping);
        Self {
            projected_operator: Arc::new(RwLock::new(projected_operator)),
            projected_state: ProjectedState::new(rhs),
            options,
            reference_state: TreeTN::new(),
            boundary_bond_map: HashMap::new(),
        }
    }

    /// Initialize reference_state from ket_state if not already initialized.
    ///
    /// Creates reference_state by relabeling link indices (using sim() for internal bonds,
    /// preserving boundary bonds for cache consistency).
    ///
    /// This is called lazily on the first `before_step` to ensure we have the initial ket_state.
    fn ensure_reference_state_initialized(&mut self, ket_state: &TreeTN<T, V>) -> Result<()> {
        // Check if reference_state is already initialized (has nodes)
        if !self.reference_state.node_names().is_empty() {
            return Ok(());
        }

        // Initialize reference_state by cloning ket_state and relabeling link indices
        // For boundary bonds, we'll preserve the mapping for later reuse
        let mut reference_state = ket_state.clone();

        // Get all edges to determine which are boundary bonds
        // Since we don't have a region yet, we'll relabel all links initially
        // Boundary bonds will be stabilized in after_step when we know the region
        reference_state.sim_linkinds_mut()?;

        // Initialize boundary_bond_map as empty (will be populated per-region in after_step)
        self.boundary_bond_map.clear();

        self.reference_state = reference_state;
        Ok(())
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

        let proj_op = self
            .projected_operator
            .read()
            .map_err(|e| {
                anyhow::anyhow!("Failed to acquire read lock on projected_operator: {}", e)
            })
            .context("verify: lock poisoned")?;
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
        let tensor_refs: Vec<&T> = tensors.iter().collect();
        T::contract(&tensor_refs, AllowedPairs::All)
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
                        // Create a new bond index matching decomposed bond dimension.
                        // Use sim() once for this edge to avoid ID collisions.
                        if let Some(orig_edge) = subtree.edge_between(node_a, node_b) {
                            let new_bond = decomp_bond.sim();
                            bond_mapping.insert(decomp_bond.id().clone(), new_bond.clone());

                            // Update the edge bond in subtree
                            subtree.replace_edge_bond(orig_edge, new_bond)?;
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
            let mut new_tensor = decomposed
                .tensor(decomp_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?
                .clone();

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
            let subtree_idx = subtree
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node))?;
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
        let rhs_local_raw = self
            .projected_state
            .local_constant_term(region, state, topology)?;

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
                // Permute RHS indices to init index order
                rhs_local_raw.permuteinds(&init_indices)?
            }
        } else {
            return Err(anyhow::anyhow!(
                "Index count mismatch between init ({}) and RHS ({})",
                init_indices.len(),
                rhs_indices.len()
            ));
        };

        // Convert coefficients to AnyScalar
        let a0 = AnyScalar::new_real(self.options.a0);
        let a1 = AnyScalar::new_real(self.options.a1);

        // Create local linear operator with separate reference_state
        // This prevents unintended bra↔ket link contractions in environment computation
        let linop = LocalLinOp::new(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state.clone(),
            self.reference_state.clone(),
            a0,
            a1,
        );

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

        // Note: GMRES convergence info (result.converged, result.iterations, result.residual_norm)
        // is not currently exposed. Consider adding tracing or returning via a result struct
        // if convergence diagnostics are needed.
        let _ = result.converged; // Suppress unused variable warning

        Ok(result.solution)
    }

    /// Synchronize reference_state region with ket_state, preserving boundary bond IDs.
    ///
    /// This ensures reference_state stays in sync with ket_state updates while maintaining
    /// stable boundary bond IDs for environment cache reuse.
    fn sync_reference_state_region(
        &mut self,
        step: &LocalUpdateStep<V>,
        ket_state: &TreeTN<T, V>,
    ) -> Result<()> {
        // Extract updated region from ket_state
        let ket_region = ket_state.extract_subtree(&step.nodes)?;

        // Build mapping from ket bond IDs to reference bond indices for *all* bonds incident to the region.
        //
        // Important: reference_state bond IDs must remain stable across steps, even when an edge alternates
        // between being a boundary edge and an internal edge in different steps (as happens in sweeps).
        // Therefore, we always reuse the current reference_state bond indices, and never create fresh IDs here.
        let mut ket_to_ref_bond_map: HashMap<<T::Index as IndexLike>::Id, T::Index> =
            HashMap::new();

        // Populate mapping for all edges incident to region nodes.
        // - Boundary edges (region ↔ outside): use reference_state's existing bond IDs for cache stability
        // - Internal edges (within region): use sim() to create new IDs distinct from ket
        // This ensures reference_state bond IDs are always different from ket_state bond IDs.
        let region_nodes: std::collections::HashSet<_> = step.nodes.iter().collect();
        for node in &step.nodes {
            for neighbor in ket_state.site_index_network().neighbors(node) {
                let ket_edge = match ket_state.edge_between(node, &neighbor) {
                    Some(e) => e,
                    None => continue,
                };
                let ket_bond = match ket_state.bond_index(ket_edge) {
                    Some(b) => b,
                    None => continue,
                };

                let ref_bond = if region_nodes.contains(&neighbor) {
                    // Internal edge: create new ID using sim()
                    ket_bond.sim()
                } else {
                    // Boundary edge: use reference_state's existing bond ID
                    let ref_edge = match self.reference_state.edge_between(node, &neighbor) {
                        Some(e) => e,
                        None => continue,
                    };
                    match self.reference_state.bond_index(ref_edge) {
                        Some(b) => b.clone(),
                        None => continue,
                    }
                };
                ket_to_ref_bond_map.insert(ket_bond.id().clone(), ref_bond);
            }
        }

        // Keep a small explicit cache for boundary edges (region ↔ outside) for inspection/debugging.
        // This is not used to drive the mapping logic.
        for boundary_edge in get_boundary_edges(ket_state, &step.nodes)? {
            if let Some(edge) = self.reference_state.edge_between(
                &boundary_edge.node_in_region,
                &boundary_edge.neighbor_outside,
            ) {
                if let Some(ref_bond) = self.reference_state.bond_index(edge) {
                    self.boundary_bond_map.insert(
                        (
                            boundary_edge.node_in_region.clone(),
                            boundary_edge.neighbor_outside.clone(),
                        ),
                        ref_bond.clone(),
                    );
                }
            }
        }

        // Create new ref_region by copying ket_region
        let mut ref_region = ket_region.clone();

        // First, update edge bonds in ref_region to reference-side IDs
        // This must be done before replacing tensors to ensure consistency
        let mut edges_to_update: Vec<(V, V, T::Index)> = Vec::new();
        for node in &step.nodes {
            let neighbors: Vec<V> = ref_region.site_index_network().neighbors(node).collect();
            for neighbor in neighbors {
                if let Some(edge) = ref_region.edge_between(node, &neighbor) {
                    if let Some(bond) = ref_region.bond_index(edge) {
                        if let Some(new_bond) = ket_to_ref_bond_map.get(bond.id()) {
                            edges_to_update.push((node.clone(), neighbor, new_bond.clone()));
                        }
                    }
                }
            }
        }
        // Update edges (ref_region is no longer borrowed)
        for (node, neighbor, new_bond) in edges_to_update {
            if let Some(edge) = ref_region.edge_between(&node, &neighbor) {
                ref_region.replace_edge_bond(edge, new_bond)?;
            }
        }

        // Now replace all bond indices (including boundary bonds) in ref_region tensors with reference-side IDs
        for node in &step.nodes {
            if let Some(node_idx) = ref_region.node_index(node) {
                if let Some(tensor) = ref_region.tensor(node_idx) {
                    let mut new_tensor = tensor.clone();
                    let tensor_indices = tensor.external_indices();

                    for ket_idx in &tensor_indices {
                        // Replace if this index is one of the region's bond indices (internal or boundary)
                        if let Some(ref_bond) = ket_to_ref_bond_map.get(ket_idx.id()) {
                            new_tensor = new_tensor.replaceind(ket_idx, ref_bond)?;
                        }
                        // Site indices are kept as-is (same IDs in reference and ket)
                    }

                    ref_region.replace_tensor(node_idx, new_tensor)?;
                }
            }
        }

        // Replace the region back into reference_state
        self.reference_state
            .replace_subtree(&step.nodes, &ref_region)?;

        Ok(())
    }
}

impl<T, V> LocalUpdater<T, V> for SquareLinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn before_step(
        &mut self,
        _step: &LocalUpdateStep<V>,
        full_treetn_before: &TreeTN<T, V>,
    ) -> Result<()> {
        // Initialize reference_state lazily on first call
        self.ensure_reference_state_initialized(full_treetn_before)?;
        Ok(())
    }

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
        let mut factorize_options = FactorizeOptions::svd();
        if let Some(max_rank) = self.options.truncation.max_rank() {
            factorize_options = factorize_options.with_max_rank(max_rank);
        }
        let decomposed =
            factorize_tensor_to_treetn_with(&solved_local, &topology, factorize_options)?;

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

        // Synchronize reference_state with ket_state (full_treetn_after) for the updated region
        // while preserving boundary bond IDs for cache consistency
        self.sync_reference_state_region(step, full_treetn_after)?;

        // Invalidate all caches affected by the updated region
        {
            let mut proj_op = self
                .projected_operator
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))
                .context("after_step: lock poisoned")?;
            proj_op.invalidate(&step.nodes, topology);
        }
        self.projected_state.invalidate(&step.nodes, topology);

        Ok(())
    }
}
