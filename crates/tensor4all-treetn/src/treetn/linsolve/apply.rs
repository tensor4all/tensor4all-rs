//! Apply LinearOperator to TreeTN state.
//!
//! This module provides the `apply_linear_operator` function for computing `A|x⟩`
//! where A is a LinearOperator (MPO with index mappings) and |x⟩ is a TreeTN state.
//!
//! # Algorithm
//!
//! The application works as follows:
//! 1. **Partial Site Handling**: If the operator only covers some nodes of the state,
//!    use `compose_exclusive_linear_operators` to fill gaps with identity operators.
//! 2. **Index Transformation**: Replace state's site indices with operator's input indices.
//! 3. **Contraction**: Contract the transformed state with the operator using
//!    `contract_zipup`, `contract_fit`, or `contract_naive` depending on options.
//! 4. **Output Transformation**: Replace operator's output indices with true output indices.
//!
//! # Example
//!
//! ```ignore
//! let result = apply_linear_operator(&operator, &state, ApplyOptions::default())?;
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{Context, Result};

use tensor4all_core::{IndexLike, TensorIndex, TensorLike};

use super::linear_operator::LinearOperator;
use super::projected_operator::IndexMapping;
use crate::operator::{compose_exclusive_linear_operators, Operator};
use crate::treetn::contraction::{contract, ContractionMethod, ContractionOptions};
use crate::treetn::TreeTN;

/// Options for apply_linear_operator.
#[derive(Debug, Clone)]
pub struct ApplyOptions {
    /// Contraction method to use.
    pub method: ContractionMethod,
    /// Maximum bond dimension for truncation.
    pub max_rank: Option<usize>,
    /// Relative tolerance for truncation.
    pub rtol: Option<f64>,
    /// Number of full sweeps for Fit method.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Convergence tolerance for Fit method.
    pub convergence_tol: Option<f64>,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            method: ContractionMethod::Zipup,
            max_rank: None,
            rtol: None,
            nfullsweeps: 1,
            convergence_tol: None,
        }
    }
}

impl ApplyOptions {
    /// Create options with ZipUp method (default).
    pub fn zipup() -> Self {
        Self::default()
    }

    /// Create options with Fit method.
    pub fn fit() -> Self {
        Self {
            method: ContractionMethod::Fit,
            ..Default::default()
        }
    }

    /// Create options with Naive method.
    pub fn naive() -> Self {
        Self {
            method: ContractionMethod::Naive,
            ..Default::default()
        }
    }

    /// Set maximum bond dimension.
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set number of full sweeps for Fit method.
    pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self {
        self.nfullsweeps = nfullsweeps;
        self
    }
}

/// Apply a LinearOperator to a TreeTN state: compute `A|x⟩`.
///
/// This function handles:
/// - Partial operators (fills gaps with identity via compose_exclusive_linear_operators)
/// - Index transformations (input/output mappings)
/// - Multiple contraction algorithms (ZipUp, Fit, Naive)
///
/// # Arguments
///
/// * `operator` - The LinearOperator to apply
/// * `state` - The input state |x⟩
/// * `options` - Options controlling the contraction algorithm
///
/// # Returns
///
/// The result `A|x⟩` as a TreeTN, or an error if application fails.
///
/// # Example
///
/// ```ignore
/// use tensor4all_treetn::linsolve::{apply_linear_operator, ApplyOptions};
///
/// // Apply with default options (ZipUp)
/// let result = apply_linear_operator(&operator, &state, ApplyOptions::default())?;
///
/// // Apply with truncation
/// let result = apply_linear_operator(
///     &operator,
///     &state,
///     ApplyOptions::zipup().with_max_rank(50).with_rtol(1e-10),
/// )?;
/// ```
pub fn apply_linear_operator<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
    options: ApplyOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // 1. Check if operator covers all state nodes
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();
    let op_nodes: HashSet<V> = operator.node_names();

    let full_operator = if op_nodes == state_nodes {
        // Operator covers all nodes - use directly
        operator.clone()
    } else if op_nodes.is_subset(&state_nodes) {
        // Partial operator - need to compose with identity on gaps
        extend_operator_to_full_space(operator, state)?
    } else {
        return Err(anyhow::anyhow!(
            "Operator nodes {:?} are not a subset of state nodes {:?}",
            op_nodes,
            state_nodes
        ));
    };

    // 2. Transform state's site indices to operator's input indices
    let transformed_state = transform_state_to_input(&full_operator, state)?;

    // 3. Contract state with operator MPO
    // Choose a center node (use first node in sorted order for determinism)
    let mut node_names: Vec<_> = state.node_names();
    node_names.sort();
    let center = node_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("Empty state"))?;

    let contraction_options = ContractionOptions {
        method: options.method,
        max_rank: options.max_rank,
        rtol: options.rtol,
        nfullsweeps: options.nfullsweeps,
        convergence_tol: options.convergence_tol,
        ..Default::default()
    };

    let contracted = contract(
        &transformed_state,
        full_operator.mpo(),
        center,
        contraction_options,
    )
    .context("Failed to contract state with operator")?;

    // 4. Transform operator's output indices to true output indices
    let result = transform_output_to_true(&full_operator, contracted)?;

    Ok(result)
}

/// Extend a partial operator to cover the full state space.
///
/// Uses `compose_exclusive_linear_operators` to fill gap nodes with identity operators.
fn extend_operator_to_full_space<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let state_network = state.site_index_network();
    let op_nodes: HashSet<V> = operator.node_names();
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();
    let gap_nodes: Vec<V> = state_nodes.difference(&op_nodes).cloned().collect();

    // Build gap site indices: for each gap node, create identity mapping
    // (input index = output index for identity)
    #[allow(clippy::type_complexity)]
    let mut gap_site_indices: HashMap<V, Vec<(T::Index, T::Index)>> = HashMap::new();

    for gap_name in &gap_nodes {
        let site_space = state
            .site_space(gap_name)
            .ok_or_else(|| anyhow::anyhow!("Gap node {:?} has no site space", gap_name))?;

        // For identity, we need (input, output) pairs with same dimension
        // Create sim indices for internal use
        let pairs: Vec<(T::Index, T::Index)> = site_space
            .iter()
            .map(|idx| {
                let input_internal = idx.sim();
                let output_internal = idx.sim();
                (input_internal, output_internal)
            })
            .collect();

        gap_site_indices.insert(gap_name.clone(), pairs);
    }

    compose_exclusive_linear_operators(state_network, &[operator], &gap_site_indices)
        .context("Failed to compose operator with identity gaps")
}

/// Transform state's site indices to operator's input indices.
fn transform_state_to_input<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut result = state.clone();

    for (node, mapping) in operator.input_mappings() {
        // Replace true_index with internal_index in the state
        result = result
            .replaceind(&mapping.true_index, &mapping.internal_index)
            .with_context(|| format!("Failed to transform input index at node {:?}", node))?;
    }

    Ok(result)
}

/// Transform operator's output indices to true output indices.
fn transform_output_to_true<T, V>(
    operator: &LinearOperator<T, V>,
    mut result: TreeTN<T, V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    for (node, mapping) in operator.output_mappings() {
        // Replace internal_index with true_index in the result
        result = result
            .replaceind(&mapping.internal_index, &mapping.true_index)
            .with_context(|| format!("Failed to transform output index at node {:?}", node))?;
    }

    Ok(result)
}

// ============================================================================
// TensorIndex implementation for LinearOperator
// ============================================================================

impl<T, V> TensorIndex for LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    type Index = T::Index;

    /// Return all external indices (true input and output indices).
    fn external_indices(&self) -> Vec<Self::Index> {
        let mut result: Vec<Self::Index> = self
            .input_mapping
            .values()
            .map(|m| m.true_index.clone())
            .collect();
        result.extend(self.output_mapping.values().map(|m| m.true_index.clone()));
        result
    }

    fn num_external_indices(&self) -> usize {
        self.input_mapping.len() + self.output_mapping.len()
    }

    /// Replace an external index (true index) in this operator.
    ///
    /// This updates the mapping but does NOT modify the internal MPO tensors.
    fn replaceind(&self, old_index: &Self::Index, new_index: &Self::Index) -> Result<Self> {
        // Validate dimension match
        if old_index.dim() != new_index.dim() {
            return Err(anyhow::anyhow!(
                "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                old_index.dim(),
                new_index.dim()
            ));
        }

        let mut result = self.clone();

        // Check input mappings
        for (node, mapping) in &self.input_mapping {
            if mapping.true_index.same_id(old_index) {
                result.input_mapping.insert(
                    node.clone(),
                    IndexMapping {
                        true_index: new_index.clone(),
                        internal_index: mapping.internal_index.clone(),
                    },
                );
                return Ok(result);
            }
        }

        // Check output mappings
        for (node, mapping) in &self.output_mapping {
            if mapping.true_index.same_id(old_index) {
                result.output_mapping.insert(
                    node.clone(),
                    IndexMapping {
                        true_index: new_index.clone(),
                        internal_index: mapping.internal_index.clone(),
                    },
                );
                return Ok(result);
            }
        }

        Err(anyhow::anyhow!(
            "Index {:?} not found in LinearOperator mappings",
            old_index.id()
        ))
    }

    /// Replace multiple external indices.
    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> Result<Self> {
        if old_indices.len() != new_indices.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} old indices, {} new indices",
                old_indices.len(),
                new_indices.len()
            ));
        }

        let mut result = self.clone();
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            result = result.replaceind(old, new)?;
        }
        Ok(result)
    }
}

// ============================================================================
// Arc-based CoW wrapper for LinearOperator
// ============================================================================

/// LinearOperator with Arc-based Copy-on-Write semantics.
///
/// This wrapper uses `Arc` for the internal MPO to enable cheap cloning
/// and efficient sharing. When mutation is needed, `make_mut` performs
/// a clone only if there are other references.
#[derive(Debug, Clone)]
pub struct ArcLinearOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The MPO with internal index IDs (wrapped in Arc for CoW)
    pub mpo: Arc<TreeTN<T, V>>,
    /// Input index mapping: node -> (true s_in, internal s_in_tmp)
    pub input_mapping: HashMap<V, IndexMapping<T::Index>>,
    /// Output index mapping: node -> (true s_out, internal s_out_tmp)
    pub output_mapping: HashMap<V, IndexMapping<T::Index>>,
}

impl<T, V> ArcLinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create from an existing LinearOperator.
    pub fn from_linear_operator(op: LinearOperator<T, V>) -> Self {
        Self {
            mpo: Arc::new(op.mpo),
            input_mapping: op.input_mapping,
            output_mapping: op.output_mapping,
        }
    }

    /// Create a new ArcLinearOperator.
    pub fn new(
        mpo: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
    ) -> Self {
        Self {
            mpo: Arc::new(mpo),
            input_mapping,
            output_mapping,
        }
    }

    /// Get a mutable reference to the MPO, cloning if necessary.
    ///
    /// This implements Copy-on-Write semantics: if this is the only reference,
    /// no copy is made. If there are other references, the MPO is cloned first.
    pub fn mpo_mut(&mut self) -> &mut TreeTN<T, V> {
        Arc::make_mut(&mut self.mpo)
    }

    /// Get an immutable reference to the MPO.
    pub fn mpo(&self) -> &TreeTN<T, V> {
        &self.mpo
    }

    /// Convert back to a LinearOperator (unwraps Arc if possible).
    pub fn into_linear_operator(self) -> LinearOperator<T, V> {
        LinearOperator {
            mpo: Arc::try_unwrap(self.mpo).unwrap_or_else(|arc| (*arc).clone()),
            input_mapping: self.input_mapping,
            output_mapping: self.output_mapping,
        }
    }

    /// Get input mapping for a node.
    pub fn get_input_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.input_mapping.get(node)
    }

    /// Get output mapping for a node.
    pub fn get_output_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.output_mapping.get(node)
    }

    /// Get all input mappings.
    pub fn input_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.input_mapping
    }

    /// Get all output mappings.
    pub fn output_mappings(&self) -> &HashMap<V, IndexMapping<T::Index>> {
        &self.output_mapping
    }

    /// Get node names covered by this operator.
    pub fn node_names(&self) -> HashSet<V> {
        self.mpo
            .site_index_network()
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{random_treetn_f64, LinkSpace};
    use crate::SiteIndexNetwork;
    use std::collections::HashSet;
    use tensor4all_core::index::{DynId, Index, TagSet};

    type DynIndex = Index<DynId, TagSet>;

    fn make_index(dim: usize) -> DynIndex {
        Index::new_dyn(dim)
    }

    fn create_chain_site_network(n: usize) -> SiteIndexNetwork<String, DynIndex> {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        for i in 0..n {
            let name = format!("N{}", i);
            let site_idx = make_index(2);
            net.add_node(name, [site_idx].into_iter().collect::<HashSet<_>>())
                .unwrap();
        }
        for i in 0..(n - 1) {
            net.add_edge(&format!("N{}", i), &format!("N{}", i + 1))
                .unwrap();
        }
        net
    }

    #[test]
    fn test_apply_options_builder() {
        let opts = ApplyOptions::zipup().with_max_rank(50).with_rtol(1e-10);
        assert_eq!(opts.method, ContractionMethod::Zipup);
        assert_eq!(opts.max_rank, Some(50));
        assert_eq!(opts.rtol, Some(1e-10));
    }

    #[test]
    fn test_linear_operator_tensor_index() {
        // Create a simple LinearOperator
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        net.add_node(
            "N0".to_string(),
            [s0_in.clone(), s0_out.clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo = random_treetn_f64(&mut rng, &net, link_space);

        let true_s0 = make_index(2);
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        input_mapping.insert(
            "N0".to_string(),
            IndexMapping {
                true_index: true_s0.clone(),
                internal_index: s0_in.clone(),
            },
        );
        output_mapping.insert(
            "N0".to_string(),
            IndexMapping {
                true_index: true_s0.clone(),
                internal_index: s0_out.clone(),
            },
        );

        let lin_op = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Test external_indices
        let ext_indices = lin_op.external_indices();
        assert_eq!(ext_indices.len(), 2);

        // Test replaceind
        let new_idx = make_index(2);
        let replaced = lin_op.replaceind(&true_s0, &new_idx).unwrap();
        assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());
    }

    #[test]
    fn test_arc_linear_operator_cow() {
        let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0 = make_index(2);
        net.add_node(
            "N0".to_string(),
            [s0.clone()].into_iter().collect::<HashSet<_>>(),
        )
        .unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo = random_treetn_f64(&mut rng, &net, link_space);

        let arc_op = ArcLinearOperator::new(mpo, HashMap::new(), HashMap::new());

        // Clone should share the Arc
        let arc_op2 = arc_op.clone();
        assert!(Arc::ptr_eq(&arc_op.mpo, &arc_op2.mpo));

        // Mutating one should not affect the other (CoW)
        let mut arc_op3 = arc_op.clone();
        let _mpo_mut = arc_op3.mpo_mut();
        // After make_mut, the Arcs should be different if there were other refs
        // (In this case, arc_op still holds a reference)
        assert!(!Arc::ptr_eq(&arc_op.mpo, &arc_op3.mpo));
    }
}
