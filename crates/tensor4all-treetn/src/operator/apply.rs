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

use super::index_mapping::IndexMapping;
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::operator::compose_exclusive_linear_operators;
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
/// use tensor4all_treetn::operator::{apply_linear_operator, ApplyOptions};
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
/// For gap nodes, creates proper index mappings where:
/// - True indices = state's actual site indices
/// - Internal indices = new simulated indices for the MPO tensor
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

    // Build gap site indices: for each gap node, create internal indices for the identity tensor.
    // The (input_internal, output_internal) pairs are used to build the delta tensor.
    #[allow(clippy::type_complexity)]
    let mut gap_site_indices: HashMap<V, Vec<(T::Index, T::Index)>> = HashMap::new();

    // Also track true<->internal mappings for gap nodes
    #[allow(clippy::type_complexity)]
    let mut gap_input_mappings: HashMap<V, IndexMapping<T::Index>> = HashMap::new();
    #[allow(clippy::type_complexity)]
    let mut gap_output_mappings: HashMap<V, IndexMapping<T::Index>> = HashMap::new();

    for gap_name in &gap_nodes {
        let site_space = state
            .site_space(gap_name)
            .ok_or_else(|| anyhow::anyhow!("Gap node {:?} has no site space", gap_name))?;

        // For identity at gap nodes:
        // - True indices = state's site indices (what apply_linear_operator maps from/to)
        // - Internal indices = new simulated indices for the MPO tensor
        let mut pairs: Vec<(T::Index, T::Index)> = Vec::new();

        for (i, true_idx) in site_space.iter().enumerate() {
            let input_internal = true_idx.sim();
            let output_internal = true_idx.sim();
            pairs.push((input_internal.clone(), output_internal.clone()));

            // Store mapping for the first site index of each gap node
            if i == 0 {
                gap_input_mappings.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_idx.clone(),
                        internal_index: input_internal,
                    },
                );
                gap_output_mappings.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_idx.clone(),
                        internal_index: output_internal,
                    },
                );
            }
        }

        gap_site_indices.insert(gap_name.clone(), pairs);
    }

    // Compose the operator with identity at gaps
    let mut composed =
        compose_exclusive_linear_operators(state_network, &[operator], &gap_site_indices)
            .context("Failed to compose operator with identity gaps")?;

    // Override the mappings for gap nodes to use the correct true indices
    // (compose_exclusive_linear_operators uses the internal indices as true indices for gaps)
    for (gap_name, mapping) in gap_input_mappings {
        composed.input_mapping.insert(gap_name, mapping);
    }
    for (gap_name, mapping) in gap_output_mappings {
        composed.output_mapping.insert(gap_name, mapping);
    }

    Ok(composed)
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
    use tensor4all_core::TensorDynLen;

    type DynIndex = Index<DynId, TagSet>;

    fn make_index(dim: usize) -> DynIndex {
        Index::new_dyn(dim)
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

    #[test]
    fn test_apply_linear_operator_full_coverage() {
        use crate::operator::apply_linear_operator;
        use crate::operator::ApplyOptions;

        // Create a 2-site state
        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = make_index(2);
        let s1 = make_index(2);
        let b01 = make_index(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), b01.clone()], vec![1.0; 4]);
        let t1 = TensorDynLen::from_dense_f64(vec![b01.clone(), s1.clone()], vec![1.0; 4]);

        let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
        let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
        state.connect(n0, &b01, n1, &b01).unwrap();

        // Create identity operator covering both sites
        let mut mpo = TreeTN::<TensorDynLen, String>::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        let s1_in = make_index(2);
        let s1_out = make_index(2);
        let b_mpo = make_index(1);

        let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
        let t0_mpo = TensorDynLen::from_dense_f64(
            vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
            id_data.clone(),
        );
        let t1_mpo = TensorDynLen::from_dense_f64(
            vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()],
            id_data,
        );

        let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
        let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
        mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        input_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_in.clone(),
            },
        );
        input_mapping.insert(
            "site1".to_string(),
            IndexMapping {
                true_index: s1.clone(),
                internal_index: s1_in.clone(),
            },
        );
        output_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_out.clone(),
            },
        );
        output_mapping.insert(
            "site1".to_string(),
            IndexMapping {
                true_index: s1.clone(),
                internal_index: s1_out.clone(),
            },
        );

        let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Test apply with default options
        let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
        assert_eq!(result.node_count(), 2);

        // Test apply with different methods
        let result_fit = apply_linear_operator(&operator, &state, ApplyOptions::fit()).unwrap();
        assert_eq!(result_fit.node_count(), 2);

        let result_naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
        assert_eq!(result_naive.node_count(), 2);
    }

    #[test]
    fn test_apply_linear_operator_partial() {
        use crate::operator::apply_linear_operator;
        use crate::operator::ApplyOptions;

        // Create a 3-site state
        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = make_index(2);
        let s1 = make_index(2);
        let s2 = make_index(2);
        let b01 = make_index(2);
        let b12 = make_index(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), b01.clone()], vec![1.0; 4]);
        let t1 =
            TensorDynLen::from_dense_f64(vec![b01.clone(), s1.clone(), b12.clone()], vec![1.0; 8]);
        let t2 = TensorDynLen::from_dense_f64(vec![b12.clone(), s2.clone()], vec![1.0; 4]);

        let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
        let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
        let n2 = state.add_tensor("site2".to_string(), t2).unwrap();
        state.connect(n0, &b01, n1, &b01).unwrap();
        state.connect(n1, &b12, n2, &b12).unwrap();

        // Create operator covering only site0 and site1 (partial)
        let mut mpo = TreeTN::<TensorDynLen, String>::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        let s1_in = make_index(2);
        let s1_out = make_index(2);
        let b_mpo = make_index(1);

        let id_data = vec![1.0, 0.0, 0.0, 1.0];
        let t0_mpo = TensorDynLen::from_dense_f64(
            vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
            id_data.clone(),
        );
        let t1_mpo = TensorDynLen::from_dense_f64(
            vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()],
            id_data,
        );

        let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
        let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
        mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        input_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_in.clone(),
            },
        );
        input_mapping.insert(
            "site1".to_string(),
            IndexMapping {
                true_index: s1.clone(),
                internal_index: s1_in.clone(),
            },
        );
        output_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_out.clone(),
            },
        );
        output_mapping.insert(
            "site1".to_string(),
            IndexMapping {
                true_index: s1.clone(),
                internal_index: s1_out.clone(),
            },
        );

        let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Test apply with partial operator (should extend with identity on site2)
        let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
        assert_eq!(result.node_count(), 3);
    }

    #[test]
    fn test_apply_linear_operator_error_cases() {
        use crate::operator::apply_linear_operator;
        use crate::operator::ApplyOptions;

        // Create a 2-site state
        let mut state = TreeTN::<TensorDynLen, String>::new();
        let s0 = make_index(2);
        let s1 = make_index(2);
        let b01 = make_index(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), b01.clone()], vec![1.0; 4]);
        let t1 = TensorDynLen::from_dense_f64(vec![b01.clone(), s1.clone()], vec![1.0; 4]);

        let n0 = state.add_tensor("site0".to_string(), t0).unwrap();
        let n1 = state.add_tensor("site1".to_string(), t1).unwrap();
        state.connect(n0, &b01, n1, &b01).unwrap();

        // Create operator with extra node not in state (should error)
        let mut mpo = TreeTN::<TensorDynLen, String>::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        let s2_in = make_index(2); // site2 doesn't exist in state
        let s2_out = make_index(2);
        let b_mpo = make_index(1);

        let id_data = vec![1.0, 0.0, 0.0, 1.0];
        let t0_mpo = TensorDynLen::from_dense_f64(
            vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
            id_data.clone(),
        );
        let t2_mpo = TensorDynLen::from_dense_f64(
            vec![b_mpo.clone(), s2_out.clone(), s2_in.clone()],
            id_data,
        );

        let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
        let n2_mpo = mpo.add_tensor("site2".to_string(), t2_mpo).unwrap();
        mpo.connect(n0_mpo, &b_mpo, n2_mpo, &b_mpo).unwrap();

        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        input_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_in.clone(),
            },
        );
        input_mapping.insert(
            "site2".to_string(),
            IndexMapping {
                true_index: s1.clone(), // Using s1 as true index for site2
                internal_index: s2_in.clone(),
            },
        );
        output_mapping.insert(
            "site0".to_string(),
            IndexMapping {
                true_index: s0.clone(),
                internal_index: s0_out.clone(),
            },
        );
        output_mapping.insert(
            "site2".to_string(),
            IndexMapping {
                true_index: s1.clone(),
                internal_index: s2_out.clone(),
            },
        );

        let operator = LinearOperator::new(mpo, input_mapping, output_mapping);

        // Should error because operator has node "site2" not in state
        let result = apply_linear_operator(&operator, &state, ApplyOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_operator_replaceinds() {
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

        // Test replaceinds
        let new_idx1 = make_index(2);
        let new_idx2 = make_index(2);
        let replaced = lin_op
            .replaceinds(
                std::slice::from_ref(&true_s0),
                std::slice::from_ref(&new_idx1),
            )
            .unwrap();
        assert!(replaced.get_input_mapping(&"N0".to_string()).is_some());

        // Test replaceinds with multiple indices
        let new_idx3 = make_index(2);
        let replaced2 = lin_op
            .replaceinds(&[true_s0.clone(), true_s0.clone()], &[new_idx2, new_idx3])
            .unwrap();
        assert!(replaced2.get_input_mapping(&"N0".to_string()).is_some());

        // Test replaceinds error case (length mismatch)
        let result = lin_op.replaceinds(
            std::slice::from_ref(&true_s0),
            &[new_idx1.clone(), new_idx1],
        );
        assert!(result.is_err());
    }
}
