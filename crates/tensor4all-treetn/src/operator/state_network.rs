//! A [`LinearOperator`] written on a state's whole tree.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::Context;
use tensor4all_core::{IndexLike, TensorLike};

use super::apply::extend_operator_on_network;
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::error::TreeTNOperationError;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

/// A [`LinearOperator`] as a plain [`TreeTN`] on a state's whole tree, built
/// by [`LinearOperator::to_state_network`].
///
/// The network has the state's nodes and edges. Its site legs are the state's
/// site indices, where it contracts with the state, and one output leg for each
/// declared output index of the operator. Nodes the operator does not act on
/// carry identities. Every tensor has the operators' common scalar type, so the
/// network can be an operand of operations that require one dtype per network,
/// such as `PartitionedTreeTN` contraction.
///
/// [`apply_linear_operator`](crate::apply_linear_operator) is the direct way to
/// apply an operator to a `TreeTN`; this type is for callers that need the
/// operator itself as a network.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike};
/// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A two-site state; the operator swaps the two values of site "a" only.
/// let (a, b, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(1));
/// let state = TreeTN::<IdxTensor, &str>::from_tensors(
///     vec![
///         IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0, 2.0])?,
///         IdxTensor::from_dense(vec![bond, b.clone()], vec![3.0, 4.0])?,
///     ],
///     vec!["a", "b"],
/// )?;
/// let (input, output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// // Column-major [output, input]: the swap matrix.
/// let swap = IdxTensor::from_dense(vec![output.clone(), input.clone()], vec![0.0, 1.0, 1.0, 0.0])?;
/// let operator = LinearOperator::new(
///     TreeTN::<IdxTensor, &str>::from_tensors(vec![swap], vec!["a"])?,
///     HashMap::from([("a", IndexMapping { true_index: a.clone(), internal_index: input })]),
///     HashMap::from([("a", IndexMapping { true_index: a.clone(), internal_index: output })]),
/// );
///
/// let converted = operator.to_state_network(state.site_index_network())?;
/// assert_eq!(converted.network().node_count(), 2);
/// // Both declared outputs are state indices, so both legs are fresh copies.
/// let legs: Vec<DynIndex> = converted.outputs().iter().map(|(_, leg)| leg.clone()).collect();
/// assert_eq!(converted.outputs().len(), 2);
/// assert!(converted.outputs().iter().all(|(declared, leg)| declared != leg));
/// // Contracting with the state over a and b gives the swapped values on the
/// // legs: psi(a, b) = [1, 2] x [3, 4] becomes [2, 1] x [3, 4].
/// let result = IdxTensor::contract(&[&state.to_dense()?, &converted.network().to_dense()?])?;
/// let expected = IdxTensor::from_dense(legs, vec![6.0, 3.0, 8.0, 4.0])?;
/// assert!(result.sub(&expected)?.maxabs()? < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StateOperatorNetwork<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    network: TreeTN<T, V>,
    outputs: Vec<(T::Index, T::Index)>,
}

impl<T, V> StateOperatorNetwork<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The operator network on the state's tree.
    pub fn network(&self) -> &TreeTN<T, V> {
        &self.network
    }

    /// `(declared output index, output leg)` for every output leg of
    /// [`Self::network`], node by node in sorted node order.
    ///
    /// The declared index is the operator's output index, or the state's site
    /// index at a node the operator does not act on. The leg equals it unless
    /// that index is also a site index of the state (the usual `A x` convention,
    /// where the output space is the input space): then the leg is a fresh
    /// copy, because one network cannot carry the same index twice. Replace each
    /// leg by its declared index after contracting with the state to restore
    /// the declared output space.
    pub fn outputs(&self) -> &[(T::Index, T::Index)] {
        &self.outputs
    }

    /// The network and the output pairs, by value.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (TreeTN<T, V>, Vec<(T::Index, T::Index)>) {
        (self.network, self.outputs)
    }
}

impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Write this operator on the whole tree of a state, with true indices.
    ///
    /// The operator may act on any subset of the state's nodes: a connected
    /// subtree, several disjoint subtrees, or all of them. A bond between two
    /// nodes that are not adjacent in the state is routed along the state's
    /// path between them, and every other node carries an identity, so the
    /// result has exactly the state's nodes and edges. An operator on every
    /// node must already have the state's edges. Identities and real
    /// operator tensors are complex when any operator tensor is complex, so the
    /// network has one scalar type. Each input mapping's true index must be a
    /// site index of the state at the same node, and every site index of a node
    /// the operator acts on must be one of its inputs.
    ///
    /// # Arguments
    ///
    /// * `state` - The site index network of the state the operator applies to,
    ///   for example `tree.site_index_network()`.
    ///
    /// # Returns
    ///
    /// The network and its output legs; see [`StateOperatorNetwork`]. The
    /// network contracts with the state over the state's site indices.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError::InvalidGraph`] when an operator node is
    /// not a state node, when an input's true index is not a site index of the
    /// state at that node, or when a node the operator acts on has a site index
    /// that no input maps; [`TreeTNOperationError::InvalidConfiguration`] when
    /// the operator's bonds cannot be routed on the state's tree (two bonds
    /// would share a state edge, or an operator on every node has other edges
    /// than the state); and the
    /// tensor construction errors of the identities and index replacements.
    ///
    /// # Examples
    ///
    /// See [`StateOperatorNetwork`].
    pub fn to_state_network(
        &self,
        state: &SiteIndexNetwork<V, T::Index>,
    ) -> Result<StateOperatorNetwork<T, V>, TreeTNOperationError> {
        let invalid_graph = |message: String| TreeTNOperationError::InvalidGraph {
            source: anyhow::anyhow!(message),
        };
        let op_nodes = self.node_names();
        for node in &op_nodes {
            let site_space = state.site_space(node).ok_or_else(|| {
                invalid_graph(format!("operator node {node:?} is not a node of the state"))
            })?;
            let inputs: HashSet<&T::Index> = self
                .get_input_mappings(node)
                .unwrap_or_default()
                .iter()
                .map(|mapping| &mapping.true_index)
                .collect();
            if let Some(index) = inputs.iter().find(|index| !site_space.contains(**index)) {
                return Err(invalid_graph(format!(
                    "operator input {index:?} at node {node:?} is not a site index of the state there"
                )));
            }
            if let Some(index) = site_space.iter().find(|index| !inputs.contains(index)) {
                return Err(invalid_graph(format!(
                    "state site index {index:?} at operator node {node:?} is not an operator input"
                )));
            }
        }

        let state_nodes: HashSet<V> = state.node_names().into_iter().cloned().collect();
        let site_order = |name: &V| {
            state.site_space(name).map(|space| {
                let mut indices: Vec<T::Index> = space.iter().cloned().collect();
                indices.sort_by(|a, b| a.id().cmp(b.id()));
                indices
            })
        };
        let routing = |message: &str| TreeTNOperationError::InvalidConfiguration {
            source: anyhow::anyhow!("{message}"),
        };
        let same_topology = |operator: &LinearOperator<T, V>| {
            operator
                .mpo
                .site_index_network()
                .topology()
                .same_topology(state.topology())
        };
        let extended = if op_nodes == state_nodes {
            // On the same node set, a different tree would need two operator
            // bonds on one state edge, which routing does not support.
            if !same_topology(self) {
                return Err(routing(
                    "an operator on every state node must have the state's edges",
                ));
            }
            self.clone()
        } else {
            extend_operator_on_network(self, state, &site_order)
                .map_err(|source| TreeTNOperationError::InvalidConfiguration { source })?
                .0
        };
        if !same_topology(&extended) {
            return Err(routing(
                "the operator's bonds cannot be routed on the state's tree without two bonds \
                 sharing a state edge",
            ));
        }

        // Every site index of the state, to tell which declared outputs need a
        // fresh leg.
        let state_sites: HashSet<T::Index> = state
            .node_names()
            .into_iter()
            .filter_map(|name| state.site_space(name))
            .flat_map(|space| space.iter().cloned())
            .collect();
        let mut names: Vec<V> = extended.mpo.node_names();
        names.sort();
        let mut tensors = Vec::with_capacity(names.len());
        let mut outputs = Vec::new();
        for name in &names {
            let node = extended.mpo.node_index(name).ok_or_else(|| {
                invalid_graph(format!(
                    "node {name:?} is missing from the extended operator"
                ))
            })?;
            let mut tensor = extended
                .mpo
                .tensor(node)
                .ok_or_else(|| invalid_graph(format!("node {name:?} has no tensor")))?
                .clone();
            // A gap node's identity maps the state's site index to itself.
            let gap = !op_nodes.contains(name);
            for mapping in extended.get_input_mappings(name).unwrap_or_default() {
                tensor = tensor
                    .replaceind(&mapping.internal_index, &mapping.true_index)
                    .with_context(|| format!("failed to bind an input at node {name:?}"))?;
            }
            for mapping in extended.get_output_mappings(name).unwrap_or_default() {
                let declared = mapping.true_index.clone();
                let leg = if gap || state_sites.contains(&declared) {
                    declared.sim()
                } else {
                    declared.clone()
                };
                tensor = tensor
                    .replaceind(&mapping.internal_index, &leg)
                    .with_context(|| format!("failed to bind an output at node {name:?}"))?;
                outputs.push((declared, leg));
            }
            tensors.push(tensor);
        }
        let network = TreeTN::from_tensors(tensors, names)?;
        Ok(StateOperatorNetwork { network, outputs })
    }
}

#[cfg(test)]
mod tests;
