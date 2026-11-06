//! Addition operations for TreeTN using direct-sum (block) construction.
//!
//! This module provides helper functions and types for adding two TreeTNs:
//! - [`MergedBondInfo`]: Information about merged bond indices
//! - [`compute_merged_bond_indices`]: Compute merged bond index information from two networks

use crate::error::TreeTNOperationError;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{
    sort_indices_deterministic, AnyScalar, DirectSumResult, IndexLike, TensorIndex, TensorLike,
};

use super::TreeTN;

/// Information about a merged bond index for direct-sum addition.
/// When adding two TreeTNs, each bond index in the result has dimension
/// `dim_a + dim_b`, where `dim_a` and `dim_b` are the original bond dimensions.
#[derive(Debug, Clone)]
pub struct MergedBondInfo {
    /// Bond dimension from the first TreeTN.
    pub dim_a: usize,
    /// Bond dimension from the second TreeTN.
    pub dim_b: usize,
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn sorted_site_space(site_space: &HashSet<T::Index>) -> Vec<T::Index>
    where
        T::Index: Clone,
        <T::Index as IndexLike>::Id: Ord,
    {
        let mut indices: Vec<_> = site_space.iter().cloned().collect();
        sort_indices_deterministic(&mut indices);
        indices
    }

    /// Reindex this TreeTN's site space to match a template network.
    ///
    /// The topology must match, and each corresponding node must carry the same
    /// number of site indices with the same dimensions. Site indices are paired
    /// node-by-node after sorting by [`sort_indices_deterministic`]
    /// (`dim` → `plev` → `id` → `Debug` tiebreak) for matching.
    ///
    /// # Arguments
    /// * `template` - Reference TreeTN whose site index IDs should be adopted
    ///
    /// # Returns
    /// A new TreeTN with the same tensor data as `self`, but site index IDs
    /// rewritten to match `template`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `self` and `template` have different topologies,
    /// when a node's site-index count differs, when paired site indices have
    /// different dimensions, or when rewriting the indices in the tensors fails.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// # fn make_chain(site0: DynIndex, site1: DynIndex) -> TreeTN<IdxTensor, usize> {
    /// #     let bond = DynIndex::new_dyn(1);
    /// #     let t0 = IdxTensor::from_dense(vec![site0, bond.clone()], vec![1.0, 0.0]).unwrap();
    /// #     let t1 = IdxTensor::from_dense(vec![bond, site1], vec![1.0, 0.0]).unwrap();
    /// #     TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap()
    /// # }
    /// let state_a = make_chain(DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let state_b = make_chain(DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    ///
    /// let aligned = state_b.reindex_site_space_like(&state_a).unwrap();
    /// assert!(aligned.share_equivalent_site_index_network(&state_a));
    /// ```
    pub fn reindex_site_space_like(
        &self,
        template: &Self,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Clone,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if !self.same_topology(template) {
            return Err(anyhow::anyhow!(
                "reindex_site_space_like: networks have incompatible topologies"
            )
            .into());
        }

        let mut old_indices = Vec::new();
        let mut new_indices = Vec::new();

        for node_name in self.node_names() {
            let self_site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("site space not found for node {:?}", node_name))?;
            let template_site_space = template.site_space(&node_name).ok_or_else(|| {
                anyhow::anyhow!("template site space not found for node {:?}", node_name)
            })?;

            if self_site_space.len() != template_site_space.len() {
                return Err(anyhow::anyhow!(
                    "reindex_site_space_like: node {:?} has {} site indices in self but {} in template",
                    node_name,
                    self_site_space.len(),
                    template_site_space.len()
                ).into());
            }

            let self_sorted = Self::sorted_site_space(self_site_space);
            let template_sorted = Self::sorted_site_space(template_site_space);

            for (old_index, new_index) in self_sorted.iter().zip(template_sorted.iter()) {
                if old_index.dim() != new_index.dim() {
                    return Err(anyhow::anyhow!(
                        "reindex_site_space_like: node {:?} site shape mismatch {} != {}",
                        node_name,
                        old_index.dim(),
                        new_index.dim()
                    )
                    .into());
                }
                old_indices.push(old_index.clone());
                new_indices.push(new_index.clone());
            }
        }

        self.replace_indices(&old_indices, &new_indices)
    }

    /// Add two TreeTNs after reindexing the second operand's site space to match `self`.
    ///
    /// This method first calls [`reindex_site_space_like`](Self::reindex_site_space_like)
    /// on `other`, then performs strict direct-sum addition. Use it only when
    /// the node-by-node site-space matching is the intended operation. Use
    /// [`add`](Self::add) when site indices must already match exactly.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to reindex and add
    ///
    /// # Returns
    /// The direct-sum addition result with site IDs matching `self`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when reindexing `other` (see
    /// [`Self::reindex_site_space_like`]) or the following [`Self::add`] fails.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// # fn make_chain(site0: DynIndex, site1: DynIndex) -> TreeTN<IdxTensor, usize> {
    /// #     let bond = DynIndex::new_dyn(1);
    /// #     let t0 = IdxTensor::from_dense(vec![site0, bond.clone()], vec![1.0, 0.0]).unwrap();
    /// #     let t1 = IdxTensor::from_dense(vec![bond, site1], vec![1.0, 0.0]).unwrap();
    /// #     TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap()
    /// # }
    /// let state_a = make_chain(DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let state_b = make_chain(DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    ///
    /// let sum = state_a.add_reindexed_like_self(&state_b).unwrap();
    /// assert_eq!(sum.node_count(), 2);
    /// assert!(sum.share_equivalent_site_index_network(&state_a));
    /// ```
    pub fn add_reindexed_like_self(
        &self,
        other: &Self,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Clone,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let other_aligned = other.reindex_site_space_like(self)?;
        self.add(&other_aligned)
    }

    /// Compute merged bond indices for direct-sum addition.
    ///
    /// For each edge in the network, compute the merged bond dimensions from
    /// both networks. The direct-sum operation owns construction of the actual
    /// merged bond index.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to compute merged bonds with
    ///
    /// # Returns
    /// A HashMap mapping edge keys (node_name_pair in canonical order) to MergedBondInfo.
    ///
    /// # Errors
    /// Returns [`TreeTNOperationError`] when an edge or bond index cannot be resolved in either
    /// network, or when `other` has no node or edge corresponding to one in
    /// `self`. Dimensions are only recorded, not compared.
    #[allow(clippy::type_complexity)]
    pub fn compute_merged_bond_indices(
        &self,
        other: &Self,
    ) -> std::result::Result<HashMap<(V, V), MergedBondInfo>, TreeTNOperationError>
    where
        V: Ord,
    {
        let mut result = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (src, tgt) = self
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge has no endpoints"))?;

            let bond_index_a = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in self"))?;
            let dim_a = bond_index_a.dim();

            let src_name = self
                .graph
                .node_name(src)
                .ok_or_else(|| anyhow::anyhow!("Source node name not found"))?
                .clone();
            let tgt_name = self
                .graph
                .node_name(tgt)
                .ok_or_else(|| anyhow::anyhow!("Target node name not found"))?
                .clone();

            // Find corresponding edge in other
            let src_idx_other = other
                .graph
                .node_index(&src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in other"))?;
            let tgt_idx_other = other
                .graph
                .node_index(&tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in other"))?;

            // Find edge between these nodes in other
            let edge_other = other
                .graph
                .graph()
                .edges_connecting(src_idx_other, tgt_idx_other)
                .next()
                .or_else(|| {
                    other
                        .graph
                        .graph()
                        .edges_connecting(tgt_idx_other, src_idx_other)
                        .next()
                })
                .ok_or_else(|| anyhow::anyhow!("Edge not found in other"))?;

            let bond_index_b = other
                .bond_index(edge_other.id())
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in other"))?;
            let dim_b = bond_index_b.dim();

            // Store in canonical order (smaller name first). The actual
            // merged bond is created by the direct-sum implementation.
            let key = if src_name < tgt_name {
                (src_name, tgt_name)
            } else {
                (tgt_name, src_name)
            };

            result.insert(key, MergedBondInfo { dim_a, dim_b });
        }

        Ok(result)
    }

    /// Add two TreeTNs using direct-sum construction.
    ///
    /// This creates a new TreeTN where each tensor is the direct sum of the
    /// corresponding tensors from self and other, with bond dimensions merged.
    /// The two networks must share the same topology **and** the same site
    /// index IDs. Use [`add_reindexed_like_self`](Self::add_reindexed_like_self)
    /// if site index IDs differ and implicit node-by-node reindexing is intended.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to add
    ///
    /// # Returns
    /// A new TreeTN representing the sum. Each result node tensor keeps
    /// `self`'s site legs in `self`'s leg order, followed by the merged bond
    /// legs ordered by neighbor name; nodes appear in `self.node_names()` order.
    /// The layout is therefore a deterministic function of `self`, and repeated
    /// sums such as `(a + b) + c` keep the same layout.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when the two networks have different topologies or
    /// non-matching site spaces, or when the direct-sum construction of a node
    /// tensor or the resulting [`TreeTN`] fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap();
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    ///
    /// // Adding a single-node network to itself doubles the values
    /// let sum = tn.add(&tn).unwrap();
    /// let dense = sum.to_dense().unwrap();
    /// let expected = IdxTensor::from_dense(vec![s], vec![2.0, 4.0]).unwrap();
    /// assert!(dense.distance(&expected).unwrap() < 1e-12);
    /// ```
    pub fn add(&self, other: &Self) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Eq + Hash,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        Self::direct_sum_networks(&[self, other], |tensors, groups| {
            let pairs: Vec<_> = groups
                .iter()
                .map(|group| (group[0].clone(), group[1].clone()))
                .collect();
            tensors[0].direct_sum(tensors[1], &pairs)
        })
    }

    /// Sum any number of TreeTNs exactly, in one N-ary direct-sum pass.
    ///
    /// Each result node tensor is the N-ary direct sum
    /// ([`TensorContractionLike::direct_sum_many`](tensor4all_core::TensorContractionLike::direct_sum_many))
    /// of the terms' tensors at that node: term `i` occupies its own diagonal
    /// block, so every bond dimension of the result is the sum of the terms'
    /// bond dimensions there. Nothing is truncated.
    ///
    /// The result equals a chain of [`add`](Self::add) calls,
    /// `((t0 + t1) + t2) + ...`, node tensor by node tensor, but every element
    /// is written once instead of the growing partial sum being copied at
    /// every add. The built network stores about `16 * sum_k d_k B_k` bytes in
    /// `Complex64`, with `d_k` the site dimension of node `k` and `B_k` the
    /// product of its summed bond dimensions; [`TreeTN::exact_sum`] can cap
    /// that and fall back to the add chain.
    ///
    /// # Arguments
    /// * `terms` - The networks to sum; at least one. All must share the
    ///   topology **and** the site index IDs of `terms[0]`, and one scalar type.
    ///
    /// # Returns
    /// The exact sum. Each node keeps `terms[0]`'s site legs in their order,
    /// followed by the merged bond legs ordered by neighbor name, as
    /// [`add`](Self::add) does for `self`; the bond blocks appear in `terms`
    /// order. A single term is returned as a clone.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `terms` is empty, when a term has
    /// a different topology or site space than `terms[0]`, or when the
    /// direct-sum construction of a node tensor (for example mixed scalar
    /// types) or the resulting [`TreeTN`] fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// // Three two-site product states over the same site indices.
    /// let (s0, s1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let product = |left: [f64; 2], right: [f64; 2]| {
    ///     let bond = DynIndex::new_dyn(1);
    ///     let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], left.to_vec()).unwrap();
    ///     let t1 = IdxTensor::from_dense(vec![bond, s1.clone()], right.to_vec()).unwrap();
    ///     TreeTN::<IdxTensor, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap()
    /// };
    /// let a = product([1.0, 0.0], [1.0, 2.0]);
    /// let b = product([0.0, 1.0], [3.0, 4.0]);
    /// let c = product([1.0, 1.0], [1.0, 1.0]);
    ///
    /// let sum = TreeTN::add_many(&[&a, &b, &c]).unwrap();
    ///
    /// // Bond dimensions add up: 1 + 1 + 1.
    /// assert_eq!(sum.link_dims(), vec![3]);
    /// // Dense values, column-major over (s0, s1): a + b + c.
    /// let expected = IdxTensor::from_dense(vec![s0, s1], vec![2.0, 4.0, 3.0, 5.0]).unwrap();
    /// assert!(sum.to_dense().unwrap().distance(&expected).unwrap() < 1e-12);
    /// ```
    pub fn add_many(terms: &[&Self]) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Eq + Hash,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if let [only] = terms {
            return Ok((*only).clone());
        }
        Self::direct_sum_networks(terms, |tensors, groups| T::direct_sum_many(tensors, groups))
    }

    /// Direct-sum `terms` node by node with `node_sum`, which receives the
    /// terms' tensors at a node and one bond group per neighbor (sorted by
    /// name), each listing one bond index per term.
    fn direct_sum_networks(
        terms: &[&Self],
        node_sum: impl Fn(&[&T], &[Vec<T::Index>]) -> std::result::Result<DirectSumResult<T>, T::Error>,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Eq + Hash,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let first = Self::ensure_summable(terms)?;

        // Track merged indices for each edge.
        // Key: (smaller_node_name, larger_node_name) for canonical ordering
        // Value: the merged bond index to use for this edge
        let mut edge_merged_indices: HashMap<(V, V), T::Index> = HashMap::new();

        // For each node, compute the direct sum of tensors
        let mut result_tensors: Vec<T> = Vec::new();
        let mut result_node_names: Vec<V> = Vec::new();

        for node_name in first.node_names() {
            let tensors = terms
                .iter()
                .enumerate()
                .map(|(term_position, term)| {
                    term.node_index(&node_name)
                        .and_then(|node| term.tensor(node))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Tensor not found for node {:?} in term {term_position}",
                                node_name
                            )
                        })
                })
                .collect::<Result<Vec<&T>>>()?;

            // Find the bond group of every neighbor. The direct sum lays out
            // the merged bonds in group order, so iterate neighbors sorted by
            // name: the bond layout of the result then depends only on the
            // topology, not on the petgraph edge order left behind by however
            // `terms[0]` was built, and it is the same for `a + b` and for
            // chained sums such as `(a + b) + c`.
            let mut sorted_neighbors: Vec<V> =
                first.site_index_network().neighbors(&node_name).collect();
            sorted_neighbors.sort();
            let groups = sorted_neighbors
                .iter()
                .map(|neighbor| {
                    terms
                        .iter()
                        .enumerate()
                        .map(|(term_position, term)| {
                            term.edge_between(&node_name, neighbor)
                                .and_then(|edge| term.bond_index(edge))
                                .cloned()
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Bond index not found in term {term_position} between {:?} and {:?}",
                                        node_name,
                                        neighbor
                                    )
                                })
                        })
                        .collect::<Result<Vec<T::Index>>>()
                })
                .collect::<Result<Vec<_>>>()?;

            // For nodes with no bonds (single-node network), use element-wise addition
            // instead of direct_sum (which requires at least one index pair).
            if groups.is_empty() {
                let mut sum = tensors[0].clone();
                for tensor in &tensors[1..] {
                    sum = sum
                        .axpby(AnyScalar::new_real(1.0), tensor, AnyScalar::new_real(1.0))
                        .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
                }
                result_tensors.push(sum);
                result_node_names.push(node_name);
                continue;
            }

            // Compute direct sum
            let direct_sum_result = node_sum(&tensors, &groups)
                .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
            let mut result_tensor = direct_sum_result.tensor;

            // For each edge, ensure we use consistent merged indices:
            // - If we've already seen this edge, replace the auto-generated index with the stored one
            // - If this is the first time seeing this edge, store the auto-generated index
            for (neighbor, new_index) in sorted_neighbors
                .into_iter()
                .zip(&direct_sum_result.new_indices)
            {
                // Create canonical edge key (smaller name first)
                let edge_key = if node_name < neighbor {
                    (node_name.clone(), neighbor)
                } else {
                    (neighbor, node_name.clone())
                };

                if let Some(stored_index) = edge_merged_indices.get(&edge_key) {
                    // Edge already processed - replace auto-generated index with stored one
                    result_tensor = result_tensor
                        .replaceind(new_index, stored_index)
                        .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
                } else {
                    // First time seeing this edge - store the auto-generated index
                    edge_merged_indices.insert(edge_key, new_index.clone());
                }
            }

            result_tensors.push(result_tensor);
            result_node_names.push(node_name);
        }

        // Build result TreeTN
        TreeTN::from_tensors(result_tensors, result_node_names)
    }

    /// Compute a strict linear combination `a * self + b * other`.
    ///
    /// This scales two TreeTNs and adds them with [`add`](Self::add), so the two
    /// operands must already have the same topology and the same site index IDs.
    /// No site reindexing or truncation is performed.
    ///
    /// # Arguments
    /// * `a` - Scalar coefficient multiplying `self`.
    /// * `other` - Second TreeTN in the same site-index space.
    /// * `b` - Scalar coefficient multiplying `other`.
    ///
    /// # Returns
    /// A direct-sum TreeTN representing `a * self + b * other`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when scaling either operand (see [`Self::scale`]) or the
    /// subsequent [`Self::add`] fails.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let a = IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0]).unwrap();
    /// let b = IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0]).unwrap();
    /// let tn_a = TreeTN::<_, usize>::from_tensors(vec![a], vec![0]).unwrap();
    /// let tn_b = TreeTN::<_, usize>::from_tensors(vec![b], vec![0]).unwrap();
    ///
    /// let result = tn_a
    ///     .axpby(AnyScalar::new_real(2.0), &tn_b, AnyScalar::new_real(-1.0))
    ///     .unwrap();
    /// let dense = result.contract_to_tensor().unwrap();
    /// let expected = IdxTensor::from_dense(vec![site], vec![-1.0, 0.0]).unwrap();
    /// assert!(dense.isapprox(&expected, 1.0e-12, 0.0).unwrap());
    /// ```
    pub fn axpby<A, B>(
        &self,
        a: A,
        other: &Self,
        b: B,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Eq + Hash,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        A: Into<AnyScalar>,
        B: Into<AnyScalar>,
    {
        let lhs = self.scale(a.into())?;
        let rhs = other.scale(b.into())?;
        lhs.add(&rhs)
    }

    /// Check that `terms` is non-empty and that every term has the topology
    /// and the site spaces of `terms[0]`; return `terms[0]`.
    pub(crate) fn ensure_summable<'a>(
        terms: &[&'a Self],
    ) -> std::result::Result<&'a Self, TreeTNOperationError>
    where
        V: Ord,
        T::Index: Eq + Hash,
    {
        let (first, rest) = terms
            .split_first()
            .ok_or_else(|| anyhow::anyhow!("add_many requires at least one term"))?;
        for term in rest {
            // Verify same topology
            if !first.same_topology(term) {
                return Err(anyhow::anyhow!("Cannot add TreeTNs with different topologies").into());
            }
            first.ensure_same_site_spaces(term)?;
        }
        Ok(first)
    }

    fn ensure_same_site_spaces(&self, other: &Self) -> Result<()>
    where
        V: Ord,
        T::Index: Eq + Hash,
    {
        for node_name in self.node_names() {
            let self_site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("site space not found for node {:?}", node_name))?;
            let other_site_space = other
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("site space not found for node {:?}", node_name))?;
            if self_site_space != other_site_space {
                return Err(anyhow::anyhow!(
                    "Cannot add TreeTNs with different site indices at node {:?}",
                    node_name
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
