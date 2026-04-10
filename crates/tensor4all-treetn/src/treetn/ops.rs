//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - `log_norm` for computing the logarithm of the Frobenius norm
//! - `norm`, `norm_squared` for computing the Frobenius norm
//! - `inner` for computing inner products of two TreeTNs
//! - `to_dense` for contracting to a single tensor
//! - `evaluate` for evaluating at specific index values
//! - `evaluate_at` for evaluating using `Index` objects instead of raw IDs
//! - `all_site_indices` for retrieving all site indices and their owning vertices

use std::collections::HashSet;
use std::hash::Hash;

use tensor4all_core::{AnyScalar, ColMajorArrayRef, IndexLike, TensorLike};

use super::TreeTN;

// ============================================================================
// Default implementation
// ============================================================================

impl<T, V> Default for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clone implementation
// ============================================================================

impl<T, V> Clone for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            canonical_region: self.canonical_region.clone(),
            canonical_form: self.canonical_form,
            site_index_network: self.site_index_network.clone(),
            link_index_network: self.link_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

// ============================================================================
// Debug implementation
// ============================================================================

impl<T, V> std::fmt::Debug for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_region", &self.canonical_region)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use crate::CanonicalizationOptions;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Compute log(||TreeTN||_F), the log of the Frobenius norm.
    ///
    /// Uses canonicalization to avoid numerical overflow:
    /// when canonicalized to a single site with Unitary form,
    /// the Frobenius norm of the whole network equals the norm of the center tensor.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network
    /// to a single Unitary center. Use `log_norm` (without canonicalization) if you
    /// already have a properly canonicalized network.
    ///
    /// # Returns
    /// The natural logarithm of the Frobenius norm.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - Canonicalization fails
    pub fn log_norm(&mut self) -> Result<f64> {
        let n = self.node_count();
        if n == 0 {
            return Err(anyhow::anyhow!("Cannot compute log_norm of empty TreeTN"))
                .context("log_norm: network must have at least one node");
        }

        // Determine the single center site (by name)
        let center_name: V =
            if self.is_canonicalized() && self.canonical_form() == Some(CanonicalForm::Unitary) {
                if self.canonical_region.len() == 1 {
                    // Already Unitary canonicalized to single site - use it
                    self.canonical_region.iter().next().unwrap().clone()
                } else {
                    // Unitary canonicalized to multiple sites - canonicalize to min site
                    let min_center = self.canonical_region.iter().min().unwrap().clone();
                    self.canonicalize_mut(
                        std::iter::once(min_center.clone()),
                        CanonicalizationOptions::default(),
                    )
                    .context("log_norm: failed to canonicalize to single site")?;
                    min_center
                }
            } else {
                // Not canonicalized or not Unitary - canonicalize to min node name
                let min_node_name = self
                    .node_names()
                    .into_iter()
                    .min()
                    .ok_or_else(|| anyhow::anyhow!("No nodes in TreeTN"))
                    .context("log_norm: network must have nodes")?;
                self.canonicalize_mut(
                    std::iter::once(min_node_name.clone()),
                    CanonicalizationOptions::default(),
                )
                .context("log_norm: failed to canonicalize")?;
                min_node_name
            };

        // Get center node index and tensor
        let center_node = self
            .node_index(&center_name)
            .ok_or_else(|| anyhow::anyhow!("Center node not found"))
            .context("log_norm: center node must exist")?;

        let center_tensor = self
            .tensor(center_node)
            .ok_or_else(|| anyhow::anyhow!("Center tensor not found"))
            .context("log_norm: center tensor must exist")?;

        let norm_sq = center_tensor.norm_squared();
        let norm = norm_sq.sqrt();

        Ok(norm.ln())
    }

    /// Compute the Frobenius norm of the TreeTN.
    ///
    /// Uses `log_norm` internally: `norm = exp(log_norm)`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error if the network is empty or canonicalization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// // Single-node TreeTN with tensor [1, 0, 0, 1] (identity 2x2)
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(
    ///     vec![s0.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    ///
    /// let mut tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t],
    ///     vec!["A".to_string()],
    /// ).unwrap();
    ///
    /// // Frobenius norm of [[1,0],[0,1]] = sqrt(2)
    /// let n = tn.norm().unwrap();
    /// assert!((n - 2.0_f64.sqrt()).abs() < 1e-10);
    /// ```
    pub fn norm(&mut self) -> Result<f64> {
        let log_n = self
            .log_norm()
            .context("norm: failed to compute log_norm")?;
        Ok(log_n.exp())
    }

    /// Compute the squared Frobenius norm of the TreeTN.
    ///
    /// Returns `||self||^2 = norm()^2`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error if the network is empty or canonicalization fails.
    pub fn norm_squared(&mut self) -> Result<f64> {
        let n = self
            .norm()
            .context("norm_squared: failed to compute norm")?;
        Ok(n * n)
    }

    /// Compute the inner product of two TreeTNs.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both TreeTNs must have the same site indices (same IDs).
    /// Link indices may differ between the two TreeTNs.
    ///
    /// # Algorithm
    /// 1. Replace link indices in `other` with fresh IDs to avoid collision.
    /// 2. At each node, contract `conj(self_tensor) * other_tensor` pairwise.
    /// 3. Sweep from leaves to root, contracting the environment.
    ///
    /// This is equivalent to contracting the entire network
    /// `conj(self) * other` into a scalar.
    ///
    /// # Errors
    /// Returns an error if the networks have incompatible topologies.
    pub fn inner(&self, other: &Self) -> Result<AnyScalar>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if self.node_count() == 0 && other.node_count() == 0 {
            return Ok(AnyScalar::new_real(0.0));
        }

        // Use contract_naive which handles sim_internal_inds and full contraction.
        // conj(self) * other contracted over site indices gives the inner product.
        // Build conj(self) by conjugating all tensors.
        let mut conj_self = self.clone();
        for node_idx in conj_self.graph.graph().node_indices().collect::<Vec<_>>() {
            if let Some(tensor) = conj_self.graph.graph_mut().node_weight_mut(node_idx) {
                *tensor = tensor.conj();
            }
        }

        // Contract conj(self) with other using naive contraction
        let result_tensor = conj_self
            .contract_naive(other)
            .context("inner: failed to contract conj(self) with other")?;

        // The result should be a scalar (rank 0) since all site indices were contracted
        // (self and other share the same site indices, link indices were sim'd).
        // Extract the scalar value via inner_product with scalar_one.
        let scalar_one = T::scalar_one().context("inner: failed to create scalar_one")?;
        result_tensor
            .inner_product(&scalar_one)
            .context("inner: failed to extract scalar value")
    }

    /// Convert the TreeTN to a single dense tensor.
    ///
    /// This contracts all tensors in the network along their link/bond indices,
    /// producing a single tensor with only site (physical) indices.
    ///
    /// This is an alias for `contract_to_tensor()`.
    ///
    /// # Warning
    /// This operation can be very expensive for large networks,
    /// as the result size grows exponentially with the number of sites.
    ///
    /// # Errors
    /// Returns an error if the network is empty or contraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex, TensorLike};
    ///
    /// // Build a 2-node chain
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// // Identity matrices
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// ).unwrap();
    ///
    /// // Contract to a single dense tensor over site indices s0 and s1
    /// let dense = tn.to_dense().unwrap();
    /// // Result is rank-2 (two site indices s0 and s1)
    /// assert_eq!(dense.num_external_indices(), 2);
    /// ```
    pub fn to_dense(&self) -> Result<T> {
        self.contract_to_tensor()
            .context("to_dense: failed to contract network to tensor")
    }

    /// Returns all site index IDs and their owning vertex names.
    ///
    /// Returns `(index_ids, vertex_names)` where `index_ids[i]` belongs to
    /// vertex `vertex_names[i]`. Order is unspecified but consistent
    /// between the two vectors.
    ///
    /// For [`evaluate()`](Self::evaluate), pass `index_ids` and arrange
    /// values in the same order.
    #[allow(clippy::type_complexity)]
    pub fn all_site_index_ids(&self) -> Result<(Vec<<T::Index as IndexLike>::Id>, Vec<V>)>
    where
        V: Clone,
        <T::Index as IndexLike>::Id: Clone,
    {
        let mut ids = Vec::new();
        let mut vertex_names = Vec::new();
        for node_name in self.node_names() {
            let site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))
                .context("all_site_index_ids: site space must exist")?;
            for index in site_space {
                ids.push(index.id().clone());
                vertex_names.push(node_name.clone());
            }
        }
        Ok((ids, vertex_names))
    }

    /// Evaluate the TreeTN at multiple multi-indices (batch).
    ///
    /// # Arguments
    /// * `index_ids` - Identifies each site index by its ID (from
    ///   [`all_site_index_ids()`](Self::all_site_index_ids)).
    ///   Must enumerate every site index exactly once.
    /// * `values` - Column-major array of shape `[n_indices, n_points]`.
    ///   `values.get(&[i, p])` is the value of `index_ids[i]` at point `p`.
    ///
    /// # Returns
    /// A `Vec<AnyScalar>` of length `n_points`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - `values` shape is inconsistent with `index_ids`
    /// - An index ID is unknown
    /// - Index values are out of bounds
    /// - Contraction fails
    pub fn evaluate(
        &self,
        index_ids: &[<T::Index as IndexLike>::Id],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>>
    where
        <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot evaluate empty TreeTN"))
                .context("evaluate: network must have at least one node");
        }

        let n_indices = index_ids.len();
        anyhow::ensure!(
            values.shape().len() == 2,
            "evaluate: values must be 2D, got {}D",
            values.shape().len()
        );
        anyhow::ensure!(
            values.shape()[0] == n_indices,
            "evaluate: values.shape()[0] ({}) != index_ids.len() ({})",
            values.shape()[0],
            n_indices
        );
        let n_points = values.shape()[1];

        // Build index_id -> position lookup (Vec-based linear scan is fine for
        // the small number of site indices typical in practice).
        let mut known_ids: HashSet<<T::Index as IndexLike>::Id> = HashSet::new();
        let mut total_site_indices: usize = 0;
        for node_name in self.node_names() {
            let site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))
                .context("evaluate: site space must exist")?;
            for index in site_space {
                known_ids.insert(index.id().clone());
                total_site_indices += 1;
            }
        }

        // Validate: index_ids.len() must equal total number of site indices.
        anyhow::ensure!(
            n_indices == total_site_indices,
            "evaluate: index_ids.len() ({}) != total site indices ({})",
            n_indices,
            total_site_indices
        );

        // Validate: no duplicate index IDs.
        {
            let mut seen = HashSet::with_capacity(n_indices);
            for id in index_ids {
                anyhow::ensure!(seen.insert(id), "evaluate: duplicate index ID {:?}", id);
            }
        }

        // Validate: all provided IDs must be known (exist in the network).
        for id in index_ids {
            anyhow::ensure!(
                known_ids.contains(id),
                "evaluate: unknown index ID {:?}",
                id
            );
        }

        // Pre-compute per-node data: (node_name, node_index, tensor_ref,
        //   site_entries: Vec<(Index, position_in_index_ids)>)
        // This avoids HashMap lookups and repeated node_index/tensor lookups
        // inside the per-point loop.
        struct NodeEntry<'a, T: TensorLike, V> {
            name: V,
            tensor: &'a T,
            /// (site_index, position in `index_ids`)
            site_entries: Vec<(T::Index, usize)>,
        }

        let node_names = self.node_names();
        let mut node_entries: Vec<NodeEntry<'_, T, V>> = Vec::with_capacity(node_names.len());

        for node_name in &node_names {
            let node_idx = self
                .node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))
                .context("evaluate: node must exist")?;

            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("evaluate: tensor must exist")?;

            let site_space = self.site_space(node_name);
            let mut site_entries = Vec::new();
            if let Some(space) = site_space {
                for index in space {
                    let id = index.id();
                    let pos = index_ids
                        .iter()
                        .position(|x| x == id)
                        .ok_or_else(|| anyhow::anyhow!("Index ID {:?} not found in index_ids", id))
                        .context("evaluate: all site indices must be covered by index_ids")?;
                    site_entries.push((index.clone(), pos));
                }
            }

            node_entries.push(NodeEntry {
                name: node_name.clone(),
                tensor,
                site_entries,
            });
        }

        let mut results = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let mut contracted_tensors: Vec<T> = Vec::with_capacity(node_entries.len());
            let mut contracted_names: Vec<V> = Vec::with_capacity(node_entries.len());

            for entry in &node_entries {
                if entry.site_entries.is_empty() {
                    // No site indices - just use the tensor as is
                    contracted_tensors.push(entry.tensor.clone());
                    contracted_names.push(entry.name.clone());
                    continue;
                }

                let index_vals: Vec<(T::Index, usize)> = entry
                    .site_entries
                    .iter()
                    .map(|(idx, pos)| {
                        let val = *values.get(&[*pos, point]).unwrap();
                        (idx.clone(), val)
                    })
                    .collect();

                let onehot =
                    T::onehot(&index_vals).context("evaluate: failed to create one-hot tensor")?;

                let result =
                    T::contract(&[entry.tensor, &onehot], tensor4all_core::AllowedPairs::All)
                        .context("evaluate: failed to contract tensor with one-hot")?;

                contracted_tensors.push(result);
                contracted_names.push(entry.name.clone());
            }

            // Build a temporary TreeTN from the contracted tensors and contract to scalar
            let temp_tn = TreeTN::<T, V>::from_tensors(contracted_tensors, contracted_names)
                .context("evaluate: failed to build temporary TreeTN")?;
            let result_tensor = temp_tn
                .contract_to_tensor()
                .context("evaluate: failed to contract to scalar")?;

            let scalar_one = T::scalar_one().context("evaluate: failed to create scalar_one")?;
            let scalar = scalar_one
                .inner_product(&result_tensor)
                .context("evaluate: failed to extract scalar value")?;
            results.push(scalar);
        }

        Ok(results)
    }

    /// Returns all site indices and their owning vertex names.
    ///
    /// Returns `(indices, vertex_names)` where `indices[i]` belongs to
    /// vertex `vertex_names[i]`. Order is unspecified but consistent
    /// between the two vectors.
    ///
    /// This is the `Index`-based counterpart of
    /// [`all_site_index_ids()`](Self::all_site_index_ids), returning
    /// full `Index` objects instead of raw IDs.
    ///
    /// # Errors
    /// Returns an error if a node's site space cannot be found.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let s1 = DynIndex::new_dyn(2);
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    ///
    /// let (indices, vertices) = tn.all_site_indices().unwrap();
    /// assert_eq!(indices.len(), 2);
    /// assert_eq!(vertices.len(), 2);
    ///
    /// // The returned indices contain both s0 and s1
    /// let id_set: std::collections::HashSet<_> = indices.iter().map(|i| *i.id()).collect();
    /// assert!(id_set.contains(s0.id()));
    /// assert!(id_set.contains(s1.id()));
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn all_site_indices(&self) -> Result<(Vec<T::Index>, Vec<V>)>
    where
        V: Clone,
        T::Index: Clone,
    {
        let mut indices = Vec::new();
        let mut node_names = Vec::new();
        for node_name in self.node_names() {
            let site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))
                .context("all_site_indices: site space must exist")?;
            for index in site_space {
                indices.push(index.clone());
                node_names.push(node_name.clone());
            }
        }
        Ok((indices, node_names))
    }

    /// Evaluate the TreeTN at multiple multi-indices (batch), using
    /// `Index` objects instead of raw IDs.
    ///
    /// This is a convenience wrapper around [`evaluate()`](Self::evaluate)
    /// that accepts `&[T::Index]` directly, extracting the IDs
    /// internally.
    ///
    /// # Arguments
    /// * `indices` - Identifies each site index by its `Index` object
    ///   (e.g. from [`all_site_indices()`](Self::all_site_indices)).
    ///   Must enumerate every site index exactly once.
    /// * `values` - Column-major array of shape `[n_indices, n_points]`.
    ///   `values.get(&[i, p])` is the value of `indices[i]` at point `p`.
    ///
    /// # Returns
    /// A `Vec<AnyScalar>` of length `n_points`.
    ///
    /// # Errors
    /// Returns an error if the underlying [`evaluate()`](Self::evaluate)
    /// call fails (see its documentation for details).
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IndexLike, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(3);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone()], vec![10.0, 20.0, 30.0]).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();
    ///
    /// let (indices, _vertices) = tn.all_site_indices().unwrap();
    ///
    /// // Evaluate at index value 2
    /// let data = [2usize];
    /// let shape = [indices.len(), 1];
    /// let values = ColMajorArrayRef::new(&data, &shape);
    /// let result = tn.evaluate_at(&indices, values).unwrap();
    /// assert!((result[0].real() - 30.0).abs() < 1e-10);
    /// ```
    pub fn evaluate_at(
        &self,
        indices: &[T::Index],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>>
    where
        <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let index_ids: Vec<_> = indices.iter().map(|idx| idx.id().clone()).collect();
        self.evaluate(&index_ids, values)
    }
}
