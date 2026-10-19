use crate::error::TreeTNOperationError;
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use tensor4all_core::{
    contract_with_options_on_session, select_indices_on_session, AnyScalar, ColMajorArrayRef,
    ContractionOptions, DynIndex, IdxTensor, IndexLike, TensorLike,
};
use tensor4all_tensorbackend::explicit::Session;

use super::TreeTN;

#[derive(Debug, Clone)]
struct EvaluatorSiteEntry<I> {
    index: I,
    input_position: usize,
    local_axis: usize,
}

#[derive(Debug, Clone)]
struct EvaluatorNodeEntry<T, V>
where
    T: TensorLike,
{
    name: V,
    tensor: T,
    site_entries: Vec<EvaluatorSiteEntry<T::Index>>,
}

/// Reusable evaluator for batched [`TreeTN`] point evaluation.
///
/// `TreeTNEvaluator` validates a complete site-index order once and precomputes
/// the mapping from each input row to the owning TreeTN node. Reuse it when the
/// same network and site-index order are evaluated repeatedly, for example in
/// TCI-style sampling loops.
///
/// Related convenience methods such as [`TreeTN::evaluate`] and
/// [`TreeTN::evaluate`] constructs a temporary evaluator internally.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_treetn::{TreeTN, TreeTNEvaluator};
///
/// let s = DynIndex::new_dyn(3);
/// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![10.0, 20.0, 30.0])?;
/// let tree = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
///
/// let evaluator = TreeTNEvaluator::new(&tree, &[s])?;
/// let values = [0usize, 2usize];
/// let shape = [1usize, 2usize];
/// let result = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
///
/// assert_eq!(result.len(), 2);
/// assert!((result[0].real() - 10.0).abs() < 1.0e-12);
/// assert!((result[1].real() - 30.0).abs() < 1.0e-12);
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct TreeTNEvaluator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    node_entries: Vec<EvaluatorNodeEntry<T, V>>,
    n_indices: usize,
}

impl<T, V> TreeTNEvaluator<T, V>
where
    T: TensorLike,
    T::Index: Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Ord,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a reusable evaluator for a complete site-index ordering.
    ///
    /// `indices` must enumerate every site index of `tree` exactly once. The
    /// evaluator uses full index equality, so indices with the same ID but
    /// different tags, prime levels, or other metadata are distinct.
    ///
    /// # Arguments
    ///
    /// * `tree` - Tree tensor network whose local tensors and index mapping
    ///
    ///   are captured by the evaluator.
    /// * `indices` - Complete input row order for future batch value arrays.
    ///
    /// # Returns
    ///
    /// A reusable evaluator that can evaluate many value batches with shape
    /// `[indices.len(), n_points]`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `tree` is empty, when `indices` does not have one
    /// entry per registered site index or repeats one, when an index is not
    /// registered on any node, or when a node's tensor cannot be resolved.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{TreeTN, TreeTNEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0])?;
    /// let tree = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    ///
    /// let evaluator = TreeTNEvaluator::new(&tree, &[s])?;
    /// assert_eq!(evaluator.input_count(), 1);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        tree: &TreeTN<T, V>,
        indices: &[T::Index],
    ) -> std::result::Result<Self, TreeTNOperationError> {
        if tree.node_count() == 0 {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!("Cannot evaluate empty TreeTN")
                    .context("TreeTNEvaluator::new: network must have at least one node"),
            ));
        }

        let n_indices = indices.len();
        let total_site_indices = tree.site_index_network.site_index_count();
        if !(n_indices == total_site_indices) {
            return Err(anyhow::anyhow!(
                "TreeTNEvaluator::new: indices.len() ({}) != total site indices ({})",
                n_indices,
                total_site_indices
            )
            .into());
        };

        let mut seen = HashSet::with_capacity(n_indices);
        for index in indices {
            if !(seen.insert(index)) {
                return Err(
                    anyhow::anyhow!("TreeTNEvaluator::new: duplicate index {:?}", index).into(),
                );
            };
        }

        let mut entries_by_node: HashMap<V, Vec<EvaluatorSiteEntry<T::Index>>> = HashMap::new();
        let mut tensor_indices_by_node: HashMap<V, Vec<T::Index>> = HashMap::new();
        for (input_position, index) in indices.iter().enumerate() {
            let node_name = tree
                .site_index_network
                .find_node_by_index(index)
                .ok_or_else(|| anyhow::anyhow!("unknown index {:?}", index))?
                .clone();
            let node_idx = tree
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))
                .context("TreeTNEvaluator::new: node must exist")?;
            let tensor = tree
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("TreeTNEvaluator::new: tensor must exist")?;
            let tensor_indices = tensor_indices_by_node
                .entry(node_name.clone())
                .or_insert_with(|| tensor.external_indices());
            let local_axis = tensor_indices
                .iter()
                .position(|tensor_index| tensor_index == index)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "site index {:?} is registered on node {:?} but not present in its tensor",
                        index,
                        node_name
                    )
                })
                .context("TreeTNEvaluator::new: site index metadata is inconsistent")?;
            entries_by_node
                .entry(node_name)
                .or_default()
                .push(EvaluatorSiteEntry {
                    index: index.clone(),
                    input_position,
                    local_axis,
                });
        }

        let node_names = tree.node_names();
        let mut node_entries = Vec::with_capacity(node_names.len());
        for node_name in node_names {
            let node_idx = tree
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))
                .context("TreeTNEvaluator::new: node must exist")?;
            let tensor = tree
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("TreeTNEvaluator::new: tensor must exist")?;
            let mut site_entries = entries_by_node.remove(&node_name).unwrap_or_default();
            site_entries.sort_by_key(|entry| entry.local_axis);
            node_entries.push(EvaluatorNodeEntry {
                name: node_name,
                tensor: tensor.clone(),
                site_entries,
            });
        }

        Ok(Self {
            node_entries,
            n_indices,
        })
    }

    /// Return the number of site-index rows expected by this evaluator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{TreeTN, TreeTNEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0])?;
    /// let tree = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let evaluator = TreeTNEvaluator::new(&tree, &[s])?;
    ///
    /// assert_eq!(evaluator.input_count(), 1);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn input_count(&self) -> usize {
        self.n_indices
    }

    /// Evaluate all points in a column-major batch value array.
    ///
    /// `values` must have shape `[self.input_count(), n_points]`, and
    /// `values.get(&[i, p])` is the coordinate for input site index `i` at
    /// point `p`.
    ///
    /// # Arguments
    ///
    /// * `values` - Column-major coordinate array with one row per evaluator
    ///
    ///   input index and one column per point.
    ///
    /// # Returns
    ///
    /// One scalar value per point.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `values` is not a 2D array with
    /// [`input_count`](Self::input_count) rows, when a coordinate is missing or
    /// exceeds its site dimension, or when building the one-hot tensors,
    /// contracting them, or extracting the scalar fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{TreeTN, TreeTNEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0, 9.0])?;
    /// let tree = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let evaluator = TreeTNEvaluator::new(&tree, &[s])?;
    /// let values = [1usize];
    /// let shape = [1usize, 1usize];
    ///
    /// let result = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
    /// assert!((result[0].real() - 9.0).abs() < 1.0e-12);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched(
        &self,
        values: ColMajorArrayRef<'_, usize>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        if !(values.shape().len() == 2) {
            return Err(anyhow::anyhow!(
                "TreeTNEvaluator::evaluate_batched: values must be 2D, got {}D",
                values.shape().len()
            )
            .into());
        };
        if !(values.shape()[0] == self.n_indices) {
            return Err(anyhow::anyhow!(
                "TreeTNEvaluator::evaluate_batched: values.shape()[0] ({}) != indices.len() ({})",
                values.shape()[0],
                self.n_indices
            )
            .into());
        };
        let n_points = values.shape()[1];

        let mut results = Vec::with_capacity(n_points);
        for point in 0..n_points {
            results.push(self.evaluate_point(values, point)?);
        }
        Ok(results)
    }

    fn evaluate_point(
        &self,
        values: ColMajorArrayRef<'_, usize>,
        point: usize,
    ) -> Result<AnyScalar> {
        let mut contracted_tensors: Vec<T> = Vec::with_capacity(self.node_entries.len());
        let mut contracted_names: Vec<V> = Vec::with_capacity(self.node_entries.len());

        for entry in &self.node_entries {
            if entry.site_entries.is_empty() {
                contracted_tensors.push(entry.tensor.clone());
                contracted_names.push(entry.name.clone());
                continue;
            }

            let index_vals: Vec<(T::Index, usize)> = entry
                .site_entries
                .iter()
                .map(|site| {
                    let value = value_at(
                        values,
                        site.input_position,
                        point,
                        "TreeTNEvaluator::evaluate_batched",
                    )?;
                    Ok((site.index.clone(), value))
                })
                .collect::<Result<_>>()?;

            let onehot = T::onehot(&index_vals)
                .context("TreeTNEvaluator::evaluate_batched: failed to create one-hot tensor")?;

            let result = T::contract(&[&entry.tensor, &onehot]).context(
                "TreeTNEvaluator::evaluate_batched: failed to contract tensor with one-hot",
            )?;

            contracted_tensors.push(result);
            contracted_names.push(entry.name.clone());
        }

        let temp_tn = TreeTN::<T, V>::from_tensors(contracted_tensors, contracted_names)
            .context("TreeTNEvaluator::evaluate_batched: failed to build temporary TreeTN")?;
        let result_tensor = temp_tn
            .contract_to_tensor()
            .context("TreeTNEvaluator::evaluate_batched: failed to contract to scalar")?;

        let scalar_one = T::scalar_one()
            .context("TreeTNEvaluator::evaluate_batched: failed to create scalar")?;
        scalar_one
            .inner_product(&result_tensor)
            .context("TreeTNEvaluator::evaluate_batched: failed to extract scalar value")
    }
}

impl<V> TreeTNEvaluator<IdxTensor, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Evaluate all points on a caller-held explicit session.
    ///
    /// This is the session counterpart of [`TreeTNEvaluator::evaluate_batched`]
    /// for indexed tensors: each node's site legs are fixed with
    /// [`select_indices_on_session`], and the remaining tree is contracted with
    /// [`contract_with_options_on_session`], so no one-hot eager value, no
    /// process-global runtime entry and no dense `contract_to_tensor` is used.
    /// The public global [`TreeTN::contract_to_tensor`] / [`TreeTN::to_dense`]
    /// routes remain the compatibility frontend.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same shape and coordinate
    /// conditions as [`TreeTNEvaluator::evaluate_batched`], and when the supplied
    /// session rejects a selection or the final contraction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_tensorbackend::{explicit::{HeldSession, Session}, CpuExecutionContext};
    /// use tensor4all_treetn::{TreeTN, TreeTNEvaluator};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 9.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let evaluator = TreeTNEvaluator::new(&tree, &[s.clone()])?;
    /// let values = [1usize];
    /// let points = ColMajorArrayRef::new(&values, &[1, 1])?;
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let backend: CpuBackend = context.with_backend(|backend| backend.clone());
    /// let mut held = HeldSession::open(&backend)?;
    /// let result = held.with_session(|view| {
    ///     let mut session = Session::from_backend_session(view.backend());
    ///     evaluator.evaluate_batched_on_session(&mut session, points)
    /// })?;
    /// held.close()?;
    /// assert!((result[0].real() - 9.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate_batched_on_session(
        &self,
        session: &mut Session<'_>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        if values.shape().len() != 2 {
            return Err(anyhow::anyhow!(
                "TreeTNEvaluator::evaluate_batched_on_session: values must be 2D, got {}D",
                values.shape().len()
            )
            .into());
        }
        if values.shape()[0] != self.n_indices {
            return Err(anyhow::anyhow!(
                "TreeTNEvaluator::evaluate_batched_on_session: values.shape()[0] ({}) != indices.len() ({})",
                values.shape()[0],
                self.n_indices
            )
            .into());
        }
        let mut results = Vec::with_capacity(values.shape()[1]);
        for point in 0..values.shape()[1] {
            results.push(self.evaluate_point_on_session(session, values, point)?);
        }
        Ok(results)
    }

    fn evaluate_point_on_session(
        &self,
        session: &mut Session<'_>,
        values: ColMajorArrayRef<'_, usize>,
        point: usize,
    ) -> Result<AnyScalar> {
        let mut contracted = Vec::with_capacity(self.node_entries.len());
        for entry in &self.node_entries {
            if entry.site_entries.is_empty() {
                contracted.push(entry.tensor.clone());
                continue;
            }
            let index_vals: Vec<(DynIndex, usize)> = entry
                .site_entries
                .iter()
                .map(|site| {
                    let value = value_at(
                        values,
                        site.input_position,
                        point,
                        "TreeTNEvaluator::evaluate_batched_on_session",
                    )?;
                    Ok((site.index.clone(), value))
                })
                .collect::<Result<_>>()?;
            let selected = index_vals
                .iter()
                .map(|(index, _)| index.clone())
                .collect::<Vec<_>>();
            let positions = index_vals
                .iter()
                .map(|(_, position)| *position)
                .collect::<Vec<_>>();
            contracted.push(
                select_indices_on_session(session, &entry.tensor, &selected, &positions)
                    .map_err(anyhow::Error::from)
                    .context(
                        "TreeTNEvaluator::evaluate_batched_on_session: failed to slice a node",
                    )?,
            );
        }

        let refs = contracted.iter().collect::<Vec<_>>();
        let scalar = contract_with_options_on_session(session, &refs, ContractionOptions::new())
            .map_err(anyhow::Error::from)
            .context(
                "TreeTNEvaluator::evaluate_batched_on_session: failed to contract the network",
            )?;
        if scalar.is_f64() {
            let value = scalar.dense_host_values::<f64>()?;
            return Ok(AnyScalar::from_value(value[0]));
        }
        if scalar.is_f32() {
            let value = scalar.dense_host_values::<f32>()?;
            return Ok(AnyScalar::from_value(value[0]));
        }
        if scalar.is_c64() {
            let value = scalar.dense_host_values::<num_complex::Complex64>()?;
            return Ok(AnyScalar::from_value(value[0]));
        }
        if scalar.is_c32() {
            let value = scalar.dense_host_values::<num_complex::Complex32>()?;
            return Ok(AnyScalar::from_value(value[0]));
        }
        Err(anyhow::anyhow!(
            "TreeTNEvaluator::evaluate_batched_on_session: unsupported scalar kind {:?}",
            scalar.storage_kind()
        ))
    }
}

pub(super) fn value_at(
    values: ColMajorArrayRef<'_, usize>,
    input_position: usize,
    point: usize,
    context: &str,
) -> Result<usize> {
    values
        .get(&[input_position, point])
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{context}: missing coordinate at row {} point {} for shape {:?}",
                input_position,
                point,
                values.shape()
            )
        })
}
