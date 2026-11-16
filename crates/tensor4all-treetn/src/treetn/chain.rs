use anyhow::{anyhow, Context};
use petgraph::stable_graph::NodeIndex;
use tensor4all_core::{
    factorize_full_rank_on_session, index_select_on_session,
    tensor::compute_permutation_from_indices, validate_svd_truncation_options, Canonical,
    ColMajorArray, DynIndex, FactorizeAlg, IdxTensor, IndexLike, SvdTruncationPolicy,
    TensorElement,
};
use tensor4all_tensorbackend::{explicit::Session, StorageKind, StorageScalar};

use super::TreeTN;
use crate::{CanonicalForm, TreeTNOperationError};

impl TreeTN<IdxTensor, usize> {
    /// Build a chain with node names `0..cores.len()` and one fresh site index per node.
    ///
    /// Each owned core is a column-major host array with shape
    /// `[left_bond, site, right_bond]`. The first axis varies fastest. The two
    /// boundary bond dimensions must be one and are omitted from the network
    /// tensors. The returned site indices follow the input order. An empty
    /// input produces an empty network.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] for a core with the wrong rank, a zero
    /// dimension, a mismatched bond dimension, a non-unit boundary bond, or a
    /// failure to construct the tensors or network.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::ColMajorArray;
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let left = ColMajorArray::new(vec![2.0_f64, 3.0], vec![1, 2, 1])?;
    /// let right = ColMajorArray::new(vec![5.0_f64, 7.0], vec![1, 2, 1])?;
    /// let (chain, sites) = TreeTN::from_chain_cores(vec![left, right])?;
    /// let dense = chain.contract_to_tensor()?.permute_indices(&sites)?;
    /// assert_eq!(dense.to_vec::<f64>()?, vec![10.0, 15.0, 14.0, 21.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_chain_cores<T: TensorElement>(
        cores: Vec<ColMajorArray<T>>,
    ) -> Result<(Self, Vec<DynIndex>), TreeTNOperationError> {
        let nsites = cores.len();
        let mut previous_right = 1;
        for (position, core) in cores.iter().enumerate() {
            let &[left, site, right] = core.shape() else {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "chain core {position} must have rank three [left bond, site, right bond]"
                )));
            };
            if site == 0 || left == 0 || right == 0 {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "chain core {position} dimensions must be positive"
                )));
            }
            if left != previous_right {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "chain core {position} has left bond dimension {left}, expected {previous_right}"
                )));
            }
            if position + 1 == nsites && right != 1 {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "last chain core has right bond dimension {right}, expected 1"
                )));
            }
            previous_right = right;
        }

        let mut site_indices = Vec::with_capacity(nsites);
        let mut tensors = Vec::with_capacity(nsites);
        let mut left_bond = None;
        for (position, core) in cores.into_iter().enumerate() {
            // INVARIANT: the validation pass established rank three and positive dimensions.
            let site = DynIndex::new_dyn(core.shape()[1]);
            let right_bond = (position + 1 < nsites)
                .then(|| DynIndex::new_bond(core.shape()[2]))
                .transpose()
                .with_context(|| format!("allocate right bond of chain core {position}"))?;
            let indices = left_bond
                .into_iter()
                .chain(std::iter::once(site.clone()))
                .chain(right_bond.iter().cloned())
                .collect();
            tensors.push(IdxTensor::from_dense_compact(indices, core.into_data())?);
            site_indices.push(site);
            left_bond = right_bond;
        }
        Ok((
            Self::from_tensors(tensors, (0..nsites).collect())?,
            site_indices,
        ))
    }

    /// Read the chain's current local tensors as column-major host cores.
    ///
    /// Pass the site indices in node-name order `0..n`, with their exact
    /// identities. Each returned array has shape `[left_bond, site, right_bond]`,
    /// including singleton boundary axes. This copies only the local tensors,
    /// preserving their values and gauge. `T` must match the stored scalar type.
    /// Compact host cores can be read while holding an explicit session.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] if the network is not a chain with one
    /// site index per node, the supplied indices do not match, or a local tensor
    /// is structured, tracks gradients, or cannot be read as the requested host scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::ColMajorArray;
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let core = ColMajorArray::new(vec![2.0_f64, 3.0], vec![1, 2, 1])?;
    /// let (chain, sites) = TreeTN::from_chain_cores(vec![core.clone()])?;
    /// assert_eq!(chain.chain_cores::<f64>(&sites)?, vec![core]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn chain_cores<T: StorageScalar>(
        &self,
        site_indices: &[DynIndex],
    ) -> Result<Vec<ColMajorArray<T>>, TreeTNOperationError> {
        let layout = self.chain_layout()?;
        if site_indices.len() != layout.len() {
            return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                "expected {} chain site indices, got {}",
                layout.len(),
                site_indices.len()
            )));
        }
        for (position, (core, site)) in layout.iter().zip(site_indices).enumerate() {
            if core.site_index != site || core.site_index.dim() != site.dim() {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "site index at position {position} does not match chain node {position}"
                )));
            }
            if core.tensor.tracks_grad() || core.tensor.storage_kind() != StorageKind::Dense {
                return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                    "chain core {position} must have untracked dense storage for readout"
                )));
            }
        }
        layout
            .into_iter()
            .enumerate()
            .map(|(position, core)| {
                core.read()
                    .with_context(|| format!("read chain core {position}"))
                    .map_err(TreeTNOperationError::from)
            })
            .collect()
    }

    /// Compress an ordered, one-site-per-node host chain on the supplied session.
    ///
    /// A full-rank QR sweep goes towards the last node, then an SVD sweep
    /// truncates each bond going back towards node zero. It retains singular
    /// values at least `tolerance` times the largest value on their bond.
    /// `max_bond_dim` caps retained ranks. Zero tolerance retains the available
    /// SVD width up to the cap. Each bond keeps at least one direction.
    ///
    /// Returns the chain with its site identities preserved. An empty or
    /// single-node chain is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] for a non-finite or negative tolerance,
    /// a zero cap, an invalid chain layout, tracked or non-dense tensors, or a
    /// factorization or contraction failure on `session`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::ColMajorArray;
    /// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let left = ColMajorArray::new(vec![1.0_f64, 0.0, 0.0, 1.0], vec![1, 2, 2])?;
    /// let right = ColMajorArray::new(vec![4.0_f64, 0.0, 0.0, 1.0], vec![2, 2, 1])?;
    /// let (chain, sites) = TreeTN::from_chain_cores(vec![left, right])?;
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// let compressed = {
    ///     let mut session = Session::new(backend.engine_mut());
    ///     chain.compress_chain_on_session(0.5, None, &mut session)?
    /// };
    /// let dense = compressed.contract_to_tensor()?.permute_indices(&sites)?.to_vec::<f64>()?;
    /// assert!((dense[0] - 4.0).abs() < 1e-12);
    /// assert!(dense[1..].iter().all(|value| value.abs() < 1e-12));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress_chain_on_session(
        mut self,
        tolerance: f64,
        max_bond_dim: Option<usize>,
        session: &mut Session<'_>,
    ) -> Result<Self, TreeTNOperationError> {
        let policy = SvdTruncationPolicy::new(tolerance);
        validate_svd_truncation_options(max_bond_dim, Some(policy))
            .map_err(TreeTNOperationError::invalid_configuration)?;
        let nodes = self
            .chain_layout()?
            .into_iter()
            .enumerate()
            .map(|(position, core)| {
                if core.tensor.tracks_grad() || core.tensor.storage_kind() != StorageKind::Dense {
                    return Err(TreeTNOperationError::invalid_configuration(anyhow!(
                        "chain core {position} must have untracked dense storage for compression"
                    )));
                }
                Ok(core.node)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if nodes.len() <= 1 {
            return Ok(self);
        }

        let operation = "compress_chain_on_session";
        self.canonicalize_impl_on_session(
            [nodes.len() - 1],
            CanonicalForm::Unitary,
            operation,
            session,
        )?;

        for edge in nodes.windows(2).rev() {
            // With the destination bond on the right of the split, the kept
            // source factor is isometric towards the destination.
            self.sweep_edge_with_on_session(
                edge[1],
                edge[0],
                FactorizeAlg::SVD,
                operation,
                session,
                |session, tensor, left| {
                    let mut factors = factorize_full_rank_on_session(
                        session,
                        tensor,
                        left,
                        FactorizeAlg::SVD,
                        Canonical::Left,
                    )?;
                    if tolerance == 0.0 && max_bond_dim.is_none() {
                        return Ok(factors);
                    }
                    let values = factors
                        .singular_values
                        .as_mut()
                        .context("missing SVD spectrum")?;
                    let largest = values.first().copied().context("empty SVD spectrum")?;
                    let threshold = tolerance * largest;
                    let rank = values
                        .iter()
                        .take(max_bond_dim.unwrap_or(values.len()))
                        .take_while(|&&value| value >= threshold)
                        .count()
                        .max(1);
                    if rank < factors.rank {
                        let bond = DynIndex::new_bond(rank)?;
                        let kept: Vec<_> = (0..rank).collect();
                        factors.left = index_select_on_session(
                            session,
                            &factors.left,
                            &factors.bond_index,
                            bond.clone(),
                            &kept,
                        )?;
                        factors.right = index_select_on_session(
                            session,
                            &factors.right,
                            &factors.bond_index,
                            bond.clone(),
                            &kept,
                        )?;
                        factors.bond_index = bond;
                        factors.rank = rank;
                        values.truncate(rank);
                    }
                    Ok(factors)
                },
            )?;
        }
        self.set_canonical_region([0])?;
        Ok(self)
    }

    fn chain_layout(&self) -> Result<Vec<ChainCore<'_>>, TreeTNOperationError> {
        let count = self.node_count();
        if self.edge_count() != count.saturating_sub(1) {
            return Err(TreeTNOperationError::invalid_graph(anyhow!(
                "chain with {count} nodes must have {} edges",
                count.saturating_sub(1)
            )));
        }

        let mut layout = Vec::with_capacity(count);
        let mut left = None;
        for position in 0..count {
            let invalid = |reason: &str| {
                TreeTNOperationError::invalid_graph(anyhow!("chain node {position}: {reason}"))
            };
            let node = self
                .node_index(&position)
                .ok_or_else(|| invalid("node is missing"))?;
            let tensor = self
                .tensor(node)
                .ok_or_else(|| invalid("tensor is missing"))?;
            let sites = self
                .site_space(&position)
                .filter(|indices| indices.len() == 1)
                .ok_or_else(|| invalid("expected exactly one site index"))?;
            let site_index = sites
                .iter()
                .next()
                .ok_or_else(|| invalid("site index is missing"))?;
            let right = if position + 1 == count {
                None
            } else {
                Some(
                    self.edge_between(&position, &(position + 1))
                        .and_then(|edge| self.bond_index(edge))
                        .ok_or_else(|| invalid("bond to the next node is missing"))?,
                )
            };

            let indices = [left, Some(site_index), right];
            let shape = indices.map(|index| index.map_or(1, IndexLike::dim));
            let ordered: Vec<_> = indices.into_iter().flatten().cloned().collect();
            let perm =
                compute_permutation_from_indices(tensor.indices(), &ordered).map_err(|error| {
                    TreeTNOperationError::invalid_graph(
                        anyhow::Error::new(error)
                            .context(format!("chain node {position} has unexpected indices")),
                    )
                })?;
            if shape.contains(&0)
                || ordered
                    .iter()
                    .zip(&perm)
                    .any(|(index, &axis)| index.dim() != tensor.indices()[axis].dim())
            {
                return Err(invalid(
                    "tensor index dimensions must match the positive chain dimensions",
                ));
            }
            let mut source_axes = [None; 3];
            for ((role, _), axis) in indices
                .into_iter()
                .enumerate()
                .filter(|(_, index)| index.is_some())
                .zip(perm)
            {
                source_axes[role] = Some(axis);
            }
            layout.push(ChainCore {
                node,
                tensor,
                site_index,
                shape,
                source_axes,
            });
            left = right;
        }
        Ok(layout)
    }
}

struct ChainCore<'a> {
    node: NodeIndex,
    tensor: &'a IdxTensor,
    site_index: &'a DynIndex,
    shape: [usize; 3],
    source_axes: [Option<usize>; 3],
}

impl ChainCore<'_> {
    fn read<T: StorageScalar>(self) -> anyhow::Result<ColMajorArray<T>> {
        let mut strides = [0; 3];
        let mut len = 1usize;
        // INVARIANT: the validated chain has at most three axes, with matching dimensions.
        for (axis, index) in self.tensor.indices().iter().enumerate() {
            strides[axis] = len;
            len = len
                .checked_mul(index.dim())
                .context("chain core shape overflow")?;
        }
        let identity = self
            .source_axes
            .iter()
            .flatten()
            .copied()
            .eq(0..self.tensor.indices().len());
        let data = self.tensor.with_dense_host_slice::<T, _>(|source| {
            anyhow::ensure!(
                source.len() == len,
                "chain core payload has an unexpected length"
            );
            if identity {
                return Ok(source.to_vec());
            }

            let [left, site, right] = self
                .source_axes
                .map(|axis| axis.map_or(0, |axis| strides[axis]));
            let [nl, ns, nr] = self.shape;
            let mut reordered = Vec::with_capacity(len);
            for r in 0..nr {
                for s in 0..ns {
                    let start = r * right + s * site;
                    for l in 0..nl {
                        // INVARIANT: the axes are a permutation of the validated
                        // source shape. The singleton boundary axes have stride zero.
                        reordered.push(source[start + l * left]);
                    }
                }
            }
            Ok(reordered)
        })??;
        Ok(ColMajorArray::new(data, self.shape.to_vec())?)
    }
}
