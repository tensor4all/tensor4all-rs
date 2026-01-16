//! Contraction optimization for diagonal tensors.
//!
//! This module provides utilities for handling diagonal (Diag) tensors in
//! multi-tensor contractions. When Diag tensors share indices, their diagonal
//! axes should be unified to create hyperedges in the einsum optimizer.
//!
//! # Example
//!
//! Consider `Diag(i,j) * Diag(j,k)`:
//! - Diag(i,j) has diagonal axes i and j (same index)
//! - Diag(j,k) has diagonal axes j and k (same index)
//! - After union-find: i, j, k all map to the same representative ID
//! - This creates a hyperedge that the einsum optimizer handles correctly
//!
//! # Note on TensorData
//!
//! TensorData processes permutations lazily. When building the union-find,
//! we need to look at each `TensorComponent`'s storage and `index_ids`
//! to determine which IDs belong to Diag storage. The external order in
//! `TensorData.external_index_ids` may differ due to lazy permutations.

use crate::defaults::{DynId, DynIndex, TensorComponent, TensorData, TensorDynLen};
use crate::index_like::IndexLike;
use crate::storage::{DenseStorageC64, Storage};
use crate::tensor_like::AllowedPairs;

use anyhow::Result;
use mdarray_einsum::{einsum_optimized, Naive};
use num_complex::Complex64;
use petgraph::unionfind::UnionFind;
use std::collections::HashMap;
use std::sync::Arc;
use tensor4all_tensorbackend::mdarray::{DynRank, Tensor};

/// Union-Find data structure for grouping axis IDs.
///
/// Used to merge diagonal axes from Diag tensors so that they share
/// the same representative ID when passed to einsum.
#[derive(Debug, Clone)]
pub struct AxisUnionFind {
    /// Maps each ID to its parent. If parent[id] == id, it's a root.
    parent: HashMap<DynId, DynId>,
    /// Rank for union by rank optimization.
    rank: HashMap<DynId, usize>,
}

impl AxisUnionFind {
    /// Create a new empty union-find structure.
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    /// Add an ID to the structure (as its own set).
    pub fn make_set(&mut self, id: DynId) {
        if !self.parent.contains_key(&id) {
            self.parent.insert(id, id);
            self.rank.insert(id, 0);
        }
    }

    /// Find the representative (root) of the set containing `id`.
    /// Uses path compression for efficiency.
    pub fn find(&mut self, id: DynId) -> DynId {
        self.make_set(id);
        if self.parent[&id] != id {
            let root = self.find(self.parent[&id]);
            self.parent.insert(id, root);
        }
        self.parent[&id]
    }

    /// Union the sets containing `a` and `b`.
    /// Uses union by rank for efficiency.
    pub fn union(&mut self, a: DynId, b: DynId) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return;
        }

        let rank_a = self.rank[&root_a];
        let rank_b = self.rank[&root_b];

        if rank_a < rank_b {
            self.parent.insert(root_a, root_b);
        } else if rank_a > rank_b {
            self.parent.insert(root_b, root_a);
        } else {
            self.parent.insert(root_b, root_a);
            *self.rank.get_mut(&root_a).unwrap() += 1;
        }
    }

    /// Remap an ID to its representative.
    pub fn remap(&mut self, id: DynId) -> DynId {
        self.find(id)
    }

    /// Remap a slice of IDs to their representatives.
    pub fn remap_ids(&mut self, ids: &[DynId]) -> Vec<DynId> {
        ids.iter().map(|id| self.find(*id)).collect()
    }
}

impl Default for AxisUnionFind {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a union-find structure from TensorData components.
///
/// For each Diag component, all its indices are unified (they share the same
/// diagonal dimension). This creates hyperedges when multiple Diag components
/// share indices.
///
/// Note: This operates on TensorComponent level, not TensorDynLen level,
/// because TensorData handles permutations lazily. We need to look at
/// each component's storage and index_ids directly.
pub fn build_diag_union_from_components(components: &[&TensorComponent]) -> AxisUnionFind {
    let mut uf = AxisUnionFind::new();

    for component in components {
        // Add all indices to the union-find
        for &id in component.index_ids.iter() {
            uf.make_set(id);
        }

        // For Diag storage, union all diagonal axes
        if component.storage.is_diag() && component.index_ids.len() >= 2 {
            let first_id = component.index_ids[0];
            for &id in component.index_ids.iter().skip(1) {
                uf.union(first_id, id);
            }
        }
    }

    uf
}

/// Build a union-find structure from TensorData.
///
/// Processes all components in the TensorData, unifying diagonal axes
/// from Diag storage components.
pub fn build_diag_union_from_data(data: &TensorData) -> AxisUnionFind {
    let component_refs: Vec<&TensorComponent> = data.components.iter().collect();
    build_diag_union_from_components(&component_refs)
}

/// Build a union-find structure from a collection of tensors.
///
/// For each Diag tensor component, all its indices are unified (they share the same
/// diagonal dimension). This creates hyperedges when multiple Diag tensors
/// share indices.
///
/// # Example
///
/// ```ignore
/// // Diag(i,j) and Diag(j,k)
/// // After build_diag_union:
/// //   i, j, k all have the same representative
/// let uf = build_diag_union(&[diag_ij, diag_jk]);
/// ```
pub fn build_diag_union(tensors: &[&TensorDynLen]) -> AxisUnionFind {
    // Collect all components from all tensors
    let all_components: Vec<&TensorComponent> = tensors
        .iter()
        .flat_map(|t| t.tensor_data().components.iter())
        .collect();

    build_diag_union_from_components(&all_components)
}

/// Remap tensor indices using the union-find structure.
///
/// Returns a vector of remapped IDs for each tensor, suitable for passing
/// to einsum. The original tensors are not modified.
pub fn remap_tensor_ids(
    tensors: &[&TensorDynLen],
    uf: &mut AxisUnionFind,
) -> Vec<Vec<DynId>> {
    tensors
        .iter()
        .map(|t| {
            t.indices
                .iter()
                .map(|idx| uf.find(*idx.id()))
                .collect()
        })
        .collect()
}

/// Remap output IDs using the union-find structure.
pub fn remap_output_ids(output: &[DynIndex], uf: &mut AxisUnionFind) -> Vec<DynId> {
    output.iter().map(|idx| uf.find(*idx.id())).collect()
}

/// Collect dimension sizes for remapped IDs.
///
/// For unified IDs (from Diag tensors), all axes must have the same dimension,
/// so we just take the first occurrence.
pub fn collect_sizes(
    tensors: &[&TensorDynLen],
    uf: &mut AxisUnionFind,
) -> HashMap<DynId, usize> {
    let mut sizes = HashMap::new();

    for tensor in tensors {
        for (idx, &dim) in tensor.indices.iter().zip(tensor.dims.iter()) {
            let rep = uf.find(*idx.id());
            sizes.entry(rep).or_insert(dim);
        }
    }

    sizes
}

// ============================================================================
// Diag-aware contraction
// ============================================================================

/// Contract multiple tensors with Diag-aware optimization.
///
/// This function handles Diag tensors by:
/// 1. Building union-find to unify diagonal axes
/// 2. Remapping IDs so diagonal axes share the same internal ID
/// 3. Using omeco's GreedyMethod to find optimal contraction order
///
/// For Diag(i,j) * Diag(j,k), this creates a hyperedge where i,j,k all
/// share the same internal ID, allowing omeco to optimize the contraction.
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract
/// * `allowed` - Specifies which tensor pairs can have their indices contracted
///
/// # Returns
/// The result of contracting all tensors over allowed contractable indices.
pub fn contract_multi_diag_aware(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok((*tensors[0]).clone()),
        _ => contract_multi_diag_aware_impl(tensors, allowed),
    }
}

/// Implementation of Diag-aware multi-tensor contraction.
///
/// For Diag tensors, we pass them as 1D tensors (the diagonal elements) with
/// a single hyperedge ID. The einsum hyperedge optimizer will handle them correctly.
fn contract_multi_diag_aware_impl(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
) -> Result<TensorDynLen> {
    use tensor4all_tensorbackend::mdarray::{Dense, Slice};

    // 1. Build union-find from Diag tensors to unify diagonal axes
    let mut diag_uf = build_diag_union(tensors);

    // 2. Build internal IDs with Diag-awareness
    //    For Diag tensors, all axes map to the same internal ID (hyperedge)
    let (ixs, internal_id_to_original) = build_internal_ids_diag_aware(tensors, allowed, &mut diag_uf)?;

    // 3. Output = count == 1 internal IDs (external indices)
    let mut idx_count: HashMap<usize, usize> = HashMap::new();
    for ix in &ixs {
        for &i in ix {
            *idx_count.entry(i).or_insert(0) += 1;
        }
    }
    let mut output: Vec<usize> = idx_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&idx, _)| idx)
        .collect();
    output.sort(); // deterministic order

    // 4. Check connectivity using Union-Find (O(E α(N)) where E = edges, N = tensors)
    // Build index -> first tensor mapping, then union tensors sharing indices
    {
        let n = tensors.len();
        let mut tensor_uf: UnionFind<usize> = UnionFind::new(n);

        // Map: internal_id -> first tensor that has this index
        let mut id_to_first_tensor: HashMap<usize, usize> = HashMap::new();

        for (tensor_idx, ix) in ixs.iter().enumerate() {
            for &internal_id in ix {
                if let Some(&first_tensor) = id_to_first_tensor.get(&internal_id) {
                    // This index already seen in another tensor - union them
                    tensor_uf.union(first_tensor, tensor_idx);
                } else {
                    id_to_first_tensor.insert(internal_id, tensor_idx);
                }
            }
        }

        // Check if all tensors are in the same component
        let labels = tensor_uf.into_labeling();
        let unique_labels: std::collections::HashSet<usize> = labels.into_iter().collect();
        let num_components = unique_labels.len();

        if num_components > 1 {
            return Err(anyhow::anyhow!(
                "Disconnected tensor network: {} components found",
                num_components
            ));
        }
    }

    // 5. Build sizes from unique internal IDs
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (pos, &dim) in tensor.dims.iter().enumerate() {
            let internal_id = ixs[tensor_idx][pos];
            sizes.entry(internal_id).or_insert(dim);
        }
    }

    // 6. Convert TensorDynLen to mdarray tensors
    //    - Diag storage: pass as 1D tensor (hyperedge representation)
    //    - Dense storage: pass as-is with original shape
    let mut mdarray_tensors: Vec<Tensor<Complex64, DynRank>> = Vec::new();
    let mut einsum_ids: Vec<Vec<usize>> = Vec::new();

    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        let storage = tensor.materialize_storage()?;
        let is_diag = storage.is_diag();

        if is_diag {
            // Diag tensor: extract diagonal as 1D tensor
            // All axes have the same internal ID (hyperedge)
            let diag_data: Vec<Complex64> = match &*storage {
                Storage::DiagC64(ds) => ds.as_slice().to_vec(),
                Storage::DiagF64(ds) => ds.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect(),
                _ => return Err(anyhow::anyhow!("Expected Diag storage")),
            };
            let diag_len = diag_data.len();
            let tensor_1d = Tensor::from(diag_data).into_shape([diag_len].as_slice()).into_dyn();
            mdarray_tensors.push(tensor_1d);

            // Diag tensor with unified axes: use single hyperedge ID
            // All original axes map to the same internal ID
            let hyperedge_id = ixs[tensor_idx][0]; // All should be the same
            einsum_ids.push(vec![hyperedge_id]);
        } else {
            // Dense tensor: convert to Complex64 and pass as-is
            let tensor_data: Tensor<Complex64, DynRank> = match &*storage {
                Storage::DenseC64(ds) => ds.tensor().clone(),
                Storage::DenseF64(ds) => {
                    let data: Vec<Complex64> = ds.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect();
                    Tensor::from(data).into_shape(tensor.dims.as_slice()).into_dyn()
                }
                _ => return Err(anyhow::anyhow!("Expected Dense storage")),
            };
            mdarray_tensors.push(tensor_data);
            einsum_ids.push(ixs[tensor_idx].clone());
        }
    }

    // 6. Build einsum inputs: (&[ID], &Slice)
    let inputs: Vec<(&[usize], &Slice<Complex64, DynRank, Dense>)> = einsum_ids
        .iter()
        .zip(mdarray_tensors.iter())
        .map(|(ids, tensor)| (ids.as_slice(), tensor.as_ref()))
        .collect();

    // 7. Use mdarray-einsum with hyperedge support
    let result_mdarray = einsum_optimized(&Naive, &inputs, &output, &sizes);

    // 8. Convert result back to TensorDynLen
    let result_dims: Vec<usize> = result_mdarray.shape().dims().to_vec();

    // Build result indices from output internal IDs
    let restored_indices: Vec<DynIndex> = output
        .iter()
        .map(|&internal_id| {
            let (tensor_idx, pos) = internal_id_to_original[&internal_id];
            tensors[tensor_idx].indices[pos].clone()
        })
        .collect();

    // Handle scalar output: einsum returns [1] but we need empty dims
    let (final_dims, final_indices) = if output.is_empty() && result_dims == vec![1] {
        // Scalar result: use empty dims and indices
        (vec![], vec![])
    } else {
        (result_dims.clone(), restored_indices)
    };

    // Create storage from result data
    let result_data: Vec<Complex64> = result_mdarray.into_vec();
    let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
        result_data,
        &final_dims,
    )));

    Ok(TensorDynLen::new(final_indices, final_dims, storage))
}

/// Build internal IDs with Diag-awareness.
///
/// Similar to the regular `build_internal_ids` but uses the union-find
/// to ensure diagonal axes from Diag tensors share the same internal ID.
///
/// Returns: (ixs, internal_id_to_original)
#[allow(clippy::type_complexity)]
fn build_internal_ids_diag_aware(
    tensors: &[&TensorDynLen],
    allowed: AllowedPairs<'_>,
    diag_uf: &mut AxisUnionFind,
) -> Result<(Vec<Vec<usize>>, HashMap<usize, (usize, usize)>)> {
    let mut next_id = 0usize;
    // Maps (remapped DynId) -> internal_id
    let mut dynid_to_internal: HashMap<DynId, usize> = HashMap::new();
    // (tensor_idx, index_position) -> internal_id
    let mut assigned: HashMap<(usize, usize), usize> = HashMap::new();
    // internal_id -> (tensor_idx, index_position) for restoring
    let mut internal_id_to_original: HashMap<usize, (usize, usize)> = HashMap::new();

    // Process contractable pairs
    let pairs_to_process: Vec<(usize, usize)> = match allowed {
        AllowedPairs::All => {
            let mut pairs = Vec::new();
            for ti in 0..tensors.len() {
                for tj in (ti + 1)..tensors.len() {
                    pairs.push((ti, tj));
                }
            }
            pairs
        }
        AllowedPairs::Specified(pairs) => pairs.to_vec(),
    };

    for (ti, tj) in pairs_to_process {
        for (pi, idx_i) in tensors[ti].indices.iter().enumerate() {
            for (pj, idx_j) in tensors[tj].indices.iter().enumerate() {
                if idx_i.is_contractable(idx_j) {
                    let key_i = (ti, pi);
                    let key_j = (tj, pj);

                    // Get remapped IDs through union-find
                    let remapped_i = diag_uf.find(*idx_i.id());
                    let remapped_j = diag_uf.find(*idx_j.id());

                    match (assigned.get(&key_i).copied(), assigned.get(&key_j).copied()) {
                        (None, None) => {
                            // Neither assigned: create new shared ID
                            let internal_id = if let Some(&id) = dynid_to_internal.get(&remapped_i)
                            {
                                id
                            } else {
                                let id = next_id;
                                next_id += 1;
                                dynid_to_internal.insert(remapped_i, id);
                                internal_id_to_original.insert(id, key_i);
                                id
                            };
                            assigned.insert(key_i, internal_id);
                            assigned.insert(key_j, internal_id);
                            // Also map the other remapped ID if different
                            if remapped_i != remapped_j {
                                dynid_to_internal.insert(remapped_j, internal_id);
                            }
                        }
                        (Some(id), None) => {
                            assigned.insert(key_j, id);
                            dynid_to_internal.insert(remapped_j, id);
                        }
                        (None, Some(id)) => {
                            assigned.insert(key_i, id);
                            dynid_to_internal.insert(remapped_i, id);
                        }
                        (Some(_id_i), Some(_id_j)) => {
                            // Both already assigned - should have same ID
                        }
                    }
                }
            }
        }
    }

    // Assign IDs for unassigned indices (external indices)
    for (tensor_idx, tensor) in tensors.iter().enumerate() {
        for (pos, idx) in tensor.indices.iter().enumerate() {
            let key = (tensor_idx, pos);
            if let std::collections::hash_map::Entry::Vacant(e) = assigned.entry(key) {
                // Remap through union-find first
                let remapped_id = diag_uf.find(*idx.id());

                let internal_id = if let Some(&id) = dynid_to_internal.get(&remapped_id) {
                    id
                } else {
                    let id = next_id;
                    next_id += 1;
                    dynid_to_internal.insert(remapped_id, id);
                    internal_id_to_original.insert(id, key);
                    id
                };
                e.insert(internal_id);
            }
        }
    }

    // Build ixs
    let ixs: Vec<Vec<usize>> = tensors
        .iter()
        .enumerate()
        .map(|(tensor_idx, tensor)| {
            (0..tensor.indices.len())
                .map(|pos| assigned[&(tensor_idx, pos)])
                .collect()
        })
        .collect();

    Ok((ixs, internal_id_to_original))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find_basic() {
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);

        uf.make_set(a);
        uf.make_set(b);
        uf.make_set(c);

        // Initially all separate
        assert_ne!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(b), uf.find(c));

        // Union a and b
        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(a), uf.find(c));

        // Union b and c (transitively unifies a, b, c)
        uf.union(b, c);
        assert_eq!(uf.find(a), uf.find(b));
        assert_eq!(uf.find(b), uf.find(c));
        assert_eq!(uf.find(a), uf.find(c));
    }

    #[test]
    fn test_union_find_chain() {
        let mut uf = AxisUnionFind::new();

        // Simulate Diag(i,j) * Diag(j,k) * Diag(k,l)
        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        // Diag(i,j): union i and j
        uf.union(i, j);

        // Diag(j,k): union j and k
        uf.union(j, k);

        // Diag(k,l): union k and l
        uf.union(k, l);

        // All should be in the same set
        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
    }

    #[test]
    fn test_remap_ids() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);

        // Union i and j (simulating a Diag tensor)
        uf.union(i, j);

        let ids = vec![i, j, k];
        let remapped = uf.remap_ids(&ids);

        // i and j should map to the same rep
        assert_eq!(remapped[0], remapped[1]);
        // k should be different
        assert_ne!(remapped[0], remapped[2]);
    }

    #[test]
    fn test_three_diag_chain() {
        // Diag(i,j) * Diag(j,k) * Diag(k,l) -> all 4 IDs unified
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        // Diag(i,j)
        uf.union(i, j);
        // Diag(j,k)
        uf.union(j, k);
        // Diag(k,l)
        uf.union(k, l);

        // All should have the same representative
        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
    }

    #[test]
    fn test_three_diag_star() {
        // Diag(a,b), Diag(a,c), Diag(a,d) -> star topology, all unified via 'a'
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);

        // Diag(a,b)
        uf.union(a, b);
        // Diag(a,c)
        uf.union(a, c);
        // Diag(a,d)
        uf.union(a, d);

        // All should have the same representative
        let rep = uf.find(a);
        assert_eq!(uf.find(b), rep);
        assert_eq!(uf.find(c), rep);
        assert_eq!(uf.find(d), rep);
    }

    #[test]
    fn test_diag_with_three_axes() {
        // Diag tensor with 3 axes: Diag(i,j,k) means i=j=k on diagonal
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4); // separate axis

        // Diag(i,j,k): all three should be unified
        uf.union(i, j);
        uf.union(j, k);

        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        // l is separate
        assert_ne!(uf.find(l), rep);
    }

    #[test]
    fn test_two_separate_diag_groups() {
        // Diag(a,b) and Diag(c,d) - no connection
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);

        // Diag(a,b)
        uf.union(a, b);
        // Diag(c,d)
        uf.union(c, d);

        // a,b unified; c,d unified; but groups are separate
        assert_eq!(uf.find(a), uf.find(b));
        assert_eq!(uf.find(c), uf.find(d));
        assert_ne!(uf.find(a), uf.find(c));
    }

    #[test]
    fn test_diag_and_dense_mixed() {
        // Diag(i,j) * Dense(j,k) -> only i,j unified, k separate
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);

        // Diag(i,j)
        uf.union(i, j);
        // Dense(j,k) - no union (Dense doesn't unify axes)
        uf.make_set(k);

        assert_eq!(uf.find(i), uf.find(j));
        assert_ne!(uf.find(j), uf.find(k));
    }

    #[test]
    fn test_complex_network() {
        // Complex network:
        // Diag(a,b), Diag(b,c), Dense(c,d), Diag(d,e), Diag(e,f)
        // -> a,b,c unified; d,e,f unified; but the two groups are separate
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);
        let e = DynId(5);
        let f = DynId(6);

        // Diag(a,b)
        uf.union(a, b);
        // Diag(b,c)
        uf.union(b, c);
        // Dense(c,d) - no union
        uf.make_set(d);
        // Diag(d,e)
        uf.union(d, e);
        // Diag(e,f)
        uf.union(e, f);

        // First group: a,b,c
        let rep1 = uf.find(a);
        assert_eq!(uf.find(b), rep1);
        assert_eq!(uf.find(c), rep1);

        // Second group: d,e,f
        let rep2 = uf.find(d);
        assert_eq!(uf.find(e), rep2);
        assert_eq!(uf.find(f), rep2);

        // Groups are separate
        assert_ne!(rep1, rep2);
    }

    #[test]
    fn test_single_diag_tensor() {
        // Just one Diag(i,j)
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);

        uf.union(i, j);

        assert_eq!(uf.find(i), uf.find(j));
    }

    #[test]
    fn test_empty_union_find() {
        let mut uf = AxisUnionFind::new();

        // New ID should self-parent
        let x = DynId(42);
        uf.make_set(x);
        assert_eq!(uf.find(x), x);
    }

    #[test]
    fn test_idempotent_union() {
        // Union same pair multiple times
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);

        uf.union(a, b);
        let rep1 = uf.find(a);

        uf.union(a, b);
        let rep2 = uf.find(a);

        uf.union(b, a);
        let rep3 = uf.find(a);

        assert_eq!(rep1, rep2);
        assert_eq!(rep2, rep3);
    }

    #[test]
    fn test_self_union() {
        // Union element with itself
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        uf.union(a, a);

        assert_eq!(uf.find(a), a);
    }

    #[test]
    fn test_four_diag_tensors_chain() {
        // Diag(i,j) * Diag(j,k) * Diag(k,l) * Diag(l,m)
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);
        let m = DynId(5);

        uf.union(i, j); // Diag(i,j)
        uf.union(j, k); // Diag(j,k)
        uf.union(k, l); // Diag(k,l)
        uf.union(l, m); // Diag(l,m)

        // All should be unified
        let rep = uf.find(i);
        assert_eq!(uf.find(j), rep);
        assert_eq!(uf.find(k), rep);
        assert_eq!(uf.find(l), rep);
        assert_eq!(uf.find(m), rep);
    }

    #[test]
    fn test_diag_tensors_merge_two_chains() {
        // Two chains that merge:
        // Chain 1: Diag(a,b), Diag(b,c)
        // Chain 2: Diag(d,e), Diag(e,c)  <- shares 'c' with chain 1
        let mut uf = AxisUnionFind::new();

        let a = DynId(1);
        let b = DynId(2);
        let c = DynId(3);
        let d = DynId(4);
        let e = DynId(5);

        // Chain 1
        uf.union(a, b);
        uf.union(b, c);

        // Chain 2 (merges via c)
        uf.union(d, e);
        uf.union(e, c);

        // All should be unified
        let rep = uf.find(a);
        assert_eq!(uf.find(b), rep);
        assert_eq!(uf.find(c), rep);
        assert_eq!(uf.find(d), rep);
        assert_eq!(uf.find(e), rep);
    }

    #[test]
    fn test_remap_preserves_order() {
        let mut uf = AxisUnionFind::new();

        let i = DynId(1);
        let j = DynId(2);
        let k = DynId(3);
        let l = DynId(4);

        // Diag(i,j) and Diag(k,l) - two separate groups
        uf.union(i, j);
        uf.union(k, l);

        let ids = vec![i, j, k, l, i, k];
        let remapped = uf.remap_ids(&ids);

        // Check structure is preserved
        assert_eq!(remapped.len(), 6);
        assert_eq!(remapped[0], remapped[1]); // i,j same
        assert_eq!(remapped[2], remapped[3]); // k,l same
        assert_ne!(remapped[0], remapped[2]); // different groups
        assert_eq!(remapped[0], remapped[4]); // i appears again
        assert_eq!(remapped[2], remapped[5]); // k appears again
    }

    // ========================================================================
    // contract_multi_diag_aware tests
    // ========================================================================

    use crate::defaults::Index;
    use crate::storage::{DenseStorageC64, Storage};
    use num_complex::Complex64;
    use std::sync::Arc;

    fn make_dense_tensor(shape: &[usize], ids: &[u128]) -> TensorDynLen {
        let indices: Vec<DynIndex> = ids
            .iter()
            .zip(shape.iter())
            .map(|(&id, &dim)| Index::new(DynId(id), dim))
            .collect();
        let dims = shape.to_vec();
        let total_size: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total_size)
            .map(|i| Complex64::new((i + 1) as f64, 0.0))
            .collect();
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            data, &dims,
        )));
        TensorDynLen::new(indices, dims, storage)
    }

    #[test]
    fn test_contract_diag_aware_empty() {
        let tensors: Vec<&TensorDynLen> = vec![];
        let result = contract_multi_diag_aware(&tensors, AllowedPairs::All);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_diag_aware_single() {
        let tensor = make_dense_tensor(&[2, 3], &[1, 2]);
        let result = contract_multi_diag_aware(&[&tensor], AllowedPairs::All).unwrap();
        assert_eq!(result.dims, tensor.dims);
    }

    #[test]
    fn test_contract_diag_aware_pair_dense() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let result = contract_multi_diag_aware(&[&a, &b], AllowedPairs::All).unwrap();
        assert_eq!(result.dims, vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_diag_aware_three_dense() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_dense_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let result = contract_multi_diag_aware(&[&a, &b, &c], AllowedPairs::All).unwrap();
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
    }

    #[test]
    fn test_contract_diag_aware_disconnected_error() {
        // Disconnected graphs should error
        let a = make_dense_tensor(&[2, 3], &[1, 2]);
        let b = make_dense_tensor(&[4, 5], &[3, 4]); // No common indices
        let result = contract_multi_diag_aware(&[&a, &b], AllowedPairs::All);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().to_lowercase().contains("disconnected"));
    }
}
