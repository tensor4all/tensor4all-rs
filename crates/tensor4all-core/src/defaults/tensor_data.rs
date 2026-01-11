//! TensorData: Lazy tensor representation as outer product of storages.
//!
//! This module provides a lazy tensor representation that tracks outer products
//! and permutations without actually performing them until necessary.
//!
//! # Architecture
//!
//! ```text
//! TensorDynLen (high-level API with Index info)
//!     ↓
//! TensorData (outer product structure, lazy permutation)
//!     ↓
//! Storage (raw data, passed to backend)
//! ```

use std::sync::Arc;

use crate::defaults::DynId;
use crate::storage::Storage;

/// A component of a TensorData, representing a single Storage with its index mapping.
#[derive(Debug, Clone)]
pub struct TensorComponent {
    /// The underlying storage.
    pub storage: Arc<Storage>,
    /// Index IDs for each dimension of this storage (in storage memory order).
    pub index_ids: Vec<DynId>,
    /// Dimension sizes (same order as index_ids).
    pub dims: Vec<usize>,
}

impl TensorComponent {
    /// Create a new TensorComponent.
    pub fn new(storage: Arc<Storage>, index_ids: Vec<DynId>, dims: Vec<usize>) -> Self {
        debug_assert_eq!(
            index_ids.len(),
            dims.len(),
            "index_ids and dims must have the same length"
        );
        Self {
            storage,
            index_ids,
            dims,
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.index_ids.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Lazy tensor representation as an outer product of component storages.
///
/// TensorData tracks:
/// - Multiple component storages (from outer products)
/// - The external index order (which may differ from storage order due to permutations)
///
/// Operations like `outer_product` and `permute` are lazy - they only update
/// the structure without moving data. Actual computation happens during
/// `materialize` or `contraction`.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Component storages forming the outer product.
    pub components: Vec<TensorComponent>,
    /// External index IDs in current user-facing order.
    /// This is the order that will be used when materializing.
    pub external_index_ids: Vec<DynId>,
    /// Dimension for each external index (same order as external_index_ids).
    pub external_dims: Vec<usize>,
}

impl TensorData {
    /// Create a new TensorData from a single storage.
    pub fn new(storage: Arc<Storage>, index_ids: Vec<DynId>, dims: Vec<usize>) -> Self {
        let component = TensorComponent::new(storage, index_ids.clone(), dims.clone());
        Self {
            components: vec![component],
            external_index_ids: index_ids,
            external_dims: dims,
        }
    }

    /// Create TensorData from components with explicit external order.
    pub fn from_components(
        components: Vec<TensorComponent>,
        external_index_ids: Vec<DynId>,
        external_dims: Vec<usize>,
    ) -> Self {
        Self {
            components,
            external_index_ids,
            external_dims,
        }
    }

    /// Check if this is a simple tensor (single component, no permutation needed).
    pub fn is_simple(&self) -> bool {
        self.components.len() == 1 && self.components[0].index_ids == self.external_index_ids
    }

    /// Get the underlying storage if this is a simple tensor.
    pub fn storage(&self) -> Option<&Arc<Storage>> {
        if self.is_simple() {
            Some(&self.components[0].storage)
        } else {
            None
        }
    }

    /// Get the number of external dimensions.
    pub fn ndim(&self) -> usize {
        self.external_index_ids.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.external_dims.iter().product()
    }

    /// Get the external dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.external_dims
    }

    /// Get the external index IDs.
    pub fn index_ids(&self) -> &[DynId] {
        &self.external_index_ids
    }

    /// Compute outer product of two TensorData (lazy).
    ///
    /// This just concatenates the components and index lists without
    /// actually computing the outer product data.
    pub fn outer_product(a: &Self, b: &Self) -> Self {
        let mut components = a.components.clone();
        components.extend(b.components.iter().cloned());

        let mut external_index_ids = a.external_index_ids.clone();
        external_index_ids.extend(b.external_index_ids.iter().cloned());

        let mut external_dims = a.external_dims.clone();
        external_dims.extend(b.external_dims.iter().cloned());

        Self {
            components,
            external_index_ids,
            external_dims,
        }
    }

    /// Permute the external index order (lazy).
    ///
    /// This only updates the external_index_ids order without touching
    /// the underlying storage data.
    ///
    /// # Arguments
    /// * `new_order` - The new index order as a list of DynIds.
    ///                 Must be a permutation of external_index_ids.
    pub fn permute(&self, new_order: &[DynId]) -> Self {
        assert_eq!(
            new_order.len(),
            self.external_index_ids.len(),
            "new_order must have the same length as external_index_ids"
        );

        // Build the permutation and new dims
        let mut new_dims = Vec::with_capacity(new_order.len());
        for new_id in new_order {
            let pos = self
                .external_index_ids
                .iter()
                .position(|id| id == new_id)
                .expect("new_order must be a permutation of external_index_ids");
            new_dims.push(self.external_dims[pos]);
        }

        Self {
            components: self.components.clone(),
            external_index_ids: new_order.to_vec(),
            external_dims: new_dims,
        }
    }

    /// Permute using a permutation array.
    ///
    /// # Arguments
    /// * `perm` - Permutation array where `perm[i]` is the old position of
    ///            the index that should be at new position `i`.
    pub fn permute_by_perm(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.external_index_ids.len(),
            "perm must have the same length as external_index_ids"
        );

        let new_order: Vec<DynId> = perm
            .iter()
            .map(|&old_pos| self.external_index_ids[old_pos].clone())
            .collect();

        let new_dims: Vec<usize> = perm
            .iter()
            .map(|&old_pos| self.external_dims[old_pos])
            .collect();

        Self {
            components: self.components.clone(),
            external_index_ids: new_order,
            external_dims: new_dims,
        }
    }

    /// Materialize the tensor into a single Storage with the external index order.
    ///
    /// This contracts all components and permutes the result to match
    /// the external_index_ids order.
    ///
    /// # Returns
    /// A tuple of (Storage, dims) where dims matches external_dims.
    pub fn materialize(&self) -> anyhow::Result<(Arc<Storage>, Vec<usize>)> {
        use crate::defaults::{contract_multi, DynIndex, Index, TensorDynLen};
        use crate::tensor_like::TensorLike;

        if self.components.is_empty() {
            return Err(anyhow::anyhow!("Cannot materialize empty TensorData"));
        }

        // Convert components to TensorDynLen for contraction
        let tensors: Vec<TensorDynLen> = self
            .components
            .iter()
            .map(|comp| {
                let indices: Vec<DynIndex> = comp
                    .index_ids
                    .iter()
                    .zip(comp.dims.iter())
                    .map(|(id, &dim)| Index::new(id.clone(), dim))
                    .collect();
                TensorDynLen::new(indices, comp.dims.clone(), comp.storage.clone())
            })
            .collect();

        // Contract all components (outer product if no common indices)
        let contracted = contract_multi(&tensors)?;

        // Check if we need to permute to match external order
        let contracted_ids: Vec<DynId> = contracted
            .indices
            .iter()
            .map(|idx| idx.id.clone())
            .collect();

        if contracted_ids == self.external_index_ids {
            // Already in the right order
            Ok((contracted.storage().clone(), contracted.dims))
        } else {
            // Need to permute
            let target_indices: Vec<DynIndex> = self
                .external_index_ids
                .iter()
                .zip(self.external_dims.iter())
                .map(|(id, &dim)| Index::new(id.clone(), dim))
                .collect();

            let permuted = contracted.permuteinds(&target_indices)?;
            Ok((permuted.storage().clone(), permuted.dims))
        }
    }

    /// Get all components (for passing to contraction).
    pub fn into_components(self) -> Vec<TensorComponent> {
        self.components
    }

    /// Get a reference to all components.
    pub fn components(&self) -> &[TensorComponent] {
        &self.components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{DenseStorageF64, Storage};

    fn make_test_storage(data: Vec<f64>) -> Arc<Storage> {
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)))
    }

    fn new_id() -> DynId {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        DynId(COUNTER.fetch_add(1, Ordering::Relaxed) as u128)
    }

    #[test]
    fn test_tensor_data_simple() {
        let storage = make_test_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let id_i = new_id();
        let id_j = new_id();

        let data = TensorData::new(storage, vec![id_i.clone(), id_j.clone()], vec![2, 3]);

        assert!(data.is_simple());
        assert_eq!(data.ndim(), 2);
        assert_eq!(data.numel(), 6);
        assert_eq!(data.dims(), &[2, 3]);
    }

    #[test]
    fn test_outer_product() {
        let storage_a = make_test_storage(vec![1.0, 2.0]);
        let storage_b = make_test_storage(vec![3.0, 4.0, 5.0]);

        let id_i = new_id();
        let id_j = new_id();

        let a = TensorData::new(storage_a, vec![id_i.clone()], vec![2]);
        let b = TensorData::new(storage_b, vec![id_j.clone()], vec![3]);

        let c = TensorData::outer_product(&a, &b);

        assert!(!c.is_simple());
        assert_eq!(c.components.len(), 2);
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.numel(), 6);
        assert_eq!(c.external_index_ids, vec![id_i, id_j]);
        assert_eq!(c.external_dims, vec![2, 3]);
    }

    #[test]
    fn test_permute() {
        let storage = make_test_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let id_i = new_id();
        let id_j = new_id();

        let data = TensorData::new(storage, vec![id_i.clone(), id_j.clone()], vec![2, 3]);

        // Permute to [j, i]
        let permuted = data.permute(&[id_j.clone(), id_i.clone()]);

        // After permute, external order differs from storage order, so not simple
        assert!(!permuted.is_simple());
        assert_eq!(permuted.external_index_ids, vec![id_j, id_i]);
        assert_eq!(permuted.external_dims, vec![3, 2]);

        // Internal storage order unchanged
        assert_eq!(permuted.components[0].index_ids, data.components[0].index_ids);
    }

    #[test]
    fn test_permute_outer_product() {
        let storage_a = make_test_storage(vec![1.0, 2.0]);
        let storage_b = make_test_storage(vec![3.0, 4.0, 5.0]);

        let id_i = new_id();
        let id_j = new_id();

        let a = TensorData::new(storage_a, vec![id_i.clone()], vec![2]);
        let b = TensorData::new(storage_b, vec![id_j.clone()], vec![3]);

        let c = TensorData::outer_product(&a, &b);

        // Permute to [j, i]
        let permuted = c.permute(&[id_j.clone(), id_i.clone()]);

        assert_eq!(permuted.external_index_ids, vec![id_j, id_i]);
        assert_eq!(permuted.external_dims, vec![3, 2]);

        // Components unchanged
        assert_eq!(permuted.components.len(), 2);
        assert_eq!(permuted.components[0].index_ids, vec![id_i]);
        assert_eq!(permuted.components[1].index_ids, vec![id_j]);
    }

    #[test]
    fn test_materialize_simple() {
        let storage = make_test_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let id_i = new_id();
        let id_j = new_id();

        let data = TensorData::new(storage.clone(), vec![id_i.clone(), id_j.clone()], vec![2, 3]);

        let (materialized, dims) = data.materialize().unwrap();
        assert_eq!(dims, vec![2, 3]);

        // For a simple tensor, materialized storage should match original
        match (materialized.as_ref(), storage.as_ref()) {
            (Storage::DenseF64(m), Storage::DenseF64(orig)) => {
                assert_eq!(m.as_slice(), orig.as_slice());
            }
            _ => panic!("Expected DenseF64 storage"),
        }
    }

    #[test]
    fn test_materialize_outer_product() {
        // a = [1, 2], b = [3, 4, 5]
        // outer product a ⊗ b should be:
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
        // In row-major: [3, 4, 5, 6, 8, 10]
        let storage_a = make_test_storage(vec![1.0, 2.0]);
        let storage_b = make_test_storage(vec![3.0, 4.0, 5.0]);

        let id_i = new_id();
        let id_j = new_id();

        let a = TensorData::new(storage_a, vec![id_i.clone()], vec![2]);
        let b = TensorData::new(storage_b, vec![id_j.clone()], vec![3]);

        let c = TensorData::outer_product(&a, &b);

        let (materialized, dims) = c.materialize().unwrap();
        assert_eq!(dims, vec![2, 3]);

        match materialized.as_ref() {
            Storage::DenseF64(m) => {
                let data = m.as_slice();
                assert_eq!(data.len(), 6);
                // Expected: [3, 4, 5, 6, 8, 10]
                let expected = vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
                for (a, b) in data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
            _ => panic!("Expected DenseF64 storage"),
        }
    }

    #[test]
    fn test_materialize_with_permute() {
        // a = [1, 2], b = [3, 4, 5]
        // outer product a ⊗ b with indices [i, j] gives [[3, 4, 5], [6, 8, 10]]
        // After permute to [j, i], should be transposed:
        // [[3, 6], [4, 8], [5, 10]] in row-major: [3, 6, 4, 8, 5, 10]
        let storage_a = make_test_storage(vec![1.0, 2.0]);
        let storage_b = make_test_storage(vec![3.0, 4.0, 5.0]);

        let id_i = new_id();
        let id_j = new_id();

        let a = TensorData::new(storage_a, vec![id_i.clone()], vec![2]);
        let b = TensorData::new(storage_b, vec![id_j.clone()], vec![3]);

        let c = TensorData::outer_product(&a, &b);
        let permuted = c.permute(&[id_j.clone(), id_i.clone()]);

        let (materialized, dims) = permuted.materialize().unwrap();
        assert_eq!(dims, vec![3, 2]);

        match materialized.as_ref() {
            Storage::DenseF64(m) => {
                let data = m.as_slice();
                assert_eq!(data.len(), 6);
                // Expected after transpose: [3, 6, 4, 8, 5, 10]
                let expected = vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0];
                for (a, b) in data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
            _ => panic!("Expected DenseF64 storage"),
        }
    }
}
