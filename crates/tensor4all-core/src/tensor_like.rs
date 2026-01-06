//! TensorLike trait for unifying tensor types.
//!
//! This trait provides a common interface for:
//! - `TensorDynLen<Id, Symm>` - Dense tensors
//! - `TreeTN<Id, Symm, V>` - Tree tensor networks
//! - Third-party tensor-like objects
//!
//! The trait exposes:
//! - **External indices**: Physical/site indices of the object
//! - **Explicit contraction (tensordot)**: Binary contraction with specified index pairs

use crate::tensor::TensorDynLen;
use anyhow::Result;
use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use crate::index::{Index, Symmetry};
use crate::tagset::TagSetLike;

/// Trait for tensor-like objects that expose external indices and support contraction.
///
/// This is the primary abstraction for treating both dense tensors (`TensorDynLen`)
/// and tensor networks (`TreeTN`) through a common interface.
///
/// # Design Principles
///
/// - **Minimal interface**: Only external indices and explicit contraction
/// - **Object-safe**: Uses `Vec` returns instead of iterators for trait object compatibility
/// - **Clonable trait objects**: Uses `dyn-clone` for `Box<dyn TensorLike<...>>` cloneability
/// - **Stable ordering**: `external_indices()` returns indices in deterministic order
///
/// # Associated Types
///
/// - `Id`: Index identity type (e.g., `DynId` for runtime identity)
/// - `Symm`: Symmetry type (e.g., `NoSymmSpace` for no symmetry)
/// - `Tags`: Tag type (e.g., `DefaultTagSet`)
///
/// # Example
///
/// ```ignore
/// use tensor4all::TensorLike;
///
/// fn contract_external<T: TensorLike>(a: &T, b: &T) -> Result<TensorDynLen<T::Id, T::Symm>> {
///     // Get common external indices and contract
///     let pairs = find_common_indices(a.external_indices(), b.external_indices());
///     a.tensordot(b, &pairs)
/// }
/// ```
pub trait TensorLike: DynClone + Send + Sync + Debug {
    /// Index identity type.
    type Id: Clone + Hash + Eq + Debug + Send + Sync;

    /// Symmetry type.
    type Symm: Clone + Symmetry + Send + Sync;

    /// Tag type.
    type Tags: Clone + TagSetLike + Send + Sync;

    /// Return flattened external indices for this object.
    ///
    /// - For `TensorDynLen`: returns the tensor's indices
    /// - For `TreeTN`: returns union of all site/physical indices across nodes
    ///
    /// # Ordering
    ///
    /// The ordering MUST be stable (deterministic). Implementations should:
    /// - Sort indices by their `id` field, or
    /// - Use insertion-ordered storage
    ///
    /// This ensures consistent behavior for hashing, serialization, and comparison.
    fn external_indices(&self) -> Vec<Index<Self::Id, Self::Symm, Self::Tags>>;

    /// Number of external indices.
    ///
    /// Default implementation calls `external_indices().len()`, but implementations
    /// SHOULD override this for efficiency when the count can be computed without
    /// allocating the full index list.
    fn num_external_indices(&self) -> usize {
        self.external_indices().len()
    }

    /// Convert this object to a dense tensor.
    ///
    /// - For `TensorDynLen`: returns a clone of self
    /// - For `TreeTN`: contracts all nodes to produce a single tensor
    ///
    /// This method is required for implementing `tensordot` via trait objects,
    /// since we need access to the underlying tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails (e.g., empty TreeTN).
    fn to_tensor(&self) -> Result<TensorDynLen<Self::Id, Self::Symm>>;

    /// Return `self` as `Any` for optional downcasting / runtime type inspection.
    ///
    /// This allows callers to attempt downcasting a trait object back to its
    /// concrete type when needed (similar to C++'s `dynamic_cast`).
    ///
    /// # Implementation
    ///
    /// Implementers should simply return `self`:
    ///
    /// ```ignore
    /// fn as_any(&self) -> &dyn Any { self }
    /// ```
    ///
    /// This requires the concrete type to be `'static` (the usual `Any` constraint).
    ///
    /// # Usage
    ///
    /// ```ignore
    /// let tensor_like: &dyn TensorLike<...> = &my_tensor;
    /// if let Some(tensor) = tensor_like.as_any().downcast_ref::<TensorDynLen<DynId>>() {
    ///     // Use tensor directly
    /// }
    /// ```
    fn as_any(&self) -> &dyn Any;

    /// Replace an index in this tensor-like object.
    ///
    /// This replaces the index matching `old_index` by ID with `new_index`.
    /// The storage data is not modified, only the index metadata is changed.
    ///
    /// # Arguments
    ///
    /// * `old_index` - The index to replace (matched by ID)
    /// * `new_index` - The new index to use
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` with the index replaced.
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts to `TensorDynLen` via `to_tensor()`
    /// and then uses `TensorDynLen::replaceind`. Implementations may override
    /// for better performance (e.g., TreeTN could replace directly in nodes).
    fn replaceind(
        &self,
        old_index: &Index<Self::Id, Self::Symm, Self::Tags>,
        new_index: &Index<Self::Id, Self::Symm, Self::Tags>,
    ) -> Result<TensorDynLen<Self::Id, Self::Symm>> {
        let tensor = self.to_tensor()?;
        // Convert from Index<Id, Symm, Tags> to Index<Id, Symm> for TensorDynLen
        let old_idx = Index::new(old_index.id.clone(), old_index.symm.clone());
        let new_idx = Index::new(new_index.id.clone(), new_index.symm.clone());
        Ok(tensor.replaceind(&old_idx, &new_idx))
    }

    /// Replace multiple indices in this tensor-like object.
    ///
    /// This replaces each index in `old_indices` (matched by ID) with the
    /// corresponding index in `new_indices`. The storage data is not modified.
    ///
    /// # Arguments
    ///
    /// * `old_indices` - The indices to replace (matched by ID)
    /// * `new_indices` - The new indices to use
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` with the indices replaced.
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts to `TensorDynLen` via `to_tensor()`
    /// and then uses `TensorDynLen::replaceinds`. Implementations may override
    /// for better performance.
    fn replaceinds(
        &self,
        old_indices: &[Index<Self::Id, Self::Symm, Self::Tags>],
        new_indices: &[Index<Self::Id, Self::Symm, Self::Tags>],
    ) -> Result<TensorDynLen<Self::Id, Self::Symm>> {
        let tensor = self.to_tensor()?;
        // Convert from Index<Id, Symm, Tags> to Index<Id, Symm> for TensorDynLen
        let old_inds: Vec<_> = old_indices
            .iter()
            .map(|idx| Index::new(idx.id.clone(), idx.symm.clone()))
            .collect();
        let new_inds: Vec<_> = new_indices
            .iter()
            .map(|idx| Index::new(idx.id.clone(), idx.symm.clone()))
            .collect();
        Ok(tensor.replaceinds(&old_inds, &new_inds))
    }

    /// Explicit contraction between two tensor-like objects.
    ///
    /// This performs binary contraction over the specified index pairs.
    /// Each pair `(idx_self, idx_other)` specifies:
    /// - An index from `self` to contract
    /// - An index from `other` to contract with
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor-like object to contract with
    /// * `pairs` - List of (self_index, other_index) pairs to contract
    ///
    /// # Returns
    ///
    /// A new `TensorDynLen` representing the contracted result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `pairs` is empty
    /// - An index in `pairs` doesn't exist in the corresponding object
    /// - Index dimensions don't match
    /// - There are common indices not in `pairs` (ambiguous contraction)
    ///
    /// # Default Implementation
    ///
    /// The default implementation converts both operands to `TensorDynLen` via
    /// `to_tensor()` and then uses `TensorDynLen::tensordot`. Implementations
    /// may override this for better performance.
    fn tensordot(
        &self,
        other: &dyn TensorLike<Id = Self::Id, Symm = Self::Symm, Tags = Self::Tags>,
        pairs: &[(
            Index<Self::Id, Self::Symm, Self::Tags>,
            Index<Self::Id, Self::Symm, Self::Tags>,
        )],
    ) -> Result<TensorDynLen<Self::Id, Self::Symm>>
    where
        Self::Id: Ord,
    {
        // Convert both operands to TensorDynLen
        let self_tensor = self.to_tensor()?;
        let other_tensor = other.to_tensor()?;

        // Convert pairs to the format expected by TensorDynLen::tensordot
        let converted_pairs: Vec<(Index<Self::Id, Self::Symm>, Index<Self::Id, Self::Symm>)> = pairs
            .iter()
            .map(|(a, b)| {
                (
                    Index::new(a.id.clone(), a.symm.clone()),
                    Index::new(b.id.clone(), b.symm.clone()),
                )
            })
            .collect();

        // Use TensorDynLen's tensordot
        self_tensor.tensordot(&other_tensor, &converted_pairs)
    }
}

// Make trait objects cloneable
dyn_clone::clone_trait_object!(<Id, Symm, Tags> TensorLike<Id=Id, Symm=Symm, Tags=Tags> where
    Id: Clone + Hash + Eq + Debug + Send + Sync,
    Symm: Clone + Symmetry + Send + Sync,
    Tags: Clone + TagSetLike + Send + Sync,
);

// ============================================================================
// Helper methods on trait objects for downcasting
// ============================================================================

/// Extension trait for downcasting `dyn TensorLike` trait objects.
///
/// This provides convenient methods for runtime type checking and downcasting.
pub trait TensorLikeDowncast {
    /// Check if the underlying type is `T`.
    fn is<T: 'static>(&self) -> bool;

    /// Attempt to downcast to a reference of type `T`.
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
}

impl<Id, Symm, Tags> TensorLikeDowncast for dyn TensorLike<Id = Id, Symm = Symm, Tags = Tags>
where
    Id: Clone + Hash + Eq + Debug + Send + Sync,
    Symm: Clone + Symmetry + Send + Sync,
    Tags: Clone + TagSetLike + Send + Sync,
{
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

impl<Id, Symm, Tags> TensorLikeDowncast for dyn TensorLike<Id = Id, Symm = Symm, Tags = Tags> + Send
where
    Id: Clone + Hash + Eq + Debug + Send + Sync,
    Symm: Clone + Symmetry + Send + Sync,
    Tags: Clone + TagSetLike + Send + Sync,
{
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

impl<Id, Symm, Tags> TensorLikeDowncast for dyn TensorLike<Id = Id, Symm = Symm, Tags = Tags> + Send + Sync
where
    Id: Clone + Hash + Eq + Debug + Send + Sync,
    Symm: Clone + Symmetry + Send + Sync,
    Tags: Clone + TagSetLike + Send + Sync,
{
    fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

// ============================================================================
// Implementation for TensorDynLen
// ============================================================================

impl<Id, Symm> TensorLike for TensorDynLen<Id, Symm>
where
    Id: Clone + Hash + Eq + Debug + Send + Sync + 'static,
    Symm: Clone + Symmetry + Debug + Send + Sync + 'static,
{
    type Id = Id;
    type Symm = Symm;
    type Tags = crate::DefaultTagSet;

    fn external_indices(&self) -> Vec<Index<Self::Id, Self::Symm, Self::Tags>> {
        // For TensorDynLen, all indices are external.
        // Convert from Index<Id, Symm> to Index<Id, Symm, DefaultTagSet> by adding default tags.
        self.indices
            .iter()
            .map(|idx| Index::new_with_tags(
                idx.id.clone(),
                idx.symm.clone(),
                crate::DefaultTagSet::default(),
            ))
            .collect()
    }

    fn num_external_indices(&self) -> usize {
        self.indices.len()
    }

    fn to_tensor(&self) -> Result<TensorDynLen<Self::Id, Self::Symm>> {
        // TensorDynLen is already a tensor, just clone it
        Ok(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    // Use the default implementation of tensordot which calls to_tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile-time check that the trait is object-safe
    fn _assert_object_safe<Id, Symm, Tags>()
    where
        Id: Clone + Hash + Eq + Debug + Send + Sync + 'static,
        Symm: Clone + Symmetry + Send + Sync + 'static,
        Tags: Clone + TagSetLike + Send + Sync + 'static,
    {
        fn _takes_trait_object(
            _obj: &dyn TensorLike<Id = (), Symm = (), Tags = ()>,
        ) {
            // This won't compile if TensorLike is not object-safe
        }
    }
}
