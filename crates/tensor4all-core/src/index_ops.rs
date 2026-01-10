use crate::IndexLike;

/// Error type for index replacement operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplaceIndsError {
    /// The symmetry space of the replacement index does not match the original.
    SpaceMismatch {
        /// The dimension/size of the original index
        from_dim: usize,
        /// The dimension/size of the replacement index
        to_dim: usize,
    },
    /// Duplicate indices found in the collection.
    DuplicateIndices {
        /// The position of the first duplicate index
        first_pos: usize,
        /// The position of the duplicate index
        duplicate_pos: usize,
    },
}

impl std::fmt::Display for ReplaceIndsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplaceIndsError::SpaceMismatch { from_dim, to_dim } => {
                write!(
                    f,
                    "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                    from_dim, to_dim
                )
            }
            ReplaceIndsError::DuplicateIndices {
                first_pos,
                duplicate_pos,
            } => {
                write!(
                    f,
                    "Duplicate indices found: index at position {} has the same ID as index at position {}",
                    duplicate_pos, first_pos
                )
            }
        }
    }
}

impl std::error::Error for ReplaceIndsError {}

/// Check if a collection of indices contains any duplicates (by ID).
///
/// # Arguments
/// * `indices` - Collection of indices to check
///
/// # Returns
/// `Ok(())` if all indices are unique, or `Err(ReplaceIndsError::DuplicateIndices)` if duplicates are found.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::check_unique_indices;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let indices = vec![i.clone(), j.clone()];
/// assert!(check_unique_indices(&indices).is_ok());
///
/// let duplicate = vec![i.clone(), i.clone()];
/// assert!(check_unique_indices(&duplicate).is_err());
/// ```
pub fn check_unique_indices<I: IndexLike>(indices: &[I]) -> Result<(), ReplaceIndsError> {
    use std::collections::HashMap;
    let mut seen: HashMap<&I::Id, usize> = HashMap::with_capacity(indices.len());
    for (pos, idx) in indices.iter().enumerate() {
        if let Some(&first_pos) = seen.get(idx.id()) {
            return Err(ReplaceIndsError::DuplicateIndices {
                first_pos,
                duplicate_pos: pos,
            });
        }
        seen.insert(idx.id(), pos);
    }
    Ok(())
}

/// Replace indices in a collection based on ID matching.
///
/// This corresponds to ITensors.jl's `replaceinds` function. It replaces indices
/// in `indices` that match (by ID) any of the `(old, new)` pairs in `replacements`.
/// The replacement index must have the same dimension as the original.
///
/// # Arguments
/// * `indices` - Collection of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices matching `old_index.id` are replaced with `new_index`
///
/// # Returns
/// A new vector with replacements applied, or an error if any replacement has a dimension mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different dimension than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replaceinds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);  // Same size as j
///
/// let indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// let replaced = replaceinds(indices, &replacements).unwrap();
/// assert_eq!(replaced.len(), 3);
/// assert_eq!(replaced[1].id, new_j.id);
/// ```
pub fn replaceinds<I: IndexLike>(
    indices: Vec<I>,
    replacements: &[(I, I)],
) -> Result<Vec<I>, ReplaceIndsError> {
    // Check for duplicates in input indices
    check_unique_indices(&indices)?;

    // Build a map from old ID to new index for fast lookup
    let mut replacement_map = std::collections::HashMap::with_capacity(replacements.len());
    for (old, new) in replacements {
        // Validate dimension match
        if old.dim() != new.dim() {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.dim(),
                to_dim: new.dim(),
            });
        }
        replacement_map.insert(old.id(), new);
    }

    // Apply replacements
    let mut result = Vec::with_capacity(indices.len());
    for idx in indices {
        if let Some(new_idx) = replacement_map.get(idx.id()) {
            result.push((*new_idx).clone());
        } else {
            result.push(idx);
        }
    }

    // Check for duplicates in result indices
    check_unique_indices(&result)?;
    Ok(result)
}

/// Replace indices in-place based on ID matching.
///
/// This is an in-place variant of `replaceinds` that modifies the input slice directly.
/// Useful for performance-critical code where you want to avoid allocations.
///
/// # Arguments
/// * `indices` - Mutable slice of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices matching `old_index.id` are replaced with `new_index`
///
/// # Returns
/// `Ok(())` on success, or an error if any replacement has a dimension mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different dimension than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replaceinds_in_place;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);
///
/// let mut indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// replaceinds_in_place(&mut indices, &replacements).unwrap();
/// assert_eq!(indices[1].id, new_j.id);
/// ```
pub fn replaceinds_in_place<I: IndexLike>(
    indices: &mut [I],
    replacements: &[(I, I)],
) -> Result<(), ReplaceIndsError> {
    // Check for duplicates in input indices
    check_unique_indices(indices)?;

    // Build a map from old ID to new index for fast lookup
    let mut replacement_map = std::collections::HashMap::with_capacity(replacements.len());
    for (old, new) in replacements {
        // Validate dimension match
        if old.dim() != new.dim() {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.dim(),
                to_dim: new.dim(),
            });
        }
        replacement_map.insert(old.id(), new);
    }

    // Apply replacements in-place
    for idx in indices.iter_mut() {
        if let Some(new_idx) = replacement_map.get(idx.id()) {
            *idx = (*new_idx).clone();
        }
    }

    // Check for duplicates in result indices
    check_unique_indices(indices)?;
    Ok(())
}

/// Find indices that are unique to the first collection (set difference A \ B).
///
/// Returns indices that appear in `indices_a` but not in `indices_b` (matched by ID).
/// This corresponds to ITensors.jl's `uniqueinds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices from `indices_a` that are not in `indices_b`.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::unique_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let unique = unique_inds(&indices_a, &indices_b);
/// assert_eq!(unique.len(), 1);
/// assert_eq!(unique[0].id, i.id);
/// ```
pub fn unique_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| idx.id()).collect();
    indices_a
        .iter()
        .filter(|idx| !b_ids.contains(idx.id()))
        .cloned()
        .collect()
}

/// Find indices that are not common between two collections (symmetric difference).
///
/// Returns indices that appear in either `indices_a` or `indices_b` but not in both
/// (matched by ID). This corresponds to ITensors.jl's `noncommoninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices from both collections that are not common to both.
/// Order: indices from A first (in original order), then indices from B (in original order).
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::noncommon_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let noncommon = noncommon_inds(&indices_a, &indices_b);
/// assert_eq!(noncommon.len(), 2);  // i and k
/// ```
pub fn noncommon_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    let a_ids: std::collections::HashSet<_> = indices_a.iter().map(|idx| idx.id()).collect();
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| idx.id()).collect();

    // Pre-allocate with estimated capacity (worst case: no common indices)
    let mut result = Vec::with_capacity(indices_a.len() + indices_b.len());

    // Add indices from A that are not in B
    result.extend(
        indices_a
            .iter()
            .filter(|idx| !b_ids.contains(idx.id()))
            .cloned(),
    );
    // Add indices from B that are not in A
    result.extend(
        indices_b
            .iter()
            .filter(|idx| !a_ids.contains(idx.id()))
            .cloned(),
    );
    result
}

/// Find the union of two index collections.
///
/// Returns all unique indices from both collections (matched by ID).
/// This corresponds to ITensors.jl's `unioninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing all unique indices from both collections.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::union_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let union = union_inds(&indices_a, &indices_b);
/// assert_eq!(union.len(), 3);  // i, j, k
/// ```
pub fn union_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    let mut seen: std::collections::HashSet<&I::Id> =
        std::collections::HashSet::with_capacity(indices_a.len() + indices_b.len());
    let mut result = Vec::with_capacity(indices_a.len() + indices_b.len());

    for idx in indices_a {
        if seen.insert(idx.id()) {
            result.push(idx.clone());
        }
    }
    for idx in indices_b {
        if seen.insert(idx.id()) {
            result.push(idx.clone());
        }
    }
    result
}

/// Check if a collection contains a specific index (by ID).
///
/// This corresponds to ITensors.jl's `hasind` function.
///
/// # Arguments
/// * `indices` - Collection of indices to search
/// * `index` - The index to look for
///
/// # Returns
/// `true` if an index with matching ID is found, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hasind;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let indices = vec![i.clone(), j.clone()];
///
/// assert!(hasind(&indices, &i));
/// assert!(!hasind(&indices, &Index::new_dyn(4)));
/// ```
pub fn hasind<I: IndexLike>(indices: &[I], index: &I) -> bool {
    indices.iter().any(|idx| idx.id() == index.id())
}

/// Check if a collection contains all of the specified indices (by ID).
///
/// This corresponds to ITensors.jl's `hasinds` function.
///
/// # Arguments
/// * `indices` - Collection of indices to search
/// * `targets` - The indices to look for
///
/// # Returns
/// `true` if all target indices (by ID) are found, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hasinds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let indices = vec![i.clone(), j.clone(), k.clone()];
///
/// assert!(hasinds(&indices, &[i.clone(), j.clone()]));
/// assert!(!hasinds(&indices, &[i.clone(), Index::new_dyn(5)]));
/// ```
pub fn hasinds<I: IndexLike>(indices: &[I], targets: &[I]) -> bool {
    let index_ids: std::collections::HashSet<_> = indices.iter().map(|idx| idx.id()).collect();
    targets.iter().all(|target| index_ids.contains(target.id()))
}

/// Check if two collections have any common indices (by ID).
///
/// This corresponds to ITensors.jl's `hascommoninds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// `true` if there is at least one common index (by ID), `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hascommoninds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// assert!(hascommoninds(&indices_a, &indices_b));
/// assert!(!hascommoninds(&[i.clone()], &[k.clone()]));
/// ```
pub fn hascommoninds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> bool {
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| idx.id()).collect();
    indices_a.iter().any(|idx| b_ids.contains(idx.id()))
}

/// Find common indices between two index collections.
///
/// Returns a vector of indices that appear in both `indices_a` and `indices_b`
/// (set intersection). This is similar to ITensors.jl's `commoninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices that are common to both collections (matched by ID).
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::common_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let common = common_inds(&indices_a, &indices_b);
/// assert_eq!(common.len(), 1);
/// assert_eq!(common[0].id, j.id);
/// ```
pub fn common_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| idx.id()).collect();
    indices_a
        .iter()
        .filter(|idx| b_ids.contains(idx.id()))
        .cloned()
        .collect()
}
