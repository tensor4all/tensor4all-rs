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
    indices.iter().any(|idx| idx == index)
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

/// Find common indices between two slices and return their positions.
///
/// Returns a vector of `(pos_a, pos_b)` tuples where each tuple indicates
/// that `indices_a[pos_a]` and `indices_b[pos_b]` have the same ID.
///
/// # Example
/// ```
/// use tensor4all_core::index::DefaultIndex as Index;
/// use tensor4all_core::index_ops::common_ind_positions;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let positions = common_ind_positions(&indices_a, &indices_b);
/// assert_eq!(positions, vec![(1, 0)]); // j is at position 1 in a, position 0 in b
/// ```
pub fn common_ind_positions<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for (pos_a, idx_a) in indices_a.iter().enumerate() {
        for (pos_b, idx_b) in indices_b.iter().enumerate() {
            if idx_a.id() == idx_b.id() {
                positions.push((pos_a, pos_b));
                break; // Each index in a can match at most one in b
            }
        }
    }
    positions
}

/// Result of preparing a tensor contraction.
///
/// Contains all the information needed to perform the contraction:
/// - Which axes to contract from each tensor
/// - The resulting indices and dimensions after contraction
#[derive(Debug, Clone)]
pub struct ContractionSpec<I: IndexLike> {
    /// Axes to contract from the first tensor (positions in indices_a)
    pub axes_a: Vec<usize>,
    /// Axes to contract from the second tensor (positions in indices_b)
    pub axes_b: Vec<usize>,
    /// Indices of the result tensor (non-contracted indices from both tensors)
    pub result_indices: Vec<I>,
    /// Dimensions of the result tensor
    pub result_dims: Vec<usize>,
}

/// Error type for contraction preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionError {
    /// No common indices found for contraction
    NoCommonIndices,
    /// Dimension mismatch for a common index
    DimensionMismatch {
        pos_a: usize,
        pos_b: usize,
        dim_a: usize,
        dim_b: usize,
    },
    /// Duplicate axis specified in contraction
    DuplicateAxis { tensor: &'static str, pos: usize },
    /// Index not found in tensor
    IndexNotFound { tensor: &'static str },
    /// Batch contraction not yet implemented
    BatchContractionNotImplemented,
}

impl std::fmt::Display for ContractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractionError::NoCommonIndices => {
                write!(f, "No common indices found for contraction")
            }
            ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a,
                dim_b,
            } => {
                write!(
                    f,
                    "Dimension mismatch: tensor_a[{}]={} != tensor_b[{}]={}",
                    pos_a, dim_a, pos_b, dim_b
                )
            }
            ContractionError::DuplicateAxis { tensor, pos } => {
                write!(f, "Duplicate axis {} in {} tensor", pos, tensor)
            }
            ContractionError::IndexNotFound { tensor } => {
                write!(f, "Index not found in {} tensor", tensor)
            }
            ContractionError::BatchContractionNotImplemented => {
                write!(f, "Batch contraction not yet implemented")
            }
        }
    }
}

impl std::error::Error for ContractionError {}

/// Prepare contraction data for two tensors that share common indices.
///
/// This function finds common indices and computes the axes to contract
/// and the resulting indices/dimensions.
///
/// # Example
/// ```
/// use tensor4all_core::index::DefaultIndex as Index;
/// use tensor4all_core::index_ops::prepare_contraction;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let dims_a = vec![2, 3];
/// let indices_b = vec![j.clone(), k.clone()];
/// let dims_b = vec![3, 4];
///
/// let spec = prepare_contraction(&indices_a, &dims_a, &indices_b, &dims_b).unwrap();
/// assert_eq!(spec.axes_a, vec![1]);  // j is at position 1 in a
/// assert_eq!(spec.axes_b, vec![0]);  // j is at position 0 in b
/// assert_eq!(spec.result_dims, vec![2, 4]);  // [i, k]
/// ```
pub fn prepare_contraction<I: IndexLike>(
    indices_a: &[I],
    dims_a: &[usize],
    indices_b: &[I],
    dims_b: &[usize],
) -> Result<ContractionSpec<I>, ContractionError> {
    // Find common indices and their positions
    let positions = common_ind_positions(indices_a, indices_b);
    if positions.is_empty() {
        return Err(ContractionError::NoCommonIndices);
    }

    let (axes_a, axes_b): (Vec<_>, Vec<_>) = positions.iter().copied().unzip();

    // Verify dimensions match
    for &(pos_a, pos_b) in &positions {
        if dims_a[pos_a] != dims_b[pos_b] {
            return Err(ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a: dims_a[pos_a],
                dim_b: dims_b[pos_b],
            });
        }
    }

    // Build result indices and dimensions (non-contracted indices)
    let mut result_indices = Vec::new();
    let mut result_dims = Vec::new();

    for (i, idx) in indices_a.iter().enumerate() {
        if !axes_a.contains(&i) {
            result_indices.push(idx.clone());
            result_dims.push(dims_a[i]);
        }
    }

    for (i, idx) in indices_b.iter().enumerate() {
        if !axes_b.contains(&i) {
            result_indices.push(idx.clone());
            result_dims.push(dims_b[i]);
        }
    }

    Ok(ContractionSpec {
        axes_a,
        axes_b,
        result_indices,
        result_dims,
    })
}

/// Prepare contraction data for explicit index pairs (like tensordot).
///
/// Unlike `prepare_contraction`, this function takes explicit pairs of indices
/// to contract, allowing contraction of indices with different IDs.
///
/// # Example
/// ```
/// use tensor4all_core::index::DefaultIndex as Index;
/// use tensor4all_core::index_ops::prepare_contraction_pairs;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(3);  // Same dim as j but different ID
/// let l = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let dims_a = vec![2, 3];
/// let indices_b = vec![k.clone(), l.clone()];
/// let dims_b = vec![3, 4];
///
/// // Contract j with k
/// let spec = prepare_contraction_pairs(
///     &indices_a, &dims_a,
///     &indices_b, &dims_b,
///     &[(j.clone(), k.clone())]
/// ).unwrap();
/// assert_eq!(spec.axes_a, vec![1]);
/// assert_eq!(spec.axes_b, vec![0]);
/// assert_eq!(spec.result_dims, vec![2, 4]);
/// ```
pub fn prepare_contraction_pairs<I: IndexLike>(
    indices_a: &[I],
    dims_a: &[usize],
    indices_b: &[I],
    dims_b: &[usize],
    pairs: &[(I, I)],
) -> Result<ContractionSpec<I>, ContractionError> {
    use std::collections::HashSet;

    if pairs.is_empty() {
        return Err(ContractionError::NoCommonIndices);
    }

    // Check for batch contraction (common indices not in pairs)
    let contracted_a_ids: HashSet<_> = pairs.iter().map(|(idx, _)| idx.id()).collect();
    let contracted_b_ids: HashSet<_> = pairs.iter().map(|(_, idx)| idx.id()).collect();

    let common_positions = common_ind_positions(indices_a, indices_b);
    for (pos_a, pos_b) in &common_positions {
        let id_a = indices_a[*pos_a].id();
        let id_b = indices_b[*pos_b].id();
        if !contracted_a_ids.contains(id_a) || !contracted_b_ids.contains(id_b) {
            return Err(ContractionError::BatchContractionNotImplemented);
        }
    }

    // Find positions and validate
    let mut axes_a = Vec::new();
    let mut axes_b = Vec::new();

    for (idx_a, idx_b) in pairs {
        let pos_a = indices_a
            .iter()
            .position(|idx| idx.id() == idx_a.id())
            .ok_or(ContractionError::IndexNotFound { tensor: "self" })?;

        let pos_b = indices_b
            .iter()
            .position(|idx| idx.id() == idx_b.id())
            .ok_or(ContractionError::IndexNotFound { tensor: "other" })?;

        // Verify dimensions match
        if dims_a[pos_a] != dims_b[pos_b] {
            return Err(ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a: dims_a[pos_a],
                dim_b: dims_b[pos_b],
            });
        }

        // Check for duplicate axes
        if axes_a.contains(&pos_a) {
            return Err(ContractionError::DuplicateAxis {
                tensor: "self",
                pos: pos_a,
            });
        }
        if axes_b.contains(&pos_b) {
            return Err(ContractionError::DuplicateAxis {
                tensor: "other",
                pos: pos_b,
            });
        }

        axes_a.push(pos_a);
        axes_b.push(pos_b);
    }

    // Build result indices and dimensions
    let mut result_indices = Vec::new();
    let mut result_dims = Vec::new();

    for (i, idx) in indices_a.iter().enumerate() {
        if !axes_a.contains(&i) {
            result_indices.push(idx.clone());
            result_dims.push(dims_a[i]);
        }
    }

    for (i, idx) in indices_b.iter().enumerate() {
        if !axes_b.contains(&i) {
            result_indices.push(idx.clone());
            result_dims.push(dims_b[i]);
        }
    }

    Ok(ContractionSpec {
        axes_a,
        axes_b,
        result_indices,
        result_dims,
    })
}
