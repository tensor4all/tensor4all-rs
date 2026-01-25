//! Projector type for defining subdomains in tensor index space
//!
//! A projector maps tensor indices (DynIndex) to fixed values, defining a subdomain
//! where specific indices are fixed to particular values.

use std::collections::HashMap;
use tensor4all_core::DynIndex;

/// A projector maps tensor indices to fixed integer values.
///
/// Used to define subdomains in the tensor index space. Each index
/// can be projected to a specific value (0-indexed).
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::Projector;
/// use tensor4all_core::index::{Index, DynId};
///
/// let idx0 = Index::new_dyn(2);
/// let idx1 = Index::new_dyn(3);
///
/// // Create a projector that fixes idx0 to value 1
/// let p = Projector::from_pairs([(idx0.clone(), 1)]);
///
/// assert!(p.is_projected_at(&idx0));
/// assert!(!p.is_projected_at(&idx1));
/// assert_eq!(p.get(&idx0), Some(1));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Projector {
    /// Maps index -> projected value (0-indexed)
    data: HashMap<DynIndex, usize>,
}

impl Projector {
    /// Create an empty projector (no indices projected)
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a projector from pairs of (index, projected_value)
    pub fn from_pairs(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Self {
        Self {
            data: pairs.into_iter().collect(),
        }
    }

    /// Check if an index is projected
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.data.contains_key(index)
    }

    /// Get the projected value at an index (None if not projected)
    pub fn get(&self, index: &DynIndex) -> Option<usize> {
        self.data.get(index).copied()
    }

    /// Get all projected indices
    pub fn projected_indices(&self) -> impl Iterator<Item = &DynIndex> {
        self.data.keys()
    }

    /// Number of projected indices
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if no indices are projected
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the internal data as a reference
    pub fn data(&self) -> &HashMap<DynIndex, usize> {
        &self.data
    }

    /// Insert or update a projection
    pub fn insert(&mut self, index: DynIndex, value: usize) {
        self.data.insert(index, value);
    }

    /// Remove a projection
    pub fn remove(&mut self, index: &DynIndex) -> Option<usize> {
        self.data.remove(index)
    }

    /// Intersection: returns Some(merged) if projectors are compatible, None if conflicting.
    ///
    /// Two projectors are compatible if they don't have conflicting values at the same index.
    /// The result contains all projections from both projectors.
    ///
    /// This corresponds to Julia's `a & b` operator.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let mut result = self.data.clone();

        for (index, &value) in &other.data {
            if let Some(&existing) = result.get(index) {
                if existing != value {
                    // Conflict: same index has different values
                    return None;
                }
            } else {
                result.insert(index.clone(), value);
            }
        }

        Some(Self { data: result })
    }

    /// Union: keeps only indices where both projectors agree.
    ///
    /// The result contains only indices that are projected in both projectors
    /// with the same value.
    ///
    /// This corresponds to Julia's `a | b` operator.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = HashMap::new();

        for (index, &value) in &self.data {
            if other.get(index) == Some(value) {
                result.insert(index.clone(), value);
            }
        }

        Self { data: result }
    }

    /// Check if two projectors have overlapping (compatible) regions.
    ///
    /// Returns true if the projectors can be merged without conflict.
    pub fn has_overlap(&self, other: &Self) -> bool {
        self.intersection(other).is_some()
    }

    /// Check if self is a subset of other.
    ///
    /// `a.is_subset_of(b)` means that `a` is more restrictive than `b`:
    /// `a` projects at least as many indices as `b`, and all indices projected
    /// in `b` have the same values in `a`.
    ///
    /// This corresponds to Julia's `a < b` (where a < b means a is more restrictive).
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // All indices in other must be in self with the same value
        for (index, &value) in &other.data {
            match self.data.get(index) {
                Some(&v) if v == value => continue,
                _ => return false,
            }
        }

        // self must project at least as many indices as other
        self.data.len() >= other.data.len()
    }

    /// Check if a vector of projectors are mutually disjoint (non-overlapping).
    ///
    /// Returns true if no two projectors have overlapping regions.
    pub fn are_disjoint(projectors: &[Self]) -> bool {
        for (i, a) in projectors.iter().enumerate() {
            for b in projectors.iter().skip(i + 1) {
                if a.has_overlap(b) {
                    return false;
                }
            }
        }
        true
    }

    /// Filter projector to only include indices from the given set.
    ///
    /// Returns a new projector containing only the indices that are
    /// present in both this projector and the given index set.
    pub fn filter_indices(&self, indices: &[DynIndex]) -> Self {
        let index_set: std::collections::HashSet<_> = indices.iter().collect();
        Self {
            data: self
                .data
                .iter()
                .filter(|(k, _)| index_set.contains(k))
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }
}

impl PartialEq for Projector {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Projector {}

impl std::hash::Hash for Projector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Sort entries by index ID for consistent hashing
        let mut entries: Vec<_> = self.data.iter().collect();
        entries.sort_by_key(|(k, _)| k.id);
        for (k, v) in entries {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl PartialOrd for Projector {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.is_subset_of(other) {
            Some(std::cmp::Ordering::Less)
        } else if other.is_subset_of(self) {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}

impl FromIterator<(DynIndex, usize)> for Projector {
    fn from_iter<I: IntoIterator<Item = (DynIndex, usize)>>(iter: I) -> Self {
        Self::from_pairs(iter)
    }
}

impl<'a> IntoIterator for &'a Projector {
    type Item = (&'a DynIndex, &'a usize);
    type IntoIter = std::collections::hash_map::Iter<'a, DynIndex, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl IntoIterator for Projector {
    type Item = (DynIndex, usize);
    type IntoIter = std::collections::hash_map::IntoIter<DynIndex, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::index::Index;

    fn make_index(size: usize) -> DynIndex {
        Index::new_dyn(size)
    }

    #[test]
    fn test_projector_new() {
        let p = Projector::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_projector_from_pairs() {
        let idx0 = make_index(2);
        let idx1 = make_index(3);
        let idx2 = make_index(4);

        let p = Projector::from_pairs([(idx0.clone(), 1), (idx2.clone(), 3)]);
        assert_eq!(p.len(), 2);
        assert!(p.is_projected_at(&idx0));
        assert!(!p.is_projected_at(&idx1));
        assert!(p.is_projected_at(&idx2));
        assert_eq!(p.get(&idx0), Some(1));
        assert_eq!(p.get(&idx1), None);
        assert_eq!(p.get(&idx2), Some(3));
    }

    #[test]
    fn test_projector_intersection_compatible() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);
        let idx2 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let b = Projector::from_pairs([(idx1.clone(), 0), (idx2.clone(), 1)]);

        let merged = a.intersection(&b).unwrap();
        assert_eq!(merged.len(), 3);
        assert_eq!(merged.get(&idx0), Some(1));
        assert_eq!(merged.get(&idx1), Some(0));
        assert_eq!(merged.get(&idx2), Some(1));
    }

    #[test]
    fn test_projector_intersection_conflict() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let b = Projector::from_pairs([(idx1.clone(), 1)]); // Conflict at idx1

        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_projector_union() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);
        let idx2 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let b = Projector::from_pairs([(idx1.clone(), 0), (idx2.clone(), 1)]);

        let u = a.union(&b);
        assert_eq!(u.len(), 1);
        assert!(!u.is_projected_at(&idx0));
        assert!(u.is_projected_at(&idx1));
        assert_eq!(u.get(&idx1), Some(0));
        assert!(!u.is_projected_at(&idx2));
    }

    #[test]
    fn test_projector_has_overlap() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1)]);
        let b = Projector::from_pairs([(idx1.clone(), 0)]);
        let c = Projector::from_pairs([(idx0.clone(), 0)]); // Different value at same index

        assert!(a.has_overlap(&b)); // No common indices, compatible
        assert!(!a.has_overlap(&c)); // Same index, different values
    }

    #[test]
    fn test_projector_is_subset_of() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);
        let idx2 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);
        let b = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let c = Projector::from_pairs([(idx0.clone(), 1)]);

        assert!(a.is_subset_of(&b)); // a projects more indices
        assert!(a.is_subset_of(&c));
        assert!(b.is_subset_of(&c));
        assert!(!b.is_subset_of(&a));
        assert!(!c.is_subset_of(&a));
    }

    #[test]
    fn test_projector_are_disjoint() {
        let idx0 = make_index(2);

        // Disjoint projectors: different values at same index
        let p1 = Projector::from_pairs([(idx0.clone(), 0)]);
        let p2 = Projector::from_pairs([(idx0.clone(), 1)]);

        assert!(Projector::are_disjoint(&[p1.clone(), p2.clone()]));

        // Non-disjoint: same projection
        let p3 = Projector::from_pairs([(idx0.clone(), 0)]);
        assert!(!Projector::are_disjoint(&[p1, p3]));
    }

    #[test]
    fn test_projector_partial_ord() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let b = Projector::from_pairs([(idx0.clone(), 1)]);
        let c = Projector::from_pairs([(idx0.clone(), 0)]); // Incompatible with a and b

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.partial_cmp(&c), None);
        assert_eq!(b.partial_cmp(&c), None);
    }

    #[test]
    fn test_projector_iteration() {
        let idx0 = make_index(2);
        let idx1 = make_index(3);

        let p = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 2)]);

        let pairs: Vec<_> = p.into_iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_projector_equality_and_hash() {
        use std::collections::HashSet;

        let idx0 = make_index(2);
        let idx1 = make_index(2);

        let a = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0)]);
        let b = Projector::from_pairs([(idx1.clone(), 0), (idx0.clone(), 1)]); // Same content
        let c = Projector::from_pairs([(idx0.clone(), 1)]);

        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn test_projector_filter_indices() {
        let idx0 = make_index(2);
        let idx1 = make_index(2);
        let idx2 = make_index(2);

        let p = Projector::from_pairs([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);

        let filtered = p.filter_indices(&[idx0.clone(), idx2.clone()]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.is_projected_at(&idx0));
        assert!(!filtered.is_projected_at(&idx1));
        assert!(filtered.is_projected_at(&idx2));
    }
}
