//! Point evaluation of a partition.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{AnyScalar, ColMajorArrayRef, DynIndex};

use super::PartitionedTreeTN;
use crate::error::{PartitionedTreeTNError, Result};
use crate::subdomain_tree_tn::{ScalarKind, SubDomainTreeTN};

/// Patch ids of one group, keyed by the values the group's rows are fixed to.
type PatchLookup = HashMap<Vec<usize>, usize>;

/// Finds the patch of a partition that covers a point, by hashing.
///
/// Patches are grouped by the rows (site positions) their projectors fix;
/// within a group the fixed values identify the patch. A point is looked up
/// once per group, hashing as many coordinates as the group fixes, until a
/// group matches; the patches are disjoint, so at most one does. The cost per
/// point is therefore at most the total size of the distinct fixed-site
/// sets, independent of how many patches share a set: a layout whose patches
/// all fix the same sites costs one lookup, while an irregular layout can
/// have as many sets as patches.
pub(crate) struct PatchLocator<'a, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    patches: Vec<&'a SubDomainTreeTN<V>>,
    groups: Vec<(Vec<usize>, PatchLookup)>,
}

impl<'a, V> PatchLocator<'a, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// A locator over the patches of `partition` in canonical order, with
    /// `positions` mapping each site index to its row in a point.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::SiteIndexMismatch`] when a projector
    /// fixes an index absent from `positions`.
    pub(crate) fn new(
        partition: &'a PartitionedTreeTN<V>,
        positions: &HashMap<&DynIndex, usize>,
    ) -> Result<Self> {
        Self::from_patches(partition.ordered_patches(), positions)
    }

    fn from_patches(
        patches: Vec<&'a SubDomainTreeTN<V>>,
        positions: &HashMap<&DynIndex, usize>,
    ) -> Result<Self> {
        let mut groups: HashMap<Vec<usize>, PatchLookup> = HashMap::new();
        for (patch_id, patch) in patches.iter().enumerate() {
            let mut entries: Vec<(usize, usize)> = patch
                .projector()
                .iter()
                .map(|(index, &value)| {
                    positions
                        .get(index)
                        .map(|&row| (row, value))
                        .ok_or(PartitionedTreeTNError::SiteIndexMismatch)
                })
                .collect::<Result<_>>()?;
            entries.sort_unstable();
            let (rows_fixed, fixed_values): (Vec<usize>, Vec<usize>) = entries.into_iter().unzip();
            groups
                .entry(rows_fixed)
                .or_default()
                .insert(fixed_values, patch_id);
        }
        let mut groups: Vec<(Vec<usize>, PatchLookup)> = groups.into_iter().collect();
        groups.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(Self { patches, groups })
    }

    /// The id of the patch covering the point `column` (one value per row),
    /// or `None` when no patch covers it.
    pub(crate) fn owner(&self, column: &[usize]) -> Option<usize> {
        let mut key = Vec::new();
        self.groups.iter().find_map(|(rows_fixed, lookup)| {
            key.clear();
            key.extend(rows_fixed.iter().map(|&row| column[row]));
            lookup.get(&key).copied()
        })
    }

    /// The patch with id `patch_id`.
    pub(crate) fn patch(&self, patch_id: usize) -> &'a SubDomainTreeTN<V> {
        self.patches[patch_id]
    }
}

fn invalid(reason: &'static str) -> PartitionedTreeTNError {
    PartitionedTreeTNError::InvalidOptions {
        operation: "PartitionedTreeTN::evaluate_batched",
        reason,
    }
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Evaluate the partition at many points.
    ///
    /// Each point is evaluated by the one patch whose projector it satisfies,
    /// through that patch's [`TreeTN::evaluate`](tensor4all_treetn::TreeTN::evaluate).
    /// A point that no patch covers evaluates to zero, because a partition
    /// treats an absent patch as zero. The patches are disjoint, so no point
    /// is covered twice.
    ///
    /// The owning patch of a point is found by hashing: patches are grouped by
    /// the set of sites their projectors fix, and a point is looked up once per
    /// group until one matches. The cost per point is at most the total size
    /// of the distinct fixed-site sets, independent of how many patches share
    /// a set; an irregular layout can have as many sets as patches. Points of
    /// one patch are evaluated together with one evaluator.
    ///
    /// # Arguments
    ///
    /// * `indices` - Every site index of the partition exactly once, in the
    ///   order of the rows of `values`; the dimension must match the
    ///   partition's.
    /// * `values` - Column-major array of shape `[indices.len(), n_points]`;
    ///   `values.get(&[i, p])` is the value of `indices[i]` at point `p`, below
    ///   that index's dimension.
    ///
    /// # Returns
    ///
    /// One value per point, in point order: real ([`AnyScalar::new_real`]
    /// representation) for a real partition, complex for a complex one. An
    /// empty partition has no dtype and no site space: it checks `values`
    /// against `indices` (shape, repeated indices, bounds), not against a site
    /// space, and returns real zeros.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] when `values` is not
    /// two-dimensional, its row count differs from `indices.len()`, `indices`
    /// repeats an index, or a value is out of bounds (checked for every point,
    /// covered or not); [`PartitionedTreeTNError::SiteIndexMismatch`] when
    /// `indices` is not the partition's site space or a dimension differs; and
    /// [`PartitionedTreeTNError::TreeTN`] when a patch evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// // f(x, y) = 10 x + y on a 2 x 3 grid, stored only on the patch x = 1.
    /// let (x, y) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    /// let values: Vec<f64> = (0..6).map(|i| (10 * (i % 2) + i / 2) as f64).collect();
    /// let tensor = IdxTensor::from_dense(vec![x.clone(), y.clone()], values)?;
    /// let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
    /// let patch = SubDomainTreeTN::new(tree, Projector::from_pairs([(x.clone(), 1)])?)?;
    /// let partition = PartitionedTreeTN::from_subdomain(patch)?;
    ///
    /// // Rows follow `indices` (here y before x); the point x = 0 is not
    /// // covered by any patch and evaluates to zero.
    /// let points = [2usize, 1, 0, 0];
    /// let shape = [2usize, 2];
    /// let result = partition
    ///     .evaluate_batched(&[y, x], ColMajorArrayRef::new(&points, &shape)?)?;
    /// assert_eq!(result[0].real(), 12.0);
    /// assert_eq!(result[1].real(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate_batched(
        &self,
        indices: &[DynIndex],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>> {
        let &[rows, n_points] = values.shape() else {
            return Err(invalid("values must be a two-dimensional array"));
        };
        if rows != indices.len() {
            return Err(invalid(
                "values must have one row per index (shape [indices.len(), n_points])",
            ));
        }
        let mut positions: HashMap<&DynIndex, usize> = HashMap::with_capacity(rows);
        for (row, index) in indices.iter().enumerate() {
            if positions.insert(index, row).is_some() {
                return Err(invalid("indices must list every site index exactly once"));
            }
        }
        let data = values.data();
        for (row, index) in indices.iter().enumerate() {
            if (0..n_points).any(|point| data[row + point * rows] >= index.dim) {
                return Err(invalid("a point value is out of bounds for its site index"));
            }
        }

        let patches: Vec<&SubDomainTreeTN<V>> = self.ordered_patches();
        let Some(template) = patches.first() else {
            return Ok(vec![AnyScalar::new_real(0.0); n_points]);
        };
        let (sites, _) = template.data().all_site_indices()?;
        if sites.len() != rows {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
        for site in &sites {
            match positions.get(site) {
                Some(&row) if indices[row].dim == site.dim => {}
                _ => return Err(PartitionedTreeTNError::SiteIndexMismatch),
            }
        }

        let locator = PatchLocator::from_patches(patches.clone(), &positions)?;
        let mut owned: Vec<Vec<usize>> = vec![Vec::new(); patches.len()];
        for point in 0..n_points {
            if let Some(patch_id) = locator.owner(&data[point * rows..(point + 1) * rows]) {
                owned[patch_id].push(point);
            }
        }

        let zero = match template.scalar_kind()? {
            Some(ScalarKind::C32 | ScalarKind::C64) => AnyScalar::new_complex(0.0, 0.0),
            _ => AnyScalar::new_real(0.0),
        };
        let mut result = vec![zero; n_points];
        let mut columns = Vec::new();
        for (patch, points) in patches.iter().zip(&owned) {
            if points.is_empty() {
                continue;
            }
            columns.clear();
            for &point in points {
                columns.extend_from_slice(&data[point * rows..(point + 1) * rows]);
            }
            let shape = [rows, points.len()];
            let batch = ColMajorArrayRef::new(&columns, &shape).map_err(|source| {
                PartitionedTreeTNError::TensorConstruction {
                    source: anyhow::Error::new(source),
                }
            })?;
            let patch_values = patch.data().evaluator(indices)?.evaluate_batched(batch)?;
            for (&point, value) in points.iter().zip(patch_values) {
                result[point] = value;
            }
        }
        Ok(result)
    }
}
