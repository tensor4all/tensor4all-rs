//! Patch-wise addition and truncated addition on caller-owned scheduler lanes
//! (#1495, #1498).

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use hataori::{map_in_lanes, MapInError};
use hataori_tenferro::TenferroLanes;
use tensor4all_core::DynIndex;
use tensor4all_treetn::TruncationOptions;

use super::{
    disjoint_cover, ensure_center, ensure_refinement_within, ensure_same_dtype,
    ensure_same_tree_structure, ensure_union_disjoint, intersect_coverages, nonempty_templates,
    projector_covered_by, validate_truncation_options, PartitionedTreeTN,
    AUTOMATIC_REFINEMENT_LIMIT,
};
use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

#[derive(Eq, PartialEq)]
struct CanonicalProjector(Projector);

impl Ord for CanonicalProjector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.canonical_cmp(&other.0)
    }
}

impl PartialOrd for CanonicalProjector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Group sorted patch keys into deterministic lane items.
///
/// When `item_sites` is `None`, each patch forms its own item in canonical
/// sorted [`Projector`] order. When `item_sites` is `Some(sites)`, every site
/// in `sites` must be projected by every patch in `keys`; patches that agree on
/// the projected values of `sites` are grouped into one item, groups are
/// ordered canonically by their restricted [`Projector`] key (independent of
/// the order of `sites`), and patches within each group keep their canonical
/// sorted [`Projector`] order.
fn group_patch_keys(
    keys: Vec<Projector>,
    item_sites: Option<&[DynIndex]>,
) -> Result<Vec<Vec<Projector>>> {
    let Some(sites) = item_sites else {
        return Ok(keys.into_iter().map(|key| vec![key]).collect());
    };

    for site in sites {
        if keys.iter().any(|key| !key.is_projected_at(site)) {
            return Err(PartitionedTreeTNError::InvalidItemSites {
                index: site.clone(),
            });
        }
    }

    let mut groups: BTreeMap<CanonicalProjector, Vec<Projector>> = BTreeMap::new();
    for key in keys {
        let group_key = Projector::from_pairs(sites.iter().map(|site| {
            let coord = key
                .get(site)
                .expect("validated that every patch projects every site in item_sites");
            (site.clone(), coord)
        }))?;
        groups
            .entry(CanonicalProjector(group_key))
            .or_default()
            .push(key);
    }

    Ok(groups.into_values().collect())
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Add two partitions patch-wise on caller-owned scheduler lanes.
    ///
    /// This is the lanes counterpart of [`Self::add`]: it performs the same
    /// validation, automatic common refinement, and coverage intersection and
    /// clipping, then distributes the patch additions over `lanes`. Every lane
    /// runs its assigned item's patches sequentially through the same code path
    /// as the serial routine, with no `lanes.len() == 1` special case.
    ///
    /// When `item_sites` is `None`, each patch is one lane item in canonical
    /// sorted [`Projector`] order. When `item_sites` is `Some(sites)`, patches
    /// that agree on the projected coordinates of `sites` form one item and run
    /// sequentially on one lane in canonical [`Projector`] order; groups are
    /// ordered canonically by their restricted [`Projector`] key, independent
    /// of the order of `sites`.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand partition to add to `self`.
    /// * `item_sites` - Optional subset of projected sites carried by every
    ///   patch that defines one parallel item, or `None` for one patch per
    ///   item.
    /// * `lanes` - Caller-owned lane executor (`TenferroLanes`).
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before patch shortcuts,
    /// [`PartitionedTreeTNError::OverlappingProjectors`] for an operand whose
    /// own stored layout is invalid,
    /// [`PartitionedTreeTNError::RefinementTooLarge`] when the common refinement
    /// has more than [`AUTOMATIC_REFINEMENT_LIMIT`] atoms,
    /// [`PartitionedTreeTNError::InvalidItemSites`] when `item_sites` names a
    /// site outside the layout or not projected by every patch,
    /// [`PartitionedTreeTNError::LaneFanOut`] when the lane executor fails, or
    /// a typed TreeTN error for addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuEngine;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let patch = |values: Vec<f64>, coordinate: usize|
    ///     -> Result<_, Box<dyn std::error::Error>> {
    ///     let tree = TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], values)?],
    ///         vec![0usize],
    ///     )?;
    ///     Ok(SubDomainTreeTN::new(
    ///         tree,
    ///         Projector::from_pairs([(site.clone(), coordinate)])?,
    ///     )?)
    /// };
    /// let left = PartitionedTreeTN::from_subdomains(vec![
    ///     patch(vec![1.0, 0.0], 0)?,
    ///     patch(vec![0.0, 2.0], 1)?,
    /// ])?;
    /// let right = PartitionedTreeTN::from_subdomain(patch(vec![10.0, 0.0], 0)?)?;
    /// let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])?;
    ///
    /// let sum = left.add_on_lanes(&right, None, &mut lanes)?;
    /// assert_eq!(sum.len(), 2);
    /// assert_eq!(sum.to_treetn()?.to_dense()?.to_vec::<f64>()?, vec![11.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_on_lanes(
        &self,
        other: &Self,
        item_sites: Option<Vec<DynIndex>>,
        lanes: &mut TenferroLanes,
    ) -> Result<Self> {
        self.add_with_lanes(
            other,
            None,
            item_sites.as_deref(),
            AUTOMATIC_REFINEMENT_LIMIT,
            lanes,
        )
    }

    /// Add two partitions patch-wise and truncate every key present in both,
    /// on caller-owned scheduler lanes.
    ///
    /// This is the lanes counterpart of [`Self::add_truncated`]: it performs
    /// the same validation, automatic common refinement, and coverage
    /// intersection and clipping, then distributes the patch additions and
    /// truncations over `lanes`. Every lane runs its assigned item's patches
    /// sequentially through the same code path as the serial routine, with no
    /// `lanes.len() == 1` special case.
    ///
    /// See [`Self::add_on_lanes`] for the `item_sites` grouping contract.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand partition to add to `self`.
    /// * `center` - Node of the shared topology that each truncated key's
    ///   TreeTN is canonicalized towards.
    /// * `options` - Truncation applied to each key present in both operands.
    /// * `item_sites` - Optional subset of projected sites carried by every
    ///   patch that defines one parallel item, or `None` for one patch per
    ///   item.
    /// * `lanes` - Caller-owned lane executor (`TenferroLanes`).
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] for an invalid
    /// truncation option before any other check, the errors documented by
    /// [`Self::add_on_lanes`], [`PartitionedTreeTNError::InvalidCenter`] for a
    /// center absent from either operand's topology, or a typed TreeTN error
    /// for truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuEngine;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let k = DynIndex::new_dyn(2);
    /// let a = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![
    ///         IdxTensor::from_dense(vec![k.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 5.0])?,
    ///         IdxTensor::from_dense(vec![bond, a], vec![1.0_f64, 0.0, 0.0, 1.0])?,
    ///     ],
    ///     vec![0usize, 1],
    /// )?;
    /// let key = Projector::from_pairs([(k.clone(), 0)])?;
    /// let partition =
    ///     PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(tree, key.clone())?)?;
    /// let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])?;
    ///
    /// let truncated = partition.add_truncated_on_lanes(
    ///     &partition,
    ///     &0,
    ///     TruncationOptions::default().with_max_bond_dim(1),
    ///     Some(vec![k]),
    ///     &mut lanes,
    /// )?;
    /// assert_eq!(truncated.get(&key).ok_or("missing key")?.data().link_dims(), vec![1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_truncated_on_lanes(
        &self,
        other: &Self,
        center: &V,
        options: TruncationOptions,
        item_sites: Option<Vec<DynIndex>>,
        lanes: &mut TenferroLanes,
    ) -> Result<Self> {
        validate_truncation_options(&options)?;
        self.add_with_lanes(
            other,
            Some((center, options)),
            item_sites.as_deref(),
            AUTOMATIC_REFINEMENT_LIMIT,
            lanes,
        )
    }

    fn add_with_lanes(
        &self,
        other: &Self,
        truncation: Option<(&V, TruncationOptions)>,
        item_sites: Option<&[DynIndex]>,
        limit: usize,
        lanes: &mut TenferroLanes,
    ) -> Result<Self> {
        self.validate_contents()?;
        other.validate_contents()?;
        let (left_template, right_template) = nonempty_templates(self, other)?;
        ensure_same_tree_structure(left_template.data(), right_template.data())?;
        ensure_same_dtype(left_template.scalar_kind()?, right_template.scalar_kind()?)?;
        if let Some((center, _)) = truncation {
            ensure_center(left_template.data(), center)?;
            ensure_center(right_template.data(), center)?;
        }

        if ensure_union_disjoint(&self.data, &other.data).is_ok() {
            return self.add_patchwise_on_lanes(other, truncation, item_sites, lanes);
        }
        let atoms = Self::common_refinement(&[self, other])?;
        ensure_refinement_within(atoms.len(), limit)?;
        let left = self.refine_to(&atoms)?;
        let right = other.refine_to(&atoms)?;
        left.add_patchwise_on_lanes(&right, truncation, item_sites, lanes)
    }

    fn add_patchwise_on_lanes(
        &self,
        other: &Self,
        truncation: Option<(&V, TruncationOptions)>,
        item_sites: Option<&[DynIndex]>,
        lanes: &mut TenferroLanes,
    ) -> Result<Self> {
        ensure_union_disjoint(&self.data, &other.data)?;

        let result_coverage = intersect_coverages(self.coverage(), other.coverage());
        if let Some(ref cov) = result_coverage {
            let needs_clip = self
                .data
                .keys()
                .chain(other.data.keys())
                .any(|projector| !projector_covered_by(projector, cov));
            if needs_clip {
                let cells = disjoint_cover(cov);
                let left = self.refine_to(&cells)?;
                let right = other.refine_to(&cells)?;
                let mut summed =
                    left.add_patchwise_on_lanes(&right, truncation, item_sites, lanes)?;
                summed.coverage = result_coverage;
                return Ok(summed);
            }
        }

        let mut all_keys: Vec<Projector> = self.data.keys().cloned().collect();
        for key in other.data.keys() {
            if !self.data.contains_key(key) {
                all_keys.push(key.clone());
            }
        }
        all_keys.sort_by(Projector::canonical_cmp);

        let groups = group_patch_keys(all_keys, item_sites)?;
        let indexed_groups: Vec<(usize, Vec<Projector>)> = groups.into_iter().enumerate().collect();
        let errors: Mutex<Vec<Option<PartitionedTreeTNError>>> =
            Mutex::new((0..indexed_groups.len()).map(|_| None).collect());

        let outcome = map_in_lanes(lanes, indexed_groups, |(group_idx, keys), _lane| {
            let mut results: Vec<(Projector, SubDomainTreeTN<V>)> = Vec::with_capacity(keys.len());
            for projector in keys {
                let patch = match (self.data.get(&projector), other.data.get(&projector)) {
                    (Some(left), Some(right)) => {
                        let mut sum = match left.add(right) {
                            Ok(sum) => sum,
                            Err(err) => {
                                let msg = err.to_string();
                                if let Ok(mut slots) = errors.lock()
                                    && let Some(slot) = slots.get_mut(group_idx)
                                {
                                    *slot = Some(err);
                                }
                                return Err(msg);
                            }
                        };
                        if let Some((center, options)) = truncation
                            && let Err(err) = sum.truncate(center, options)
                        {
                            let msg = err.to_string();
                            if let Ok(mut slots) = errors.lock()
                                && let Some(slot) = slots.get_mut(group_idx)
                            {
                                *slot = Some(err);
                            }
                            return Err(msg);
                        }
                        sum
                    }
                    (Some(left), None) => left.clone(),
                    (None, Some(right)) => right.clone(),
                    (None, None) => unreachable!("key came from union of operand keys"),
                };
                results.push((projector, patch));
            }
            Ok(results)
        });

        let group_results = match outcome {
            Ok(results) => results,
            Err(MapInError::Callback(error)) => {
                let stored = errors
                    .into_inner()
                    .unwrap_or_default()
                    .into_iter()
                    .flatten()
                    .next();
                return match stored {
                    Some(err) => Err(err),
                    None => Err(PartitionedTreeTNError::LaneFanOut {
                        message: format!(
                            "a lane callback failed without a recorded error: {error}"
                        ),
                    }),
                };
            }
            Err(error) => {
                return Err(PartitionedTreeTNError::LaneFanOut {
                    message: format!("the lane executor failed: {error}"),
                });
            }
        };

        let mut data = HashMap::with_capacity(self.data.len() + other.data.len());
        for group in group_results {
            for (projector, patch) in group {
                data.insert(projector, patch);
            }
        }
        Ok(Self {
            data,
            coverage: result_coverage,
        })
    }
}
