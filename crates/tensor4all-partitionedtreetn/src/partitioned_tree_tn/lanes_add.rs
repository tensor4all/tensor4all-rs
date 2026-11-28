//! Patch-wise addition and truncated addition on caller-owned scheduler lanes
//! (#1495, #1498).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use hataori::{map_in_lanes, MapInError};
use hataori_tenferro::TenferroLanes;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{TreeTN, TruncationOptions};

use super::{
    disjoint_cover, ensure_center, ensure_refinement_within, ensure_same_dtype,
    ensure_same_tree_structure, ensure_union_disjoint, intersect_coverages, nonempty_templates,
    projector_covered_by, validate_truncation_options, PartitionedTreeTN,
    AUTOMATIC_REFINEMENT_LIMIT,
};
use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

type SitePositions<V> = HashMap<DynIndex, (V, usize, usize)>;
type LogicalKey<V> = Vec<(V, usize, usize)>;

/// Map every site index in `template` to its logical `(node_name, axis, dim)`
/// position, independent of `DynIndex` allocation IDs.
fn template_site_positions<V>(template: &TreeTN<IdxTensor, V>) -> SitePositions<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut names = template.node_names();
    names.sort();
    let mut positions = HashMap::new();
    for name in names {
        for (axis, index) in template
            .node_site_indices(&name)
            .unwrap_or_default()
            .into_iter()
            .enumerate()
        {
            positions.insert(index.clone(), (name.clone(), axis, index.dim));
        }
    }
    positions
}

/// Validate `item_sites` against the shared template site space and return the
/// deduplicated `(node_name, axis, index)` entries in logical `(node_name, axis)`
/// order.
fn resolve_item_sites<V>(
    site_positions: &SitePositions<V>,
    item_sites: &[DynIndex],
) -> Result<Vec<(V, usize, DynIndex)>>
where
    V: Clone + Ord,
{
    let mut resolved: Vec<(V, usize, DynIndex)> = Vec::with_capacity(item_sites.len());
    for site in item_sites {
        let Some((node, axis, template_dim)) = site_positions.get(site) else {
            return Err(PartitionedTreeTNError::InvalidItemSites {
                index: site.clone(),
            });
        };
        if *template_dim != site.dim {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
        resolved.push((node.clone(), *axis, site.clone()));
    }
    resolved.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
    resolved.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    Ok(resolved)
}

/// Logical ordering key for a patch projector: sorted `(node_name, axis, coord)`
/// tuples in the shared template topology, independent of `DynIndex` allocation IDs.
fn patch_logical_key<V>(projector: &Projector, site_positions: &SitePositions<V>) -> LogicalKey<V>
where
    V: Clone + Ord,
{
    let mut entries: LogicalKey<V> = projector
        .iter()
        .filter_map(|(index, &coord)| {
            site_positions
                .get(index)
                .map(|(node, axis, _)| (node.clone(), *axis, coord))
        })
        .collect();
    entries.sort_by(|a, b| (&a.0, a.1, a.2).cmp(&(&b.0, b.1, b.2)));
    entries
}

/// Group patch keys into deterministic lane items using logical
/// `(node_name, axis, coord)` keys from `template`.
///
/// - When `item_sites` is `None`, each patch forms its own item, ordered by its
///   logical `(node_name, axis, coord)` key.
/// - When `item_sites` is `Some(sites)`, every site in `sites` must belong to
///   `template` with matching dimension and be projected by every patch in
///   `keys`. Patches that agree on the projected coordinates of `sites` form
///   one item; groups are ordered by their sorted `(node_name, axis, coord)`
///   key (independent of `sites` order and `DynIndex` allocation IDs), and
///   patches within each group keep their logical `(node_name, axis, coord)`
///   order.
fn group_patch_keys<V>(
    site_positions: &SitePositions<V>,
    mut keys: Vec<Projector>,
    item_sites: Option<&[DynIndex]>,
) -> Result<Vec<Vec<Projector>>>
where
    V: Clone + Ord,
{
    keys.sort_by_key(|key| patch_logical_key(key, site_positions));

    let Some(sites) = item_sites else {
        return Ok(keys.into_iter().map(|key| vec![key]).collect());
    };

    let resolved = resolve_item_sites(site_positions, sites)?;

    let mut groups: BTreeMap<LogicalKey<V>, Vec<Projector>> = BTreeMap::new();
    for key in keys {
        let mut group_key: LogicalKey<V> = Vec::with_capacity(resolved.len());
        for (node, axis, site) in &resolved {
            let Some(coord) = key.get(site) else {
                return Err(PartitionedTreeTNError::InvalidItemSites {
                    index: site.clone(),
                });
            };
            group_key.push((node.clone(), *axis, coord));
        }
        groups.entry(group_key).or_default().push(key);
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
    /// # Grouping and deterministic order
    ///
    /// - When `item_sites` is `None`, each patch is one lane item, ordered by
    ///   its logical `(node_name, axis, coord)` key in the shared topology
    ///   (independent of `DynIndex` allocation IDs and insertion order).
    /// - When `item_sites` is `Some(sites)`, patches that agree on the
    ///   projected coordinates of `sites` form one item and run sequentially on
    ///   one lane. Groups are enumerated in deterministic order sorted by their
    ///   `(node_name, axis, coord)` tuple on `sites` (independent of `sites`
    ///   slice order, insertion order, and `DynIndex` allocation IDs), and
    ///   patches within each item keep the logical `(node_name, axis, coord)`
    ///   patch order.
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
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] (including when an
    /// `item_sites` index has a dimension different from the partition's site
    /// space), or [`PartitionedTreeTNError::DTypeMismatch`] before patch
    /// shortcuts, [`PartitionedTreeTNError::OverlappingProjectors`] for an
    /// operand whose own stored layout is invalid,
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
    /// See [`Self::add_on_lanes`] for the `item_sites` grouping and
    /// deterministic ordering contract.
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

        let site_positions = template_site_positions(left_template.data());

        // Compute the unclipped target projector set purely from projector
        // metadata on the caller thread; tensor projections run inside
        // `map_in_lanes` on each lane's explicit `Session`.
        let unclipped_keys: Vec<Projector> =
            if ensure_union_disjoint(&self.data, &other.data).is_ok() {
                let mut keys: Vec<Projector> = self.data.keys().cloned().collect();
                for key in other.data.keys() {
                    if !self.data.contains_key(key) {
                        keys.push(key.clone());
                    }
                }
                keys
            } else {
                let atoms = Self::common_refinement(&[self, other])?;
                ensure_refinement_within(atoms.len(), limit)?;
                atoms
            };

        if let Some(sites) = item_sites {
            // Validate item_sites against the template site space and against
            // every unclipped patch up front so invalid or unprojected sites
            // are rejected even when coverage clipping empties the active set.
            let resolved = resolve_item_sites(&site_positions, sites)?;
            for (_, _, site) in &resolved {
                if unclipped_keys.iter().any(|key| !key.is_projected_at(site)) {
                    return Err(PartitionedTreeTNError::InvalidItemSites {
                        index: site.clone(),
                    });
                }
            }
        }

        let result_coverage = intersect_coverages(self.coverage(), other.coverage());
        let target_keys: Vec<Projector> = if let Some(ref cov) = result_coverage {
            let needs_clip = unclipped_keys
                .iter()
                .any(|projector| !projector_covered_by(projector, cov));
            if needs_clip {
                let cells = disjoint_cover(cov);
                let mut clipped: Vec<Projector> = Vec::new();
                for key in &unclipped_keys {
                    for cell in &cells {
                        if let Some(inter) = key.intersection(cell)
                            && !clipped.contains(&inter)
                        {
                            clipped.push(inter);
                        }
                    }
                }
                clipped
            } else {
                unclipped_keys
            }
        } else {
            unclipped_keys
        };

        let groups = group_patch_keys(&site_positions, target_keys, item_sites)?;
        let indexed_groups: Vec<(usize, Vec<Projector>)> = groups.into_iter().enumerate().collect();
        let errors: Mutex<Vec<Option<PartitionedTreeTNError>>> =
            Mutex::new((0..indexed_groups.len()).map(|_| None).collect());

        let outcome = map_in_lanes(lanes, indexed_groups, |(group_idx, keys), lane| {
            let mut session = tensor4all_tensorbackend::explicit::Session::new(lane);
            let mut results: Vec<(Projector, SubDomainTreeTN<V>)> = Vec::with_capacity(keys.len());
            let record_err = |err: PartitionedTreeTNError| -> String {
                let msg = err.to_string();
                if let Ok(mut slots) = errors.lock()
                    && let Some(slot) = slots.get_mut(group_idx)
                {
                    *slot = Some(err);
                }
                msg
            };
            let resolve_patch = |partition: &Self,
                                 projector: &Projector,
                                 session: &mut tensor4all_tensorbackend::explicit::Session<'_>|
             -> Result<Option<SubDomainTreeTN<V>>> {
                if let Some(exact) = partition.data.get(projector) {
                    return Ok(Some(exact.clone()));
                }
                for src in partition.data.values() {
                    if src.projector().intersection(projector).is_some() {
                        return src.project_on_session(projector, session);
                    }
                }
                Ok(None)
            };
            for projector in keys {
                let left_patch = match resolve_patch(self, &projector, &mut session) {
                    Ok(patch) => patch,
                    Err(err) => return Err(record_err(err)),
                };
                let right_patch = match resolve_patch(other, &projector, &mut session) {
                    Ok(patch) => patch,
                    Err(err) => return Err(record_err(err)),
                };
                let patch = match (left_patch, right_patch) {
                    (Some(left), Some(right)) => {
                        let mut sum = match left.add_on_session(&right, &mut session) {
                            Ok(sum) => sum,
                            Err(err) => return Err(record_err(err)),
                        };
                        if let Some((center, options)) = truncation
                            && let Err(err) = sum.truncate_on_session(center, options, &mut session)
                        {
                            return Err(record_err(err));
                        }
                        sum
                    }
                    (Some(left), None) => left,
                    (None, Some(right)) => right,
                    (None, None) => continue,
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
