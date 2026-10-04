//! The validated site layout of a patched interpolation problem.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArray, DynIndex, IndexLike};
use tensor4all_treetn::interpolation::{validate_layout, InterpolationError, InterpolationProblem};
use tensor4all_treetn::NodeNameNetwork;

use super::sampling::point_list_capacity;
use super::{invalid, PatchedInterpolationError, PatchedInterpolationOptions};
use crate::{ErrorNorm, L2Reference, Projector};

/// The validated layout of the problem.
pub(super) struct SiteLayout<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    pub(super) topology: NodeNameNetwork<V>,
    pub(super) node_sites: BTreeMap<V, Vec<DynIndex>>,
    /// Every node with the positions of its sites in the site order.
    pub(super) node_positions: Vec<(V, Vec<usize>)>,
    /// The derived site order.
    pub(super) sites: Vec<DynIndex>,
    pub(super) dims: Vec<usize>,
    /// Node of every site.
    pub(super) site_nodes: Vec<V>,
    pub(super) edges: Vec<(V, V)>,
    /// Positions fixed by successive splits.
    pub(super) split_order: Vec<usize>,
    /// `|X|` as `f64`; finite under `ErrorNorm::L2`, possibly infinite
    /// under `SampledMax`.
    pub(super) domain_points: f64,
}

impl<V> SiteLayout<V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    /// Validate every input in the documented order, before any evaluation.
    pub(super) fn validated(
        topology: NodeNameNetwork<V>,
        node_sites: BTreeMap<V, Vec<DynIndex>>,
        initial_pivots: &ColMajorArray<usize>,
        options: &PatchedInterpolationOptions,
    ) -> Result<Self, PatchedInterpolationError> {
        validate_layout(&topology, &node_sites).map_err(|error| match error {
            InterpolationError::InvalidProblem { message } => invalid(message),
            other => invalid(other.to_string()),
        })?;

        let sites = InterpolationProblem::derive_site_order(&node_sites);
        let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
        let split_order = resolve_patch_order(&options.patch_order, &sites)?;
        validate_options(options)?;
        if point_list_capacity(dims.len(), options.verification.samples).is_none() {
            return Err(invalid(format!(
                "verification.samples ({}) times the number of sites ({}) exceeds the \
                 representable point-list capacity",
                options.verification.samples,
                dims.len()
            )));
        }
        validate_initial_pivots(initial_pivots, &dims)?;
        let domain_points: f64 = dims.iter().map(|&dim| dim as f64).product();
        if let ErrorNorm::L2 { reference, .. } = options.error_norm {
            if !domain_points.is_finite() {
                return Err(invalid(format!(
                    "the domain has more points than f64 can represent (a product of {} site \
                     dimensions); ErrorNorm::L2 needs a finite point count; use \
                     ErrorNorm::sampled_max() for this domain",
                    dims.len()
                )));
            }
            if reference == L2Reference::Required && options.tolerance.rtol > 0.0 && sites.len() > 1
            {
                return Err(invalid(
                    "ErrorNorm::L2 needs a reference norm when tolerance.rtol > 0 and the root \
                     patch has more than one site; give L2Reference::Given(norm), set rtol = 0 \
                     with a positive atol, or opt into L2Reference::MonteCarlo",
                ));
            }
        }

        let mut node_positions = Vec::with_capacity(node_sites.len());
        let mut site_nodes = Vec::with_capacity(sites.len());
        for (node, node_site_list) in &node_sites {
            let start = site_nodes.len();
            site_nodes.extend(std::iter::repeat_n(node.clone(), node_site_list.len()));
            node_positions.push((node.clone(), (start..site_nodes.len()).collect()));
        }

        let graph = topology.graph();
        let edges = graph
            .edge_indices()
            .filter_map(|edge| graph.edge_endpoints(edge))
            .filter_map(|(left, right)| {
                Some((
                    topology.node_name(left)?.clone(),
                    topology.node_name(right)?.clone(),
                ))
            })
            .collect();

        Ok(Self {
            topology,
            node_sites,
            node_positions,
            sites,
            dims,
            site_nodes,
            edges,
            split_order,
            domain_points,
        })
    }

    pub(super) fn projector(
        &self,
        path: &[(usize, usize)],
    ) -> Result<Projector, PatchedInterpolationError> {
        Ok(Projector::from_pairs(path.iter().map(
            |&(position, value)| (self.sites[position].clone(), value),
        ))?)
    }

    /// The active sites of every node, as an interpolation problem takes them.
    pub(super) fn active_node_sites(&self, fixed: &[Option<usize>]) -> BTreeMap<V, Vec<DynIndex>> {
        self.node_positions
            .iter()
            .map(|(node, positions)| {
                let active = positions
                    .iter()
                    .filter(|&&position| fixed[position].is_none())
                    .map(|&position| self.sites[position].clone())
                    .collect();
                (node.clone(), active)
            })
            .collect()
    }
}

/// Positions of the `patch_order` entries in the site order, or every
/// position for an empty order.
fn resolve_patch_order(
    patch_order: &[DynIndex],
    sites: &[DynIndex],
) -> Result<Vec<usize>, PatchedInterpolationError> {
    if patch_order.is_empty() {
        return Ok((0..sites.len()).collect());
    }
    let positions: HashMap<&DynIndex, usize> = sites
        .iter()
        .enumerate()
        .map(|(position, site)| (site, position))
        .collect();
    let mut listed = HashSet::with_capacity(patch_order.len());
    patch_order
        .iter()
        .map(|entry| {
            let Some(&position) = positions.get(entry) else {
                return Err(invalid(format!(
                    "patch_order entry {entry:?} is not a site index of the problem"
                )));
            };
            if entry.dim != sites[position].dim {
                return Err(invalid(format!(
                    "patch_order entry {entry:?} has dimension {}, but the problem site has \
                     dimension {}",
                    entry.dim, sites[position].dim
                )));
            }
            if !listed.insert(position) {
                return Err(invalid(format!(
                    "patch_order entry {entry:?} appears more than once"
                )));
            }
            Ok(position)
        })
        .collect()
}

fn validate_options(
    options: &PatchedInterpolationOptions,
) -> Result<(), PatchedInterpolationError> {
    let tolerance = options.tolerance;
    for (name, value) in [("rtol", tolerance.rtol), ("atol", tolerance.atol)] {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(format!(
                "tolerance.{name} must be finite and nonnegative, got {value}"
            )));
        }
    }
    match options.error_norm {
        ErrorNorm::SampledMax {
            max_reference: Some(scale),
            ..
        } if !scale.is_finite() || scale <= 0.0 => {
            return Err(invalid(format!(
                "max_reference of ErrorNorm::SampledMax must be finite and positive, got {scale}"
            )));
        }
        ErrorNorm::L2 {
            reference: L2Reference::Given(norm),
            ..
        } if !norm.is_finite() || norm <= 0.0 => {
            return Err(invalid(format!(
                "the reference norm of L2Reference::Given must be finite and positive, got {norm}"
            )));
        }
        _ => {}
    }
    if options.max_bond_dim < 2 {
        return Err(invalid(format!(
            "max_bond_dim must be at least 2 (a cap of one only accepts patches nonzero on a \
             single node), got {}",
            options.max_bond_dim
        )));
    }
    if options.n_initial_pivots == 0 {
        return Err(invalid("n_initial_pivots must be at least 1"));
    }
    if options.max_patches == Some(0) {
        return Err(invalid("max_patches must be positive when given"));
    }
    super::acceptance::validate_bounds(options).map_err(invalid)?;
    if options.verification.samples < 2 {
        return Err(invalid(format!(
            "verification.samples must be at least 2, got {}",
            options.verification.samples
        )));
    }
    Ok(())
}

fn validate_initial_pivots(
    pivots: &ColMajorArray<usize>,
    dims: &[usize],
) -> Result<(), PatchedInterpolationError> {
    let (Some(n_rows), Some(n_cols)) = (pivots.nrows(), pivots.ncols()) else {
        return Err(invalid(format!(
            "initial_pivots must be a 2D array, got shape {:?}",
            pivots.shape()
        )));
    };
    if n_rows != dims.len() {
        return Err(invalid(format!(
            "initial_pivots has {n_rows} rows but the problem has {} sites",
            dims.len()
        )));
    }
    for column in 0..n_cols {
        let point = pivots.column(column).unwrap_or_default();
        if let Some((row, &value)) = point
            .iter()
            .enumerate()
            .find(|&(row, &value)| value >= dims[row])
        {
            return Err(invalid(format!(
                "initial pivot {column} has coordinate {value} at row {row}, out of range for \
                 site dimension {}",
                dims[row]
            )));
        }
    }
    Ok(())
}
