//! Engine-independent contract for interpolating a function on a tree.
//!
//! A caller describes one interpolation problem with an
//! [`InterpolationProblem`]: a tree topology with named nodes, the active site
//! indices of every node, initial pivots, an absolute tolerance, an optional
//! bond cap, and a seed. An engine implementing [`TreeInterpolator`] samples
//! the function through a batch evaluator and returns an
//! [`InterpolationOutcome`]: a [`TreeTN`] with the caller's node names and
//! site indices, an [`InterpolationTermination`] verdict, the unnormalized
//! error estimate and maximum sampled magnitude, and optional full-domain
//! pivots.
//!
//! Code that runs an engine (for example a patch driver) depends only on this
//! module; each engine implements the trait in its own crate. The TreeTCI
//! engine is `tensor4all_treetci::TreeTciInterpolator`. Such code checks its
//! topology and sites with [`validate_layout`], the same checks
//! [`InterpolationProblem::new`] performs.
//!
//! # Site order
//!
//! Batches passed to the evaluator and pivots use one site order, derived
//! from the problem: nodes in ascending name order, and each node's sites in
//! the order given in `node_sites`
//! ([`InterpolationProblem::derive_site_order`]). A batch is a column-major
//! `[n_active_sites, n_points]` array whose column `p` is one point.
//!
//! # Examples
//!
//! ```
//! use std::collections::BTreeMap;
//! use tensor4all_core::{ColMajorArray, DynIndex};
//! use tensor4all_treetn::interpolation::InterpolationProblem;
//! use tensor4all_treetn::NodeNameNetwork;
//!
//! let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
//! let mut topology = NodeNameNetwork::new();
//! topology.add_node("left".to_string())?;
//! topology.add_node("right".to_string())?;
//! topology.add_edge(&"left".to_string(), &"right".to_string())?;
//! let node_sites = BTreeMap::from([
//!     ("left".to_string(), vec![a.clone()]),
//!     ("right".to_string(), vec![b.clone()]),
//! ]);
//!
//! // One initial pivot (a = 1, b = 2) in the derived site order [a, b].
//! let pivots = ColMajorArray::new(vec![1, 2], vec![2, 1])?;
//! let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-10, None, 7)?;
//! assert_eq!(problem.site_order(), &[a, b][..]);
//! assert_eq!(problem.initial_pivots().column(0), Some(&[1, 2][..]));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};

use crate::{NodeNameNetwork, TreeTN};

/// One tree interpolation problem, validated on construction.
///
/// Built only through [`InterpolationProblem::new`], which checks the
/// invariants engines rely on: the topology is a tree whose nodes are exactly
/// the keys of `node_sites`, every active site index appears once with a
/// positive dimension, at least one active site exists, the initial pivots
/// are in-range points in site order, and the tolerance is finite and
/// nonnegative. Fields are private; use the accessors.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use std::num::NonZeroUsize;
/// use tensor4all_core::{ColMajorArray, DynIndex};
/// use tensor4all_treetn::interpolation::InterpolationProblem;
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // A three-node chain 0 - 1 - 2; node 1 carries two sites, node 2 none.
/// let (s0, s1a, s1b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// for node in 0..3usize {
///     topology.add_node(node)?;
/// }
/// topology.add_edge(&0, &1)?;
/// topology.add_edge(&1, &2)?;
/// let node_sites = BTreeMap::from([
///     (0usize, vec![s0.clone()]),
///     (1, vec![s1a.clone(), s1b.clone()]),
///     (2, vec![]),
/// ]);
///
/// let pivots = ColMajorArray::new(vec![0, 1, 2], vec![3, 1])?;
/// let problem = InterpolationProblem::new(
///     topology,
///     node_sites,
///     pivots,
///     1e-8,
///     NonZeroUsize::new(4),
///     42,
/// )?;
/// assert_eq!(problem.site_order(), &[s0, s1a, s1b][..]);
/// assert_eq!(problem.max_bond_dim().map(NonZeroUsize::get), Some(4));
/// assert_eq!(problem.seed(), 42);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct InterpolationProblem<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    site_order: Vec<DynIndex>,
    initial_pivots: ColMajorArray<usize>,
    absolute_tolerance: f64,
    max_bond_dim: Option<NonZeroUsize>,
    seed: u64,
}

impl<V> InterpolationProblem<V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    /// Validate and build an interpolation problem.
    ///
    /// # Arguments
    ///
    /// * `topology` - Tree topology with named nodes. Its node set must equal
    ///   the keys of `node_sites`.
    /// * `node_sites` - Active site indices of every node, possibly none for
    ///   a node. A node's sites are listed in the order the site order uses.
    /// * `initial_pivots` - Column-major `[n_active_sites, n_pivots]` array in
    ///   site order ([`Self::derive_site_order`]), with at least one column.
    ///   Engines start from these points; if all of them evaluate to exactly
    ///   zero the engine returns [`InterpolationError::AllSamplesZero`].
    /// * `absolute_tolerance` - Bound compared with the engine's raw
    ///   (unnormalized) error estimate. The caller derives it from its
    ///   relative tolerance and a reference scale; engines do not normalize.
    /// * `max_bond_dim` - Bond cap, or `None` for no cap. A run converged at
    ///   a rank equal to the cap is reported as
    ///   [`InterpolationTermination::BondCapReached`].
    /// * `seed` - The only source of randomness; engines override any seed in
    ///   their own configuration with it.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolationError::InvalidProblem`] when the topology is not
    /// a tree (disconnected, or an edge count other than the node count minus
    /// one), its node set differs from the keys of `node_sites`, a site index
    /// appears more than once or has dimension zero, there is no active site,
    /// `initial_pivots` is not a 2D array with one row per active site and at
    /// least one column, a pivot coordinate is out of range for its site, or
    /// `absolute_tolerance` is negative or not finite. The topology and site
    /// checks are those of [`validate_layout`], which runs first.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use tensor4all_core::{ColMajorArray, DynIndex};
    /// use tensor4all_treetn::interpolation::{InterpolationError, InterpolationProblem};
    /// use tensor4all_treetn::NodeNameNetwork;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// let node_sites = BTreeMap::from([(0usize, vec![site.clone()])]);
    ///
    /// // Coordinate 2 is out of range for a dimension-2 site.
    /// let bad = ColMajorArray::new(vec![2], vec![1, 1])?;
    /// let error = InterpolationProblem::new(
    ///     topology.clone(), node_sites.clone(), bad, 1e-8, None, 0,
    /// ).unwrap_err();
    /// assert!(matches!(error, InterpolationError::InvalidProblem { .. }));
    ///
    /// let good = ColMajorArray::new(vec![1], vec![1, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, good, 1e-8, None, 0)?;
    /// assert_eq!(problem.site_order(), &[site][..]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        topology: NodeNameNetwork<V>,
        node_sites: BTreeMap<V, Vec<DynIndex>>,
        initial_pivots: ColMajorArray<usize>,
        absolute_tolerance: f64,
        max_bond_dim: Option<NonZeroUsize>,
        seed: u64,
    ) -> Result<Self, InterpolationError> {
        validate_layout(&topology, &node_sites)?;
        let site_order = Self::derive_site_order(&node_sites);
        validate_pivots(&initial_pivots, &site_order)?;
        if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
            return Err(invalid(format!(
                "absolute_tolerance must be finite and nonnegative, got {absolute_tolerance}"
            )));
        }
        Ok(Self {
            topology,
            node_sites,
            site_order,
            initial_pivots,
            absolute_tolerance,
            max_bond_dim,
            seed,
        })
    }

    /// Return the site order that [`Self::new`] derives from `node_sites`.
    ///
    /// Nodes are taken in ascending name order and each node's sites in the
    /// given order. Callers use it to lay out `initial_pivots` before
    /// constructing the problem. The input is not validated here.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_treetn::interpolation::InterpolationProblem;
    ///
    /// let (x, y, z) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(4));
    /// let node_sites = BTreeMap::from([
    ///     ("b".to_string(), vec![z.clone()]),
    ///     ("a".to_string(), vec![y.clone(), x.clone()]),
    /// ]);
    /// let order = InterpolationProblem::<String>::derive_site_order(&node_sites);
    /// assert_eq!(order, vec![y, x, z]);
    /// ```
    pub fn derive_site_order(node_sites: &BTreeMap<V, Vec<DynIndex>>) -> Vec<DynIndex> {
        node_sites.values().flatten().cloned().collect()
    }

    /// Borrow the tree topology.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(1usize)?;
    /// topology.add_node(2usize)?;
    /// topology.add_edge(&1, &2)?;
    /// let node_sites = BTreeMap::from([(1usize, vec![DynIndex::new_dyn(2)]), (2, vec![])]);
    /// let pivots = ColMajorArray::new(vec![0], vec![1, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0)?;
    /// assert_eq!(problem.topology().node_count(), 2);
    /// assert_eq!(problem.topology().edge_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn topology(&self) -> &NodeNameNetwork<V> {
        &self.topology
    }

    /// Borrow the active site indices of every node.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let site = DynIndex::new_dyn(3);
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(1usize)?;
    /// topology.add_node(2usize)?;
    /// topology.add_edge(&1, &2)?;
    /// let node_sites = BTreeMap::from([(1usize, vec![]), (2, vec![site.clone()])]);
    /// let pivots = ColMajorArray::new(vec![2], vec![1, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0)?;
    /// assert!(problem.node_sites()[&1].is_empty());
    /// assert_eq!(problem.node_sites()[&2], vec![site]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn node_sites(&self) -> &BTreeMap<V, Vec<DynIndex>> {
        &self.node_sites
    }

    /// Borrow the site order used by batches and pivots.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(5usize)?;
    /// topology.add_node(3usize)?;
    /// topology.add_edge(&5, &3)?;
    /// // Node 3 comes first in the site order.
    /// let node_sites = BTreeMap::from([(5usize, vec![a.clone()]), (3, vec![b.clone()])]);
    /// let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0)?;
    /// assert_eq!(problem.site_order(), &[b, a][..]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn site_order(&self) -> &[DynIndex] {
        &self.site_order
    }

    /// Borrow the initial pivots, shape `[n_active_sites, n_pivots]` in site
    /// order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// let node_sites = BTreeMap::from([(0usize, vec![DynIndex::new_dyn(4)])]);
    /// let pivots = ColMajorArray::new(vec![1, 3], vec![1, 2])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0)?;
    /// assert_eq!(problem.initial_pivots().shape(), &[1, 2]);
    /// assert_eq!(problem.initial_pivots().column(1), Some(&[3][..]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn initial_pivots(&self) -> &ColMajorArray<usize> {
        &self.initial_pivots
    }

    /// Return the absolute tolerance compared with the engine's raw error
    /// estimate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// let node_sites = BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]);
    /// let pivots = ColMajorArray::new(vec![0], vec![1, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-6, None, 0)?;
    /// assert_eq!(problem.absolute_tolerance(), 1e-6);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Return the bond cap, or `None` when uncapped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use std::num::NonZeroUsize;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// let node_sites = BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]);
    /// let pivots = ColMajorArray::new(vec![0], vec![1, 1])?;
    /// let cap = NonZeroUsize::new(8);
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, cap, 0)?;
    /// assert_eq!(problem.max_bond_dim(), cap);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_bond_dim(&self) -> Option<NonZeroUsize> {
        self.max_bond_dim
    }

    /// Return the seed, the only source of randomness for engines.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::BTreeMap;
    /// # use tensor4all_core::{ColMajorArray, DynIndex};
    /// # use tensor4all_treetn::interpolation::InterpolationProblem;
    /// # use tensor4all_treetn::NodeNameNetwork;
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// let node_sites = BTreeMap::from([(0usize, vec![DynIndex::new_dyn(2)])]);
    /// let pivots = ColMajorArray::new(vec![0], vec![1, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 99)?;
    /// assert_eq!(problem.seed(), 99);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

fn invalid(message: String) -> InterpolationError {
    InterpolationError::InvalidProblem { message }
}

/// Check the topology and site layout of an interpolation problem.
///
/// These are the layout checks of [`InterpolationProblem::new`], which calls
/// this function, exposed so that callers such as a patch driver validate
/// their inputs with the same code before any evaluation. The initial pivots
/// and the tolerance are not part of the layout and are not checked here.
///
/// # Arguments
///
/// * `topology` - Tree topology with named nodes. Its node set must equal the
///   keys of `node_sites`.
/// * `node_sites` - Site indices of every node, possibly none for a node.
///
/// # Returns
///
/// `Ok(())` when the topology is a tree whose node set equals the keys of
/// `node_sites`, every site index appears once (full identity: ID, tags, and
/// prime level) with a positive dimension, and at least one site exists.
///
/// # Errors
///
/// Returns [`InterpolationError::InvalidProblem`] when the topology has no
/// node, is not a tree (disconnected, or an edge count other than the node
/// count minus one), or its node set differs from the keys of `node_sites`;
/// when a site index appears more than once or has dimension zero; or when
/// there is no site at all. The message names the violated condition.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::DynIndex;
/// use tensor4all_treetn::interpolation::{validate_layout, InterpolationError};
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // A star whose center 0 has degree three; the center carries no site.
/// let mut topology = NodeNameNetwork::new();
/// for node in 0..4usize {
///     topology.add_node(node)?;
/// }
/// for leaf in 1..4usize {
///     topology.add_edge(&0, &leaf)?;
/// }
/// let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
/// let mut node_sites = BTreeMap::from([(0usize, vec![])]);
/// for (leaf, site) in (1..4usize).zip(&sites) {
///     node_sites.insert(leaf, vec![site.clone()]);
/// }
/// assert!(validate_layout(&topology, &node_sites).is_ok());
///
/// // The same full index on two nodes is rejected.
/// node_sites.insert(0, vec![sites[0].clone()]);
/// let error = validate_layout(&topology, &node_sites).unwrap_err();
/// match error {
///     InterpolationError::InvalidProblem { message } => {
///         assert!(message.contains("appears more than once"));
///     }
///     other => panic!("unexpected error {other:?}"),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn validate_layout<V>(
    topology: &NodeNameNetwork<V>,
    node_sites: &BTreeMap<V, Vec<DynIndex>>,
) -> Result<(), InterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    validate_topology(topology, node_sites)?;
    validate_sites(node_sites.values().flatten())
}

fn validate_topology<V>(
    topology: &NodeNameNetwork<V>,
    node_sites: &BTreeMap<V, Vec<DynIndex>>,
) -> Result<(), InterpolationError>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if topology.node_count() != node_sites.len() {
        return Err(invalid(format!(
            "topology has {} nodes but node_sites has {} entries",
            topology.node_count(),
            node_sites.len()
        )));
    }
    if let Some(missing) = node_sites.keys().find(|node| !topology.has_node(node)) {
        return Err(invalid(format!(
            "node_sites names node {missing:?}, which is not in the topology"
        )));
    }
    let node_count = topology.node_count();
    if node_count == 0 {
        return Err(invalid("topology has no nodes".to_string()));
    }
    if topology.edge_count() != node_count - 1 {
        return Err(invalid(format!(
            "a tree with {node_count} nodes has {} edges, got {}",
            node_count - 1,
            topology.edge_count()
        )));
    }
    let nodes: HashSet<_> = topology.graph().node_indices().collect();
    if !topology.is_connected_subset(&nodes) {
        return Err(invalid("topology is not connected".to_string()));
    }
    Ok(())
}

fn validate_sites<'a>(sites: impl Iterator<Item = &'a DynIndex>) -> Result<(), InterpolationError> {
    let mut seen = HashSet::new();
    for site in sites {
        if site.dim() == 0 {
            return Err(invalid(format!("site index {site:?} has dimension zero")));
        }
        if !seen.insert(site) {
            return Err(invalid(format!(
                "site index {site:?} appears more than once"
            )));
        }
    }
    if seen.is_empty() {
        return Err(invalid("the problem has no active site".to_string()));
    }
    Ok(())
}

fn validate_pivots(
    pivots: &ColMajorArray<usize>,
    site_order: &[DynIndex],
) -> Result<(), InterpolationError> {
    let (Some(n_rows), Some(n_cols)) = (pivots.nrows(), pivots.ncols()) else {
        return Err(invalid(format!(
            "initial_pivots must be a 2D array, got shape {:?}",
            pivots.shape()
        )));
    };
    if n_rows != site_order.len() {
        return Err(invalid(format!(
            "initial_pivots has {n_rows} rows but the problem has {} active sites",
            site_order.len()
        )));
    }
    if n_cols == 0 {
        return Err(invalid(
            "initial_pivots must contain at least one pivot".to_string(),
        ));
    }
    for column in 0..n_cols {
        let point = pivots.column(column).ok_or_else(|| {
            invalid(format!(
                "initial_pivots column {column} is out of range for shape {:?}",
                pivots.shape()
            ))
        })?;
        for (row, (&value, site)) in point.iter().zip(site_order).enumerate() {
            if value >= site.dim() {
                return Err(invalid(format!(
                    "initial pivot {column} has coordinate {value} at row {row}, out of range for site dimension {}",
                    site.dim()
                )));
            }
        }
    }
    Ok(())
}

/// Verdict of one interpolation run.
///
/// Callers that accept a result only when it converged below the bond cap
/// accept [`InterpolationTermination::Converged`] and treat every other
/// variant, including variants added later, as not accepted.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::interpolation::InterpolationTermination;
///
/// fn accepted(termination: InterpolationTermination) -> bool {
///     matches!(termination, InterpolationTermination::Converged)
/// }
/// assert!(accepted(InterpolationTermination::Converged));
/// assert!(!accepted(InterpolationTermination::BondCapReached));
/// assert!(!accepted(InterpolationTermination::IterationLimit));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InterpolationTermination {
    /// The engine's error criterion holds against the absolute tolerance and
    /// the final maximum bond dimension is strictly below the cap (or no cap
    /// is set).
    Converged,
    /// The run stopped at the bond cap, including a run whose error
    /// criterion holds at a rank equal to the cap. The result may be
    /// inaccurate.
    BondCapReached,
    /// The run stopped at the engine's iteration limit.
    IterationLimit,
}

/// Result of one interpolation run.
///
/// A plain struct with public fields so that engines in other crates build it
/// with a struct literal.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor};
/// use tensor4all_treetn::interpolation::{InterpolationOutcome, InterpolationTermination};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, -3.0])?;
/// let outcome = InterpolationOutcome {
///     network: TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?,
///     termination: InterpolationTermination::Converged,
///     error_estimate: 0.0,
///     max_sample_magnitude: 3.0,
///     pivots: Some(ColMajorArray::new(vec![1], vec![1, 1])?),
/// };
/// let value = outcome.network.evaluate_point(&[site], &[1])?;
/// assert_eq!(value.real(), -3.0);
/// assert_eq!(outcome.pivots.as_ref().map(|p| p.shape().to_vec()), Some(vec![1, 1]));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct InterpolationOutcome<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Network over the active sites with the problem's node names and
    /// topology. A node without active sites carries no site index. Its bond
    /// dimensions never exceed [`InterpolationProblem::max_bond_dim`], whatever
    /// the termination: a caller may use the network of a run that did not
    /// converge, for example a patch driver that accepts small capped patches.
    pub network: TreeTN<IdxTensor, V>,
    /// Why the run stopped.
    pub termination: InterpolationTermination,
    /// The engine's final raw (unnormalized) error estimate, the quantity
    /// compared with [`InterpolationProblem::absolute_tolerance`].
    pub error_estimate: f64,
    /// Largest magnitude among the samples used by the interpolation itself,
    /// excluding samples taken only for global pivot search or
    /// materialization.
    pub max_sample_magnitude: f64,
    /// Full-domain pivots of the result, shape `[n_active_sites, n_pivots]`
    /// in site order, or `None` when the engine does not produce pivots.
    /// They are valid points, may include points where the function is zero,
    /// and are meant only to seed further runs.
    pub pivots: Option<ColMajorArray<usize>>,
}

/// Error returned by [`InterpolationProblem::new`] and
/// [`TreeInterpolator::interpolate`].
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::interpolation::InterpolationError;
///
/// let error = InterpolationError::Evaluator {
///     source: anyhow::anyhow!("function undefined at this point"),
/// };
/// assert_eq!(error.to_string(), "evaluator failed: function undefined at this point");
/// assert!(std::error::Error::source(&error).is_some());
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InterpolationError {
    /// The problem violates an invariant of [`InterpolationProblem`], or an
    /// engine cannot represent it (for example a size overflow).
    #[error("invalid interpolation problem: {message}")]
    InvalidProblem {
        /// Description of the violated condition.
        message: String,
    },
    /// The evaluator returned an error, a wrong number of values, or an
    /// invalid (non-finite) value where the engine requires a finite one.
    #[error("evaluator failed: {source}")]
    Evaluator {
        /// The evaluator's error, or a description of the invalid result.
        #[source]
        source: anyhow::Error,
    },
    /// Every initial pivot evaluated to exactly zero.
    #[error("every initial pivot evaluates to zero")]
    AllSamplesZero,
    /// The engine failed for a reason unrelated to the evaluator.
    #[error("interpolation engine failed: {source}")]
    Engine {
        /// The engine's error, preserving its source chain.
        #[source]
        source: anyhow::Error,
    },
}

/// A tree interpolation engine.
///
/// An implementation samples the function through `evaluate` and returns an
/// [`InterpolationOutcome`] for the given [`InterpolationProblem`]. The scalar
/// type `T` is unbounded here; each implementation states the bounds its
/// engine needs.
///
/// Contract for implementations:
///
/// - `evaluate` receives a column-major `[n_active_sites, n_points]` batch in
///   the problem's site order and must return one value per point.
/// - The error criterion compares the raw error estimate with
///   [`InterpolationProblem::absolute_tolerance`]; the engine does not
///   normalize it.
/// - [`InterpolationTermination::Converged`] requires the criterion to hold
///   and the final maximum bond dimension to be strictly below the cap; a
///   criterion met at a rank equal to the cap is
///   [`InterpolationTermination::BondCapReached`].
/// - [`InterpolationProblem::seed`] overrides any seed in the engine's own
///   configuration.
/// - If every initial pivot evaluates to exactly zero, the engine returns
///   [`InterpolationError::AllSamplesZero`]; a non-finite initial sample is an
///   invalid evaluator value and is reported as
///   [`InterpolationError::Evaluator`].
/// - The outcome network carries the problem's node names, topology, and
///   exactly the active site indices.
///
/// # Examples
///
/// A toy engine that handles only single-node problems by evaluating every
/// point:
///
/// ```
/// use std::collections::BTreeMap;
/// use std::hash::Hash;
/// use std::fmt::Debug;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
/// use tensor4all_treetn::interpolation::{
///     InterpolationError, InterpolationOutcome, InterpolationProblem,
///     InterpolationTermination, TreeInterpolator,
/// };
/// use tensor4all_treetn::{NodeNameNetwork, TreeTN};
///
/// struct DenseSingleNode;
///
/// impl TreeInterpolator<f64> for DenseSingleNode {
///     fn interpolate<V, F>(
///         &self,
///         problem: &InterpolationProblem<V>,
///         evaluate: F,
///     ) -> Result<InterpolationOutcome<V>, InterpolationError>
///     where
///         V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
///         F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>,
///     {
///         let (node, sites) = match problem.node_sites().iter().next() {
///             Some(entry) if problem.node_sites().len() == 1 => entry,
///             _ => {
///                 return Err(InterpolationError::InvalidProblem {
///                     message: "only single-node problems are supported".into(),
///                 })
///             }
///         };
///         // Enumerate every point column-major (first site fastest).
///         let dims: Vec<usize> = sites.iter().map(|s| s.dim()).collect();
///         let n_points: usize = dims.iter().product();
///         let mut points = Vec::with_capacity(n_points * dims.len());
///         for mut linear in 0..n_points {
///             for &dim in &dims {
///                 points.push(linear % dim);
///                 linear /= dim;
///             }
///         }
///         let shape = [dims.len(), n_points];
///         let batch = ColMajorArrayRef::new(&points, &shape)
///             .map_err(|e| InterpolationError::Engine { source: e.into() })?;
///         let values = evaluate(batch).map_err(|source| InterpolationError::Evaluator { source })?;
///         let max = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
///         if max == 0.0 {
///             return Err(InterpolationError::AllSamplesZero);
///         }
///         let tensor = IdxTensor::from_dense(sites.clone(), values)
///             .map_err(|e| InterpolationError::Engine { source: e.into() })?;
///         let network = TreeTN::from_tensors(vec![tensor], vec![node.clone()])
///             .map_err(|e| InterpolationError::Engine { source: e.into() })?;
///         Ok(InterpolationOutcome {
///             network,
///             termination: InterpolationTermination::Converged,
///             error_estimate: 0.0,
///             max_sample_magnitude: max,
///             pivots: None,
///         })
///     }
/// }
///
/// let (x, y) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node("only".to_string())?;
/// let node_sites = BTreeMap::from([("only".to_string(), vec![x.clone(), y.clone()])]);
/// let pivots = ColMajorArray::new(vec![1, 2], vec![2, 1])?;
/// let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-12, None, 0)?;
///
/// // f(x, y) = x + 10 y
/// let outcome = DenseSingleNode.interpolate(&problem, |batch: ColMajorArrayRef<'_, usize>| {
///     let n = batch.shape()[1];
///     Ok((0..n)
///         .map(|p| (batch.get(&[0, p]).unwrap() + 10 * batch.get(&[1, p]).unwrap()) as f64)
///         .collect())
/// })?;
/// assert_eq!(outcome.termination, InterpolationTermination::Converged);
/// assert_eq!(outcome.max_sample_magnitude, 21.0);
/// assert_eq!(outcome.network.evaluate_point(&[x, y], &[1, 2])?.real(), 21.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TreeInterpolator<T> {
    /// Interpolate the function sampled by `evaluate` on `problem`.
    ///
    /// # Arguments
    ///
    /// * `problem` - Validated problem: topology, active sites, initial
    ///   pivots, tolerance, bond cap, and seed.
    /// * `evaluate` - Batch evaluator. It receives a column-major
    ///   `[n_active_sites, n_points]` array in site order and returns one
    ///   value per point.
    ///
    /// # Returns
    ///
    /// An [`InterpolationOutcome`] whose `termination` tells whether the
    /// result converged below the bond cap.
    ///
    /// # Errors
    ///
    /// - [`InterpolationError::InvalidProblem`] when the engine cannot
    ///   represent the problem (for example a size derived by the engine
    ///   overflows).
    /// - [`InterpolationError::Evaluator`] when `evaluate` returns an error, a
    ///   number of values other than the number of points, or a non-finite
    ///   value at an initial pivot.
    /// - [`InterpolationError::AllSamplesZero`] when every initial pivot
    ///   evaluates to exactly zero.
    /// - [`InterpolationError::Engine`] for any other engine failure.
    ///
    /// # Examples
    ///
    /// See the trait-level example, which implements and calls this method.
    ///
    /// ```
    /// use tensor4all_treetn::interpolation::{InterpolationError, InterpolationTermination};
    ///
    /// // Callers branch on the documented error variants.
    /// fn describe(result: Result<InterpolationTermination, InterpolationError>) -> &'static str {
    ///     match result {
    ///         Ok(InterpolationTermination::Converged) => "accepted",
    ///         Ok(_) => "not accepted",
    ///         Err(InterpolationError::AllSamplesZero) => "zero patch",
    ///         Err(_) => "failed",
    ///     }
    /// }
    /// assert_eq!(describe(Ok(InterpolationTermination::BondCapReached)), "not accepted");
    /// assert_eq!(describe(Err(InterpolationError::AllSamplesZero)), "zero patch");
    /// ```
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>;
}

#[cfg(test)]
mod tests;
