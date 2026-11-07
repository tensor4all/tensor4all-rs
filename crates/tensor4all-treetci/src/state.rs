use crate::error::Result as TreeTciResult;
use crate::{assemble::MultiIndex, column_2d, ncols_2d, SubtreeKey, TreeTciEdge, TreeTciGraph};
use anyhow::Result;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::marker::PhantomData;
use tensor4all_core::ColMajorArray;

/// TreeTCI state mirroring the upstream `SimpleTCI` layout.
///
/// Stores the current pivot sets, bond errors, and tree graph metadata
/// for tree tensor cross interpolation. The type parameter `T` is the
/// scalar type (e.g., `f64`, `Complex64`).
///
/// Use [`TreeTCI2::new`] to create a fresh state, then seed it with
/// [`TreeTCI2::add_global_pivots`] before passing to
/// [`optimize_default`](crate::optimize_default) or
/// [`crossinterpolate2`](crate::crossinterpolate2).
///
/// Continue optimization on the same function, topology and local dimensions
/// by calling an `optimize_*` function again with a larger `max_bond_dim` or
/// `None`. Pivot sets and the maximum sampled magnitude carry over. Pivot
/// history is not copied: proposers retain the current edge's pivots. Diagnostics,
/// convergence windows and iteration budgets restart per call; repeated calls
/// need not match one longer run. Caller-stream APIs continue the supplied RNG,
/// while seeded high-level calls restart their named generators.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{TreeTCI2, TreeTciEdge, TreeTciGraph};
///
/// let graph = TreeTciGraph::new(3, &[
///     TreeTciEdge::new(0, 1),
///     TreeTciEdge::new(1, 2),
/// ]).unwrap();
///
/// let mut state = TreeTCI2::<f64>::new(vec![2, 3, 2], graph).unwrap();
/// assert_eq!(state.max_bond_dim(), 0);
/// assert!((state.max_bond_error() - 0.0).abs() < f64::EPSILON);
///
/// // Seed with a global pivot
/// state.add_global_pivots(&[vec![0, 0, 0]]).unwrap();
/// assert_eq!(state.max_bond_dim(), 1);
///
/// let evaluate = |batch: tensor4all_treetci::GlobalIndexBatch<'_>| {
///     Ok((0..batch.n_points()).map(|p| {
///         if batch.get(0, p) == batch.get(1, p) { 1.0_f64 } else { 0.0 }
///     }).collect::<Vec<_>>())
/// };
/// // Use a two-site identity so the required rank is known to be two.
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2)?)?;
/// state.add_global_pivots(&[vec![0, 0]])?;
/// let capped = tensor4all_treetci::TreeTciOptions {
///     max_bond_dim: Some(1), max_iter: 3, enable_global_pivots: false,
///     ..Default::default()
/// };
/// tensor4all_treetci::optimize_default(&mut state, evaluate, &capped)?;
/// assert_eq!(state.max_bond_dim(), 1);
/// let uncapped = tensor4all_treetci::TreeTciOptions { max_bond_dim: None, ..capped };
/// tensor4all_treetci::optimize_default(&mut state, evaluate, &uncapped)?;
/// assert_eq!(state.max_bond_dim(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct TreeTCI2<T> {
    /// Pivot sets keyed by canonical subtree keys.
    /// Shape of each entry: [n_subtree_sites, n_pivots].
    pub(crate) ijset: HashMap<SubtreeKey, ColMajorArray<usize>>,
    /// Local dimensions for each site.
    pub(crate) local_dims: Vec<usize>,
    /// Tree graph metadata.
    pub(crate) graph: TreeTciGraph,
    /// Error estimate per edge bipartition.
    pub(crate) bond_errors: BTreeMap<TreeTciEdge, f64>,
    /// Edges whose most recent update could not measure a bond error: one
    /// side of the candidate matrix held a single candidate although both
    /// sides of the edge have more than one configuration, so the selected
    /// pivot exhausted that side and the recorded error is zero by
    /// construction (issue #1153).
    pub(crate) unmeasured_bonds: BTreeSet<TreeTciEdge>,
    /// Back-truncation style pivot errors.
    pub(crate) pivot_errors: Vec<f64>,
    /// Maximum observed sample magnitude for normalization.
    pub(crate) max_sample_value: f64,
    marker: PhantomData<T>,
}

/// Deprecated alias for `TreeTCI2`.
#[deprecated(note = "use TreeTCI2")]
pub type SimpleTreeTci<T> = TreeTCI2<T>;

impl<T> TreeTCI2<T> {
    /// Create a new TreeTCI state from local dimensions and a tree graph.
    /// # Errors
    ///
    /// Returns [`crate::TreeTciError::Operation`] when `local_dims` has fewer
    /// than two entries, its length differs from the graph's site count, or an
    /// entry is zero.
    ///
    pub fn new(local_dims: Vec<usize>, graph: TreeTciGraph) -> TreeTciResult<Self> {
        if !(local_dims.len() > 1) {
            return Err(anyhow::anyhow!("local_dims should have at least 2 elements").into());
        };
        if !(local_dims.len() == graph.n_sites()) {
            return Err(anyhow::anyhow!(
                "local_dims length {} must match graph site count {}",
                local_dims.len(),
                graph.n_sites()
            )
            .into());
        };
        if let Some((site, _)) = local_dims.iter().enumerate().find(|&(_, &dim)| dim == 0) {
            return Err(anyhow::anyhow!("local dimension at site {site} must be positive").into());
        }

        let bond_errors = graph
            .edges()
            .into_iter()
            .map(|edge| (edge, 0.0))
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            ijset: HashMap::new(),
            local_dims,
            graph,
            bond_errors,
            unmeasured_bonds: BTreeSet::new(),
            pivot_errors: Vec::new(),
            max_sample_value: 0.0,
            marker: PhantomData,
        })
    }

    /// Add global pivots and project them to every edge bipartition.
    ///
    /// Each side of an edge is deduplicated independently, so pivots added
    /// this way are only paired consistently across each edge (both sides
    /// holding the same number of pivots) after an optimization sweep. Sweep
    /// the state with [`optimize_default`](crate::optimize_default) or
    /// [`optimize_with_proposer`](crate::optimize_with_proposer) before
    /// passing it to [`to_treetn`](crate::to_treetn).
    ///
    /// # Errors
    ///
    /// Returns [`crate::TreeTciError::Operation`] when a pivot has fewer or
    /// more entries than there are sites or holds an out-of-range value, when
    /// an edge is absent from the graph, or when a pivot column cannot be
    /// appended.
    pub fn add_global_pivots(&mut self, pivots: &[MultiIndex]) -> TreeTciResult<()> {
        let n_sites = self.local_dims.len();
        if !(pivots.iter().all(|pivot| pivot.len() == n_sites)) {
            return Err(
                anyhow::anyhow!("each global pivot must contain one index per site").into(),
            );
        };
        for pivot in pivots {
            for (site, &value) in pivot.iter().enumerate() {
                if value >= self.local_dims[site] {
                    return Err(anyhow::anyhow!(
                        "global pivot value {value} is out of bounds for site {site} with dimension {}",
                        self.local_dims[site]
                    )
                    .into());
                }
            }
        }

        for pivot in pivots {
            for edge in self.graph.edges() {
                let (left_key, right_key) = self.graph.subregion_vertices(edge)?;
                let left_projection = project_pivot(pivot, &left_key);
                let right_projection = project_pivot(pivot, &right_key);
                let n_left = left_key.as_slice().len();
                let n_right = right_key.as_slice().len();
                match self.ijset.entry(left_key) {
                    Entry::Occupied(mut entry) => {
                        push_unique_column(entry.get_mut(), &left_projection)?;
                    }
                    Entry::Vacant(entry) => {
                        let mut array = empty_2d(n_left)?;
                        push_unique_column(&mut array, &left_projection)?;
                        entry.insert(array);
                    }
                }
                match self.ijset.entry(right_key) {
                    Entry::Occupied(mut entry) => {
                        push_unique_column(entry.get_mut(), &right_projection)?;
                    }
                    Entry::Vacant(entry) => {
                        let mut array = empty_2d(n_right)?;
                        push_unique_column(&mut array, &right_projection)?;
                        entry.insert(array);
                    }
                }
            }
        }

        let full_key = SubtreeKey::new((0..n_sites).collect());
        if let Entry::Vacant(entry) = self.ijset.entry(full_key) {
            entry.insert(empty_2d(n_sites)?);
        }
        Ok(())
    }

    /// Enrich the pivots around vertices of local dimension 1, as xfac's
    /// `TensorTreeCI` does with `addPivotsAllBonds(enrich_pivot1())`.
    ///
    /// A vertex of local dimension 1 carries no site, so the junction-side
    /// candidate set of an adjacent bond is the product of the other adjacent
    /// bonds' pivot sets. When those bonds have rank 1 the product has a single
    /// candidate and no local update can grow them (issue #1153).
    ///
    /// The rule: collect the vertices of local dimension greater than 1 that
    /// are adjacent to a vertex of local dimension 1. For every unordered pair
    /// `(i, j)` of them, add one pivot per pair of local values `(x_i, x_j)`:
    /// the base pivot with sites `i` and `j` set to `x_i` and `x_j`. The base
    /// pivot is the first pivot pair of the first edge adjacent to a vertex of
    /// local dimension 1. The pivots are projected like
    /// [`TreeTCI2::add_global_pivots`], so the next sweep must process them.
    ///
    /// Nothing is added when no edge adjacent to a vertex of local dimension 1
    /// has a side with at most one pivot, when fewer than two such adjacent
    /// vertices exist, or when the state holds no pivots yet.
    ///
    /// Reference: xfac's [`enrich_pivot1`] and its call site
    /// `addPivotsAllBonds(enrich_pivot1())` in the constructor.
    ///
    /// [`enrich_pivot1`]: https://github.com/tensor4all/xfac/blob/da5ff8f9b25ad1e3ebdcece9bd1d2d214441296b/include/xfac/tree/tensor_tree_ci.h#L185
    ///
    /// Returns the number of pivots passed to the projection (before
    /// per-side deduplication).
    pub(crate) fn enrich_pivots_around_unit_vertices(&mut self) -> TreeTciResult<usize> {
        let n_sites = self.local_dims.len();
        let mut unit_edges = Vec::new();
        let mut neighbours = BTreeSet::new();
        for vertex in (0..n_sites).filter(|&v| self.local_dims[v] == 1) {
            for neighbour in self.graph.neighbors(vertex)? {
                unit_edges.push(self.graph.edge_between(vertex, neighbour)?);
                if self.local_dims[neighbour] > 1 {
                    neighbours.insert(neighbour);
                }
            }
        }
        if neighbours.len() < 2 {
            return Ok(0);
        }
        unit_edges.sort();
        unit_edges.dedup();

        let mut base: Option<MultiIndex> = None;
        let mut pinned = false;
        for &edge in &unit_edges {
            let (left_key, right_key) = self.graph.subregion_vertices(edge)?;
            let (Some(left), Some(right)) = (self.ijset.get(&left_key), self.ijset.get(&right_key))
            else {
                return Ok(0);
            };
            let (n_left, n_right) = (ncols_2d(left)?, ncols_2d(right)?);
            if n_left == 0 || n_right == 0 {
                return Ok(0);
            }
            pinned |= n_left <= 1 || n_right <= 1;
            if base.is_none() {
                let mut pivot = vec![0; n_sites];
                for (key, array) in [(&left_key, left), (&right_key, right)] {
                    for (&site, &value) in key.as_slice().iter().zip(column_2d(array, 0)?) {
                        pivot[site] = value;
                    }
                }
                base = Some(pivot);
            }
        }
        let Some(base) = base else {
            return Ok(0);
        };
        if !pinned {
            return Ok(0);
        }

        let neighbours: Vec<usize> = neighbours.into_iter().collect();
        let mut pivots = Vec::new();
        for (a, &i) in neighbours.iter().enumerate() {
            for &j in &neighbours[a + 1..] {
                for xi in 0..self.local_dims[i] {
                    for xj in 0..self.local_dims[j] {
                        let mut pivot = base.clone();
                        pivot[i] = xi;
                        pivot[j] = xj;
                        pivots.push(pivot);
                    }
                }
            }
        }
        self.add_global_pivots(&pivots)?;
        Ok(pivots.len())
    }

    /// Reset the sweep-local pivot error accumulator.
    pub fn flush_pivot_errors(&mut self) {
        self.pivot_errors.clear();
    }

    /// Update one bond error.
    pub fn update_bond_error(&mut self, edge: TreeTciEdge, error: f64) {
        self.bond_errors.insert(edge, error);
    }

    /// Merge a new pivot-error vector into the sweep-local maximum.
    pub fn update_pivot_errors(&mut self, errors: &[f64]) {
        let len = self.pivot_errors.len().max(errors.len());
        self.pivot_errors.resize(len, 0.0);
        for (idx, &error) in errors.iter().enumerate() {
            self.pivot_errors[idx] = self.pivot_errors[idx].max(error);
        }
    }

    /// Maximum bond error across all edges.
    pub fn max_bond_error(&self) -> f64 {
        self.bond_errors.values().copied().fold(0.0, f64::max)
    }

    /// Maximum current bond dimension across stored subtree pivot sets.
    pub fn max_bond_dim(&self) -> usize {
        self.ijset
            .values()
            .filter_map(|arr| arr.ncols())
            .max()
            .unwrap_or(0)
    }

    /// Tree graph metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciGraph};
    ///
    /// let state = TreeTCI2::<f64>::new(vec![2, 2, 2], TreeTciGraph::linear_chain(3)?)?;
    /// assert_eq!(state.graph().n_sites(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn graph(&self) -> &TreeTciGraph {
        &self.graph
    }

    /// Local dimension of each site.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciGraph};
    ///
    /// let state = TreeTCI2::<f64>::new(vec![2, 3], TreeTciGraph::linear_chain(2)?)?;
    /// assert_eq!(state.local_dims(), &[2, 3]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// The pivot set stored for the subtree `key`, if any.
    ///
    /// The array has shape `[n_subtree_sites, n_pivots]`: one column per pivot, one row per site of the subtree in
    /// the key's order. The keys of an edge come from
    /// [`TreeTciGraph::subregion_vertices`](crate::TreeTciGraph::subregion_vertices).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciEdge, TreeTciGraph};
    ///
    /// let edge = TreeTciEdge::new(0, 1);
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::new(2, &[edge])?)?;
    /// let (left, _right) = state.graph().subregion_vertices(edge)?;
    /// assert!(state.pivot_set(&left).is_none());
    /// state.add_global_pivots(&[vec![0, 1]])?;
    /// let pivots = state.pivot_set(&left).expect("a pivot set after seeding");
    /// assert_eq!(pivots.ncols(), Some(1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pivot_set(&self, key: &SubtreeKey) -> Option<&ColMajorArray<usize>> {
        self.ijset.get(key)
    }

    /// Error estimate per edge bipartition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciEdge, TreeTciGraph};
    ///
    /// let edge = TreeTciEdge::new(0, 1);
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::new(2, &[edge])?)?;
    /// assert_eq!(state.bond_errors().get(&edge), Some(&0.0));
    /// state.update_bond_error(edge, 0.25);
    /// assert_eq!(state.bond_errors().get(&edge), Some(&0.25));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bond_errors(&self) -> &BTreeMap<TreeTciEdge, f64> {
        &self.bond_errors
    }

    /// Back-truncation style pivot errors of the current sweep.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciGraph};
    ///
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2)?)?;
    /// assert!(state.pivot_errors().is_empty());
    /// state.update_pivot_errors(&[0.5, 0.1]);
    /// assert_eq!(state.pivot_errors(), &[0.5, 0.1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }

    /// Maximum observed sample magnitude, the scale of the relative tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciGraph};
    ///
    /// let state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2)?)?;
    /// assert_eq!(state.max_sample_value(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Set the maximum sample magnitude that normalizes the error when
    /// [`TreeTciOptions::scale_tolerance`](crate::TreeTciOptions::scale_tolerance) is on.
    ///
    /// Call it before optimizing a state built by hand, with the largest absolute value of the target seen so far;
    /// the high-level entry points do this themselves. Pass a finite positive value: zero, a negative value and NaN
    /// leave the error unscaled, and an infinite value would scale every error to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTCI2, TreeTciGraph};
    ///
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2)?)?;
    /// state.set_max_sample_value(4.0);
    /// assert_eq!(state.max_sample_value(), 4.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set_max_sample_value(&mut self, value: f64) {
        self.max_sample_value = value;
    }
}

fn project_pivot(pivot: &MultiIndex, key: &SubtreeKey) -> MultiIndex {
    key.as_slice().iter().map(|&site| pivot[site]).collect()
}

/// Create an empty 2D ColMajorArray with shape [nrows, 0].
fn empty_2d(nrows: usize) -> Result<ColMajorArray<usize>> {
    Ok(ColMajorArray::new(vec![], vec![nrows, 0])?)
}

/// Push a column to a ColMajorArray if it is not already present.
pub(crate) fn push_unique_column(array: &mut ColMajorArray<usize>, column: &[usize]) -> Result<()> {
    for j in 0..ncols_2d(array)? {
        if column_2d(array, j)? == column {
            return Ok(()); // duplicate
        }
    }
    array.push_column(column)?;
    Ok(())
}

#[cfg(test)]
mod tests;
