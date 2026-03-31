use crate::{assemble::MultiIndex, SubtreeKey, TreeTciEdge, TreeTciGraph};
use anyhow::{ensure, Result};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use tensor4all_core::ColMajorArray;

/// TreeTCI state mirroring the upstream `SimpleTCI` layout.
#[derive(Clone, Debug)]
pub struct TreeTCI2<T> {
    /// Pivot sets keyed by canonical subtree keys.
    /// Shape of each entry: [n_subtree_sites, n_pivots].
    pub ijset: HashMap<SubtreeKey, ColMajorArray<usize>>,
    /// Local dimensions for each site.
    pub local_dims: Vec<usize>,
    /// Tree graph metadata.
    pub graph: TreeTciGraph,
    /// Error estimate per edge bipartition.
    pub bond_errors: BTreeMap<TreeTciEdge, f64>,
    /// Back-truncation style pivot errors.
    pub pivot_errors: Vec<f64>,
    /// Maximum observed sample magnitude for normalization.
    pub max_sample_value: f64,
    /// Previous pivot sets for candidate-generation history.
    pub ijset_history: Vec<HashMap<SubtreeKey, ColMajorArray<usize>>>,
    marker: PhantomData<T>,
}

/// Deprecated alias for `TreeTCI2`.
#[deprecated(note = "use TreeTCI2")]
pub type SimpleTreeTci<T> = TreeTCI2<T>;

impl<T> TreeTCI2<T> {
    /// Create a new TreeTCI state from local dimensions and a tree graph.
    pub fn new(local_dims: Vec<usize>, graph: TreeTciGraph) -> Result<Self> {
        ensure!(
            local_dims.len() > 1,
            "local_dims should have at least 2 elements"
        );
        ensure!(
            local_dims.len() == graph.n_sites(),
            "local_dims length {} must match graph site count {}",
            local_dims.len(),
            graph.n_sites()
        );

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
            pivot_errors: Vec::new(),
            max_sample_value: 0.0,
            ijset_history: Vec::new(),
            marker: PhantomData,
        })
    }

    /// Add global pivots and project them to every edge bipartition.
    pub fn add_global_pivots(&mut self, pivots: &[MultiIndex]) -> Result<()> {
        let n_sites = self.local_dims.len();
        ensure!(
            pivots.iter().all(|pivot| pivot.len() == n_sites),
            "each global pivot must contain one index per site"
        );

        for pivot in pivots {
            for edge in self.graph.edges() {
                let (left_key, right_key) = self.graph.subregion_vertices(edge)?;
                let left_projection = project_pivot(pivot, &left_key);
                let right_projection = project_pivot(pivot, &right_key);
                let n_left = left_key.as_slice().len();
                let n_right = right_key.as_slice().len();
                push_unique_column(
                    self.ijset
                        .entry(left_key)
                        .or_insert_with(|| empty_2d(n_left)),
                    &left_projection,
                );
                push_unique_column(
                    self.ijset
                        .entry(right_key)
                        .or_insert_with(|| empty_2d(n_right)),
                    &right_projection,
                );
            }
        }

        let full_key = SubtreeKey::new((0..n_sites).collect());
        self.ijset
            .entry(full_key)
            .or_insert_with(|| empty_2d(n_sites));
        Ok(())
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
    pub fn max_rank(&self) -> usize {
        self.ijset
            .values()
            .map(|arr| arr.ncols())
            .max()
            .unwrap_or(0)
    }
}

fn project_pivot(pivot: &MultiIndex, key: &SubtreeKey) -> MultiIndex {
    key.as_slice().iter().map(|&site| pivot[site]).collect()
}

/// Create an empty 2D ColMajorArray with shape [nrows, 0].
fn empty_2d(nrows: usize) -> ColMajorArray<usize> {
    ColMajorArray::new(vec![], vec![nrows, 0]).expect("empty 2D array creation should not fail")
}

/// Push a column to a ColMajorArray if it is not already present.
pub(crate) fn push_unique_column(array: &mut ColMajorArray<usize>, column: &[usize]) {
    for j in 0..array.ncols() {
        if array.column(j) == Some(column) {
            return; // duplicate
        }
    }
    array
        .push_column(column)
        .expect("push_column should not fail for matching nrows");
}

#[cfg(test)]
mod tests;
