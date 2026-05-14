//! Cached batch evaluation for tree tensor networks.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{bail, Context, Result};
use tensor4all_core::{AllowedPairs, AnyScalar, ColMajorArrayRef, IndexLike, TensorLike};

use super::TreeTN;

type KeyId = usize;

#[derive(Clone, Debug)]
struct SiteEntry<I> {
    index: I,
    input_position: usize,
    local_axis: usize,
}

#[derive(Clone, Debug)]
struct EvaluatorLayout<I, V> {
    entries_by_node: HashMap<V, Vec<SiteEntry<I>>>,
    n_indices: usize,
}

#[derive(Default)]
struct KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    ids: HashMap<T, KeyId>,
}

impl<T> KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    fn intern(&mut self, key: T) -> KeyId {
        let next = self.ids.len();
        *self.ids.entry(key).or_insert(next)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ComponentAssignment {
    values: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ComponentBatch<I, V> {
    neighbor: V,
    nodes: Vec<V>,
    entries: Vec<SiteEntry<I>>,
    entry_offsets_by_node: HashMap<V, Vec<(SiteEntry<I>, usize)>>,
    point_to_assignment: Vec<usize>,
    assignments: Vec<ComponentAssignment>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CachedEvaluationStats {
    subtree_environment_count: usize,
}

#[derive(Debug)]
struct ComponentCostIndex<V> {
    neighbors: HashMap<V, Vec<V>>,
    directed_counts: HashMap<(V, V), usize>,
    node_costs: Option<HashMap<V, usize>>,
}

impl<V> ComponentCostIndex<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn new<T>(
        tree: &TreeTN<T, V>,
        indices: &[T::Index],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self>
    where
        T: TensorLike,
        T::Index: Clone + Eq + Hash,
        <T::Index as IndexLike>::Id: Ord,
    {
        let layout = build_layout(tree, indices)?;
        Self::from_layout(tree, &layout, values)
    }

    fn from_layout<T>(
        tree: &TreeTN<T, V>,
        layout: &EvaluatorLayout<T::Index, V>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self>
    where
        T: TensorLike,
        T::Index: Clone + Eq + Hash,
        <T::Index as IndexLike>::Id: Ord,
    {
        validate_values_shape(values, layout.n_indices, "ComponentCostIndex::new")?;
        let n_points = values.shape()[1];

        let neighbors = sorted_neighbors(tree);
        if neighbors.is_empty() {
            return Ok(Self {
                neighbors,
                directed_counts: HashMap::new(),
                node_costs: None,
            });
        }

        let mut node_names: Vec<V> = neighbors.keys().cloned().collect();
        node_names.sort();
        let root = node_names[0].clone();

        let (parent, order) = rooted_tree(&neighbors, &root)?;
        let mut local_interner = KeyInterner::<Vec<usize>>::default();
        let mut local_keys: HashMap<V, Vec<KeyId>> = HashMap::with_capacity(node_names.len());
        for node in &node_names {
            let entries = layout
                .entries_by_node
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut keys = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let key = entries
                    .iter()
                    .map(|entry| *values.get(&[entry.input_position, point]).unwrap())
                    .collect::<Vec<_>>();
                validate_entry_values(entries, &key, "ComponentCostIndex::new")?;
                keys.push(local_interner.intern(key));
            }
            local_keys.insert(node.clone(), keys);
        }

        let mut component_interner = KeyInterner::<Vec<KeyId>>::default();
        let mut directed_keys: HashMap<(V, V), Vec<KeyId>> =
            HashMap::with_capacity(tree.edge_count() * 2);

        for node in order.iter().rev() {
            let Some(parent_node) = parent.get(node).and_then(Clone::clone) else {
                continue;
            };
            let incoming = neighbors
                .get(node)
                .unwrap()
                .iter()
                .filter(|neighbor| *neighbor != &parent_node)
                .map(|neighbor| {
                    directed_keys
                        .get(&(neighbor.clone(), node.clone()))
                        .with_context(|| {
                            format!(
                                "ComponentCostIndex::new: missing child key {:?}->{:?}",
                                neighbor, node
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let keys = intern_component_keys(
                local_keys.get(node).unwrap(),
                &incoming,
                n_points,
                &mut component_interner,
            );
            directed_keys.insert((node.clone(), parent_node), keys);
        }

        for node in &order {
            for child in neighbors.get(node).unwrap().iter().filter(|neighbor| {
                parent.get(*neighbor).and_then(Clone::clone) == Some(node.clone())
            }) {
                let incoming = neighbors
                    .get(node)
                    .unwrap()
                    .iter()
                    .filter(|neighbor| *neighbor != child)
                    .map(|neighbor| {
                        directed_keys
                            .get(&(neighbor.clone(), node.clone()))
                            .with_context(|| {
                                format!(
                                    "ComponentCostIndex::new: missing incoming key {:?}->{:?}",
                                    neighbor, node
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let keys = intern_component_keys(
                    local_keys.get(node).unwrap(),
                    &incoming,
                    n_points,
                    &mut component_interner,
                );
                directed_keys.insert((node.clone(), child.clone()), keys);
            }
        }

        let directed_counts = directed_keys
            .into_iter()
            .map(|(edge, keys)| {
                let count = keys.into_iter().collect::<HashSet<_>>().len();
                (edge, count)
            })
            .collect();

        Ok(Self {
            neighbors,
            directed_counts,
            node_costs: None,
        })
    }

    fn all_nodes(&self) -> Vec<V> {
        let mut nodes: Vec<V> = self.neighbors.keys().cloned().collect();
        nodes.sort();
        nodes
    }

    fn component_count(&self, edge: &(V, V)) -> Option<usize> {
        self.directed_counts.get(edge).copied()
    }

    fn center_cost(&self, center: &V) -> Result<usize> {
        if let Some(node_costs) = &self.node_costs {
            return node_costs.get(center).copied().ok_or_else(|| {
                anyhow::anyhow!("center {:?} is not present in cost index", center)
            });
        }
        let neighbors = self
            .neighbors
            .get(center)
            .ok_or_else(|| anyhow::anyhow!("center {:?} is not present in cost index", center))?;
        neighbors.iter().try_fold(0usize, |acc, neighbor| {
            self.component_count(&(neighbor.clone(), center.clone()))
                .map(|count| acc + count)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing component cost for directed edge {:?}->{:?}",
                        neighbor,
                        center
                    )
                })
        })
    }

    #[cfg(test)]
    fn from_parts_for_test(
        mut neighbors: HashMap<V, Vec<V>>,
        node_costs: HashMap<V, usize>,
    ) -> Self {
        for neighbor_list in neighbors.values_mut() {
            neighbor_list.sort();
        }
        Self {
            neighbors,
            directed_counts: HashMap::new(),
            node_costs: Some(node_costs),
        }
    }
}

fn intern_component_keys(
    local_keys: &[KeyId],
    incoming: &[&Vec<KeyId>],
    n_points: usize,
    interner: &mut KeyInterner<Vec<KeyId>>,
) -> Vec<KeyId> {
    let mut keys = Vec::with_capacity(n_points);
    for point in 0..n_points {
        let mut tuple = Vec::with_capacity(1 + incoming.len());
        tuple.push(local_keys[point]);
        for incoming_keys in incoming {
            tuple.push(incoming_keys[point]);
        }
        keys.push(interner.intern(tuple));
    }
    keys
}

/// Options controlling cached batch evaluation for [`TreeTN`].
///
/// Use this to pin the contraction center or to configure the greedy automatic
/// center search. When in doubt, leave all fields at their defaults.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CachedEvaluatorOptions;
///
/// let options = CachedEvaluatorOptions::<usize>::default();
/// assert!(options.center.is_none());
/// assert!(options.initial_centers.is_empty());
/// assert!(options.max_greedy_steps_per_start.is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedEvaluatorOptions<V> {
    /// Fixed center node for evaluation.
    ///
    /// When set, greedy center search is skipped. Use this when the caller
    /// already knows where repeated batch structure is concentrated.
    pub center: Option<V>,
    /// Candidate starting centers for greedy automatic center search.
    ///
    /// Empty means all nodes are eligible as starts. Supplying a short list can
    /// reduce center-search overhead for large trees.
    pub initial_centers: Vec<V>,
    /// Maximum number of greedy moves from each initial center.
    ///
    /// `None` means no explicit step limit; the search stops at a local minimum.
    pub max_greedy_steps_per_start: Option<usize>,
}

impl<V> Default for CachedEvaluatorOptions<V> {
    fn default() -> Self {
        Self {
            center: None,
            initial_centers: Vec::new(),
            max_greedy_steps_per_start: None,
        }
    }
}

/// Result of greedy center search for cached TreeTN evaluation.
///
/// The result records the selected node, its estimated cost, and the path taken
/// by greedy descent from the chosen start.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CenterSearchResult;
///
/// let result = CenterSearchResult {
///     center: 2_usize,
///     cost: 7,
///     path: vec![0, 1, 2],
/// };
/// assert_eq!(result.center, 2);
/// assert_eq!(result.cost, 7);
/// assert_eq!(result.path.last(), Some(&2));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CenterSearchResult<V> {
    /// Selected center node.
    pub center: V,
    /// Estimated cache cost at `center`.
    pub cost: usize,
    /// Greedy descent path that produced `center`.
    pub path: Vec<V>,
}

/// Greedy local search for TreeTN cached-evaluation centers.
///
/// This type is intentionally separate from [`TreeTNCachedEvaluator`] so future
/// center-selection algorithms can share the same cost model.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::GreedyCenterSearch;
///
/// let search = GreedyCenterSearch::<usize>::default();
/// assert!(search.max_steps().is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedyCenterSearch<V> {
    max_steps: Option<usize>,
    _marker: std::marker::PhantomData<V>,
}

impl<V> Default for GreedyCenterSearch<V> {
    fn default() -> Self {
        Self {
            max_steps: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<V> GreedyCenterSearch<V> {
    /// Creates a greedy center search with an optional step limit.
    ///
    /// `max_steps` limits the number of edge moves from each start. `None`
    /// searches until no neighbor has lower cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(Some(3));
    /// assert_eq!(search.max_steps(), Some(3));
    /// ```
    pub fn with_max_steps(max_steps: Option<usize>) -> Self {
        Self {
            max_steps,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the optional greedy-step limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(None);
    /// assert_eq!(search.max_steps(), None);
    /// ```
    pub fn max_steps(&self) -> Option<usize> {
        self.max_steps
    }
}

impl<V> GreedyCenterSearch<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn search(
        &self,
        cost_index: &ComponentCostIndex<V>,
        starts: &[V],
    ) -> Result<CenterSearchResult<V>> {
        let owned_starts;
        let starts = if starts.is_empty() {
            owned_starts = cost_index.all_nodes();
            owned_starts.as_slice()
        } else {
            starts
        };
        if starts.is_empty() {
            bail!("GreedyCenterSearch::search: cost index has no nodes");
        }

        let mut best: Option<CenterSearchResult<V>> = None;
        for start in starts {
            if !cost_index.neighbors.contains_key(start) {
                bail!(
                    "GreedyCenterSearch::search: initial center {:?} is not present in TreeTN",
                    start
                );
            }
            let result = self.descend_from(cost_index, start)?;
            match &best {
                None => best = Some(result),
                Some(current) => {
                    if (result.cost, result.center.clone()) < (current.cost, current.center.clone())
                    {
                        best = Some(result);
                    }
                }
            }
        }

        best.ok_or_else(|| anyhow::anyhow!("GreedyCenterSearch::search: no start centers"))
    }

    fn descend_from(
        &self,
        cost_index: &ComponentCostIndex<V>,
        start: &V,
    ) -> Result<CenterSearchResult<V>> {
        let mut center = start.clone();
        let mut cost = cost_index.center_cost(&center)?;
        let mut path = vec![center.clone()];
        let mut steps = 0usize;

        loop {
            if self.max_steps.is_some_and(|max_steps| steps >= max_steps) {
                break;
            }

            let mut candidates = Vec::new();
            for neighbor in cost_index.neighbors.get(&center).unwrap() {
                candidates.push((cost_index.center_cost(neighbor)?, neighbor.clone()));
            }
            candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let Some((next_cost, next_center)) = candidates.into_iter().next() else {
                break;
            };
            if next_cost >= cost {
                break;
            }

            center = next_center;
            cost = next_cost;
            path.push(center.clone());
            steps += 1;
        }

        Ok(CenterSearchResult { center, cost, path })
    }
}

/// Cached batch evaluator for [`TreeTN`].
///
/// Use this when many batch points share repeated assignments on subtrees. It
/// chooses a center node and caches contractions from neighboring components
/// into that center.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen};
/// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
///
/// let s = DynIndex::new_dyn(2);
/// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
/// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
/// let values = [0usize, 1usize];
/// let shape = [1usize, 2usize];
/// let points = ColMajorArrayRef::new(&values, &shape);
///
/// let mut evaluator = TreeTNCachedEvaluator::new(
///     &tree,
///     &[s],
///     CachedEvaluatorOptions::<usize>::default(),
/// )?;
/// let result = evaluator.evaluate_batch(points)?;
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].real(), 4.0);
/// assert_eq!(result[1].real(), 6.0);
/// assert_eq!(evaluator.center(), Some(&0));
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    tree: &'a TreeTN<T, V>,
    layout: EvaluatorLayout<T::Index, V>,
    options: CachedEvaluatorOptions<V>,
    center: Option<V>,
    last_stats: CachedEvaluationStats,
}

impl<'a, T, V> TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
    T::Index: Clone + Eq + Hash,
    <T::Index as IndexLike>::Id: Clone + Eq + Hash + Ord + Debug + Send + Sync,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// Creates a cached evaluator for `tree` and the requested physical indices.
    ///
    /// If `options.center` is set, that node is used directly. Otherwise, the
    /// first call to [`Self::evaluate_batch`] chooses a center with greedy search
    /// using that batch's repeated-subtree structure.
    ///
    /// # Errors
    ///
    /// Returns an error when `indices` are not all present in `tree`, when the
    /// tree is empty, or when a fixed or initial center does not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![5])?;
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions { center: Some(5), ..Default::default() },
    /// )?;
    /// assert_eq!(evaluator.center(), Some(&5));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        tree: &'a TreeTN<T, V>,
        indices: &[T::Index],
        options: CachedEvaluatorOptions<V>,
    ) -> Result<Self> {
        let layout = build_layout(tree, indices)?;
        if let Some(center) = &options.center {
            ensure_node_exists(tree, center, "TreeTNCachedEvaluator::new: center")?;
        }
        for initial_center in &options.initial_centers {
            ensure_node_exists(
                tree,
                initial_center,
                "TreeTNCachedEvaluator::new: initial center",
            )?;
        }
        let center = options.center.clone();
        Ok(Self {
            tree,
            layout,
            options,
            center,
            last_stats: CachedEvaluationStats::default(),
        })
    }

    /// Returns the selected center node, if one has been selected.
    ///
    /// A fixed center is available immediately after [`Self::new`]. An automatic
    /// center is selected during the first [`Self::evaluate_batch`] call.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// assert_eq!(evaluator.center(), None);
    /// let values = [0usize];
    /// let shape = [1usize, 1usize];
    /// let _ = evaluator.evaluate_batch(ColMajorArrayRef::new(&values, &shape))?;
    /// assert_eq!(evaluator.center(), Some(&0));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn center(&self) -> Option<&V> {
        self.center.as_ref()
    }

    /// Evaluates all batch points using cached subtree environments.
    ///
    /// `values` must have shape `[indices.len(), n_points]` in column-major
    /// layout. The returned vector contains one scalar per column.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` has the wrong row count, if any site value
    /// is outside the corresponding index dimension, or if tensor contraction
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let values = [0usize, 1usize];
    /// let shape = [1usize, 2usize];
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// let result = evaluator.evaluate_batch(ColMajorArrayRef::new(&values, &shape))?;
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0].real(), 4.0);
    /// assert_eq!(result[1].real(), 6.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batch(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>> {
        validate_values_shape(
            values,
            self.layout.n_indices,
            "TreeTNCachedEvaluator::evaluate_batch",
        )?;
        let center = self.ensure_center(values)?.clone();
        let component_batches = self.build_component_batches(&center, values)?;
        let environment_cache = self.build_environment_cache(&center, &component_batches)?;
        self.contract_center_for_points(&center, values, &component_batches, &environment_cache)
    }

    fn ensure_center(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<&V> {
        if self.center.is_none() {
            let cost_index = ComponentCostIndex::from_layout(self.tree, &self.layout, values)?;
            let search =
                GreedyCenterSearch::<V>::with_max_steps(self.options.max_greedy_steps_per_start);
            let result = search.search(&cost_index, &self.options.initial_centers)?;
            self.center = Some(result.center);
        }
        Ok(self.center.as_ref().unwrap())
    }

    fn build_component_batches(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<ComponentBatch<T::Index, V>>> {
        let mut batches = Vec::new();
        let n_points = values.shape()[1];
        for neighbor in sorted_neighbors(self.tree)
            .get(center)
            .cloned()
            .unwrap_or_default()
        {
            let nodes = component_nodes(self.tree, &neighbor, center)?;
            let mut entries = Vec::new();
            for node in &nodes {
                if let Some(node_entries) = self.layout.entries_by_node.get(node) {
                    entries.extend(node_entries.iter().cloned());
                }
            }
            entries.sort_by_key(|entry| entry.input_position);

            let mut entry_offsets_by_node: HashMap<V, Vec<(SiteEntry<T::Index>, usize)>> =
                HashMap::new();
            for (offset, entry) in entries.iter().cloned().enumerate() {
                let owner = self
                    .tree
                    .site_index_network()
                    .find_node_by_index(&entry.index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batch: unknown index {:?}",
                            entry.index
                        )
                    })?
                    .clone();
                entry_offsets_by_node
                    .entry(owner)
                    .or_default()
                    .push((entry, offset));
            }

            let mut assignment_ids = HashMap::<ComponentAssignment, usize>::new();
            let mut assignments = Vec::<ComponentAssignment>::new();
            let mut point_to_assignment = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let assignment = ComponentAssignment {
                    values: entries
                        .iter()
                        .map(|entry| *values.get(&[entry.input_position, point]).unwrap())
                        .collect(),
                };
                validate_entry_values(
                    &entries,
                    &assignment.values,
                    "TreeTNCachedEvaluator::evaluate_batch",
                )?;
                let next_id = assignment_ids.len();
                let assignment_id =
                    *assignment_ids.entry(assignment.clone()).or_insert_with(|| {
                        assignments.push(assignment);
                        next_id
                    });
                point_to_assignment.push(assignment_id);
            }

            batches.push(ComponentBatch {
                neighbor,
                nodes,
                entries,
                entry_offsets_by_node,
                point_to_assignment,
                assignments,
            });
        }
        Ok(batches)
    }

    fn build_environment_cache(
        &mut self,
        center: &V,
        component_batches: &[ComponentBatch<T::Index, V>],
    ) -> Result<HashMap<V, Vec<T>>> {
        let mut cache = HashMap::new();
        let mut count = 0usize;
        for batch in component_batches {
            let mut environments = Vec::with_capacity(batch.assignments.len());
            for assignment in &batch.assignments {
                environments.push(self.compute_component_environment(center, batch, assignment)?);
                count += 1;
            }
            cache.insert(batch.neighbor.clone(), environments);
        }
        self.last_stats.subtree_environment_count = count;
        Ok(cache)
    }

    fn compute_component_environment(
        &self,
        _center: &V,
        batch: &ComponentBatch<T::Index, V>,
        assignment: &ComponentAssignment,
    ) -> Result<T> {
        let mut tensors = Vec::with_capacity(batch.nodes.len());
        let mut names = Vec::with_capacity(batch.nodes.len());
        for node in &batch.nodes {
            let tensor = tensor_for_node(self.tree, node)?;
            let index_vals = batch
                .entry_offsets_by_node
                .get(node)
                .map(|offsets| {
                    offsets
                        .iter()
                        .map(|(entry, offset)| (entry.index.clone(), assignment.values[*offset]))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            tensors.push(slice_tensor(tensor, &index_vals).with_context(|| {
                format!(
                    "TreeTNCachedEvaluator::evaluate_batch: failed to slice component node {:?}",
                    node
                )
            })?);
            names.push(node.clone());
        }

        if tensors.len() == 1 {
            return Ok(tensors.remove(0));
        }

        TreeTN::<T, V>::from_tensors(tensors, names)
            .context("TreeTNCachedEvaluator::evaluate_batch: failed to build component TreeTN")?
            .contract_to_tensor()
            .context("TreeTNCachedEvaluator::evaluate_batch: failed to contract component")
    }

    fn contract_center_for_points(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<T::Index, V>],
        environment_cache: &HashMap<V, Vec<T>>,
    ) -> Result<Vec<AnyScalar>> {
        let n_points = values.shape()[1];
        let center_entries = self
            .layout
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let center_tensor = tensor_for_node(self.tree, center)?;
        let scalar_one = T::scalar_one()
            .context("TreeTNCachedEvaluator::evaluate_batch: failed to create scalar")?;

        let mut results = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let center_index_vals = center_entries
                .iter()
                .map(|entry| {
                    let value = *values.get(&[entry.input_position, point]).unwrap();
                    (entry.index.clone(), value)
                })
                .collect::<Vec<_>>();
            validate_index_vals(&center_index_vals, "TreeTNCachedEvaluator::evaluate_batch")?;
            let center_partial = slice_tensor(center_tensor, &center_index_vals)
                .context("TreeTNCachedEvaluator::evaluate_batch: failed to slice center tensor")?;

            let mut tensor_refs = Vec::with_capacity(1 + component_batches.len());
            tensor_refs.push(&center_partial);
            for batch in component_batches {
                let assignment_id = batch.point_to_assignment[point];
                let environment = environment_cache
                    .get(&batch.neighbor)
                    .and_then(|environments| environments.get(assignment_id))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batch: missing cached environment"
                        )
                    })?;
                tensor_refs.push(environment);
            }

            let result_tensor = T::contract(&tensor_refs, AllowedPairs::All)
                .context("TreeTNCachedEvaluator::evaluate_batch: failed to contract center")?;
            results.push(
                scalar_one
                    .inner_product(&result_tensor)
                    .context("TreeTNCachedEvaluator::evaluate_batch: failed to extract scalar")?,
            );
        }

        Ok(results)
    }

    #[cfg(test)]
    fn stats_for_test(&self) -> CachedEvaluationStats {
        self.last_stats.clone()
    }
}

fn build_layout<T, V>(
    tree: &TreeTN<T, V>,
    indices: &[T::Index],
) -> Result<EvaluatorLayout<T::Index, V>>
where
    T: TensorLike,
    T::Index: Clone + Eq + Hash,
    <T::Index as IndexLike>::Id: Ord,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    if tree.node_count() == 0 {
        bail!("TreeTNCachedEvaluator::new: network must have at least one node");
    }

    let total_site_indices = tree.site_index_network().site_index_count();
    anyhow::ensure!(
        indices.len() == total_site_indices,
        "TreeTNCachedEvaluator::new: indices.len() ({}) != total site indices ({})",
        indices.len(),
        total_site_indices
    );

    let mut seen = HashSet::with_capacity(indices.len());
    for index in indices {
        anyhow::ensure!(
            seen.insert(index.clone()),
            "TreeTNCachedEvaluator::new: duplicate index {:?}",
            index
        );
    }

    let mut entries_by_node: HashMap<V, Vec<SiteEntry<T::Index>>> = HashMap::new();
    let mut tensor_indices_by_node: HashMap<V, Vec<T::Index>> = HashMap::new();
    for (input_position, index) in indices.iter().enumerate() {
        let node_name = tree
            .site_index_network()
            .find_node_by_index(index)
            .ok_or_else(|| {
                anyhow::anyhow!("TreeTNCachedEvaluator::new: unknown index {:?}", index)
            })?
            .clone();
        let tensor = tensor_for_node(tree, &node_name)?;
        let tensor_indices = tensor_indices_by_node
            .entry(node_name.clone())
            .or_insert_with(|| tensor.external_indices());
        let local_axis = tensor_indices
            .iter()
            .position(|tensor_index| tensor_index == index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::new: site index {:?} is registered on node {:?} but not present in its tensor",
                    index,
                    node_name
                )
            })?;
        entries_by_node
            .entry(node_name)
            .or_default()
            .push(SiteEntry {
                index: index.clone(),
                input_position,
                local_axis,
            });
    }

    for entries in entries_by_node.values_mut() {
        entries.sort_by_key(|entry| entry.local_axis);
    }

    Ok(EvaluatorLayout {
        entries_by_node,
        n_indices: indices.len(),
    })
}

fn validate_values_shape(
    values: ColMajorArrayRef<'_, usize>,
    n_indices: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        values.shape().len() == 2,
        "{context}: values must be 2D, got {}D",
        values.shape().len()
    );
    anyhow::ensure!(
        values.shape()[0] == n_indices,
        "{context}: row count {} does not match indices.len() {}",
        values.shape()[0],
        n_indices
    );
    Ok(())
}

fn validate_entry_values<I>(entries: &[SiteEntry<I>], values: &[usize], context: &str) -> Result<()>
where
    I: IndexLike,
{
    let index_vals = entries
        .iter()
        .zip(values.iter().copied())
        .map(|(entry, value)| (entry.index.clone(), value))
        .collect::<Vec<_>>();
    validate_index_vals(&index_vals, context)
}

fn validate_index_vals<I>(index_vals: &[(I, usize)], context: &str) -> Result<()>
where
    I: IndexLike,
{
    for (index, value) in index_vals {
        anyhow::ensure!(
            *value < index.dim(),
            "{context}: coordinate {} is out of range for index {:?} with dim {}",
            value,
            index,
            index.dim()
        );
    }
    Ok(())
}

fn ensure_node_exists<T, V>(tree: &TreeTN<T, V>, node: &V, context: &str) -> Result<()>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    if tree.node_index(node).is_none() {
        bail!("{context} {:?} is not present in TreeTN", node);
    }
    Ok(())
}

fn tensor_for_node<'a, T, V>(tree: &'a TreeTN<T, V>, node: &V) -> Result<&'a T>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    let node_idx = tree
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("node {:?} is not present in TreeTN", node))?;
    tree.tensor(node_idx)
        .ok_or_else(|| anyhow::anyhow!("tensor for node {:?} is not present", node))
}

fn slice_tensor<T>(tensor: &T, index_vals: &[(T::Index, usize)]) -> Result<T>
where
    T: TensorLike,
{
    if index_vals.is_empty() {
        return Ok(tensor.clone());
    }
    validate_index_vals(index_vals, "slice_tensor")?;
    let onehot = T::onehot(index_vals)?;
    T::contract(&[tensor, &onehot], AllowedPairs::All)
}

fn sorted_neighbors<T, V>(tree: &TreeTN<T, V>) -> HashMap<V, Vec<V>>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let mut map = HashMap::new();
    let mut node_names = tree.node_names();
    node_names.sort();
    for node in node_names {
        let mut neighbors: Vec<V> = tree.site_index_network().neighbors(&node).collect();
        neighbors.sort();
        map.insert(node, neighbors);
    }
    map
}

fn rooted_tree<V>(
    neighbors: &HashMap<V, Vec<V>>,
    root: &V,
) -> Result<(HashMap<V, Option<V>>, Vec<V>)>
where
    V: Clone + Eq + Hash + Ord + Debug,
{
    let mut parent = HashMap::<V, Option<V>>::new();
    let mut order = Vec::<V>::new();
    let mut stack = vec![(root.clone(), None)];
    while let Some((node, parent_node)) = stack.pop() {
        if parent.contains_key(&node) {
            continue;
        }
        parent.insert(node.clone(), parent_node.clone());
        order.push(node.clone());
        let mut children = neighbors
            .get(&node)
            .ok_or_else(|| anyhow::anyhow!("node {:?} is missing from neighbor map", node))?
            .iter()
            .filter(|neighbor| Some(*neighbor) != parent_node.as_ref())
            .cloned()
            .collect::<Vec<_>>();
        children.sort_by(|a, b| b.cmp(a));
        for child in children {
            stack.push((child, Some(node.clone())));
        }
    }

    anyhow::ensure!(
        parent.len() == neighbors.len(),
        "TreeTN topology is disconnected: reached {} of {} nodes",
        parent.len(),
        neighbors.len()
    );
    Ok((parent, order))
}

fn component_nodes<T, V>(tree: &TreeTN<T, V>, start: &V, blocked: &V) -> Result<Vec<V>>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let neighbors = sorted_neighbors(tree);
    let mut nodes = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![start.clone()];
    while let Some(node) = stack.pop() {
        if &node == blocked || !seen.insert(node.clone()) {
            continue;
        }
        nodes.push(node.clone());
        for neighbor in neighbors.get(&node).cloned().unwrap_or_default() {
            if &neighbor != blocked && !seen.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }
    nodes.sort();
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen};

    fn assert_scalars_close(actual: &[AnyScalar], expected: &[AnyScalar]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual.real() - expected.real()).abs() < 1.0e-12,
                "actual={} expected={}",
                actual.real(),
                expected.real()
            );
        }
    }

    fn two_node_tree() -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);

        let t0 =
            TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap();
        let t1 =
            TensorDynLen::from_dense(vec![bond, s1.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();

        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
        (tree, vec![s0, s1])
    }

    fn three_node_chain() -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64; 4]).unwrap();
        let t1 =
            TensorDynLen::from_dense(vec![b01, s1.clone(), b12.clone()], vec![1.0_f64; 8]).unwrap();
        let t2 = TensorDynLen::from_dense(vec![b12, s2.clone()], vec![1.0_f64; 4]).unwrap();
        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
        (tree, vec![s0, s1, s2])
    }

    fn star_tree() -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
        let sc = DynIndex::new_dyn(2);
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);
        let b0 = DynIndex::new_dyn(2);
        let b1 = DynIndex::new_dyn(2);
        let b2 = DynIndex::new_dyn(2);
        let center_data: Vec<f64> = (0..16).map(|value| value as f64 + 1.0).collect();
        let center = TensorDynLen::from_dense(
            vec![sc.clone(), b0.clone(), b1.clone(), b2.clone()],
            center_data,
        )
        .unwrap();
        let leaf0 =
            TensorDynLen::from_dense(vec![b0, s0.clone()], vec![1.0_f64, 0.5, 1.5, 2.0]).unwrap();
        let leaf1 =
            TensorDynLen::from_dense(vec![b1, s1.clone()], vec![0.25_f64, 1.0, 1.25, 2.0]).unwrap();
        let leaf2 =
            TensorDynLen::from_dense(vec![b2, s2.clone()], vec![2.0_f64, 1.0, 0.75, 1.5]).unwrap();
        let tree =
            TreeTN::<_, usize>::from_tensors(vec![center, leaf0, leaf1, leaf2], vec![0, 1, 2, 3])
                .unwrap();
        (tree, vec![sc, s0, s1, s2])
    }

    #[test]
    fn cached_evaluator_matches_tree_evaluate_on_two_node_chain() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 0, 1, 0, 0, 1, 1, 1];
        let shape = [2, 4];
        let points = ColMajorArrayRef::new(&values, &shape);

        let expected = tree.evaluate(&indices, points).unwrap();
        let options = CachedEvaluatorOptions {
            center: Some(0),
            ..CachedEvaluatorOptions::default()
        };
        let mut evaluator = TreeTNCachedEvaluator::new(&tree, &indices, options).unwrap();
        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.center(), Some(&0));
    }

    #[test]
    fn component_cost_index_counts_unique_directed_components() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
        let shape = [3, 3];
        let points = ColMajorArrayRef::new(&values, &shape);

        let cost_index = ComponentCostIndex::new(&tree, &indices, points).unwrap();

        assert_eq!(cost_index.component_count(&(0, 1)).unwrap(), 2);
        assert_eq!(cost_index.component_count(&(1, 0)).unwrap(), 2);
        assert_eq!(cost_index.component_count(&(1, 2)).unwrap(), 3);
        assert_eq!(cost_index.component_count(&(2, 1)).unwrap(), 2);
        assert_eq!(cost_index.center_cost(&0).unwrap(), 2);
        assert_eq!(cost_index.center_cost(&1).unwrap(), 4);
        assert_eq!(cost_index.center_cost(&2).unwrap(), 3);
    }

    #[test]
    fn component_cost_index_rejects_wrong_batch_row_count() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 1, 1, 0];
        let shape = [2, 2];
        let points = ColMajorArrayRef::new(&values, &shape);

        let err = ComponentCostIndex::new(&tree, &indices, points)
            .err()
            .unwrap();
        assert!(err.to_string().contains("row count"));
    }

    #[test]
    fn greedy_center_search_descends_to_lower_cost_neighbor() {
        let cost_index = ComponentCostIndex::from_parts_for_test(
            HashMap::from([(0, vec![1]), (1, vec![0, 2]), (2, vec![1, 3]), (3, vec![2])]),
            HashMap::from([(0, 40), (1, 20), (2, 10), (3, 15)]),
        );

        let result = GreedyCenterSearch::<usize>::default()
            .search(&cost_index, &[0])
            .unwrap();

        assert_eq!(result.center, 2);
        assert_eq!(result.cost, 10);
        assert_eq!(result.path, vec![0, 1, 2]);
    }

    #[test]
    fn greedy_center_search_uses_best_of_multiple_starts() {
        let cost_index = ComponentCostIndex::from_parts_for_test(
            HashMap::from([
                ("a", vec!["b"]),
                ("b", vec!["a", "c"]),
                ("c", vec!["b", "d"]),
                ("d", vec!["c"]),
            ]),
            HashMap::from([("a", 8), ("b", 6), ("c", 5), ("d", 2)]),
        );

        let result = GreedyCenterSearch::<&str>::with_max_steps(Some(1))
            .search(&cost_index, &["a", "d"])
            .unwrap();

        assert_eq!(result.center, "d");
        assert_eq!(result.cost, 2);
    }

    #[test]
    fn cached_evaluator_selects_greedy_center_when_center_is_not_fixed() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
        let shape = [3, 3];
        let points = ColMajorArrayRef::new(&values, &shape);

        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                initial_centers: vec![1],
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_eq!(evaluator.center(), Some(&0));
        assert_scalars_close(&actual, &expected);
    }

    #[test]
    fn cached_evaluator_rejects_unknown_initial_center() {
        let (tree, indices) = three_node_chain();
        let err = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                initial_centers: vec![99],
                ..Default::default()
            },
        )
        .err()
        .unwrap();

        assert!(err.to_string().contains("initial center"));
    }

    #[test]
    fn cached_evaluator_computes_one_environment_per_unique_subtree_assignment() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1];
        let shape = [3, 4];
        let points = ColMajorArrayRef::new(&values, &shape);

        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.stats_for_test().subtree_environment_count, 4);
    }

    #[test]
    fn cached_evaluator_rejects_wrong_value_row_count() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 1, 1];
        let shape = [1, 3];
        let points = ColMajorArrayRef::new(&values, &shape);
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let err = evaluator.evaluate_batch(points).unwrap_err();
        assert!(err.to_string().contains("row count"));
    }

    #[test]
    fn cached_evaluator_rejects_out_of_range_site_value() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 2];
        let shape = [2, 1];
        let points = ColMajorArrayRef::new(&values, &shape);
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let err = evaluator.evaluate_batch(points).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn cached_evaluator_handles_repeated_points_without_changing_order() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let shape = [2, 4];
        let points = ColMajorArrayRef::new(&values, &shape);
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(actual[0].real(), actual[2].real());
        assert_eq!(actual[1].real(), actual[3].real());
    }

    #[test]
    fn cached_evaluator_matches_tree_evaluate_on_star_tree() {
        let (tree, indices) = star_tree();
        let values = vec![
            0, 0, 0, 0, //
            1, 0, 1, 0, //
            0, 1, 0, 1, //
            1, 1, 1, 1,
        ];
        let shape = [4, 4];
        let points = ColMajorArrayRef::new(&values, &shape);
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_scalars_close(&actual, &expected);
    }
}
