# TreeTN Cached Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a cached batch evaluator for `TreeTN` with greedy center selection, correctness tests, and benchmarks comparing TreeTN chain performance with `TTCache`.

**Architecture:** The implementation adds a public `TreeTNCachedEvaluator` API under `treetn::cached_evaluator`. Center selection is separated into a reusable greedy search over an internal component-cost index, and evaluation caches subtree environments for repeated batch restrictions around the selected center. Benchmarks compare current uncached `TreeTN::evaluate`, new cached TreeTN evaluation on a linear tree, and `TTCache::evaluate_many`.

**Tech Stack:** Rust, `tensor4all-core` tensor abstractions, existing `TreeTN`/`TreeTNEvaluator`, `tensor4all-simplett::cache::TTCache`, `criterion`, `cargo nextest`.

---

### Task 1: Add the public API shape with a failing correctness test

**Files:**
- Create: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/mod.rs`
- Modify: `crates/tensor4all-treetn/src/lib.rs`

**Step 1: Write the failing test**

Add this test module at the bottom of `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen, TensorLike};

    fn two_node_tree() -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense(
            vec![s0.clone(), bond.clone()],
            vec![1.0_f64, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let t1 = TensorDynLen::from_dense(
            vec![bond.clone(), s1.clone()],
            vec![0.5_f64, 1.5, 2.5, 3.5],
        )
        .unwrap();

        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
        (tree, vec![s0, s1])
    }

    #[test]
    fn cached_evaluator_matches_tree_evaluate_on_two_node_chain() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 0, 1, 0, 0, 1, 1, 1];
        let points = ColMajorArrayRef::new(&values, &[2, 4]).unwrap();

        let expected = tree.evaluate(&indices, points.clone()).unwrap();
        let options = CachedEvaluatorOptions {
            center: Some(0),
            ..CachedEvaluatorOptions::default()
        };
        let mut evaluator = TreeTNCachedEvaluator::new(&tree, &indices, options).unwrap();
        let actual = evaluator.evaluate_batch(points).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.real(), expected.real(), epsilon = 1e-12);
        }
        assert_eq!(evaluator.center(), &0);
    }
}
```

**Step 2: Run the test to verify it fails**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_matches_tree_evaluate_on_two_node_chain
```

Expected: FAIL to compile because `cached_evaluator`, `TreeTNCachedEvaluator`, and `CachedEvaluatorOptions` do not exist.

**Step 3: Add the minimal public module and fallback implementation**

In `crates/tensor4all-treetn/src/treetn/mod.rs`, add:

```rust
mod cached_evaluator;
pub use cached_evaluator::{
    CachedEvaluatorOptions, CenterSearchResult, GreedyCenterSearch, TreeTNCachedEvaluator,
};
```

In `crates/tensor4all-treetn/src/lib.rs`, add these names to the existing `pub use treetn::{ ... }` list:

```rust
CachedEvaluatorOptions,
CenterSearchResult,
GreedyCenterSearch,
TreeTNCachedEvaluator,
```

Create `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` with the minimal API and rustdoc:

```rust
//! Cached batch evaluation for tree tensor networks.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{bail, Result};
use tensor4all_core::{AnyScalar, ColMajorArrayRef, TensorLike};

use super::{TreeTN, TreeTNEvaluator};

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
    /// Empty means all nodes are eligible as starts. Supplying a short list
    /// can reduce center-search overhead for large trees.
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

/// Cached batch evaluator for [`TreeTN`].
///
/// Use this when many batch points share repeated assignments on subtrees. It
/// chooses a center node and caches contractions from neighboring components
/// into that center.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen, TensorLike};
/// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
///
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(vec![s0.clone()], vec![2.0_f64, 3.0]).unwrap();
/// let t1 = TensorDynLen::from_dense(vec![s1.clone()], vec![5.0_f64, 7.0]).unwrap();
/// let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
/// let indices = vec![s0, s1];
/// let values = vec![0, 0, 1, 1];
/// let points = ColMajorArrayRef::new(&values, &[2, 2]).unwrap();
///
/// let mut evaluator = TreeTNCachedEvaluator::new(
///     &tree,
///     &indices,
///     CachedEvaluatorOptions { center: Some(0), ..Default::default() },
/// ).unwrap();
/// let result = evaluator.evaluate_batch(points).unwrap();
/// assert_eq!(result.len(), 2);
/// assert_eq!(evaluator.center(), &0);
/// ```
pub struct TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
{
    evaluator: TreeTNEvaluator<'a, T, V>,
    center: V,
}

impl<'a, T, V> TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug,
{
    /// Creates a cached evaluator for `tree` and the requested physical indices.
    ///
    /// If `options.center` is set, that node is used directly. Otherwise, this
    /// initially uses the first tree node; later tasks replace this fallback with
    /// greedy center selection.
    ///
    /// # Errors
    ///
    /// Returns an error when `indices` are not all present in `tree`, when the
    /// tree is empty, or when a fixed center does not exist in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap();
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![t], vec![5]).unwrap();
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions { center: Some(5), ..Default::default() },
    /// ).unwrap();
    /// assert_eq!(evaluator.center(), &5);
    /// ```
    pub fn new(
        tree: &'a TreeTN<T, V>,
        indices: &[T::Index],
        options: CachedEvaluatorOptions<V>,
    ) -> Result<Self> {
        let evaluator = TreeTNEvaluator::new(tree, indices)?;
        let center = if let Some(center) = options.center {
            if !tree.contains_node(&center) {
                bail!("center node {:?} is not present in TreeTN", center);
            }
            center
        } else {
            tree.node_names()
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("cannot create cached evaluator for empty TreeTN"))?
        };
        Ok(Self { evaluator, center })
    }

    /// Returns the center node used by this evaluator.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap();
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// ).unwrap();
    /// assert_eq!(evaluator.center(), &0);
    /// ```
    pub fn center(&self) -> &V {
        &self.center
    }

    /// Evaluates all batch points.
    ///
    /// `values` must have shape `[indices.len(), n_points]` in column-major
    /// layout. The returned vector contains one scalar per column.
    ///
    /// # Errors
    ///
    /// Returns an error if `values` has the wrong row count or if any site value
    /// is outside the corresponding index dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s.clone()], vec![4.0_f64, 6.0]).unwrap();
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    /// let values = vec![0, 1];
    /// let points = ColMajorArrayRef::new(&values, &[1, 2]).unwrap();
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// ).unwrap();
    /// let result = evaluator.evaluate_batch(points).unwrap();
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0].real(), 4.0);
    /// assert_eq!(result[1].real(), 6.0);
    /// ```
    pub fn evaluate_batch(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<Vec<AnyScalar>> {
        self.evaluator.evaluate_batch(values)
    }
}
```

If `tree.contains_node` or `tree.node_names` do not exist with these exact names, use the existing public accessors from `TreeTN` and keep the API behavior identical.

**Step 4: Run the test to verify it passes**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_matches_tree_evaluate_on_two_node_chain
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/src/treetn/mod.rs crates/tensor4all-treetn/src/lib.rs
git commit -m "feat(treetn): add cached evaluator API"
```

### Task 2: Add exact component-cost precomputation

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing tests for component counts**

Add private tests in the existing test module:

```rust
#[test]
fn component_cost_index_counts_unique_directed_components() {
    let (tree, indices) = three_node_chain();
    let values = vec![
        0, 0, 0,
        0, 1, 1,
        1, 1, 1,
    ];
    let points = ColMajorArrayRef::new(&values, &[3, 3]).unwrap();

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
    let points = ColMajorArrayRef::new(&values, &[2, 2]).unwrap();

    let err = ComponentCostIndex::new(&tree, &indices, points).unwrap_err();
    assert!(err.to_string().contains("row count"));
}
```

Add this helper:

```rust
fn three_node_chain() -> (TreeTN<TensorDynLen, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64; 4]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        vec![1.0_f64; 8],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(vec![b12, s2.clone()], vec![1.0_f64; 4]).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-treetn component_cost_index
```

Expected: FAIL because `ComponentCostIndex` does not exist.

**Step 3: Implement exact component-cost indexing**

Add a private exact key interner and `ComponentCostIndex`:

```rust
type KeyId = usize;

#[derive(Default)]
struct KeyInterner<T>
where
    T: Eq + Hash + Clone,
{
    ids: HashMap<T, KeyId>,
}

impl<T> KeyInterner<T>
where
    T: Eq + Hash + Clone,
{
    fn intern(&mut self, key: T) -> KeyId {
        let next = self.ids.len();
        *self.ids.entry(key).or_insert(next)
    }
}

struct ComponentCostIndex<V> {
    neighbors: HashMap<V, Vec<V>>,
    directed_counts: HashMap<(V, V), usize>,
}
```

Implementation requirements:

- Validate `values.shape()[0] == indices.len()`.
- Build a deterministic map from each requested index to the owning TreeTN node using the same ownership rules as `TreeTNEvaluator`.
- For each tree node and batch point, intern the node-local site assignment into a `KeyId`.
- Root the tree at the smallest node name (`Ord`) for deterministic traversal.
- Compute exact component keys for all child-to-parent directed edges in postorder.
- Compute exact component keys for all parent-to-child directed edges in preorder using parent outside keys plus sibling child keys.
- Count unique `KeyId` values per directed edge.
- Define `center_cost(center)` as the sum of `component_count((neighbor, center))` for every neighbor of `center`.
- The implementation may store `Vec<KeyId>` per directed edge during construction, but only `directed_counts` is needed after construction.
- Do not use hash-only fingerprints as semantic keys; interning must compare exact tuples so collisions cannot change costs.

**Step 4: Run tests**

Run:

```bash
cargo test --release -p tensor4all-treetn component_cost_index
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): add cached evaluator component costs"
```

### Task 3: Implement reusable greedy center search

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing greedy-search tests**

Add:

```rust
#[test]
fn greedy_center_search_descends_to_lower_cost_neighbor() {
    let cost_index = ComponentCostIndex::from_parts_for_test(
        HashMap::from([
            (0, vec![1]),
            (1, vec![0, 2]),
            (2, vec![1, 3]),
            (3, vec![2]),
        ]),
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
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-treetn greedy_center_search
```

Expected: FAIL because `search` and the test-only constructor do not exist.

**Step 3: Implement search**

Add:

```rust
impl<V> GreedyCenterSearch<V>
where
    V: Clone + Eq + Hash + Ord + Debug,
{
    pub(crate) fn search(
        &self,
        cost_index: &ComponentCostIndex<V>,
        starts: &[V],
    ) -> Result<CenterSearchResult<V>> {
        // For each start, repeatedly move to the lowest-cost neighbor when it
        // is strictly lower than the current center cost. Break ties by node
        // order for deterministic results.
        // Return the best terminal result among all starts.
    }
}
```

Rules:

- Empty `starts` means use all nodes in sorted order.
- A start not present in the tree is an error.
- Only move on strict cost improvement; equal-cost neighbors do not move.
- Tie-break terminal candidates by `(cost, center)` in ascending order.
- Respect `max_steps` when set.

**Step 4: Run tests**

Run:

```bash
cargo test --release -p tensor4all-treetn greedy_center_search
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): add greedy center search"
```

### Task 4: Wire automatic center selection into `TreeTNCachedEvaluator`

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing center-selection integration tests**

Add:

```rust
#[test]
fn cached_evaluator_selects_greedy_center_when_center_is_not_fixed() {
    let (tree, indices) = three_node_chain();
    let values = vec![
        0, 0, 0,
        0, 1, 1,
        1, 1, 1,
    ];
    let points = ColMajorArrayRef::new(&values, &[3, 3]).unwrap();

    let expected = tree.evaluate(&indices, points.clone()).unwrap();
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

    assert_eq!(evaluator.center(), &0);
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.real(), expected.real(), epsilon = 1e-12);
    }
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
    .unwrap_err();

    assert!(err.to_string().contains("initial center"));
}
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_selects_greedy_center
```

Expected: FAIL because `TreeTNCachedEvaluator::new` still uses the first node instead of cost-based center selection.

**Step 3: Implement center selection**

Update `TreeTNCachedEvaluator::new`:

- If `options.center` is `Some`, validate it and use it.
- If `options.center` is `None`, delay center selection until the first `evaluate_batch`, because the cost model depends on the batch values.
- Store `options` in the evaluator.
- Change the struct fields to:

```rust
pub struct TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
{
    evaluator: TreeTNEvaluator<'a, T, V>,
    options: CachedEvaluatorOptions<V>,
    center: Option<V>,
}
```

- Change `center(&self)` to return `Option<&V>` or keep `&V` only if `new` computes a provisional center. Prefer preserving the planned API `center(&self) -> &V` by computing the center in `new` only for fixed-center construction and requiring the first `evaluate_batch` test to inspect `center()` after evaluation. If this is too awkward, add `selected_center(&self) -> Option<&V>` and keep `center()` only after selection with a documented panic. Avoid panics in normal API; the preferred public API is:

```rust
pub fn center(&self) -> Option<&V> {
    self.center.as_ref()
}
```

Update rustdoc examples and Task 1 assertions accordingly.

- In `evaluate_batch`, when `center` is `None`, build `ComponentCostIndex`, run `GreedyCenterSearch`, and set `self.center`.

**Step 4: Run tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): select cached evaluator center greedily"
```

### Task 5: Replace fallback evaluation with cached subtree environments

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing tests proving environment reuse**

Add a private test-only stats accessor:

```rust
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CachedEvaluationStats {
    subtree_environment_count: usize,
}
```

Add tests:

```rust
#[test]
fn cached_evaluator_computes_one_environment_per_unique_subtree_assignment() {
    let (tree, indices) = three_node_chain();
    let values = vec![
        0, 0, 0,
        1, 0, 0,
        0, 1, 1,
        1, 1, 1,
    ];
    let points = ColMajorArrayRef::new(&values, &[3, 4]).unwrap();

    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let expected = tree.evaluate(&indices, points.clone()).unwrap();
    let actual = evaluator.evaluate_batch(points).unwrap();

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.real(), expected.real(), epsilon = 1e-12);
    }
    assert_eq!(evaluator.stats_for_test().subtree_environment_count, 4);
}
```

For this batch and center `1`, the left component has two unique assignments and the right component has two unique assignments, so the cached evaluator should build four subtree environments instead of eight point-wise neighbor environments.

**Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_computes_one_environment
```

Expected: FAIL because fallback evaluation does not compute or report cached environments.

**Step 3: Implement cached evaluation**

Add internal structures:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ComponentAssignment {
    values: Vec<usize>,
}

struct ComponentBatch<V> {
    neighbor: V,
    nodes: Vec<V>,
    point_to_assignment: Vec<usize>,
    assignments: Vec<ComponentAssignment>,
}
```

Implementation requirements:

- For the selected center, enumerate each neighbor component with DFS from the neighbor while excluding the center.
- For each component, collect requested index rows owned by nodes in that component in deterministic node/index order.
- Deduplicate each point's component assignment with `HashMap<ComponentAssignment, usize>`.
- For each unique assignment, compute exactly one environment tensor from the component into the center.
- For each batch point, contract the center tensor sliced by center-owned physical indices with the matching neighbor environments.
- Keep using existing `TensorLike` and `TreeTNEvaluator` helpers where possible; do not access low-level tensor internals from another crate.
- It is acceptable to introduce private helper methods in `TreeTNEvaluator` if they are the correct abstraction for mapping requested indices to nodes and slicing node tensors.
- Do not add public scalar-specific APIs.
- Preserve the public `evaluate_batch` result order and error behavior.

Suggested helper layout:

```rust
impl<'a, T, V> TreeTNCachedEvaluator<'a, T, V>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug,
{
    fn evaluate_with_cache(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<Vec<AnyScalar>> {
        let center = self.ensure_center(values.clone())?.clone();
        let component_batches = self.build_component_batches(&center, values.clone())?;
        let environment_cache = self.build_environment_cache(&center, &component_batches)?;
        self.contract_center_for_points(&center, values, &component_batches, &environment_cache)
    }
}
```

**Step 4: Run focused tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): cache batch subtree environments"
```

### Task 6: Add error-path and layout tests

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write tests**

Add tests for:

```rust
#[test]
fn cached_evaluator_rejects_wrong_value_row_count() { /* wrong rows */ }

#[test]
fn cached_evaluator_rejects_out_of_range_site_value() { /* value == dim */ }

#[test]
fn cached_evaluator_handles_repeated_points_without_changing_order() { /* duplicate columns */ }

#[test]
fn cached_evaluator_matches_tree_evaluate_on_star_tree() { /* center with degree > 2 */ }
```

Each test should compare error messages or compare full result vectors to `TreeTN::evaluate` using `AnyScalar::real()` and `assert_abs_diff_eq!`.

**Step 2: Run tests**

```bash
cargo test --release -p tensor4all-treetn cached_evaluator
```

Expected: PASS.

**Step 3: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "test(treetn): cover cached evaluator edge cases"
```

### Task 7: Add TreeTN cached evaluator benchmark

**Files:**
- Modify: `crates/tensor4all-treetn/Cargo.toml`
- Create: `crates/tensor4all-treetn/benches/cached_evaluator.rs`

**Step 1: Update Cargo manifest**

Add to `[dev-dependencies]`:

```toml
criterion.workspace = true
```

Add:

```toml
[[bench]]
name = "cached_evaluator"
harness = false
```

**Step 2: Create benchmark**

Implement `crates/tensor4all-treetn/benches/cached_evaluator.rs` with:

- `create_tt_with_bond_dim(n_sites, local_dim, bond_dim) -> TensorTrain<f64>` copied from the existing simplett cache benchmark, adjusted only as needed.
- `generate_tci_like_indices(n_left, n_right, n_sites, local_dim, split, seed) -> Vec<Vec<usize>>`.
- `multi_indices_to_col_major(indices: &[Vec<usize>]) -> Vec<usize>`.
- `bench_chain_size_scaling`: for `n_sites in [16, 32, 64, 128]`, compare:
  - `TTCache::evaluate_many(&indices, None)`
  - `TreeTNCachedEvaluator::new(...).evaluate_batch(...)`
  - current `tree.evaluate(&site_indices, values_ref)`
- `bench_batch_size_scaling`: for `(n_left, n_right) in [(10, 10), (30, 30), (100, 100)]`, compare the same three methods on fixed `n_sites = 64`.
- Use `tensor_train_to_treetn` to build the equivalent linear-chain TreeTN.
- Use `criterion_group!` and `criterion_main!`.

Keep sample sizes small enough for local iteration:

```rust
let mut group = c.benchmark_group("chain_size_scaling");
group.sample_size(10);
```

**Step 3: Compile and smoke-run the benchmark**

Run:

```bash
cargo bench -p tensor4all-treetn --bench cached_evaluator -- --sample-size 10
```

Expected: benchmark compiles and emits Criterion timing output. If it is too slow locally, reduce only benchmark sample sizes or benchmark matrix sizes; do not weaken correctness tests.

**Step 4: Commit**

```bash
git add crates/tensor4all-treetn/Cargo.toml crates/tensor4all-treetn/benches/cached_evaluator.rs
git commit -m "bench(treetn): compare cached batch evaluation"
```

### Task 8: Add or update rustdoc and API exports

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/src/lib.rs`
- Modify if needed: `README.md`

**Step 1: Audit public docs**

Verify every new public item has rustdoc with runnable examples and assertions:

- `CachedEvaluatorOptions`
- `CenterSearchResult`
- `GreedyCenterSearch`
- `TreeTNCachedEvaluator`
- public methods on these types

If the examples from Task 1 changed because `center()` returns `Option<&V>`, update every example to assert `Some(&center)`.

**Step 2: Run doctests**

Run:

```bash
cargo test --doc --release -p tensor4all-treetn
```

Expected: PASS.

**Step 3: Run rustdoc build**

Run:

```bash
cargo doc --workspace --no-deps
```

Expected: PASS without missing-doc warnings from the new public API.

**Step 4: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/src/lib.rs README.md
git commit -m "docs(treetn): document cached evaluator API"
```

### Task 9: Final verification before PR

**Files:**
- No code changes expected unless verification reveals an issue.

**Step 1: Format**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
```

Expected: PASS.

If formatting changed files, commit them:

```bash
git add crates/tensor4all-treetn
git commit -m "style: format cached evaluator changes"
```

**Step 2: Run focused checks**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator
cargo test --release -p tensor4all-simplett cache
cargo test --doc --release -p tensor4all-treetn
```

Expected: PASS.

**Step 3: Run lint**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS.

**Step 4: Run full release test suite if time allows**

Run:

```bash
cargo nextest run --release --workspace
```

Expected: PASS. If this is too slow for the current turn, run the focused checks above and state clearly in the PR body that full workspace nextest was not run locally.

**Step 5: Commit any verification fixes**

If any code or docs changed during verification:

```bash
git add <changed-files>
git commit -m "fix(treetn): address cached evaluator verification"
```

### Task 10: Push and open a draft PR

**Files:**
- No source edits expected.

**Step 1: Inspect final diff and status**

Run:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
```

Expected: branch contains only the design doc, implementation plan, TreeTN cached evaluator code/tests/docs, and benchmark changes.

**Step 2: Push**

Run:

```bash
git push -u origin "$(git branch --show-current)"
```

Expected: push succeeds.

**Step 3: Create draft PR**

Use the GitHub app if possible, otherwise fallback to:

```bash
gh pr create --draft --base main --head "$(git branch --show-current)" --title "[codex] Add TreeTN cached batch evaluator" --body-file /tmp/treetn-cached-evaluator-pr.md
```

PR body must include:

- Summary of the cached evaluator API and greedy center search.
- Benchmark coverage and what the benchmark compares.
- Tests/checks run.
- Mention that this is part of the batch issue-fix branch and addresses issue `#464`; mention `#380/#383` only if the final implementation directly changes their behavior.

**Step 4: Report**

Report branch, PR URL, commits, checks run, and any checks not run.
