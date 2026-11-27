//! Candidate-matrix evaluation across one tree edge (#1481).

use super::*;
use tensor4all_core::MultiIndex;

/// One side of a candidate matrix: a set of sites and the candidates on them.
///
/// `sites` are positions in the evaluator's index list (the `indices` given to
/// [`TreeTNCachedEvaluator::new`]), in any order but each at most once.
/// Candidate `c` assigns `candidates[c][k]` to site `sites[k]`, so every
/// candidate has `sites.len()` values. This is the form TreeTCI holds for the
/// two sides of an edge (a subtree key and its `MultiIndex` candidates), so it
/// is passed without a copy.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CandidateSide;
///
/// let candidates = vec![vec![0, 1], vec![1, 1]];
/// let side = CandidateSide { sites: &[2, 0], candidates: &candidates };
/// assert_eq!(side.sites.len(), side.candidates[0].len());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CandidateSide<'a> {
    /// Positions in the evaluator's index list of the sites on this side.
    pub sites: &'a [usize],
    /// One multi-index per candidate, its values in `sites` order.
    pub candidates: &'a [MultiIndex],
}

/// Which entries of a candidate matrix to evaluate.
///
/// Non-exhaustive so later selections can be added without breaking callers.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CandidateEntries;
///
/// let misses = [(0usize, 1usize), (2, 0)];
/// assert!(matches!(CandidateEntries::Pairs(&misses), CandidateEntries::Pairs(p) if p.len() == 2));
/// assert!(matches!(CandidateEntries::All, CandidateEntries::All));
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum CandidateEntries<'a> {
    /// The full `n_left x n_right` matrix, column-major with the left
    /// candidate varying fastest: entry `(i, j)` is at `i + n_left * j`.
    All,
    /// Only these `(left, right)` candidate pairs (a memo's misses, say), in
    /// the given order. Repeated pairs are allowed and answered each time.
    Pairs(&'a [(usize, usize)]),
}

/// Why a candidate-matrix request was rejected before any contraction.
///
/// Reported inside [`TreeTNOperationError::InvalidCandidates`] by
/// [`TreeTNCachedEvaluator::evaluate_candidates_typed`]. Each variant names
/// one failure and its remedy. `side` is `"left"` or `"right"`; `candidate`
/// is a position in that side's `candidates`.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CandidateInputError;
///
/// let error = CandidateInputError::MissingSite { site: 3 };
/// assert!(error.to_string().contains("site 3"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CandidateInputError {
    /// A site position is not a position in the evaluator's index list.
    ///
    /// Remedy: pass positions `0..n_indices` into the index list given to
    /// [`TreeTNCachedEvaluator::new`], not site or node labels.
    #[error(
        "{side} site {site} is out of range for an evaluator of {n_indices} indices; \
         sites are positions in the evaluator's index list"
    )]
    SiteOutOfRange {
        /// `"left"` or `"right"`.
        side: &'static str,
        /// The offending position.
        site: usize,
        /// Length of the evaluator's index list.
        n_indices: usize,
    },
    /// A site is listed twice, on one side or on both.
    ///
    /// Remedy: give every site to exactly one side, once.
    #[error("site {site} is listed more than once; give every site to exactly one side, once")]
    DuplicateSite {
        /// The repeated position.
        site: usize,
    },
    /// A site of the evaluator is on neither side.
    ///
    /// Remedy: the two sides together must list every index of the evaluator.
    #[error(
        "site {site} is on neither side; the two sides together must list every index of \
         the evaluator"
    )]
    MissingSite {
        /// The first position missing from both sides.
        site: usize,
    },
    /// The two site sets cover the evaluator, but no tree edge separates them.
    ///
    /// Remedy: the left sites must be exactly the sites on one side of one
    /// edge of the tree (the sites of a subtree hanging from that edge).
    #[error(
        "no edge of the tree separates the left sites from the right sites; the left sites \
         must be exactly the sites on one side of one edge"
    )]
    NotAnEdgeSplit,
    /// A candidate does not have one value per site of its side.
    ///
    /// Remedy: give each candidate exactly `sites.len()` values, in `sites`
    /// order.
    #[error(
        "{side} candidate {candidate} has {len} values but its side has {expected} sites; \
         give one value per site, in sites order"
    )]
    CandidateLength {
        /// `"left"` or `"right"`.
        side: &'static str,
        /// Position of the candidate in its side.
        candidate: usize,
        /// Number of values it has.
        len: usize,
        /// Number of sites on its side.
        expected: usize,
    },
    /// A candidate value is at or above the dimension of its site.
    ///
    /// Remedy: values must lie in `0..dim` of the site's index.
    #[error(
        "{side} candidate {candidate} has value {value} at site {site} of dimension {dim}; \
         values must lie in 0..dim"
    )]
    CandidateValueOutOfRange {
        /// `"left"` or `"right"`.
        side: &'static str,
        /// Position of the candidate in its side.
        candidate: usize,
        /// Position of the site in the evaluator's index list.
        site: usize,
        /// The offending value.
        value: usize,
        /// Dimension of the site's index.
        dim: usize,
    },
    /// A requested pair names a candidate that does not exist.
    ///
    /// Remedy: pair indices are positions in the two sides' `candidates`,
    /// below `n_left` and `n_right`.
    #[error(
        "pair {pair} is ({left}, {right}) but the sides have {n_left} left and {n_right} right \
         candidates; pair indices are positions in the two candidate lists"
    )]
    PairOutOfRange {
        /// Position of the pair in the request.
        pair: usize,
        /// Its left candidate.
        left: usize,
        /// Its right candidate.
        right: usize,
        /// Number of left candidates.
        n_left: usize,
        /// Number of right candidates.
        n_right: usize,
    },
    /// `n_left * n_right` entries do not fit in `usize`.
    ///
    /// Remedy: request the matrix in pieces with
    /// [`CandidateEntries::Pairs`], or split a side into smaller candidate
    /// lists.
    #[error(
        "the candidate matrix of {n_left} x {n_right} entries overflows usize; request it in \
         pieces with CandidateEntries::Pairs or split a side"
    )]
    MatrixSizeOverflow {
        /// Number of left candidates.
        n_left: usize,
        /// Number of right candidates.
        n_right: usize,
    },
}

/// Entries per GEMM block of [`CandidateEntries::All`]: the same bound TreeTCI
/// uses for the evaluator calls of one candidate matrix, so the product
/// scratch stays bounded while each block still amortizes the GEMM call.
const CANDIDATE_BLOCK_ENTRIES: usize = 65_536;

/// Order of the edges that give one split: distance of the nearer end from the
/// root, then the two ends in node order, then the orientation.
type EdgeRank<V> = (usize, V, V, V);

/// The candidates a request names on each side, in first-use order, and for a
/// pair request each pair as `(left column, right column)` into them.
struct CandidateSelection {
    left: Vec<usize>,
    right: Vec<usize>,
    request: Option<Vec<(usize, usize)>>,
}

impl<'a, V> TreeTNCachedEvaluator<'a, V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// Evaluates entries of the candidate matrix of one tree edge from the
    /// directed messages of its two sides.
    ///
    /// The value of entry `(i, j)` is the network at the global point that
    /// takes `left.candidates[i]` on the left sites and
    /// `right.candidates[j]` on the right sites. It is computed as the
    /// product of two directed messages across the edge that separates the
    /// two site sets: one per distinct left candidate toward the right side,
    /// one per distinct right candidate toward the left side. No global point
    /// is assembled per entry, and no point key is built per entry. Values
    /// equal [`Self::evaluate_batched_typed`] at the assembled points to
    /// rounding (the summation order of the final product may differ).
    ///
    /// # Arguments
    ///
    /// * `left`, `right` - The two sides. Together their `sites` must list
    ///   every position of the evaluator's index list exactly once, and the
    ///   left sites must be exactly the sites on one side of one tree edge.
    ///   That edge is inferred from them. Where site-less nodes make several
    ///   edges give the same split, the edge nearest the root is used: the
    ///   root is the first node of the plan's sorted node order, an edge's
    ///   distance is that of its nearer end, and a tie goes to the edge whose
    ///   ends come first in node order. The choice does not change the values,
    ///   only which message caches serve them.
    /// * `entries` - [`CandidateEntries::All`] or [`CandidateEntries::Pairs`].
    ///
    /// # Returns
    ///
    /// For `All`, the `n_left * n_right` values column-major with the left
    /// candidate varying fastest (entry `(i, j)` at `i + n_left * j`). The
    /// product is formed in blocks of right candidates so its scratch stays
    /// bounded, but the returned vector is the caller's full
    /// `n_left x n_right` matrix and is **not capped**: request a large
    /// matrix as slices of `Pairs` to bound it. For `Pairs`, one value per
    /// pair, in the given order. An empty side or an empty pair list gives an
    /// empty vector. Only the candidates a request names have their messages
    /// computed.
    ///
    /// # Caching
    ///
    /// The messages go through the evaluator's ordinary per-directed-edge
    /// message caches under the one shared
    /// [`CachedEvaluatorOptions::message_cache_max_bytes`] budget and its
    /// least-recently-used eviction, exactly as the messages of a hinted
    /// [`Self::evaluate_batched_typed`] call do; this method adds no cache of
    /// its own. Candidate messages are admitted like any other, so a large
    /// candidate request **can evict pivot messages** that later calls would
    /// have reused; keep requests chunked when that reuse matters. Messages
    /// are computed in internal batches of at most the evaluator's batch
    /// point limit ([`CachedEvaluatorOptions::max_batch_points`]).
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError::InvalidCandidates`] with one
    /// [`CandidateInputError`] variant per malformed request (each naming its
    /// remedy): a site position out of range, a site listed twice, a site on
    /// neither side, site sets that no edge separates, a candidate of the
    /// wrong length or with a value outside its site's dimension, a pair
    /// naming a missing candidate, and an `All` matrix whose size overflows
    /// `usize`. Returns [`TreeTNOperationError::ScalarKindMismatch`] when the
    /// tree's values cannot be decoded as `T` (the rule of
    /// [`Self::evaluate_batched_typed`]), and
    /// [`TreeTNOperationError::Operation`] when a contraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, CandidateEntries, CandidateSide, TreeTN,
    ///     TreeTNCachedEvaluator,
    /// };
    ///
    /// // f(a, b) = u[a] * w[b] on two nodes joined by a bond of dimension 1.
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let u = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let w = IdxTensor::from_dense(vec![bond, b.clone()], vec![3.0_f64, 10.0])?;
    /// let tree = TreeTN::from_tensors(vec![u, w], vec![0usize, 1])?;
    /// let mut evaluator =
    ///     TreeTNCachedEvaluator::new(&tree, &[a, b], CachedEvaluatorOptions::default())?;
    ///
    /// let left_candidates = vec![vec![0], vec![1]];
    /// let right_candidates = vec![vec![0], vec![1]];
    /// let left = CandidateSide { sites: &[0], candidates: &left_candidates };
    /// let right = CandidateSide { sites: &[1], candidates: &right_candidates };
    ///
    /// // Column-major, left fastest: (0,0), (1,0), (0,1), (1,1).
    /// let all = evaluator.evaluate_candidates_typed::<f64>(left, right, CandidateEntries::All)?;
    /// assert_eq!(all, vec![3.0, 6.0, 10.0, 20.0]);
    ///
    /// // Only the requested pairs, in the given order.
    /// let pairs = [(1, 1), (0, 0)];
    /// let some =
    ///     evaluator.evaluate_candidates_typed::<f64>(left, right, CandidateEntries::Pairs(&pairs))?;
    /// assert_eq!(some, vec![20.0, 3.0]);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_candidates_typed<T: TensorElement>(
        &mut self,
        left: CandidateSide<'_>,
        right: CandidateSide<'_>,
        entries: CandidateEntries<'_>,
    ) -> std::result::Result<Vec<T>, TreeTNOperationError> {
        self.evaluate_candidates_typed_with_parallelism(left, right, entries, Parallelism::Serial)
    }

    /// Evaluates candidate-matrix entries like
    /// [`Self::evaluate_candidates_typed`], on the parallelism the caller hands
    /// in.
    ///
    /// With [`Parallelism::Pool`], the products of
    /// [`CandidateEntries::Pairs`] are split into at most `budget` shards of
    /// whole entries that run on that pool; each entry is still summed in the
    /// serial order, so the result is bit-identical to [`Parallelism::Serial`].
    /// The block products of [`CandidateEntries::All`] and the message
    /// contractions run on the calling thread, as the message path of
    /// [`Self::evaluate_batched_typed_with_parallelism`] does. No global,
    /// thread-local or ambient pool is consulted.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::evaluate_candidates_typed`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::interpolation::Parallelism;
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, CandidateEntries, CandidateSide, TreeTN,
    ///     TreeTNCachedEvaluator,
    /// };
    ///
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(3);
    /// let bond = DynIndex::new_dyn(1);
    /// let u = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let w = IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0_f64, 10.0, 100.0])?;
    /// let tree = TreeTN::from_tensors(vec![u, w], vec![0usize, 1])?;
    /// let mut evaluator =
    ///     TreeTNCachedEvaluator::new(&tree, &[a, b], CachedEvaluatorOptions::default())?;
    ///
    /// let left_candidates = vec![vec![0], vec![1]];
    /// let right_candidates = vec![vec![0], vec![1], vec![2]];
    /// let left = CandidateSide { sites: &[0], candidates: &left_candidates };
    /// let right = CandidateSide { sites: &[1], candidates: &right_candidates };
    /// let pairs = [(0, 2), (1, 0), (1, 2), (0, 1)];
    ///
    /// let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    /// let on_pool = evaluator.evaluate_candidates_typed_with_parallelism::<f64>(
    ///     left,
    ///     right,
    ///     CandidateEntries::Pairs(&pairs),
    ///     Parallelism::Pool { pool: &pool, budget: 2 },
    /// )?;
    /// assert_eq!(on_pool, vec![100.0, 2.0, 200.0, 10.0]);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_candidates_typed_with_parallelism<T: TensorElement>(
        &mut self,
        left: CandidateSide<'_>,
        right: CandidateSide<'_>,
        entries: CandidateEntries<'_>,
        parallelism: Parallelism<'_>,
    ) -> std::result::Result<Vec<T>, TreeTNOperationError> {
        let cached = self.evaluate_candidates_cached(
            left,
            right,
            entries,
            parallelism,
            CANDIDATE_BLOCK_ENTRIES,
        )?;
        cached_values_into_typed(cached).map_err(TreeTNOperationError::from)
    }

    /// The candidate evaluation in the evaluator's lightweight scalars, with
    /// the `All` block size as a parameter so tests can force several blocks.
    pub(super) fn evaluate_candidates_cached(
        &mut self,
        left: CandidateSide<'_>,
        right: CandidateSide<'_>,
        entries: CandidateEntries<'_>,
        parallelism: Parallelism<'_>,
        block_entries: usize,
    ) -> std::result::Result<Vec<CachedScalar>, TreeTNOperationError> {
        self.validate_candidate_sites(left.sites, right.sites)?;
        let (from, to) = self
            .candidate_edge(left.sites)
            .ok_or(CandidateInputError::NotAnEdgeSplit)?;
        self.validate_candidates("left", left)?;
        self.validate_candidates("right", right)?;

        let (n_left, n_right) = (left.candidates.len(), right.candidates.len());
        let CandidateSelection {
            left: left_selected,
            right: right_selected,
            request,
        } = match entries {
            CandidateEntries::All => {
                checked_matrix_len(n_left, n_right)?;
                CandidateSelection {
                    left: (0..n_left).collect(),
                    right: (0..n_right).collect(),
                    request: None,
                }
            }
            CandidateEntries::Pairs(pairs) => select_pair_columns(pairs, n_left, n_right)?,
        };
        self.last_stats = CachedEvaluationStats::default();
        let (Some(&left_filler), Some(&right_filler)) =
            (left_selected.first(), right_selected.first())
        else {
            return Ok(Vec::new());
        };

        let kind = self.tree_scalar_kind()?;
        let bond_dim = self.candidate_bond_dim(&from, &to)?;
        let mut route = CachedEvaluationRoute::global();
        let left_columns = self.candidate_messages(
            &from,
            &to,
            left,
            &left_selected,
            (right, right_filler),
            &mut route,
        )?;
        let right_columns = self.candidate_messages(
            &to,
            &from,
            right,
            &right_selected,
            (left, left_filler),
            &mut route,
        )?;
        let product = CandidateProduct {
            bond_dim,
            n_left: left_selected.len(),
            n_right: right_selected.len(),
            request: request.as_deref(),
            block_entries,
            parallelism,
        };
        let values = match kind {
            ScalarKind::F32 => product
                .assemble(
                    &left_columns,
                    &right_columns,
                    &mut route,
                    |value| match value {
                        CachedScalar::F32(value) => Some(*value),
                        _ => None,
                    },
                )?
                .into_iter()
                .map(CachedScalar::F32)
                .collect(),
            ScalarKind::F64 => product
                .assemble(&left_columns, &right_columns, &mut route, |value| {
                    value.as_real()
                })?
                .into_iter()
                .map(CachedScalar::F64)
                .collect(),
            ScalarKind::C32 => product
                .assemble(
                    &left_columns,
                    &right_columns,
                    &mut route,
                    |value| match value {
                        CachedScalar::F32(value) => Some(Complex32::new(*value, 0.0)),
                        CachedScalar::C32(value) => Some(*value),
                        _ => None,
                    },
                )?
                .into_iter()
                .map(CachedScalar::C32)
                .collect(),
            ScalarKind::C64 => product
                .assemble(&left_columns, &right_columns, &mut route, |value| {
                    Some(value.as_complex())
                })?
                .into_iter()
                .map(CachedScalar::C64)
                .collect(),
        };
        Ok(values)
    }

    /// Range, uniqueness and coverage of the two site lists.
    fn validate_candidate_sites(
        &self,
        left: &[usize],
        right: &[usize],
    ) -> std::result::Result<(), CandidateInputError> {
        let n_indices = self.layout().n_indices;
        let mut seen = vec![false; n_indices];
        for (side, sites) in [("left", left), ("right", right)] {
            for &site in sites {
                let slot = seen
                    .get_mut(site)
                    .ok_or(CandidateInputError::SiteOutOfRange {
                        side,
                        site,
                        n_indices,
                    })?;
                if *slot {
                    return Err(CandidateInputError::DuplicateSite { site });
                }
                *slot = true;
            }
        }
        match seen.iter().position(|&seen| !seen) {
            Some(site) => Err(CandidateInputError::MissingSite { site }),
            None => Ok(()),
        }
    }

    /// Length and value range of every candidate of one side.
    fn validate_candidates(
        &self,
        side: &'static str,
        candidates: CandidateSide<'_>,
    ) -> std::result::Result<(), CandidateInputError> {
        let indices = &self.plan.inner.indices;
        for (candidate, values) in candidates.candidates.iter().enumerate() {
            if values.len() != candidates.sites.len() {
                return Err(CandidateInputError::CandidateLength {
                    side,
                    candidate,
                    len: values.len(),
                    expected: candidates.sites.len(),
                });
            }
            for (&site, &value) in candidates.sites.iter().zip(values) {
                let dim = indices[site].dim();
                if value >= dim {
                    return Err(CandidateInputError::CandidateValueOutOfRange {
                        side,
                        candidate,
                        site,
                        value,
                        dim,
                    });
                }
            }
        }
        Ok(())
    }

    /// The directed edge `(from, to)` whose `from` side holds exactly
    /// `left_sites`; among several (site-less nodes), the one nearest the
    /// plan's root, as documented on [`Self::evaluate_candidates_typed`].
    fn candidate_edge(&self, left_sites: &[usize]) -> Option<(V, V)> {
        let mut wanted = left_sites.to_vec();
        wanted.sort_unstable();
        let root = self.plan.inner.sorted_node_names.first()?;
        let (parent, order) = rooted_tree(&self.plan.inner.neighbors, root).ok()?;
        let mut depth = HashMap::<V, usize>::default();
        for node in order {
            let node_depth = match parent.get(&node).and_then(Option::as_ref) {
                Some(up) => depth.get(up).map_or(0, |d| d + 1),
                None => 0,
            };
            depth.insert(node, node_depth);
        }
        let mut best: Option<(EdgeRank<V>, (V, V))> = None;
        for ((from, to), component) in self.directed_component_layouts() {
            let mut sites = component.layout.input_positions.clone();
            sites.sort_unstable();
            if sites != wanted {
                continue;
            }
            let distance = depth.get(from)?.min(depth.get(to)?);
            let (near, far) = if from < to { (from, to) } else { (to, from) };
            // The orientation breaks the tie only when both sides are empty.
            let key = (*distance, near.clone(), far.clone(), from.clone());
            if best.as_ref().is_none_or(|(best_key, _)| key < *best_key) {
                best = Some((key, (from.clone(), to.clone())));
            }
        }
        best.map(|(_, edge)| edge)
    }

    /// Promoted scalar kind of all node tensors: the kind of every value.
    fn tree_scalar_kind(&self) -> Result<ScalarKind> {
        let mut kind = None;
        for node in &self.plan.inner.sorted_node_names {
            let node_kind = tensor_scalar_kind(tensor_for_node(self.tree, node)?)?;
            kind = Some(kind.map_or(node_kind, |kind| promote_scalar_kind(kind, node_kind)));
        }
        kind.ok_or_else(|| anyhow::anyhow!("TreeTNCachedEvaluator: the tree has no nodes"))
    }

    fn candidate_bond_dim(&self, from: &V, to: &V) -> Result<usize> {
        let edge = self.tree.edge_between(from, to).ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator: no edge between {from:?} and {to:?}")
        })?;
        let bond = self.tree.bond_index(edge).ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator: missing bond index between {from:?} and {to:?}")
        })?;
        Ok(bond.dim())
    }

    /// The directed message `from -> to` of each selected candidate of `own`,
    /// bond-fastest, one column per entry of `selected`.
    ///
    /// The existing message path reads full points, so each internal batch
    /// fills the other side's sites with one valid candidate of that side
    /// (`other`): the message `from -> to` depends only on the sites behind
    /// `from`, so the filler changes no value and no cache key of this
    /// direction. Fully cached batches build only this direction's keys.
    fn candidate_messages(
        &mut self,
        from: &V,
        to: &V,
        own: CandidateSide<'_>,
        selected: &[usize],
        other: (CandidateSide<'_>, usize),
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Vec<CachedScalar>> {
        let n_indices = self.layout().n_indices;
        let bond_dim = self.candidate_bond_dim(from, to)?;
        let plan = self.rooted_plan_for_center(to)?;
        self.raw_messages = self.can_use_raw_messages(to)?;
        let mut template = vec![0usize; n_indices];
        let (other_side, filler) = other;
        for (&site, &value) in other_side.sites.iter().zip(&other_side.candidates[filler]) {
            template[site] = value;
        }
        let directed = (from.clone(), to.clone());
        let chunk_len = self.batch_point_limit.max(1);
        let mut columns = Vec::with_capacity(bond_dim.saturating_mul(selected.len()));
        let mut data = Vec::with_capacity(n_indices * chunk_len.min(selected.len()));
        for chunk in selected.chunks(chunk_len) {
            data.clear();
            for &candidate in chunk {
                let start = data.len();
                data.extend_from_slice(&template);
                for (&site, &value) in own.sites.iter().zip(&own.candidates[candidate]) {
                    data[start + site] = value;
                }
            }
            let shape = [n_indices, chunk.len()];
            let values = ColMajorArrayRef::new(&data, &shape).map_err(anyhow::Error::from)?;
            // As in the hinted edge cut: a fully cached direction needs only
            // its own keys, not the assignments of every node.
            let warm = match self.message_caches.get(&directed) {
                Some(cache) if cache.key_count() > 0 => {
                    let batch = self.build_directed_assignment_batch(from, to, values)?;
                    batch
                        .keys
                        .iter()
                        .all(|key| cache.contains(key))
                        .then_some(batch)
                }
                _ => None,
            };
            let assignments = match warm {
                Some(batch) => HashMap::from_iter([(from.clone(), batch)]),
                None => self.build_message_assignment_batches(&plan, values)?,
            };
            let mut messages = HashMap::default();
            let message = self.get_or_compute_node_message(
                from,
                values,
                &plan,
                &assignments,
                &mut messages,
                route,
            )?;
            let batch = assignments.get(from).ok_or_else(|| {
                anyhow::anyhow!("TreeTNCachedEvaluator: missing candidate assignments for {from:?}")
            })?;
            let stacked = match message.raw_values {
                Some(values) => values,
                None => tensor_values_cached(message.tensor.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("TreeTNCachedEvaluator: candidate message has no values")
                })?)?,
            };
            ensure_typed_message_storage_shape(
                &stacked,
                message.assignment_index.dim(),
                bond_dim,
                "candidate message",
            )?;
            for &assignment in &batch.point_to_assignment {
                let start = assignment * bond_dim;
                columns.extend_from_slice(&stacked[start..start + bond_dim]);
            }
        }
        Ok(columns)
    }
}

/// The product of the two sides' message columns for one request.
struct CandidateProduct<'r, 'p> {
    bond_dim: usize,
    n_left: usize,
    n_right: usize,
    /// `None` for the whole matrix, else `(left column, right column)` per
    /// requested entry.
    request: Option<&'r [(usize, usize)]>,
    block_entries: usize,
    parallelism: Parallelism<'p>,
}

impl CandidateProduct<'_, '_> {
    /// Decodes both sides into the working type `W` and forms the requested
    /// entries: block GEMMs for the whole matrix, ordered dot products for
    /// pairs.
    fn assemble<W>(
        &self,
        left: &[CachedScalar],
        right: &[CachedScalar],
        route: &mut CachedEvaluationRoute<'_, '_>,
        decode: impl Fn(&CachedScalar) -> Option<W>,
    ) -> Result<Vec<W>>
    where
        W: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = W> + Send + Sync,
    {
        let decode_all = |columns: &[CachedScalar]| {
            columns
                .iter()
                .map(|value| {
                    decode(value).ok_or_else(|| {
                        anyhow::anyhow!("candidate message scalar kind does not match the tree")
                    })
                })
                .collect::<Result<Vec<W>>>()
        };
        let (left, right) = (decode_all(left)?, decode_all(right)?);
        let chi = self.bond_dim;
        ensure!(
            left.len() == chi * self.n_left && right.len() == chi * self.n_right,
            "candidate message columns do not match their candidate counts"
        );
        let Some(request) = self.request else {
            return self.all_entries(&left, &right, route);
        };
        run_point_shards(request.len(), 1, self.parallelism, |first, end| {
            request[first..end]
                .iter()
                .map(|&(i, j)| {
                    column_dot(
                        &left[i * chi..(i + 1) * chi],
                        &right[j * chi..(j + 1) * chi],
                        |value| Some(*value),
                    )
                })
                .collect()
        })
    }

    /// `Lᵀ R` in blocks of right columns, appended in column-major order.
    fn all_entries<W>(
        &self,
        left: &[W],
        right: &[W],
        route: &mut CachedEvaluationRoute<'_, '_>,
    ) -> Result<Vec<W>>
    where
        W: BlasMul + Copy + Default,
    {
        let (chi, n_left, n_right) = (self.bond_dim, self.n_left, self.n_right);
        // The caller checked this product with `checked_matrix_len`.
        let total = n_left * n_right;
        let mut transposed = vec![W::default(); n_left * chi];
        for i in 0..n_left {
            for k in 0..chi {
                transposed[i + n_left * k] = left[k + chi * i];
            }
        }
        let left = Matrix::from_col_major_vec(n_left, chi, transposed);
        let columns_per_block = (self.block_entries / n_left).max(1);
        let mut values = Vec::with_capacity(total);
        for start in (0..n_right).step_by(columns_per_block) {
            let end = (start + columns_per_block).min(n_right);
            let block = Matrix::from_col_major_vec(
                chi,
                end - start,
                right[start * chi..end * chi].to_vec(),
            );
            values.extend(route.mat_mul(&left, &block)?.into_col_major_vec());
        }
        Ok(values)
    }
}

/// Validates `pairs` and maps them onto the candidates they name.
fn select_pair_columns(
    pairs: &[(usize, usize)],
    n_left: usize,
    n_right: usize,
) -> std::result::Result<CandidateSelection, CandidateInputError> {
    let mut left_column = vec![usize::MAX; n_left];
    let mut right_column = vec![usize::MAX; n_right];
    let (mut left_selected, mut right_selected) = (Vec::new(), Vec::new());
    let mut request = Vec::with_capacity(pairs.len());
    for (pair, &(left, right)) in pairs.iter().enumerate() {
        if left >= n_left || right >= n_right {
            return Err(CandidateInputError::PairOutOfRange {
                pair,
                left,
                right,
                n_left,
                n_right,
            });
        }
        for (candidate, column, selected) in [
            (left, &mut left_column, &mut left_selected),
            (right, &mut right_column, &mut right_selected),
        ] {
            if column[candidate] == usize::MAX {
                column[candidate] = selected.len();
                selected.push(candidate);
            }
        }
        request.push((left_column[left], right_column[right]));
    }
    Ok(CandidateSelection {
        left: left_selected,
        right: right_selected,
        request: Some(request),
    })
}

/// `n_left * n_right`, or the overflow error of an `All` request.
pub(super) fn checked_matrix_len(
    n_left: usize,
    n_right: usize,
) -> std::result::Result<usize, CandidateInputError> {
    n_left
        .checked_mul(n_right)
        .ok_or(CandidateInputError::MatrixSizeOverflow { n_left, n_right })
}

/// The kind both operands of a product promote to.
fn promote_scalar_kind(a: ScalarKind, b: ScalarKind) -> ScalarKind {
    let complex = |kind| matches!(kind, ScalarKind::C32 | ScalarKind::C64);
    let double = |kind| matches!(kind, ScalarKind::F64 | ScalarKind::C64);
    match (complex(a) || complex(b), double(a) || double(b)) {
        (false, false) => ScalarKind::F32,
        (false, true) => ScalarKind::F64,
        (true, false) => ScalarKind::C32,
        (true, true) => ScalarKind::C64,
    }
}
