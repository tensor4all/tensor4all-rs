use crate::error::Result as TreeTciResult;
use crate::{assemble::MultiIndex, column_2d, ncols_2d, SubtreeKey, TreeTCI2, TreeTciEdge};
use anyhow::{ensure, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use tensor4all_core::ColMajorArray;

/// Generates candidate pivot sets for one edge bipartition.
///
/// Implementors produce candidate multi-indices for both sides of an edge
/// bipartition. The optimizer evaluates the function at these candidates to
/// select new pivots.
///
/// Built-in proposers:
/// - [`DefaultProposer`] -- neighbor-product candidates (recommended default)
/// - [`SimpleProposer`] -- random candidates with deterministic seed
/// - [`TruncatedDefaultProposer`] -- truncated default candidates with random sampling
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{PivotCandidateProposer, SimpleProposer};
/// assert_eq!(SimpleProposer::seeded(42).seed(), 42);
/// ```
pub trait PivotCandidateProposer {
    /// Return `(I_candidates, J_candidates)` using a fresh seeded ChaCha8 stream.
    ///
    /// `I_candidates` are multi-indices for the left (u-side) subtree,
    /// `J_candidates` for the right (v-side) subtree.
    /// # Errors
    ///
    /// Returns an error for invalid edges, missing or malformed pivot sets,
    /// or candidate-size overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{DefaultProposer, PivotCandidateProposer, TreeTCI2, TreeTciEdge, TreeTciGraph};
    /// let edge = TreeTciEdge::new(0, 1);
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::new(2, &[edge])?)?;
    /// state.add_global_pivots(&[vec![0, 0]])?;
    /// let (left, right) = DefaultProposer.candidates(&state, edge)?;
    /// assert_eq!(left, vec![vec![0], vec![1]]);
    /// assert_eq!(right, vec![vec![0], vec![1]]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn candidates<T>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed());
        self.candidates_with_rng(state, edge, &mut rng)
    }

    /// Generate candidates using the caller's stream directly.
    ///
    /// No seed is derived and no private generator is constructed. Reuse `rng`
    /// across edges and continued optimizations to advance the same stream.
    /// The deterministic [`DefaultProposer`] consumes no draws.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid edges, missing or malformed pivot sets,
    /// or candidate-size overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    /// use tensor4all_treetci::{DefaultProposer, PivotCandidateProposer, TreeTCI2, TreeTciEdge, TreeTciGraph};
    /// let edge = TreeTciEdge::new(0, 1);
    /// let mut state = TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::new(2, &[edge])?)?;
    /// state.add_global_pivots(&[vec![0, 0]])?;
    /// let mut rng = ChaCha8Rng::seed_from_u64(7);
    /// let (left, right) = DefaultProposer.candidates_with_rng(&state, edge, &mut rng)?;
    /// assert_eq!(left, vec![vec![0], vec![1]]);
    /// assert_eq!(right, vec![vec![0], vec![1]]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)>;

    /// Seed for the high-level ChaCha8 candidate stream; defaults to zero.
    ///
    /// Caller-owned stream entry points ignore this value. High-level
    /// optimizations initialize the candidate stream once per call.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{PivotCandidateProposer, SimpleProposer};
    /// assert_eq!(SimpleProposer::seeded(42).seed(), 42);
    /// ```
    fn seed(&self) -> u64 {
        0
    }
}

/// Default neighbor-product proposer that mirrors `TreeTCI.jl`.
///
/// Generates candidates by combining existing pivots from adjacent
/// subtrees with a Kronecker expansion over the local dimension of
/// the edge endpoints. This is the recommended proposer for most use cases.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::DefaultProposer;
///
/// let proposer = DefaultProposer;
/// // Typically used with crossinterpolate2 or optimize_with_proposer
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultProposer;

impl PivotCandidateProposer for DefaultProposer {
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        _rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;

        let adjacent_vp = state.graph.adjacent_edges(vp, &[edge]);
        let in_ikeys = state.graph.edge_in_ij_keys(vp, &adjacent_vp)?;
        let ipivots = pivot_set(&state.ijset, &in_ikeys, &ikey)?;
        let isite_index = subtree_position(&ikey, vp)?;
        let iset = kronecker(&ipivots, isite_index, state.local_dims[vp])?;

        let adjacent_vq = state.graph.adjacent_edges(vq, &[edge]);
        let in_jkeys = state.graph.edge_in_ij_keys(vq, &adjacent_vq)?;
        let jpivots = pivot_set(&state.ijset, &in_jkeys, &jkey)?;
        let jsite_index = subtree_position(&jkey, vq)?;
        let jset = kronecker(&jpivots, jsite_index, state.local_dims[vq])?;

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey)?;
        let jcombined = union_with_history(jset, history, &jkey)?;
        Ok((icombined, jcombined))
    }
}

/// Simple random proposer that mirrors `TreeTCI.jl`'s
/// `SimplePivotCandidateProposer`.
///
/// Generates random candidate multi-indices using a deterministic seed.
/// Seeded calls use ChaCha8; caller-stream calls ignore the seed.
/// Useful for reproducible benchmarking or when the default proposer
/// produces too many candidates.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{PivotCandidateProposer, SimpleProposer};
///
/// // Default seed (0)
/// let p = SimpleProposer::default();
/// assert_eq!(p.seed(), 0);
///
/// // Deterministic seed for reproducibility
/// let p = SimpleProposer::seeded(42);
/// assert_eq!(p.seed(), 42);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SimpleProposer {
    seed: u64,
}

impl SimpleProposer {
    /// Construct a proposer with a deterministic base seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{PivotCandidateProposer, SimpleProposer};
    ///
    /// let proposer = SimpleProposer::seeded(123);
    /// assert_eq!(proposer.seed(), 123);
    /// ```
    pub const fn seeded(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for SimpleProposer {
    fn default() -> Self {
        Self::seeded(0)
    }
}

impl PivotCandidateProposer for SimpleProposer {
    fn seed(&self) -> u64 {
        self.seed
    }

    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;

        let ichi = state.local_dims[vp]
            .checked_mul(ncols_2d(state.ijset.get(&ikey).ok_or_else(|| {
                anyhow::anyhow!("missing pivot set for subtree key {:?}", ikey)
            })?)?)
            .ok_or_else(|| anyhow::anyhow!("left candidate count overflowed usize"))?;
        let jchi = state.local_dims[vq]
            .checked_mul(ncols_2d(state.ijset.get(&jkey).ok_or_else(|| {
                anyhow::anyhow!("missing pivot set for subtree key {:?}", jkey)
            })?)?)
            .ok_or_else(|| anyhow::anyhow!("right candidate count overflowed usize"))?;

        let iset = random_candidates(rng, state.local_dims.as_slice(), &ikey, ichi);
        let jset = random_candidates(rng, state.local_dims.as_slice(), &jkey, jchi);

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey)?;
        let jcombined = union_with_history(jset, history, &jkey)?;
        Ok((icombined, jcombined))
    }
}

/// Truncated default proposer that samples an ordered subset from the default
/// candidate set, adapted from `TreeTCI.jl`'s
/// `TruncatedDefaultPivotCandidateProposer`.
///
/// Starts from the [`DefaultProposer`] candidates but truncates them to a
/// bounded size using random sampling. Useful for large problems where the
/// default candidate set would be prohibitively large, typically at
/// branching vertices, where it is the product of the incoming bond ranks.
///
/// Each side of an edge keeps at most `max(d, 2) * r` candidates, where `r`
/// is the current rank of the edge and `d` is the local dimension of the
/// side's endpoint vertex. On vertices with sites (`d >= 2`) this is the
/// `d * r` budget of `TreeTCI.jl`. A site-free vertex (`d = 1`) still offers
/// new candidates through the product of its incoming pivot sets, so it gets
/// the growth factor of a binary site instead of `1`, which would pin the
/// bond at its current rank.
///
/// Unlike `TreeTCI.jl`, the previous-pass pivots of the edge (which the
/// default proposer appends to its candidates) are always kept, and only the
/// remaining budget is sampled. A uniform sample would drop almost all of them
/// at a branching vertex, so every update would restart from a fresh random
/// subset and the bond error would not settle. The optimization loop records
/// the pivot sets at the start of every pass and visits each edge once per
/// pass, so the kept set is exactly the edge's current pivots, on every
/// vertex: chains and vertices with sites sample differently from
/// `TreeTCI.jl` too whenever they truncate.
///
/// When the default candidates fit into the budget they are returned
/// unchanged, so the proposer then behaves exactly like [`DefaultProposer`].
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{PivotCandidateProposer, TruncatedDefaultProposer};
///
/// let proposer = TruncatedDefaultProposer::seeded(42);
/// assert_eq!(proposer.seed(), 42);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TruncatedDefaultProposer {
    seed: u64,
}

impl TruncatedDefaultProposer {
    /// Construct a proposer with a deterministic base seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{PivotCandidateProposer, TruncatedDefaultProposer};
    ///
    /// let proposer = TruncatedDefaultProposer::seeded(99);
    /// assert_eq!(proposer.seed(), 99);
    /// ```
    pub const fn seeded(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for TruncatedDefaultProposer {
    fn default() -> Self {
        Self::seeded(0)
    }
}

impl PivotCandidateProposer for TruncatedDefaultProposer {
    fn seed(&self) -> u64 {
        self.seed
    }

    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;
        let (default_i, default_j) = DefaultProposer.candidates_with_rng(state, edge, rng)?;

        let ichi = truncated_candidate_budget(state, vp, &ikey)?;
        let jchi = truncated_candidate_budget(state, vq, &jkey)?;

        let history = state.ijset_history.last();
        let ikeep = history_columns(history, &ikey)?;
        let jkeep = history_columns(history, &jkey)?;
        Ok((
            sample_ordered_candidates(&default_i, &ikeep, ichi, rng),
            sample_ordered_candidates(&default_j, &jkeep, jchi, rng),
        ))
    }
}

/// Smallest per-update growth factor [`TruncatedDefaultProposer`] grants an
/// edge side.
///
/// It equals the local dimension of a binary (quantics) site, so a site-free
/// vertex can at most double the bond rank per update, as a binary site can.
const MIN_TRUNCATED_GROWTH_FACTOR: usize = 2;

/// Candidate budget of one edge side for [`TruncatedDefaultProposer`]:
/// `max(local_dims[vertex], 2) * rank`, with `rank` the number of pivots
/// currently stored for `key`.
///
/// The budget bounds how far the rank of the edge can grow in one update, so
/// it must exceed the current rank. `TreeTCI.jl` uses `local_dims[vertex] *
/// rank`, which equals the current rank at a site-free (local dimension 1)
/// vertex: the bond can then never grow there, and the optimization stalls at
/// a large error. The candidates of a site-free vertex are the product of its
/// incoming pivot sets, which can hold up to `rank_1 * rank_2 * ...` entries,
/// so it still has new directions to offer; it gets the growth factor of a
/// binary site. Vertices with sites keep the `TreeTCI.jl` budget.
///
/// A budget based on the incoming pivot sets alone (e.g. their summed
/// sizes) is not used: the edge rank can legitimately exceed that sum (it is
/// bounded by the product), which would pin the bond again. Using the full
/// Kronecker candidate count would disable the truncation at junctions,
/// which is where it matters. The sampler never returns more candidates than
/// the default proposer offers, so the budget is capped automatically.
fn truncated_candidate_budget<T>(
    state: &TreeTCI2<T>,
    vertex: usize,
    key: &SubtreeKey,
) -> Result<usize> {
    let rank = ncols_2d(
        state
            .ijset
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", key))?,
    )?;
    let local_dim = *state
        .local_dims
        .get(vertex)
        .ok_or_else(|| anyhow::anyhow!("vertex {vertex} has no local dimension"))?;
    local_dim
        .max(MIN_TRUNCATED_GROWTH_FACTOR)
        .checked_mul(rank)
        .ok_or_else(|| anyhow::anyhow!("truncated candidate budget overflowed usize"))
}

fn subtree_position(key: &SubtreeKey, site: usize) -> Result<usize> {
    key.as_slice()
        .iter()
        .position(|&value| value == site)
        .ok_or_else(|| anyhow::anyhow!("site {} not found in subtree key {:?}", site, key))
}

/// Concatenate `values` with the previous iteration's pivots for `key`,
/// dropping duplicates and preserving first-occurrence order.
///
/// Membership is tested through a `HashSet`, mirroring `TreeTCI.jl`'s
/// hash-based `union`. A linear `Vec::contains` scan here would be
/// O(n^2), which is ruinous at a branching vertex: the candidate count is
/// the *product* of the other incident bonds' dimensions (`d * chi_1 * chi_2`)
/// rather than `d * chi` as on a chain, so n reaches O(10^5) and the
/// quadratic term dominates the whole optimization.
fn union_with_history(
    values: Vec<MultiIndex>,
    history: Option<&HashMap<SubtreeKey, ColMajorArray<usize>>>,
    key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let mut unique = Vec::with_capacity(values.len());
    let mut seen: HashSet<MultiIndex> = HashSet::with_capacity(values.len());
    for candidate in values {
        if seen.insert(candidate.clone()) {
            unique.push(candidate);
        }
    }
    if let Some(arr) = history.and_then(|history| history.get(key)) {
        for j in 0..ncols_2d(arr)? {
            let col = column_2d(arr, j)?.to_vec();
            if seen.insert(col.clone()) {
                unique.push(col);
            }
        }
    }
    Ok(unique)
}

fn pivot_set(
    ijset: &HashMap<SubtreeKey, ColMajorArray<usize>>,
    in_keys: &[SubtreeKey],
    out_key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let out_len = out_key.as_slice().len();
    let mut pivots = vec![vec![0; out_len]];

    for in_key in in_keys {
        let incoming = ijset
            .get(in_key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", in_key))?;
        let incoming_cols = ncols_2d(incoming)?;
        let next_capacity = pivots
            .len()
            .checked_mul(incoming_cols)
            .ok_or_else(|| anyhow::anyhow!("pivot-set candidate count overflowed usize"))?;
        let mut next = Vec::with_capacity(next_capacity);
        for base in &pivots {
            for j in 0..incoming_cols {
                let index = column_2d(incoming, j)?;
                ensure!(
                    index.len() == in_key.as_slice().len(),
                    "pivot length {} does not match subtree key length {}",
                    index.len(),
                    in_key.as_slice().len()
                );
                let mut merged = base.clone();
                for (&site, &value) in in_key.as_slice().iter().zip(index.iter()) {
                    let out_pos = subtree_position(out_key, site)?;
                    merged[out_pos] = value;
                }
                next.push(merged);
            }
        }
        pivots = next;
    }

    Ok(pivots)
}

fn kronecker(
    pivots: &[MultiIndex],
    site_index: usize,
    local_dim: usize,
) -> Result<Vec<MultiIndex>> {
    let capacity = pivots
        .len()
        .checked_mul(local_dim)
        .ok_or_else(|| anyhow::anyhow!("Kronecker candidate count overflowed usize"))?;
    let mut result = Vec::with_capacity(capacity);
    for pivot in pivots {
        for value in 0..local_dim {
            let mut candidate = pivot.clone();
            candidate[site_index] = value;
            result.push(candidate);
        }
    }
    Ok(result)
}

fn random_candidates<R: Rng + ?Sized>(
    rng: &mut R,
    local_dims: &[usize],
    key: &SubtreeKey,
    size: usize,
) -> Vec<MultiIndex> {
    (0..size)
        .map(|_| {
            key.as_slice()
                .iter()
                .map(|&site| rng.random_range(0..local_dims[site]))
                .collect()
        })
        .collect()
}

/// Collect the previous-pass pivots stored for `key`, which the default
/// proposer appends to its candidates.
fn history_columns(
    history: Option<&HashMap<SubtreeKey, ColMajorArray<usize>>>,
    key: &SubtreeKey,
) -> Result<HashSet<MultiIndex>> {
    let mut columns = HashSet::new();
    if let Some(arr) = history.and_then(|history| history.get(key)) {
        for j in 0..ncols_2d(arr)? {
            columns.insert(column_2d(arr, j)?.to_vec());
        }
    }
    Ok(columns)
}

/// Sample at most `max_size` of `candidates`, keeping their order.
///
/// Candidates in `keep` (the previous-pass pivots) are retained first; the
/// rest of the budget is filled by a uniform random sample of the others.
/// With an empty `keep` this is a plain ordered uniform sample.
///
/// If `keep` alone exceeds the budget, a uniform sample of it is returned.
/// The built-in optimization loop never reaches that branch: there `keep` is
/// the edge's current `r` pivots and the budget is at least `2 * r`. It is
/// defensive code for states whose history was edited by hand.
fn sample_ordered_candidates<R: Rng + ?Sized>(
    candidates: &[MultiIndex],
    keep: &HashSet<MultiIndex>,
    max_size: usize,
    rng: &mut R,
) -> Vec<MultiIndex> {
    if candidates.len() <= max_size {
        return candidates.to_vec();
    }

    let (mut selected, mut others): (Vec<usize>, Vec<usize>) =
        (0..candidates.len()).partition(|&index| keep.contains(&candidates[index]));
    if selected.len() >= max_size {
        selected.shuffle(rng);
        selected.truncate(max_size);
    } else {
        others.shuffle(rng);
        others.truncate(max_size - selected.len());
        selected.extend(others);
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}

#[cfg(test)]
mod tests;
