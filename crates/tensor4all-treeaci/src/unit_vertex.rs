//! Candidate expansion across vertices of local dimension 1 (#1352, #1435).
//!
//! A commit of edge `a -> b` reads as rows the local coordinate of `a` times
//! the candidate sets of `a`'s other incoming bonds, and as columns the same
//! for `b`. When `a` has local dimension 1 (a unit vertex: no site, or only
//! sites of dimension 1), it contributes no local coordinate, so the rows are
//! just the product of the other bonds' candidates. Two bonds through the same
//! unit vertex then cap each other: neither can grow past what the other
//! already offers, and every update records a residual of about zero on its
//! own local matrix while the output is wrong (#1435, mechanism 1).
//!
//! The commit therefore makes the unit vertex transparent for one of its other
//! bonds, as TreeTCI's `DefaultProposer` does since #1434 (`side_candidates`
//! in `tensor4all-treetci`):
//!
//! - **When.** The side's other bonds offer fewer than
//!   [`UNIT_VERTEX_EXPANSION_FACTOR`] times the edge's current rank (the
//!   product of their ranks). Above that the unit vertex keeps the plain
//!   product, so high-rank junctions pay nothing extra.
//! - **Which bond.** Only the largest-rank other bond `n -> a` is expanded;
//!   ties keep the first in `incoming_to_from` order. A nearly separable bond
//!   has small rank and cannot carry the missing direction.
//! - **What.** The neighbour's own side samples, `local_dim(n)` times the
//!   candidate sets of `n`'s other incoming bonds, are interned and appended
//!   after the current entries of `candidates.ids[n -> a]`, as a two-site
//!   update across the unit vertex would see them. This is the growth path
//!   Guard injection uses ([`TreeAciState::grow_candidate_sets`]).
//! - **For how long.** Only for the one commit: afterwards the expanded set is
//!   cut back to its length before the expansion. The output cores need no new
//!   shape, because the commit's factor is restricted to the bond-basis prefix
//!   of every incoming bond (`CandidateBasisProjection`), as for Guard
//!   samples.
//!
//! Only the current candidate sets of the neighbouring bonds are read. No
//! retained pivot is re-validated and no earlier local matrix or history is
//! replayed (#1372). A tree whose unit vertices are all leaves, or that has
//! none, never expands, and its runs are unchanged.

use crate::{
    frames::InputFrameStore,
    local_update::{enumerate_candidates, side_candidate_count},
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{CandidateSets, SampleArena, SampleArenaCheckpoint},
    state::TreeAciState,
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar,
};

/// A unit vertex is expanded while its other bonds offer fewer than this many
/// times the edge's current rank.
///
/// The value 2 is the slack a binary site would add. It is the untuned
/// constant of TreeTCI's unit-vertex expansion (#1434), carried over
/// unchanged; it was not tuned for tree ACI.
pub(crate) const UNIT_VERTEX_EXPANSION_FACTOR: usize = 2;

/// The incoming bonds whose candidate sets the commit of `forward` expands.
///
/// For each side of the edge (the source of `forward`, then the source of its
/// reverse) whose vertex has local dimension 1 and at least one other bond,
/// returns that side's largest-rank other bond `n -> a` when the product of
/// the other bonds' ranks is below
/// [`UNIT_VERTEX_EXPANSION_FACTOR`] times the edge's rank. Ties keep the first
/// bond in `incoming_to_from` order. The result is empty for an edge with no
/// such side, which is every edge of a tree without unit vertices.
///
/// The test reads only the tree's structure and `edge_ranks`.
///
/// # Errors
///
/// Returns [`TreeAciError::InternalInvariant`] when `forward`, a node or a
/// bond rank is unknown to `problem` / `edge_ranks`.
pub(crate) fn unit_vertex_expansion_targets<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    edge_ranks: &[usize],
    forward: DirectedEdgeId,
) -> Result<Vec<DirectedEdgeId>> {
    let rank_of = |directed: DirectedEdgeId| {
        edge_ranks
            .get(directed / 2)
            .copied()
            .ok_or(TreeAciError::InternalInvariant {
                message: "unit-vertex expansion references an unknown edge rank",
            })
    };
    let edge = problem
        .directed_edges
        .get(forward)
        .ok_or(TreeAciError::InternalInvariant {
            message: "unit-vertex expansion references an unknown directed edge",
        })?;
    let threshold = UNIT_VERTEX_EXPANSION_FACTOR.saturating_mul(rank_of(forward)?.max(1));
    let mut targets = Vec::new();
    for side in [forward, edge.reverse] {
        let directed = problem
            .directed_edges
            .get(side)
            .ok_or(TreeAciError::InternalInvariant {
                message: "unit-vertex expansion references an unknown directed edge",
            })?;
        if directed.incoming_to_from.is_empty() {
            continue;
        }
        let node =
            *problem
                .node_positions
                .get(&directed.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "unit-vertex expansion source has no prepared node position",
                })?;
        let local_dim = problem
            .physical
            .get(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "unit-vertex expansion source has no physical plan",
            })?
            .local_dim;
        if local_dim != 1 {
            continue;
        }
        // Saturating arithmetic only rules out overflow. A saturated product
        // is at least any threshold, so the side does not expand; the exact
        // comparison can disagree only when the threshold saturates too, at
        // an edge rank near `usize::MAX / 2`, which no run reaches.
        let mut offered = 1usize;
        let mut widest: Option<(DirectedEdgeId, usize)> = None;
        for &incoming in &directed.incoming_to_from {
            let rank = rank_of(incoming)?;
            offered = offered.saturating_mul(rank);
            if widest.is_none_or(|(_, widest_rank)| rank > widest_rank) {
                widest = Some((incoming, rank));
            }
        }
        if offered >= threshold {
            continue;
        }
        if let Some((incoming, _)) = widest {
            targets.push(incoming);
        }
    }
    Ok(targets)
}

/// Candidate sets one commit expanded, and what restores the state.
///
/// Returned by [`expand_unit_vertices`]. Exactly one of
/// [`Self::finish`] (after the commit) or [`Self::abandon`] (after a failed
/// proposal or commit) must consume it.
#[must_use = "an expansion must be cut back after its commit"]
pub(crate) struct UnitVertexExpansion<T> {
    /// Expanded directed edges with their candidate-set lengths before.
    restore: Vec<(DirectedEdgeId, usize)>,
    /// Arena before the expansion, for [`Self::abandon`].
    checkpoint: SampleArenaCheckpoint,
    /// The input frames the expansion replaced, for [`Self::abandon`].
    frames: InputFrameStore<T>,
}

impl<T: TreeAciScalar> UnitVertexExpansion<T> {
    /// The expanded directed edges with their candidate-set lengths before
    /// the expansion.
    #[cfg(test)]
    pub(crate) fn restore(&self) -> &[(DirectedEdgeId, usize)] {
        &self.restore
    }

    /// Cuts every expanded candidate set back to its length before the
    /// expansion, after a successful commit.
    ///
    /// A commit replaces only the two candidate sets of its own edge, never
    /// an expanded one, so the expanded entries are exactly the ones past the
    /// recorded length.
    ///
    /// The interned samples and their frames stay in the arena. The selected
    /// ones must: the commit's pivots name them. The unselected ones are
    /// interleaved with them in the expanded bond's arena, so dropping only
    /// those would renumber samples the commit's records name; they stay, and
    /// a later expansion of the same sets reuses their records and frames.
    /// What one commit leaves is bounded by the enumerated side samples of its
    /// expanded bonds, which [`expand_unit_vertices`] keeps within the
    /// candidate limits; the measured growth over long runs is in
    /// `docs/design/tree-aci.md` (#1464).
    pub(crate) fn finish<V: TreeAciNode>(self, state: &mut TreeAciState<'_, T, V>) {
        for (edge, len) in self.restore {
            if let Some(ids) = state.candidates.ids.get_mut(edge) {
                ids.truncate(len);
            }
        }
    }

    /// Undoes the expansion entirely, after a failed proposal or commit:
    /// candidate sets, input frames and sample arena are as before it, and
    /// the candidate-frame memo forgets every entry that names a sample the
    /// rollback removes (an id the arena will hand out again).
    ///
    /// # Errors
    ///
    /// Returns [`TreeAciError::InternalInvariant`] when the arena no longer
    /// extends the expansion's checkpoint.
    pub(crate) fn abandon<V: TreeAciNode>(self, state: &mut TreeAciState<'_, T, V>) -> Result<()> {
        let Self {
            restore,
            checkpoint,
            frames,
        } = self;
        for (edge, len) in restore {
            if let Some(ids) = state.candidates.ids.get_mut(edge) {
                ids.truncate(len);
            }
        }
        state.input_frames = frames;
        // The restored store shares the candidate-frame memo, which the
        // failed proposal may have filled with frames of expanded samples.
        // Their ids are about to be reused, so those entries go too.
        state
            .input_frames
            .forget_candidates_past(&state.problem, checkpoint.record_counts());
        state.sample_arena.rollback(checkpoint)
    }
}

/// Expands the candidate sets the commit of `forward` reads across a unit
/// vertex (see the module documentation).
///
/// Each target is charged against the shape limits of the commit it feeds:
/// the side it enlarges (the rows for a target on the forward side, the
/// columns for one on the reverse side) must stay within
/// [`TreeAciOptions::max_candidate_rows`] / `max_candidate_cols`, and the
/// enlarged local matrix within the run's local-matrix element ceiling. A
/// target whose expansion would exceed one of them is skipped, so the commit
/// keeps the shape it has without the expansion: the expansion never adds a
/// shape failure to a commit that fits without it. Targets are charged in
/// order, so a reverse-side target sees the forward side as already grown.
///
/// Returns `None`, with the state unchanged, when no side of the edge
/// qualifies, every qualifying target is skipped, or every expanded sample is
/// already in its candidate set. Otherwise the expanded sets and the input
/// frames over their samples are published, and the returned expansion must
/// be finished or abandoned.
///
/// # Errors
///
/// Returns [`TreeAciError::ResourceLimit`] when the interned samples exceed
/// the sample-arena budget or their frames the frame budget, and propagates
/// other frame failures. On error the state is unchanged.
pub(crate) fn expand_unit_vertices<T, V>(
    state: &mut TreeAciState<'_, T, V>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    exec: &mut crate::exec::Exec<'_, '_>,
) -> Result<Option<UnitVertexExpansion<T>>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    let targets = unit_vertex_expansion_targets(&state.problem, &state.edge_ranks, forward)?;
    if targets.is_empty() {
        return Ok(None);
    }
    // A forward-side target feeds the rows, a reverse-side one the columns.
    let edge = &state.problem.directed_edges[forward];
    let sides: Vec<_> = targets
        .iter()
        .map(|&target| {
            if edge.incoming_to_from.contains(&target) {
                TargetSide {
                    target,
                    side: forward,
                    opposite: edge.reverse,
                    limit: options.max_candidate_rows,
                }
            } else {
                TargetSide {
                    target,
                    side: edge.reverse,
                    opposite: forward,
                    limit: options.max_candidate_cols,
                }
            }
        })
        .collect();
    let checkpoint = state.sample_arena.checkpoint();
    let grown = state.grow_candidate_sets(exec, |arena, problem, candidates| {
        let mut grown = candidates.clone();
        let mut restore = Vec::with_capacity(sides.len());
        for side in &sides {
            if let Some(before) = side.expand(arena, problem, candidates, &mut grown)? {
                restore.push((side.target, before));
            }
        }
        Ok((!restore.is_empty()).then_some((grown, restore)))
    })?;
    #[cfg(test)]
    if let Some((restore, _)) = &grown {
        expansion_debug_log::record(forward, restore);
    }
    Ok(grown.map(|(restore, frames)| UnitVertexExpansion {
        restore,
        checkpoint,
        frames,
    }))
}

/// One expansion target with the side of the commit it enlarges.
struct TargetSide {
    /// The incoming bond `n -> a` whose candidate set is expanded.
    target: DirectedEdgeId,
    /// The directed edge whose side (rows or columns) holds `target`.
    side: DirectedEdgeId,
    /// The directed edge of the other side of the commit.
    opposite: DirectedEdgeId,
    /// The candidate limit of `side`.
    limit: usize,
}

impl TargetSide {
    /// Appends `target`'s expansion to `grown` when the commit still fits its
    /// shape limits, interning the new samples in `arena`.
    ///
    /// `candidates` are the sets before any expansion, from which the samples
    /// are enumerated; `grown` already holds the targets charged before this
    /// one. Returns the candidate-set length before the expansion, or `None`
    /// when nothing was appended: the expansion would exceed a shape limit
    /// (its interned samples are then rolled back), or every sample is
    /// already present.
    fn expand<V: TreeAciNode>(
        &self,
        arena: &mut SampleArena,
        problem: &PreparedTreeProblem<V>,
        candidates: &CandidateSets,
        grown: &mut CandidateSets,
    ) -> Result<Option<usize>> {
        let within = |count: Option<usize>, limit: usize| count.is_some_and(|count| count <= limit);
        // The enlarged side holds every enumerated sample, so an enumeration
        // above the limit cannot fit; skip it before materializing it.
        if !within(
            side_candidate_count(problem, candidates, self.target)?,
            self.limit,
        ) {
            return Ok(None);
        }
        let samples = enumerate_candidates(
            problem,
            candidates,
            self.target,
            "candidate expansion",
            self.limit,
        )?;
        let before = grown.ids[self.target].len();
        let undo = arena.checkpoint();
        for sample in samples {
            let id = arena.intern_component(problem, self.target, sample)?;
            grown.push_unique(self.target, id);
        }
        if grown.ids[self.target].len() == before {
            return Ok(None);
        }
        let side = side_candidate_count(problem, grown, self.side)?;
        let opposite = side_candidate_count(problem, grown, self.opposite)?;
        let elements = side
            .zip(opposite)
            .and_then(|(side, opposite)| side.checked_mul(opposite));
        if within(side, self.limit) && within(elements, problem.max_local_matrix_elements) {
            return Ok(Some(before));
        }
        grown.ids[self.target].truncate(before);
        arena.rollback(undo)?;
        Ok(None)
    }
}

/// Test-only log of the expansions made on this thread.
#[cfg(test)]
pub(crate) mod expansion_debug_log {
    use std::cell::RefCell;

    use crate::problem::DirectedEdgeId;

    /// One expansion: the edge being committed and the expanded edges with their
    /// candidate-set lengths before.
    pub(crate) type Entry = (DirectedEdgeId, Vec<(DirectedEdgeId, usize)>);

    thread_local! {
        static LOG: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn record(forward: DirectedEdgeId, restore: &[(DirectedEdgeId, usize)]) {
        LOG.with(|log| log.borrow_mut().push((forward, restore.to_vec())));
    }

    /// Returns and clears the log.
    pub(crate) fn take() -> Vec<Entry> {
        LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }
}

#[cfg(test)]
mod tests;
