//! Result values and diagnostics returned by tree ACI.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::TreeAciNode;

/// Explains why a tree ACI run stopped.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciTermination;
/// assert_eq!(TreeAciTermination::default(), TreeAciTermination::MaxSweeps);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum TreeAciTermination {
    /// The local error target, per-edge rank stability, and enabled global
    /// guard criteria were satisfied. Local residuals and randomized guard
    /// searches are not a certificate of the full-grid maximum error: the
    /// local error target is checked on the recorded per-edge residuals, each
    /// of which measures the output only at commit time and only over its own
    /// local matrix (see [`TreeAciResult::max_errors`]).
    Converged,
    /// An inaccurate edge reached its algebraic/configured rank cap, or the
    /// guard found an unresolved residual while all cuts were saturated.
    RankLimited,
    /// The configured sweep limit was reached.
    ///
    /// This includes runs whose ranks have settled except for a few
    /// threshold-limited edges that keep gaining a pivot; see the known
    /// limitation under [`TreeAciOptions::tolerance`](crate::TreeAciOptions::tolerance).
    #[default]
    MaxSweeps,
}

/// Structural and cache statistics from a tree ACI run.
///
/// Edge tuples use deterministic node-name order, making diagnostics stable
/// across equivalent insertion orders.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciDiagnostics;
/// let diagnostics = TreeAciDiagnostics::<usize>::default();
/// assert_eq!(diagnostics.evaluated_points, 0);
/// assert!(diagnostics.edge_ranks.is_empty());
/// assert!(diagnostics.candidate_set_sizes.is_empty());
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TreeAciDiagnostics<V> {
    /// Final `(lower endpoint, upper endpoint, rank)` values.
    pub edge_ranks: Vec<(V, V, usize)>,
    /// Edges whose algebraic or configured rank limit was reached.
    pub saturated_edges: Vec<(V, V)>,
    /// Total number of full input points evaluated.
    pub evaluated_points: u64,
    /// Number of immutable component-sample records retained at termination.
    pub sample_arena_records: usize,
    /// Logical bytes retained by component records and their deduplication keys.
    pub sample_arena_retained_bytes: usize,
    /// Number of directed frames retained by the frame cache at termination.
    pub frame_records: usize,
    /// Logical bytes retained by those frames.
    ///
    /// Includes directed input frames and the candidate-frame cache sharing
    /// `max_frame_bytes`. Together with the `sample_arena_*` fields this
    /// reports the state-owned retained caches. Guard evaluator messages have
    /// their separate aggregate `message_cache_max_bytes` bound. These figures
    /// are logical payload accounting, not allocator or process measurements.
    pub frame_retained_bytes: usize,
    /// Candidate component samples retained per directed cut, as
    /// `(from, to, len)`.
    ///
    /// Candidate sets are replaced when their own edge is updated and appended
    /// to by global pivot injection, so they stay bounded without an eviction
    /// policy. They are reported so that a future phase can design one against
    /// measurements should that stop being true.
    pub candidate_set_sizes: Vec<(V, V, usize)>,
}

/// The interpolated tree and convergence history produced by tree ACI.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::{TreeAciDiagnostics, TreeAciResult, TreeAciTermination};
/// use tensor4all_treetn::TreeTN;
///
/// let result = TreeAciResult::<usize> {
///     tree: TreeTN::new(),
///     max_ranks: vec![],
///     max_errors: vec![],
///     max_sample_magnitude: 0.0,
///     global_pivots_found: vec![],
///     termination: TreeAciTermination::MaxSweeps,
///     diagnostics: TreeAciDiagnostics::default(),
/// };
/// assert_eq!(result.tree.node_count(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct TreeAciResult<V: TreeAciNode> {
    /// Interpolated output with the prepared input topology and site indices.
    pub tree: TreeTN<IdxTensor, V>,
    /// Maximum output edge rank after each pass (one entry per forward or
    /// return pass).
    pub max_ranks: Vec<usize>,
    /// Maximum normalized local error after each pass (one entry per forward
    /// or return pass).
    ///
    /// Each entry is the largest recorded per-edge residual after that pass:
    /// for every edge, the LUCI residual of the latest commit on that edge,
    /// over that commit's local matrix and in the units of
    /// [`TreeAciOptions::tolerance`](crate::TreeAciOptions::tolerance)
    /// (relative to the local matrix's largest `|f|` when
    /// [`scale_tolerance`](crate::TreeAciOptions::scale_tolerance) is on).
    ///
    /// What a residual measures: the largest error of the output itself, at
    /// commit time, over every point of that commit's local matrix (not only
    /// its pivots). This holds when every bond the commit reads is in the pivot
    /// basis, that is, each bond has one basis shared by both adjacent cores
    /// and bond axis `k` denotes the `k`-th retained candidate of that bond.
    ///
    /// What a residual does **not** bound:
    ///
    /// - the error of the dense output away from the local matrices;
    /// - the output after later commits, which change it on the same points;
    /// - the output at all for residuals recorded in pass 1, whose commits read
    ///   bonds that still carry the initial cores. The return pass revisits
    ///   only the spine, so a run that stops after pass 2 still gates on the
    ///   pass-1 residuals of the edges off the spine.
    ///
    /// See [`TreeAciTermination::Converged`].
    pub max_errors: Vec<f64>,
    /// Largest `|f|` among the operator values of the local matrices the run
    /// committed, or among all values on the exact single-node path.
    ///
    /// Values sampled only by the global guard's searches are excluded, so
    /// this is the scale of the samples the interpolation itself was built
    /// from.
    pub max_sample_magnitude: f64,
    /// Number of distinct significant pivots returned by each global guard run,
    /// including previously found points that still fail revalidation.
    pub global_pivots_found: Vec<usize>,
    /// Reason execution stopped.
    pub termination: TreeAciTermination,
    /// Final structural and cache diagnostics.
    pub diagnostics: TreeAciDiagnostics<V>,
}
