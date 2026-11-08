//! Run-owned target evaluation, execution routing and optional bounded memoization.

use crate::{GlobalIndexBatch, TreeTciResult};
use tensor4all_core::{ColMajorArrayRef, LeftSide, MultiIndexCache, RightSide, SideKeys};
use tensor4all_tensorbackend::{BackendLinalgError, FullPivLuScalar};
use tensor4all_treetn::interpolation::EvalContext;

use crate::assemble::MultiIndex;
use crate::batch::{assemble_candidate_points, candidate_entries};

/// Target evaluation and memo accounting for one TreeTCI run.
///
/// Counts oracle requests and successful evaluations, not FLOPs or avoided
/// contractions. Memo bytes are those of its hash tables, excluding allocator
/// overhead. Continued optimization starts a new run cache.
///
/// # Examples
/// ```
/// use tensor4all_treetci::{crossinterpolate2, DefaultProposer, TreeTciGraph, TreeTciOptions};
/// let result = crossinterpolate2::<f64, _, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions { evaluation_cache_bytes: Some(1024), seed: Some(0),
///         ..Default::default() }, None, &DefaultProposer, &mut ())?;
/// assert_eq!(result.evaluation.cached_entries, 4);
/// assert_eq!(result.evaluation.evaluated_points, 4);
/// assert!(result.evaluation.cache_hits > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeTciEvaluationStats {
    /// Points requested by TreeTCI, including duplicates and cache hits.
    pub requested_points: usize,
    /// Distinct misses successfully evaluated, or all points when memo is disabled.
    pub evaluated_points: usize,
    /// Persistent cache lookups served from retained values; zero when disabled.
    pub cache_hits: usize,
    /// Persistent cache lookups without a retained value, including cold-batch duplicates.
    pub cache_misses: usize,
    /// Entries retained at the end of this call, before the run cache is dropped.
    pub cached_entries: usize,
    /// Bytes of the memo's hash tables at the end of this call, excluding allocator overhead.
    pub retained_bytes: usize,
    /// Insertions skipped because the limit is below the smallest table.
    pub dropped_inserts: usize,
    /// Entries evicted to keep the memo within its limit (oldest first).
    pub evicted_entries: usize,
    /// Configured byte limit of the tables; `None` means memoization was disabled.
    pub retained_byte_limit: Option<usize>,
}

/// The per-side keys of one edge candidate matrix, prepared once per matrix.
///
/// Only a memoizing execution keys the entries; every other execution assembles
/// the points of the entries it evaluates. Both keep one scratch buffer for the
/// whole matrix.
pub(crate) enum CandidateSides {
    /// No memoization: assemble a point per entry.
    Points { buffer: Vec<usize> },
    /// Memoized: the cache's own side keys, built once for the whole matrix.
    Keyed {
        left: SideKeys<LeftSide>,
        right: SideKeys<RightSide>,
        buffer: Vec<usize>,
    },
}

/// One mutable execution owner of a TreeTCI run.
///
/// A run needs two things from its caller: the target values at batch points and the
/// session its own numerical steps use (the materialization solve, and whatever the
/// pivot readout reaches later). Both stay on the same owner so that the session has
/// exactly one mutable accessor and is reborrowed between evaluator calls and
/// numerical steps instead of being captured by a callback.
///
/// There are exactly two implementations, one per frontend:
///
/// - [`Compatibility`] keeps the context-free evaluator and the process-global route,
///   which is what the context-free public entries select.
/// - [`Explicit`] keeps the caller's [`EvalContext`] and issues the numerical steps on
///   that session, which is what a caller that owns its execution passes.
pub(crate) trait BatchExecution<T: FullPivLuScalar> {
    /// Evaluate one batch of global points.
    fn call(&mut self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>;

    /// Prepare the per-side keys of one edge candidate matrix.
    ///
    /// Called once per matrix and passed to every [`Self::call_candidate_entries`]
    /// of that matrix, so keys are built and validated once instead of once per
    /// chunk. The default prepares none, because it assembles points anyway.
    fn candidate_sides(
        &mut self,
        left: (&[usize], &[MultiIndex]),
        right: (&[usize], &[MultiIndex]),
    ) -> anyhow::Result<CandidateSides> {
        let _ = (left, right);
        Ok(CandidateSides::Points { buffer: Vec::new() })
    }

    /// Evaluate `count` consecutive entries of one edge's candidate matrix.
    ///
    /// Entry `start + offset` is the global point assembled from
    /// `left_candidates[i]` and `right_candidates[j]`, with left candidates
    /// varying fastest. `sides` is this matrix's [`Self::candidate_sides`]
    /// result. The default assembles those points and calls [`Self::call`]; an
    /// execution that memoizes overrides this to look the entries up through the
    /// prepared keys instead of assembling and re-validating a point per entry.
    fn call_candidate_entries(
        &mut self,
        sides: &mut CandidateSides,
        n_sites: usize,
        left: (&[usize], &[MultiIndex]),
        right: (&[usize], &[MultiIndex]),
        start: usize,
        count: usize,
    ) -> anyhow::Result<Vec<T>> {
        let CandidateSides::Points { buffer } = sides else {
            anyhow::bail!("this execution cannot serve keyed candidate entries");
        };
        let entries = candidate_entries(start, count, left.1.len());
        let data = assemble_candidate_points(buffer, n_sites, left, right, entries, count)?;
        self.call(GlobalIndexBatch::new(data, n_sites, count)?)
    }

    /// Solve `T * P = Pi1` for the materialization, on this owner's route.
    fn solve_right_full_piv_lu(
        &mut self,
        lhs_values: &[T],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[T],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> std::result::Result<Vec<T>, BackendLinalgError>;

    /// The concrete session this owner's numerical steps run on, when it has one.
    ///
    /// The compatibility route has no session and returns `None`; the explicit
    /// route returns its caller-owned session so a readout reached from the same
    /// run can evaluate on it instead of the process-global backend.
    fn session(&mut self) -> Option<&mut dyn tenferro_tensor::Engine>;

    /// The evaluation accounting of this owner so far.
    ///
    /// Only the memoizing owner counts; the others return the default, which is
    /// what a run without a memo reports.
    fn stats(&self) -> TreeTciEvaluationStats {
        TreeTciEvaluationStats::default()
    }
}

/// The context-free route: a batch evaluator and the process-global solve.
///
/// This is what a context-free public entry selects. It adds no execution choice of
/// its own: the evaluator is the caller's closure, and the solve is the compatibility
/// route every context-free entry already used.
pub(crate) struct Compatibility<'a, T> {
    evaluate: &'a mut dyn FnMut(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>,
}

impl<'a, T> Compatibility<'a, T> {
    pub(crate) fn new(
        evaluate: &'a mut dyn FnMut(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>,
    ) -> Self {
        Self { evaluate }
    }
}

impl<T: FullPivLuScalar> BatchExecution<T> for Compatibility<'_, T> {
    fn call(&mut self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>> {
        (self.evaluate)(batch)
    }

    fn solve_right_full_piv_lu(
        &mut self,
        lhs_values: &[T],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[T],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> std::result::Result<Vec<T>, BackendLinalgError> {
        T::solve_right_full_piv_lu(
            lhs_values,
            lhs_rows,
            lhs_cols,
            pivot_values,
            pivot_rows,
            pivot_cols,
        )
    }

    fn session(&mut self) -> Option<&mut dyn tenferro_tensor::Engine> {
        None
    }
}

/// The explicit route: a caller-owned [`EvalContext`] and a context-taking evaluator.
///
/// The evaluator receives the same context on every call, and the run's numerical
/// steps reborrow its session, so one mutable owner stays explicit for the whole run.
pub(crate) struct Explicit<'a, 'session, F> {
    context: &'a mut EvalContext<'session>,
    evaluate: F,
}

impl<'a, 'session, F> Explicit<'a, 'session, F> {
    pub(crate) fn new(context: &'a mut EvalContext<'session>, evaluate: F) -> Self {
        Self { context, evaluate }
    }
}

/// Build the run's execution owner from a session this crate entered itself.
///
/// The `_in` entries and the process-global compatibility route open one concrete
/// session for a whole run; this wraps it as an [`EvalContext`] with
/// [`Parallelism::Serial`], because a context-free entry hands its evaluator no
/// context and therefore makes no parallelism promise to the application's code.
pub(crate) fn serial_context<'a>(
    session: &'a mut tensor4all_tensorbackend::explicit::Session<'_>,
) -> EvalContext<'a> {
    EvalContext::new(
        tensor4all_tensorbackend::explicit::Session::new(session.engine()),
        tensor4all_treetn::interpolation::Parallelism::Serial,
    )
}

impl<'a, 'session, T, F> BatchExecution<T> for Explicit<'a, 'session, F>
where
    T: FullPivLuScalar,
    F: for<'batch, 'value> FnMut(
        GlobalIndexBatch<'batch>,
        &'value mut EvalContext<'session>,
    ) -> anyhow::Result<Vec<T>>,
{
    fn call(&mut self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>> {
        (self.evaluate)(batch, &mut *self.context)
    }

    fn solve_right_full_piv_lu(
        &mut self,
        lhs_values: &[T],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[T],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> std::result::Result<Vec<T>, BackendLinalgError> {
        T::solve_right_full_piv_lu_on_session(
            self.context.session().engine(),
            lhs_values,
            lhs_rows,
            lhs_cols,
            pivot_values,
            pivot_rows,
            pivot_cols,
        )
    }

    fn session(&mut self) -> Option<&mut dyn tenferro_tensor::Engine> {
        Some(self.context.session().engine())
    }
}

/// One run's target evaluation over a [`BatchExecution`], with optional bounded
/// memoization.
///
/// The cache wraps the owner rather than replacing it: misses and numerical steps go
/// to the same owner, so a memoized run keeps the caller's execution route.
pub(crate) struct CachedExecution<'a, T>
where
    T: FullPivLuScalar + Clone + Send + Sync + 'static,
{
    source: &'a mut dyn BatchExecution<T>,
    cache: Option<MultiIndexCache<T>>,
    requested: usize,
    evaluated: usize,
}

impl<'a, T> CachedExecution<'a, T>
where
    T: FullPivLuScalar + Clone + Send + Sync + 'static,
{
    pub(crate) fn new(
        source: &'a mut dyn BatchExecution<T>,
        local_dims: &[usize],
        limit: Option<usize>,
    ) -> TreeTciResult<Self> {
        let cache = limit
            .map(|bytes| MultiIndexCache::with_retained_byte_limit(local_dims, bytes))
            .transpose()
            .map_err(anyhow::Error::new)?;
        // Edge candidate matrices are served through cache-owned side keys, and
        // only u64/u128 caches build them. Fail here rather than part-way
        // through a run, and never fall back to whole-point lookups.
        if let Some(cache) = cache.as_ref().filter(|cache| !cache.supports_side_keys()) {
            return Err(
                anyhow::Error::new(tensor4all_core::CacheKeyError::SideKeyUnsupported {
                    key_type: cache.key_type(),
                })
                .into(),
            );
        }
        Ok(Self {
            source,
            cache,
            requested: 0,
            evaluated: 0,
        })
    }
}

impl<T> CachedExecution<'_, T>
where
    T: FullPivLuScalar + Clone + Send + Sync + 'static,
{
    /// Forward one already-counted evaluator call to the run's owner and add the
    /// points it actually evaluated.
    fn evaluate_and_count(&mut self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>> {
        let values = self.source.call(batch)?;
        if values.len() == batch.n_points() {
            self.evaluated = self.evaluated.saturating_add(batch.n_points());
        }
        Ok(values)
    }
}

impl<T> BatchExecution<T> for CachedExecution<'_, T>
where
    T: FullPivLuScalar + Clone + Send + Sync + 'static,
{
    fn call(&mut self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>> {
        self.requested = self.requested.saturating_add(batch.n_points());
        let shape = [batch.n_sites(), batch.n_points()];
        let Some(cache) = self.cache.as_mut() else {
            return self.evaluate_and_count(batch);
        };
        let source = &mut self.source;
        let evaluated = &mut self.evaluated;
        cache
            .evaluate_batched(ColMajorArrayRef::new(batch.data(), &shape)?, |misses| {
                let batch =
                    GlobalIndexBatch::new(misses.data(), misses.shape()[0], misses.shape()[1])?;
                let values = source.call(batch)?;
                anyhow::ensure!(
                    values.len() == batch.n_points(),
                    "target function returned {} values for {} evaluated points",
                    values.len(),
                    batch.n_points()
                );
                *evaluated = evaluated.saturating_add(batch.n_points());
                Ok(values)
            })
            .map_err(|error| match error {
                tensor4all_core::CachedBatchError::Evaluation(source) => source,
                other => anyhow::Error::new(other),
            })
    }

    fn candidate_sides(
        &mut self,
        left: (&[usize], &[MultiIndex]),
        right: (&[usize], &[MultiIndex]),
    ) -> anyhow::Result<CandidateSides> {
        // The cache owns key construction: one key per fragment of each side.
        let Some(cache) = self.cache.as_mut() else {
            return Ok(CandidateSides::Points { buffer: Vec::new() });
        };
        Ok(CandidateSides::Keyed {
            left: cache
                .left_keys(left.0, left.1)
                .map_err(anyhow::Error::new)?,
            right: cache
                .right_keys(right.0, right.1)
                .map_err(anyhow::Error::new)?,
            buffer: Vec::new(),
        })
    }

    fn call_candidate_entries(
        &mut self,
        sides: &mut CandidateSides,
        n_sites: usize,
        left: (&[usize], &[MultiIndex]),
        right: (&[usize], &[MultiIndex]),
        start: usize,
        count: usize,
    ) -> anyhow::Result<Vec<T>> {
        self.requested = self.requested.saturating_add(count);
        match sides {
            // No memo: assemble the chunk and evaluate it as one batch.
            CandidateSides::Points { buffer } => {
                let entries = candidate_entries(start, count, left.1.len());
                let data = assemble_candidate_points(buffer, n_sites, left, right, entries, count)?;
                self.evaluate_and_count(GlobalIndexBatch::new(data, n_sites, count)?)
            }
            CandidateSides::Keyed {
                left: left_keys,
                right: right_keys,
                buffer,
            } => {
                // `candidate_sides` keys the entries exactly when this run
                // memoizes, so only the misses are assembled as points.
                let Some(cache) = self.cache.as_mut() else {
                    anyhow::bail!("a keyed candidate matrix needs the run's memo cache");
                };
                let requests: Vec<(usize, usize)> =
                    candidate_entries(start, count, left_keys.len()).collect();
                let source = &mut self.source;
                let evaluated = &mut self.evaluated;
                cache
                    .evaluate_keyed(left_keys, right_keys, &requests, |misses| {
                        let data = assemble_candidate_points(
                            buffer,
                            n_sites,
                            left,
                            right,
                            misses.iter().copied(),
                            misses.len(),
                        )?;
                        let batch = GlobalIndexBatch::new(data, n_sites, misses.len())?;
                        let values = source.call(batch)?;
                        anyhow::ensure!(
                            values.len() == batch.n_points(),
                            "target function returned {} values for {} evaluated points",
                            values.len(),
                            batch.n_points()
                        );
                        *evaluated = evaluated.saturating_add(batch.n_points());
                        Ok(values)
                    })
                    .map_err(|error| match error {
                        tensor4all_core::CachedBatchError::Evaluation(source) => source,
                        other => anyhow::Error::new(other),
                    })
            }
        }
    }

    fn solve_right_full_piv_lu(
        &mut self,
        lhs_values: &[T],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[T],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> std::result::Result<Vec<T>, BackendLinalgError> {
        self.source.solve_right_full_piv_lu(
            lhs_values,
            lhs_rows,
            lhs_cols,
            pivot_values,
            pivot_rows,
            pivot_cols,
        )
    }

    fn session(&mut self) -> Option<&mut dyn tenferro_tensor::Engine> {
        self.source.session()
    }

    fn stats(&self) -> TreeTciEvaluationStats {
        let mut stats = TreeTciEvaluationStats {
            requested_points: self.requested,
            evaluated_points: self.evaluated,
            ..Default::default()
        };
        if let Some(cache) = &self.cache {
            stats.cache_hits = cache.hits();
            stats.cache_misses = cache.misses();
            stats.cached_entries = cache.len();
            stats.retained_bytes = cache.retained_bytes();
            stats.dropped_inserts = cache.dropped_inserts();
            stats.evicted_entries = cache.evicted_entries();
            stats.retained_byte_limit = Some(cache.retained_byte_limit());
        }
        stats
    }
}

#[cfg(test)]
mod tests;
