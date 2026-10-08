//! Run-owned target evaluation and optional bounded memoization.

use crate::{GlobalIndexBatch, TreeTciResult};
use std::cell::{Cell, RefCell};
use tensor4all_core::{ColMajorArrayRef, MultiIndexCache};

/// Target evaluation and logical memo payload accounting for one TreeTCI run.
///
/// Counts oracle requests and successful evaluations, not FLOPs or avoided
/// contractions. Payload bytes exclude allocator/hash-table overhead and are
/// not a bound on process RSS. Continued optimization starts a new run cache.
///
/// # Examples
/// ```
/// use tensor4all_treetci::{crossinterpolate2, DefaultProposer, TreeTciGraph, TreeTciOptions};
/// let result = crossinterpolate2::<f64, _, _>(
///     |batch| Ok(vec![2.0; batch.n_points()]), vec![2, 2],
///     TreeTciGraph::linear_chain(2)?, vec![vec![0, 0]],
///     TreeTciOptions { evaluation_cache_bytes: Some(1024), seed: Some(0),
///         ..Default::default() }, None, &DefaultProposer)?;
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
    /// Logical retained key/value bytes, excluding hash-table and allocator overhead.
    pub retained_bytes: usize,
    /// Successful unique-value insertions skipped at the retained-byte limit.
    pub dropped_inserts: usize,
    /// Configured logical byte limit; `None` means memoization was disabled.
    pub retained_byte_limit: Option<usize>,
}

#[cfg(test)]
mod tests;

pub(crate) struct RunEvaluator<T, F>
where
    T: Clone + Send + Sync + 'static,
{
    evaluate: RefCell<F>,
    cache: Option<RefCell<MultiIndexCache<T>>>,
    requested: Cell<usize>,
    evaluated: Cell<usize>,
}

impl<T, F> RunEvaluator<T, F>
where
    T: Clone + Send + Sync + 'static,
    F: FnMut(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>,
{
    pub(crate) fn new(
        evaluate: F,
        local_dims: &[usize],
        limit: Option<usize>,
    ) -> TreeTciResult<Self> {
        let cache = limit
            .map(|bytes| {
                MultiIndexCache::with_retained_byte_limit(local_dims, bytes).map(RefCell::new)
            })
            .transpose()
            .map_err(anyhow::Error::new)?;
        Ok(Self {
            evaluate: RefCell::new(evaluate),
            cache,
            requested: Cell::new(0),
            evaluated: Cell::new(0),
        })
    }

    pub(crate) fn call(&self, batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>> {
        self.requested
            .set(self.requested.get().saturating_add(batch.n_points()));
        let evaluate = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<T>> {
            let values = (self.evaluate.try_borrow_mut()?)(batch)?;
            if values.len() == batch.n_points() {
                self.evaluated
                    .set(self.evaluated.get().saturating_add(batch.n_points()));
            }
            Ok(values)
        };
        match &self.cache {
            None => evaluate(batch),
            Some(cache) => {
                let shape = [batch.n_sites(), batch.n_points()];
                Ok(cache
                    .try_borrow_mut()?
                    .evaluate_batched(ColMajorArrayRef::new(batch.data(), &shape)?, |misses| {
                        let batch = GlobalIndexBatch::new(
                            misses.data(),
                            misses.shape()[0],
                            misses.shape()[1],
                        )?;
                        let values = evaluate(batch)?;
                        anyhow::ensure!(
                            values.len() == batch.n_points(),
                            "target function returned {} values for {} evaluated points",
                            values.len(),
                            batch.n_points()
                        );
                        Ok(values)
                    })
                    .map_err(|error| match error {
                        tensor4all_core::CachedBatchError::Evaluation(source) => source,
                        other => anyhow::Error::new(other),
                    })?)
            }
        }
    }

    pub(crate) fn stats(&self) -> TreeTciEvaluationStats {
        let mut stats = TreeTciEvaluationStats {
            requested_points: self.requested.get(),
            evaluated_points: self.evaluated.get(),
            ..Default::default()
        };
        if let Some(cache) = &self.cache {
            // INVARIANT: The run boundary reads statistics after all synchronous
            // evaluation calls have returned; no mutable borrow survives a call.
            let cache = cache.borrow();
            stats.cache_hits = cache.hits();
            stats.cache_misses = cache.misses();
            stats.cached_entries = cache.len();
            stats.retained_bytes = cache.retained_bytes();
            stats.dropped_inserts = cache.dropped_inserts();
            stats.retained_byte_limit = Some(cache.retained_byte_limit());
        }
        stats
    }
}
