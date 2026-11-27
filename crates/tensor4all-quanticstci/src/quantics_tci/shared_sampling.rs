//! One memo shared by the component runs of a multicomponent call.
//!
//! The target of a multicomponent interpolation answers *all* components of a
//! point in one call. [`SharedSampling`] keeps those answers, keyed by the
//! quantics multi-index, so the second and later component runs only send the
//! callback the points no earlier run evaluated (#1463). The memo lives for one
//! call; nothing is shared across calls.
//!
//! The three choices below are deliberately small and each lives in one place:
//! the stats stay `pub(crate)` (`SharedSampling::stats`), the callback
//! precondition is documented on the public entry points, and the retained-byte
//! limit is the argument of [`SharedSampling::new`] (the public entry point
//! passes [`DEFAULT_RETAINED_BYTE_LIMIT`](tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT)).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use quanticsgrids::DiscretizedGrid;
use tensor4all_simplett::{SimpleTensorTrain, TTScalar};
use tensor4all_tensorbackend::FullPivLuScalar;

use super::{grid_pivots, new_memo_cache_with_limit, run_treetci_batch, site_evaluator, MemoCache};
use crate::batch::QuanticsBatch;
use crate::error::{QuanticsTCIError, Result as QtciResult};
use crate::options::QtciOptions;

/// Bytes one cached point adds beyond the table: all `n_components` values plus
/// the two reference-count words (strong and weak) of the `Arc<[V]>` allocation.
/// Saturating, so a point whose byte count overflows is simply never retained.
pub(crate) fn shared_point_payload_bytes<V>(n_components: usize) -> usize {
    n_components
        .saturating_mul(std::mem::size_of::<V>())
        .saturating_add(2 * std::mem::size_of::<usize>())
}

/// A component run's error with the component it came from, keeping the
/// original error as its source so callers can still recover it from the chain.
#[derive(Debug)]
struct ComponentError {
    label: String,
    source: anyhow::Error,
}

impl std::fmt::Display for ComponentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.label, self.source)
    }
}

impl std::error::Error for ComponentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Name the component a failed run belongs to; the variant is unchanged.
pub(crate) fn with_component_context(
    error: QuanticsTCIError,
    component: usize,
    n_components: usize,
) -> QuanticsTCIError {
    let label = format!("component {component} of {n_components}");
    match error {
        QuanticsTCIError::InvalidConfiguration { message } => {
            QuanticsTCIError::InvalidConfiguration {
                message: format!("{label}: {message}"),
            }
        }
        QuanticsTCIError::Operation { source } => QuanticsTCIError::Operation {
            source: anyhow::Error::new(ComponentError { label, source }),
        },
        other => other,
    }
}

/// One component's fit: tensor train, ranks, errors.
pub(crate) type ComponentFit<V> = (SimpleTensorTrain<V>, Vec<usize>, Vec<f64>);

/// The memo of one multicomponent call: all component values per quantics point.
pub(crate) struct SharedSampling<V>
where
    V: Clone + Send + Sync + 'static,
{
    cache: MemoCache<Arc<[V]>>,
    n_components: usize,
}

impl<V> SharedSampling<V>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
{
    /// Create the memo for `n_components` values per point of a quantics index
    /// space with `local_dims`, retaining at most `retained_byte_limit` bytes of
    /// tables plus payload.
    ///
    /// # Errors
    /// `InvalidConfiguration` when the index space is wider than the widest
    /// cache key; raised before any callback call.
    pub(crate) fn new(
        local_dims: &[usize],
        n_components: usize,
        retained_byte_limit: usize,
    ) -> QtciResult<Self> {
        let cache = new_memo_cache_with_limit::<Arc<[V]>>(local_dims, retained_byte_limit)?
            .with_value_payload_bytes(shared_point_payload_bytes::<V>(n_components));
        Ok(Self {
            cache: Rc::new(RefCell::new(cache)),
            n_components,
        })
    }

    /// Interpolate one component, serving points an earlier component already
    /// evaluated from the memo.
    ///
    /// Only the distinct misses of a request reach `f`, in one batch, which must
    /// return `n_misses * n_components` values (the components of each miss
    /// consecutively). The count is checked inside the miss callback, so a
    /// wrong count or an error from `f` inserts nothing and aborts the call.
    pub(crate) fn run_component<F>(
        &self,
        grid: &DiscretizedGrid,
        f: &F,
        component: usize,
        initial_pivots: Option<Vec<Vec<usize>>>,
        options: &QtciOptions,
        rng: &mut dyn rand::RngCore,
    ) -> QtciResult<ComponentFit<V>>
    where
        F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    {
        let n_components = self.n_components;
        let grid_for_evaluation = grid.clone();
        let evaluate = site_evaluator(
            move |quantics: &[usize]| {
                grid_for_evaluation
                    .quantics_to_origcoord(quantics)
                    .map_err(|error| {
                        anyhow!("failed to convert quantics index {quantics:?}: {error}")
                    })
            },
            |batch: QuanticsBatch<'_, f64>| -> Result<Vec<Arc<[V]>>> {
                let n_points = batch.n_points();
                if n_points == 0 {
                    return Ok(Vec::new());
                }
                let returned = f(batch)?;
                let expected = n_points.checked_mul(n_components).ok_or_else(|| {
                    anyhow!(
                        "expected value count overflows usize: {n_points} points x {n_components} components"
                    )
                })?;
                if returned.len() != expected {
                    return Err(anyhow!(
                        "callback returned {} values for {n_points} points, expected exactly {n_components} components per point ({expected} values)",
                        returned.len()
                    ));
                }
                Ok(returned.chunks(n_components).map(Arc::from).collect())
            },
            self.cache.clone(),
            move |values: &Arc<[V]>| values[component],
        );
        // The pivots are drawn before the first evaluation, so the stream
        // position after the run does not depend on what the memo serves.
        let local_dims = grid.local_dimensions();
        let run = || -> QtciResult<_> {
            let pivots = grid_pivots(
                initial_pivots,
                &local_dims,
                |pivot| grid.grididx_to_quantics(pivot),
                options,
                rng,
            )?;
            run_treetci_batch(local_dims.clone(), evaluate, pivots, options, None)
        };
        let (_tci, tt, ranks, errors) =
            run().map_err(|error| with_component_context(error, component, n_components))?;
        Ok((tt, ranks, errors))
    }

    /// The memo, for tests that read its length and retained bytes.
    #[cfg(test)]
    pub(crate) fn cache(&self) -> &MemoCache<Arc<[V]>> {
        &self.cache
    }

    /// Source-call accounting of the whole call: `num_evals` is the number of
    /// distinct points the callback evaluated (retained, evicted or dropped).
    #[cfg(test)]
    pub(crate) fn stats(&self) -> super::CacheStats {
        super::memo_cache_stats(&self.cache)
    }
}
