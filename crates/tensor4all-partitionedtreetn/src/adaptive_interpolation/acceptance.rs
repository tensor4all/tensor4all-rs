//! The patch-size policy of the driver: acceptance of capped patches and the
//! minimum patch size.
//!
//! Sizes are generalized bits: the number of active (unfixed) sites of a
//! patch, whatever their dimensions. See
//! `docs/design/tree-pqtci-patch-size-bounds.md`.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArrayRef, CommonScalar, TensorElement};
use tensor4all_treetn::interpolation::{InterpolationTermination, TreeInterpolator};

use super::cache::{Counters, PatchSampler};
use super::options::{CappedPatches, PatchedInterpolationOptions};
use super::report::PatchStatus;
use super::verify::MeasureTarget;
use super::{accepted, internal, Driver, PatchContext, PatchedInterpolationError, Unaccepted};
use super::{Projector, Verdict};

/// The size bounds of a run, in generalized bits.
#[derive(Clone, Copy, Debug)]
pub(super) struct SizePolicy {
    min_bits: Option<usize>,
    max_capped_bits: Option<usize>,
}

impl SizePolicy {
    pub(super) fn new(options: &PatchedInterpolationOptions) -> Self {
        let max_capped_bits = match options.capped_patches {
            CappedPatches::Split => None,
            CappedPatches::AcceptUpTo { bits } => Some(bits),
        };
        Self {
            min_bits: options.min_patch_bits,
            max_capped_bits,
        }
    }

    /// Whether a run of a patch with `active_sites` active sites may be
    /// accepted on its error check although it reached the cap. Only
    /// `BondCapReached` qualifies, never `IterationLimit`.
    pub(super) fn capped_eligible(
        self,
        termination: InterpolationTermination,
        active_sites: usize,
    ) -> bool {
        termination == InterpolationTermination::BondCapReached
            && self
                .max_capped_bits
                .is_some_and(|bits| active_sites <= bits)
    }

    /// Whether the minimum blocks the split of a patch with `active_sites`
    /// active sites: every split fixes one site, so its children would have
    /// `active_sites - 1` bits.
    pub(super) fn split_blocked(self, active_sites: usize) -> bool {
        self.min_bits
            .is_some_and(|bits| active_sites.saturating_sub(1) < bits)
    }
}

/// The `max >= min` constraint: below the minimum, a capped bound would have
/// no effect, because the minimum retains those patches first.
pub(super) fn validate_bounds(options: &PatchedInterpolationOptions) -> Result<(), String> {
    match (options.min_patch_bits, options.capped_patches) {
        (Some(min), CappedPatches::AcceptUpTo { bits }) if bits < min => Err(format!(
            "capped_patches accepts capped patches up to {bits} bits, below min_patch_bits \
             ({min}); the minimum would retain those patches first, so raise the capped bound \
             to at least the minimum"
        )),
        _ => Ok(()),
    }
}

impl<T, V, E, F> Driver<'_, T, V, E, F>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    /// Retain a patch whose split the minimum blocked, with its last engine
    /// run. A run judged already keeps its verdict (no second measurement);
    /// an unjudged one (not converged, not capped-eligible) is checked like
    /// a capped run and judged once: measured on the stream of its attempt
    /// under L2, by its engine estimate under `SampledMax`.
    pub(super) fn retain_blocked(
        &self,
        ctx: &PatchContext<'_>,
        last: Unaccepted<V>,
        sampler: &PatchSampler<'_, T, F>,
        projector: &Projector,
    ) -> Result<Verdict<T, V>, PatchedInterpolationError> {
        let Unaccepted {
            outcome,
            attempt,
            rejected,
            failure,
        } = last;
        let judged = rejected.is_some();
        let subdomain = match rejected {
            Some(subdomain) => subdomain,
            None => self.checked_subdomain(&outcome, ctx.fixed, projector)?,
        };
        let mut record = self.record(projector, &outcome, &subdomain, attempt);
        let within = if self.is_l2() {
            let target = MeasureTarget {
                network: Some(subdomain.data()),
                sites: &self.layout.sites,
                fixed: ctx.fixed,
                active_dims: ctx.active_dims,
                patch_points: ctx.patch_points,
            };
            let measurement = if judged {
                failure
                    .ok_or_else(|| {
                        internal(anyhow::anyhow!("a rejected L2 run has no measurement"))
                    })?
                    .measurement
            } else {
                let plan = self.plan(ctx.patch_count, ctx.seeds.verify(attempt));
                let measured = self.measure(sampler, &target, plan, false, projector)?;
                if measured.measurement.rms > ctx.tolerance {
                    Counters::add(&self.verification_failures, 1);
                }
                measured.measurement
            };
            // The audit removes the upward bias of a sample kept because it
            // failed, like that of any sampled contribution.
            record.audit = self.audit(sampler, &target, &measurement, ctx.seeds, projector)?;
            let within = measurement.rms <= ctx.tolerance;
            record.acceptance = Some(measurement);
            within
        } else {
            !judged && outcome.error_estimate <= ctx.tolerance
        };
        if !within {
            record.status = PatchStatus::ToleranceNotMet;
        }
        Ok(accepted(record, subdomain, ctx.patch_points))
    }
}
