//! Adaptive factor on the local rrLU truncation threshold of the sampled stop
//! path (issue #1371).
//!
//! The local truncation, the global-search acceptance and the stop all use
//! `tolerance * max_sample_value`. A point whose error is spread over several
//! bonds can sit above `tolerance` while every per-bond Schur residual is
//! below it, so the local updates drop it on every sweep and the run sits at a
//! fixed point. This factor tightens only the local threshold, and only after
//! an iteration whose recorded error fails the stop comparison; the stop
//! tolerance and the search acceptance are unchanged.
//!
//! It is part of [`StopRule::Sampled`] only. Under [`StopRule::PerBond`] the
//! recorded error is the bond pivot error, which a `max_bond_dim` cap can keep
//! above `tolerance`; tightening there would change capped runs, so the factor
//! stays exactly `1.0` and the floor never fires (#1405).

use crate::StopRule;

/// Per-call state of the local threshold factor.
///
/// The factor starts at `1.0`, is halved after each iteration whose recorded
/// error is not strictly below `tolerance` (the comparison the stop uses), never
/// grows again, and is clamped at `1 / E`, where E is the number of edges of
/// the tree (at least 1). With `tolerance == 0` the local threshold is zero at
/// every factor, so the factor stays at `1.0`. Under [`StopRule::PerBond`] it
/// is disabled: the factor stays at `1.0` and [`Self::floor_window_failed`] is
/// always false.
///
/// The floor stop is armed only when the floor is below `1.0`, that is for
/// E >= 2. With E <= 1 the floor is `1.0`, the factor can never descend, and a
/// window "at the floor" would only mean untightened failing sweeps, so
/// [`Self::floor_window_failed`] stays false and the run ends as it did before
/// the adaptive threshold (`MaxIterations` or `Converged`).
#[derive(Clone, Debug)]
pub(crate) struct LocalThreshold {
    enabled: bool,
    /// Whether [`Self::floor_window_failed`] can be true: enabled and E >= 2.
    floor_armed: bool,
    factor: f64,
    floor: f64,
    /// Factor applied by the sweep of each completed iteration.
    applied: Vec<f64>,
}

impl LocalThreshold {
    /// Start at `1.0` for a tree with `n_edges` edges, under `stop_rule`.
    pub(crate) fn new(n_edges: usize, tolerance: f64, stop_rule: StopRule) -> Self {
        let adaptive = match stop_rule {
            StopRule::PerBond => false,
            StopRule::Sampled => true,
        };
        let enabled = adaptive && tolerance > 0.0;
        let floor = 1.0 / n_edges.max(1) as f64;
        Self {
            enabled,
            // At floor 1.0 the factor never descends, so there is nothing for
            // the floor stop to report.
            floor_armed: enabled && floor < 1.0,
            factor: 1.0,
            floor,
            applied: Vec::new(),
        }
    }

    /// Factor for the next sweep's local truncation threshold.
    pub(crate) fn factor(&self) -> f64 {
        self.factor
    }

    /// Factor applied by the most recent completed sweep, `1.0` before any.
    pub(crate) fn last_applied(&self) -> f64 {
        self.applied.last().copied().unwrap_or(1.0)
    }

    /// Record a completed iteration: the factor its sweep applied and its
    /// recorded error. Halves the factor, clamped at the floor, when the error
    /// is not strictly below `tolerance`.
    pub(crate) fn record(&mut self, error: f64, tolerance: f64) {
        self.applied.push(self.factor);
        if self.enabled && not_below(error, tolerance) {
            self.factor = (self.factor * 0.5).max(self.floor);
        }
    }

    /// Whether the trailing `window` iterations all swept at the floor factor
    /// and all recorded an error not strictly below `tolerance`.
    ///
    /// Always false unless the floor stop is armed (adaptive, and E >= 2, so
    /// that reaching the floor took at least one halving).
    ///
    /// `errors` is the recorded error history of the same iterations passed to
    /// [`Self::record`].
    pub(crate) fn floor_window_failed(
        &self,
        errors: &[f64],
        tolerance: f64,
        window: usize,
    ) -> bool {
        let n = errors.len();
        self.floor_armed
            && window > 0
            && n >= window
            && self.applied.len() == n
            && self.applied[n - window..]
                .iter()
                .all(|&factor| factor == self.floor)
            && errors[n - window..]
                .iter()
                .all(|&error| not_below(error, tolerance))
    }
}

/// The negation of the stop's `error < tolerance`, so a NaN error also fails.
fn not_below(error: f64, tolerance: f64) -> bool {
    error.partial_cmp(&tolerance) != Some(std::cmp::Ordering::Less)
}

#[cfg(test)]
mod tests;
