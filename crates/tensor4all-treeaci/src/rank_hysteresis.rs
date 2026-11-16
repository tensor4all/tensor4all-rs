//! Rank hysteresis for threshold-limited edges (#784).
//!
//! On an edge whose rank is set by the local threshold, pivots whose error
//! sits within a small factor of the threshold enter and leave from pass to
//! pass. The rank then oscillates by a few units, the per-edge rank-stability
//! window never fills, and a run ends `MaxSweeps` while its state drifts.
//!
//! Hysteresis keeps such a pivot while the edge already had that many: the
//! local search runs at `threshold / RANK_HYSTERESIS_FACTOR`, and the kept
//! rank is
//!
//! ```text
//! max(n_threshold, min(previous_rank, n_lowered))
//! ```
//!
//! where `n_threshold` counts the pivots at or above the threshold (what the
//! plain update keeps) and `n_lowered` those at or above the lowered one. The
//! pivot order, the threshold that admits new pivots, Guard and the stopping
//! rule are unchanged. Every pivot the plain update keeps is still kept, so
//! hysteresis only adds pivots and never trades away accuracy at the
//! threshold.
//!
//! The floor starts at pass [`RANK_HYSTERESIS_FIRST_PASS`]: the first pass
//! grows edges from their bootstrap ranks and can overshoot, and using those
//! ranks as the floor would keep the overshoot down to the lowered threshold.

/// Ratio between the local threshold and the lowest pivot error an edge may
/// keep under hysteresis.
///
/// Measured on the #784 gate (24 G0 cases, median over stopping passes) and
/// the original DMRG fixture: 4 converged every G0 case without an accuracy
/// loss, while 2 left cases cycling and lost accuracy on two of them.
pub(crate) const RANK_HYSTERESIS_FACTOR: f64 = 4.0;

/// First pass (1-based) whose local updates use the previous rank as a floor.
pub(crate) const RANK_HYSTERESIS_FIRST_PASS: usize = 3;

/// Whether the local updates of pass `pass_number` (1-based) apply hysteresis.
pub(crate) fn applies_to_pass(pass_number: usize) -> bool {
    pass_number >= RANK_HYSTERESIS_FIRST_PASS
}

/// The rank an edge keeps from a search run at the lowered threshold.
///
/// # Arguments
///
/// * `pivot_errors` - The search's error history: one magnitude per selected
///   pivot, in selection order, then the remaining residual. All magnitudes
///   use the same units as `threshold`.
/// * `threshold` - The local threshold of the plain update.
/// * `previous_rank` - The edge's rank before this update.
///
/// # Returns
///
/// `max(n_threshold, min(previous_rank, n_lowered))`, where `n_lowered` is
/// the number of selected pivots and `n_threshold` the number the plain
/// update would keep: all pivots before the first one below `threshold`, and
/// always the first pivot.
pub(crate) fn kept_rank(pivot_errors: &[f64], threshold: f64, previous_rank: usize) -> usize {
    let lowered = pivot_errors.len().saturating_sub(1);
    let at_threshold = (1..lowered)
        .find(|&pivot| pivot_errors[pivot] < threshold)
        .unwrap_or(lowered);
    at_threshold.max(previous_rank.min(lowered))
}

#[cfg(test)]
mod tests;
