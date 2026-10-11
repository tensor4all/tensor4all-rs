//! Greedy coordinate-descent (floating-zone) search for high-error points.
//!
//! The floating-zone walk is the strongest global pivot search used in this
//! codebase: from a starting point it repeatedly sweeps every site
//! coordinate, moving each coordinate to the value with the largest
//! interpolation error, until the error stops improving or exceeds a
//! tolerance. It is a strict generalization of the single-cross search used
//! elsewhere (one sweep with no repeats equals a cross scan).

use crate::MultiIndex;

/// Invalid search inputs or an error from a floating-zone evaluator.
///
/// [`floating_zone_walk_with_initial_error`] validates the supplied starting
/// residual before evaluating any candidate. Callback failures retain their
/// original type in [`Self::Evaluation`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::{floating_zone_walk_with_initial_error, FloatingZoneError};
/// let result = floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
///     &[2], &vec![], 0.0, 0, 1.0, |_, _| unreachable!(),
/// );
/// assert_eq!(result, Err(FloatingZoneError::StartingPointLength {
///     expected: 1, actual: 0,
/// }));
/// ```
#[derive(Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum FloatingZoneError<E> {
    /// A site's local dimension is zero.
    #[error("floating-zone site {site} has zero local dimension")]
    InvalidLocalDimension {
        /// Position of the invalid site.
        site: usize,
    },
    /// The starting point has the wrong number of coordinates.
    #[error("floating-zone starting point has length {actual}; expected {expected}")]
    StartingPointLength {
        /// Number of local dimensions.
        expected: usize,
        /// Number of supplied coordinates.
        actual: usize,
    },
    /// A starting coordinate is outside its site's local dimension.
    #[error("floating-zone coordinate {coordinate} at site {site} exceeds dimension {dimension}")]
    StartingCoordinate {
        /// Position of the invalid site.
        site: usize,
        /// Supplied coordinate.
        coordinate: usize,
        /// Exclusive upper bound for the coordinate.
        dimension: usize,
    },
    /// The supplied starting error is negative or nonfinite.
    #[error("floating-zone starting error must be finite and nonnegative; got {value}")]
    InvalidInitialError {
        /// Supplied starting error.
        value: f64,
    },
    /// The stopping tolerance is negative or NaN.
    #[error("floating-zone stopping tolerance must be nonnegative and not NaN; got {value}")]
    InvalidTolerance {
        /// Supplied stopping tolerance; positive infinity is permitted.
        value: f64,
    },
    /// The coordinate scan's vector storage cannot be represented.
    #[error("floating-zone candidate storage overflows at site {site}")]
    AllocationSizeOverflow {
        /// Position of the site whose candidate batch is too large.
        site: usize,
    },
    /// The error-magnitude callback failed.
    #[error("floating-zone evaluation failed: {0}")]
    Evaluation(#[source] E),
}

/// Walk one floating-zone search trajectory.
///
/// Mirrors `TensorCrossInterpolation.jl`'s `_floatingzone`: starting from
/// `init_p`, each sweep visits every site in order and moves that site's
/// coordinate to the value with the largest error (as measured by
/// `eval_batch`), keeping the running maximum error monotonically
/// non-decreasing. The walk stops when a sweep does not increase the
/// maximum error (the trajectory is stuck on a local maximum) or when the
/// maximum error exceeds `early_stop_tol` (the point is already
/// significant), or after `max_sweeps` sweeps as a safety bound.
///
/// # Arguments
///
/// * `local_dims` - Local dimension of each site.
/// * `init_p` - Starting multi-index; must have length `local_dims.len()`.
/// * `max_sweeps` - Upper bound on the number of coordinate sweeps. The
///   no-improvement early stop almost always fires first.
/// * `early_stop_tol` - Stop walking once the maximum error exceeds this
///   value; the caller has found a significantly wrong point.
/// * `eval_batch` - Evaluates the error magnitude `|f - tt|` at a batch of
///   multi-indices. It is called once for the starting point with a scan site
///   of `None`, and then once per site per sweep with `Some(site)` and that
///   site's `local_dims[site] - 1` *other* candidate points: the candidate
///   equal to the current pivot is not re-evaluated, because the walk already
///   knows its error from the step that moved there. The scan site is passed
///   so that a caller whose evaluator exploits scan structure can declare it
///   rather than infer it from the batch, which a single-point batch cannot
///   support.
///
/// # Returns
///
/// The final pivot and the maximum error encountered along the walk. The
/// returned error may exceed `early_stop_tol`; the caller decides whether
/// the point is significant.
///
/// # Errors
///
/// Propagates the error returned by `eval_batch` unchanged - typically an
/// operation failure or an index mismatch from the underlying evaluator.
///
/// # Examples
///
/// ```
/// use tensor4all_core::floating_zone_walk;
///
/// // A separable error surface whose maximum is the all-last-coordinate
/// // point, so a greedy coordinate walk must find it exactly.
/// let local_dims = [3usize, 4, 2];
/// let error_at = |point: &Vec<usize>| point.iter().map(|&c| c as f64).sum::<f64>();
/// let mut evaluated = 0usize;
/// let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
///     &local_dims,
///     &vec![0usize, 0, 0],
///     16,
///     f64::INFINITY,
///     |_site, points| {
///         evaluated += points.len();
///         Ok(points.iter().map(error_at).collect())
///     },
/// )?;
///
/// assert_eq!(pivot, vec![2, 3, 1]);
/// assert_eq!(error, 6.0);
/// // One point for the start, then `local_dims[site] - 1` per site scan: the
/// // coordinate the pivot already holds is never re-evaluated.
/// assert_eq!(evaluated, 1 + 2 * ((3 - 1) + (4 - 1) + (2 - 1)));
/// # Ok::<(), std::convert::Infallible>(())
/// ```
pub fn floating_zone_walk<E, Err>(
    local_dims: &[usize],
    init_p: &MultiIndex,
    max_sweeps: usize,
    early_stop_tol: f64,
    mut eval_batch: E,
) -> std::result::Result<(MultiIndex, f64), Err>
where
    E: FnMut(Option<usize>, &[MultiIndex]) -> std::result::Result<Vec<f64>, Err>,
{
    // Initial error at the starting point. This also seeds `pivot_error`,
    // which is what lets every site scan below skip the candidate the pivot
    // already holds.
    let start_errors = eval_batch(None, std::slice::from_ref(init_p))?;
    floating_zone_walk_unchecked(
        local_dims,
        init_p,
        start_errors.first().copied().unwrap_or(0.0),
        max_sweeps,
        early_stop_tol,
        eval_batch,
    )
}

/// Walk a floating-zone trajectory using an already evaluated starting error.
///
/// This shares the site order, tie breaking, and stopping rule of
/// [`floating_zone_walk`], but skips its initial `eval_batch(None, ...)` call.
/// Use it when starting points were evaluated together before the search.
/// `initial_error` must be the same finite, nonnegative error magnitude that the
/// callback would return for `init_p`. The other arguments have the meaning
/// documented on [`floating_zone_walk`]. The callback receives `Some(site)`
/// for each nonempty batch of that site's other coordinate values.
///
/// Returns the final pivot and the maximum error encountered, including
/// `initial_error`. With zero sweeps, returns the starting point and error.
///
/// # Errors
///
/// Returns [`FloatingZoneError`] before evaluation for a zero local dimension,
/// a starting-point length or coordinate mismatch, a negative or nonfinite
/// `initial_error`, a negative or NaN `early_stop_tol`, or an unrepresentable
/// candidate allocation. Positive infinity is a valid stopping tolerance.
/// Empty `local_dims` are valid with an empty starting point. Validation also
/// applies with zero sweeps. Callback errors are wrapped in
/// [`FloatingZoneError::Evaluation`] with their original value and type.
///
/// # Examples
///
/// ```
/// use tensor4all_core::floating_zone_walk_with_initial_error;
/// let mut points_evaluated = 0;
/// let (pivot, error) = floating_zone_walk_with_initial_error::<_, std::convert::Infallible>(
///     &[2, 3], &vec![0, 1], 1.0, 4, f64::INFINITY,
///     |site, points| {
///         assert!(site.is_some());
///         points_evaluated += points.len();
///         Ok(points.iter().map(|p| (p[0] + p[1]) as f64).collect())
///     },
/// )?;
/// assert_eq!((pivot, error), (vec![1, 2], 3.0));
/// assert_eq!(points_evaluated, 2 * (1 + 2));
/// # Ok::<(), tensor4all_core::FloatingZoneError<std::convert::Infallible>>(())
/// ```
pub fn floating_zone_walk_with_initial_error<E, Err>(
    local_dims: &[usize],
    init_p: &MultiIndex,
    initial_error: f64,
    max_sweeps: usize,
    early_stop_tol: f64,
    eval_batch: E,
) -> std::result::Result<(MultiIndex, f64), FloatingZoneError<Err>>
where
    E: FnMut(Option<usize>, &[MultiIndex]) -> std::result::Result<Vec<f64>, Err>,
{
    if init_p.len() != local_dims.len() {
        return Err(FloatingZoneError::StartingPointLength {
            expected: local_dims.len(),
            actual: init_p.len(),
        });
    }
    for (site, (&dimension, &coordinate)) in local_dims.iter().zip(init_p).enumerate() {
        if dimension == 0 {
            return Err(FloatingZoneError::InvalidLocalDimension { site });
        }
        if coordinate >= dimension {
            return Err(FloatingZoneError::StartingCoordinate {
                site,
                coordinate,
                dimension,
            });
        }
        if max_sweeps > 0 {
            let candidate_bytes = (dimension - 1)
                .checked_mul(std::mem::size_of::<MultiIndex>())
                .and_then(|metadata| {
                    init_p
                        .len()
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|coordinates| coordinates.checked_mul(dimension - 1))
                        .and_then(|coordinates| metadata.checked_add(coordinates))
                });
            if !candidate_bytes.is_some_and(|bytes| bytes <= isize::MAX as usize) {
                return Err(FloatingZoneError::AllocationSizeOverflow { site });
            }
        }
    }
    if !initial_error.is_finite() || initial_error < 0.0 {
        return Err(FloatingZoneError::InvalidInitialError {
            value: initial_error,
        });
    }
    if early_stop_tol.is_nan() || early_stop_tol < 0.0 {
        return Err(FloatingZoneError::InvalidTolerance {
            value: early_stop_tol,
        });
    }
    floating_zone_walk_unchecked(
        local_dims,
        init_p,
        initial_error,
        max_sweeps,
        early_stop_tol,
        eval_batch,
    )
    .map_err(FloatingZoneError::Evaluation)
}

fn floating_zone_walk_unchecked<E, Err>(
    local_dims: &[usize],
    init_p: &MultiIndex,
    initial_error: f64,
    max_sweeps: usize,
    early_stop_tol: f64,
    mut eval_batch: E,
) -> std::result::Result<(MultiIndex, f64), Err>
where
    E: FnMut(Option<usize>, &[MultiIndex]) -> std::result::Result<Vec<f64>, Err>,
{
    let n = local_dims.len();
    let mut pivot = init_p.clone();
    let mut max_error = initial_error;
    let mut pivot_error = max_error;

    for _ in 0..max_sweeps {
        let prev_max_error = max_error;
        for ipos in 0..n {
            // Candidate points: every value at this site except the one the
            // pivot already holds, the rest fixed at the current pivot
            // (updated greedily within this sweep). The skipped candidate is
            // the current pivot itself, whose error is `pivot_error`.
            let held = pivot[ipos];
            let mut points = Vec::with_capacity(local_dims[ipos].saturating_sub(1));
            for value in 0..local_dims[ipos] {
                if value == held {
                    continue;
                }
                let mut point = pivot.clone();
                point[ipos] = value;
                points.push(point);
            }
            let errors = if points.is_empty() {
                Vec::new()
            } else {
                eval_batch(Some(ipos), &points)?
            };

            // Fold in value order, with the held candidate's known error in
            // its own place, so the greedy choice is the one the full batch
            // would have made.
            let mut best_local_idx = held;
            let mut best_local_error = 0.0f64;
            let mut evaluated = errors.iter();
            for value in 0..local_dims[ipos] {
                let error = if value == held {
                    pivot_error
                } else {
                    match evaluated.next() {
                        Some(&error) => error,
                        None => break,
                    }
                };
                if error > best_local_error {
                    best_local_error = error;
                    best_local_idx = value;
                }
            }
            pivot[ipos] = best_local_idx;
            // The pivot is now the winning candidate of this scan, so its
            // error is that candidate's error.
            pivot_error = best_local_error;
            max_error = max_error.max(best_local_error);
        }

        if max_error == prev_max_error || max_error > early_stop_tol {
            break;
        }
    }

    Ok((pivot, max_error))
}

#[cfg(test)]
mod tests;
