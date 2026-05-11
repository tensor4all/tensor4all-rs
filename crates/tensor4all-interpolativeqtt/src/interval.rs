//! Interval helpers used by multiscale and adaptive interpolation.

use crate::error::{invalid_argument, Result};

/// One-dimensional closed-open interval `[start, stop)`.
///
/// The constructor sorts its endpoints so that `start <= stop`. Intervals are
/// used by multiscale interpolation and by the interpolation error estimator.
///
/// # Related Types
///
/// Use [`NInterval`] for axis-aligned multidimensional boxes.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Interval {
    start: f64,
    stop: f64,
}

impl Interval {
    /// Create a one-dimensional interval.
    ///
    /// `start` and `stop` must be finite. The values are sorted, so passing
    /// them in reverse order is allowed.
    ///
    /// # Errors
    ///
    /// Returns an error when an endpoint is not finite.
    ///
    pub(crate) fn new(start: f64, stop: f64) -> Result<Self> {
        if !start.is_finite() || !stop.is_finite() {
            return Err(invalid_argument("interval endpoints must be finite"));
        }
        Ok(Self {
            start: start.min(stop),
            stop: start.max(stop),
        })
    }

    /// Lower endpoint of the interval.
    ///
    pub(crate) fn start(&self) -> f64 {
        self.start
    }

    /// Length of the interval.
    ///
    pub(crate) fn length(&self) -> f64 {
        self.stop - self.start
    }
}

/// Axis-aligned multidimensional closed-open box.
///
/// `NInterval` stores `start` and `stop` coordinates for each dimension. It is
/// used by fused multidimensional interpolation, where splitting a box returns
/// `2^N` sub-boxes with dimension zero varying fastest.
///
/// # Related Types
///
/// [`Interval`] is the one-dimensional version.
///
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NInterval {
    start: Vec<f64>,
    stop: Vec<f64>,
}

impl NInterval {
    /// Create an axis-aligned interval from lower and upper corners.
    ///
    /// `start` and `stop` must have equal nonzero length and finite values.
    /// Endpoints are sorted component-wise.
    ///
    /// # Errors
    ///
    /// Returns an error for empty intervals, length mismatches, or non-finite
    /// coordinates.
    ///
    pub(crate) fn new(start: &[f64], stop: &[f64]) -> Result<Self> {
        if start.is_empty() {
            return Err(invalid_argument(
                "NInterval must have at least one dimension",
            ));
        }
        if start.len() != stop.len() {
            return Err(invalid_argument(format!(
                "NInterval endpoint length mismatch: {} vs {}",
                start.len(),
                stop.len()
            )));
        }
        if start.iter().chain(stop.iter()).any(|x| !x.is_finite()) {
            return Err(invalid_argument("NInterval endpoints must be finite"));
        }
        Ok(Self {
            start: start
                .iter()
                .zip(stop.iter())
                .map(|(&a, &b)| a.min(b))
                .collect(),
            stop: start
                .iter()
                .zip(stop.iter())
                .map(|(&a, &b)| a.max(b))
                .collect(),
        })
    }

    /// Number of dimensions.
    ///
    pub(crate) fn ndims(&self) -> usize {
        self.start.len()
    }

    /// Lower corner coordinates.
    ///
    pub(crate) fn start(&self) -> &[f64] {
        &self.start
    }

    /// Upper corner coordinates.
    ///
    pub(crate) fn stop(&self) -> &[f64] {
        &self.stop
    }

    /// Side lengths of the box.
    ///
    pub(crate) fn lengths(&self) -> Vec<f64> {
        self.stop
            .iter()
            .zip(self.start.iter())
            .map(|(&stop, &start)| stop - start)
            .collect()
    }

    /// Return `true` when the point lies inside the closed-open box.
    ///
    /// Each upper endpoint is excluded, matching the Julia implementation.
    ///
    pub(crate) fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.ndims()
            && point
                .iter()
                .zip(self.start.iter().zip(self.stop.iter()))
                .all(|(&x, (&start, &stop))| x >= start && x < stop)
    }

    /// Split the box along every dimension.
    ///
    /// The result has `2^N` boxes. Dimension zero varies fastest in the
    /// returned order, matching the fused quantics layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension count is too large for `usize`
    /// shifting.
    ///
    pub(crate) fn split(&self) -> Result<Vec<Self>> {
        if self.ndims() >= usize::BITS as usize {
            return Err(invalid_argument("too many dimensions to split interval"));
        }
        let side_lengths = self.lengths();
        let half_lengths: Vec<_> = side_lengths.iter().map(|len| len / 2.0).collect();
        let count = 1usize << self.ndims();
        let mut pieces = Vec::with_capacity(count);

        for mask in 0..count {
            let mut start = Vec::with_capacity(self.ndims());
            let mut stop = Vec::with_capacity(self.ndims());
            for (d, half_length) in half_lengths.iter().enumerate().take(self.ndims()) {
                let offset = if (mask >> d) & 1 == 0 {
                    0.0
                } else {
                    *half_length
                };
                start.push(self.start[d] + offset);
                stop.push(self.start[d] + offset + *half_length);
            }
            pieces.push(Self::new(&start, &stop)?);
        }
        Ok(pieces)
    }
}
