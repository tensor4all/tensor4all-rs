use crate::error::Result as TreeTciResult;

pub(crate) fn checked_batch_len(n_sites: usize, n_points: usize) -> TreeTciResult<usize> {
    n_sites
        .checked_mul(n_points)
        .ok_or_else(|| anyhow::anyhow!("global index batch shape product overflowed usize"))
        .map_err(Into::into)
}

/// Maximum number of points per evaluator call when TreeTCI evaluates a large
/// point set it assembles itself (edge candidate matrices, materialized site
/// tensors).
///
/// Bounds the reused index buffer to `n_sites * EVALUATION_CHUNK_POINTS`
/// entries (16 MiB at 32 sites) instead of one entry per point and site,
/// while keeping each call large enough to amortize per-call overhead of the
/// user's batch evaluator.
pub(crate) const EVALUATION_CHUNK_POINTS: usize = 65_536;

/// Evaluate `n_points` global points in calls of at most `chunk_points`
/// points, reusing one index buffer.
///
/// `fill_point` is called once per point, in point order, with that point's
/// `n_sites` slots of the buffer. The slots still hold an earlier point, so
/// `fill_point` must assign every site. `what` names the points in the error
/// raised when the evaluator returns the wrong number of values; that error
/// gives the failing call's point range and the total point count.
///
/// Returns the values in point order, exactly as one call over all points
/// would.
pub(crate) fn evaluate_points_chunked<T, F, G>(
    n_sites: usize,
    n_points: usize,
    chunk_points: usize,
    mut fill_point: G,
    evaluate: &F,
    what: &str,
) -> anyhow::Result<Vec<T>>
where
    F: Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>,
    G: FnMut(&mut [usize]),
{
    anyhow::ensure!(chunk_points > 0, "evaluation chunk size must be positive");
    anyhow::ensure!(
        n_sites > 0 && n_points > 0,
        "at least one point with one site is required"
    );
    let buffer_points = chunk_points.min(n_points);
    let mut buffer = vec![0usize; checked_batch_len(n_sites, buffer_points)?];
    let mut values = Vec::new();
    let mut done = 0;
    while done < n_points {
        let count = buffer_points.min(n_points - done);
        let data = &mut buffer[..count * n_sites];
        for point in data.chunks_exact_mut(n_sites) {
            fill_point(point);
        }
        let chunk = evaluate(GlobalIndexBatch::new(data, n_sites, count)?)?;
        anyhow::ensure!(
            chunk.len() == count,
            "batch evaluator returned {} values for {} {what} (points {}..{} of {})",
            chunk.len(),
            count,
            done,
            done + count,
            n_points
        );
        if count == n_points {
            // Single call: hand over the evaluator's buffer without a copy.
            return Ok(chunk);
        }
        if values.is_empty() {
            values.reserve_exact(n_points);
        }
        values.extend(chunk);
        done += count;
    }
    Ok(values)
}

/// Borrowed view of a global site-order batch.
///
/// The data is stored in column-major layout with shape `(n_sites, n_points)`.
/// Each column is one multi-index point, with `data[site + n_sites * point]`
/// giving the local index at `site` for `point`.
///
/// This type is the main interface for the batch evaluator closure passed to
/// [`crossinterpolate2`](crate::crossinterpolate2),
/// [`optimize_default`](crate::optimize_default),
/// [`optimize_with_proposer`](crate::optimize_with_proposer) and
/// [`to_treetn`](crate::to_treetn).
///
/// # Batch sizes
///
/// TreeTCI splits the point sets it assembles for edge candidate matrices and
/// materialized site tensors into batches of at most 65,536 points, so one
/// matrix or tensor may take several calls. Initial-pivot evaluation and the
/// global pivot search pass their point sets in a single batch each. An
/// evaluator must not rely on how points are grouped into calls.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::GlobalIndexBatch;
///
/// // 2 sites, 3 points: column-major layout
/// // point 0: (0, 1), point 1: (1, 0), point 2: (0, 0)
/// let data = vec![0, 1, 1, 0, 0, 0];
/// let batch = GlobalIndexBatch::new(&data, 2, 3).unwrap();
///
/// assert_eq!(batch.n_sites(), 2);
/// assert_eq!(batch.n_points(), 3);
/// assert_eq!(batch.get(0, 0), Some(0)); // site 0, point 0
/// assert_eq!(batch.get(1, 0), Some(1)); // site 1, point 0
/// assert_eq!(batch.get(0, 1), Some(1)); // site 0, point 1
/// assert_eq!(batch.get(0, 5), None);    // out of bounds
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GlobalIndexBatch<'a> {
    data: &'a [usize],
    n_sites: usize,
    n_points: usize,
}

impl<'a> GlobalIndexBatch<'a> {
    /// Create a borrowed batch view with column-major `(n_sites, n_points)` storage.
    ///
    /// Returns an error if `data.len() != n_sites * n_points`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::TreeTciError::Operation`] when `n_sites * n_points`
    /// overflows `usize` or `data.len()` differs from that product.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::GlobalIndexBatch;
    ///
    /// let data = vec![0, 1, 2, 3];
    /// let batch = GlobalIndexBatch::new(&data, 2, 2).unwrap();
    /// assert_eq!(batch.n_sites(), 2);
    /// assert_eq!(batch.n_points(), 2);
    ///
    /// // Wrong length produces an error
    /// assert!(GlobalIndexBatch::new(&data, 3, 2).is_err());
    /// ```
    pub fn new(data: &'a [usize], n_sites: usize, n_points: usize) -> TreeTciResult<Self> {
        let expected = checked_batch_len(n_sites, n_points)?;
        if data.len() != expected {
            return Err(anyhow::anyhow!(
                "global index batch has length {}, expected {}",
                data.len(),
                expected
            )
            .into());
        };
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow the raw column-major backing storage.
    pub fn data(&self) -> &'a [usize] {
        self.data
    }

    /// Number of sites per point.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of points in the batch.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get one value from `(site, point)` coordinates.
    ///
    /// Returns `None` if either index is out of bounds.
    pub fn get(&self, site: usize, point: usize) -> Option<usize> {
        (site < self.n_sites && point < self.n_points)
            .then(|| self.data[site + self.n_sites * point])
    }
}

#[cfg(test)]
mod tests;
